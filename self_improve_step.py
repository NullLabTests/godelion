import argparse
import datetime
import json
import os
import re
import docker

from llm import create_client, get_response_from_llm, extract_json_between_markers
from prompts.self_improvement_prompt import get_diagnose_prompt_polyglot, get_diagnose_prompt_swe, get_problem_description_prompt
from prompts.diagnose_improvement_prompt import get_diagnose_improvement_prompt
from prompts.testrepo_prompt import get_test_description
from swe_bench.harness import harness
from polyglot.harness import harness as polyglot_harness
from swe_bench.report import make_report
from utils.common_utils import load_json_file
from utils.evo_utils import get_model_patch_paths, get_all_performance, is_compiled_self_improve
from utils.docker_utils import (
    build_dgm_container as build_godelion_container,
    cleanup_container,
    copy_from_container,
    copy_to_container,
    log_container_output,
    remove_existing_container,
    setup_logger,
    safe_log,
)
from godelion.config import config

dataset = None
diagnose_model = config.get("llm", "diagnose_model", default="claude-sonnet-4-20250514")

META_COGNITIVE_PROMPT = """You are a meta-cognitive engine analyzing a self-improvement proposal for an AI coding agent.

# Improvement Proposal
{problem_statement}

# Actual Code Changes
The following is the actual diff produced by the agent:
```diff
{patch_diff}
```

# Current System Architecture Summary
{architecture_summary}

# Your Tasks
1. **Risk Assessment**: Evaluate the risk of this modification (low/medium/high).
   - Low: Minor prompt/configuration changes, safe defaults
   - Medium: New tool or workflow that doesn't affect existing functionality
   - High: Changes to core agent loop, safety mechanisms, or evaluation pipeline

2. **Failure Mode Analysis**: Identify potential failure modes:
   - Could this improvement cause regressions on existing benchmarks?
   - Could it introduce infinite loops or excessive token usage?
   - Could it break the Docker sandbox or safety mechanisms?
   - Is there a simpler alternative that achieves the same goal?

3. **Impact Assessment**: Estimate the improvement's effect on:
   - pass@1 accuracy on coding benchmarks
   - Code quality and maintainability
   - Safety and containment
   - Runtime efficiency (token usage, wall time)

4. **Patch-Problem Alignment**: Compare the actual code changes against the diagnosed problem:
   - Does the patch actually address the diagnosed problem?
   - Are the changes proportional to the problem statement?
   - Could there be unintended side effects beyond the intended change?

5. **Validation Strategy**: What specific tests would validate this improvement works?

Respond precisely in this JSON format:
```json
{{
    "risk_level": "low|medium|high",
    "risk_reasoning": "Detailed reasoning for risk assessment",
    "failure_modes": ["list of potential failure modes"],
    "impact_pass_at_1": -1.0 to 1.0,
    "impact_code_quality": -1.0 to 1.0,
    "impact_safety": -1.0 to 1.0,
    "impact_efficiency": -1.0 to 1.0,
    "patch_problem_alignment": 0.0 to 1.0,
    "patch_alignment_reasoning": "Does the patch actually solve the diagnosed problem?",
    "validation_strategy": "How to validate this improvement",
    "has_regression_risk": true/false,
    "has_safety_concern": true/false,
    "recommendation": "approve|flag|reject",
    "recommendation_reasoning": "Why this recommendation"
}}
```

Think carefully about the architecture and potential side effects."""

def get_improvement_history(output_dir: str, commit: str, max_history: int = 5) -> list:
    """Walk lineage and collect improvement diagnosis history.

    Reads the improvement_diagnosis from each ancestor's metadata.json
    and returns chronologically ordered history entries with score,
    improvements, and regressions text.
    """
    history = []
    visited = set()
    current = commit
    while current != 'initial' and current not in visited and len(history) < max_history:
        visited.add(current)
        meta_path = os.path.join(output_dir, current, "metadata.json")
        try:
            meta = load_json_file(meta_path)
            diagnosis = meta.get('improvement_diagnosis')
            if diagnosis and isinstance(diagnosis, dict):
                score = diagnosis.get('score')
                if score is not None:
                    history.append({
                        'score': score,
                        'improvements': diagnosis.get('improvements', ''),
                        'regressions': diagnosis.get('regressions', ''),
                    })
            current = meta.get('parent_commit', 'initial')
        except Exception:
            break
    return history


def format_improvement_history(history: list) -> str:
    """Format improvement history for inclusion in diagnosis prompts."""
    if not history:
        return ""

    parts = ["## Past Self-Improvement Attempts in This Lineage\n"]
    positive = sum(1 for h in history if h.get('score', 0) > 0)
    negative = sum(1 for h in history if h.get('score', 0) < 0)
    parts.append(f"Of {len(history)} past self-modifications in this lineage: "
                 f"{positive} were net-positive, {negative} were net-negative.\n")

    for i, h in enumerate(reversed(history)):
        score = h.get('score', 0)
        label = "IMPROVEMENT" if score > 0 else ("REGRESSION" if score < 0 else "NEUTRAL")
        parts.append(f"### Attempt {i+1} ({label}, score={score:.1f})")
        imp = h.get('improvements', '')
        reg = h.get('regressions', '')
        if imp:
            parts.append(f"- What worked: {imp[:300]}")
        if reg:
            parts.append(f"- What went wrong: {reg[:300]}")
        parts.append("")

    return "\n".join(parts)


def diagnose_problem(entry, commit, root_dir, out_dir, patch_files=[], max_attempts=3, polyglot=False):
    client = create_client(diagnose_model)
    if polyglot:
        diagnose_sys_message, diagnose_prompt = get_diagnose_prompt_polyglot(
            entry, commit, root_dir, out_dir, dataset,
            patch_files=patch_files,
        )
    else:
        diagnose_sys_message, diagnose_prompt = get_diagnose_prompt_swe(
            entry, commit, root_dir, out_dir, dataset,
            patch_files=patch_files,
        )

    # Inject lineage improvement history into the system message
    improvement_history = get_improvement_history(out_dir, commit)
    if improvement_history:
        history_text = format_improvement_history(improvement_history)
        diagnose_sys_message = f"{diagnose_sys_message}\n\n{history_text}"

    try:
        response, msg_history = get_response_from_llm(
            msg=diagnose_prompt,
            client=client[0],
            model=client[1],
            system_message=diagnose_sys_message,
            print_debug=False,
            msg_history=None,
        )
        safe_log(f"Message history: {msg_history}")
        response_json = extract_json_between_markers(response)
        assert response_json, "empty response json"
        problem_statement = get_problem_description_prompt(response_json, polyglot)
    except Exception as e:
        # Exception most probably due to not having json in the response
        safe_log(f"Error while diagnosing the problem: {e}")
        if max_attempts > 0:
            return diagnose_problem(
                entry, commit, root_dir, out_dir,
                patch_files=patch_files,
                max_attempts=max_attempts-1,
                polyglot=polyglot,
            )
        else:
            return None
    return problem_statement

def diagnose_improvement(
        entry, parent_commit, root_dir, model_patch_file, out_dir, run_id,
        patch_files=[], max_attempts=3,
    ):
    """
    Diagnose the improvement of the model patch.

    Args:
        entry (str): The task entry to improve.
        parent_commit (str): The commit hash of the parent commit.
        root_dir (str): The root directory of the repository.
        model_patch_file (str): The path to the model patch file.
        out_dir (str): The output directory.
        run_id (str): The run id of the self-improvement attempt.
        patch_files (list): The list of patch files before self-improvement.
        max_attempts (int): The maximum number of attempts to diagnose the improvement.
    
    Returns:
        dict: The improvement diagnosis.
    """
    client = create_client(diagnose_model)
    diagnose_sys_message, diagnose_prompt = get_diagnose_improvement_prompt(
        entry, parent_commit, root_dir, model_patch_file, out_dir, run_id, dataset,
        patch_files=patch_files,
    )
    safe_log(f"Diagnosing the improvement: {diagnose_prompt}")
    try:
        response, msg_history = get_response_from_llm(
            msg=diagnose_prompt,
            client=client[0],
            model=client[1],
            system_message=diagnose_sys_message,
            print_debug=False,
            msg_history=None,
        )
        safe_log(f"Message history: {msg_history}")
        response_json = extract_json_between_markers(response)
        assert response_json, "empty response json"
        improvement_diagnosis = response_json
    except Exception as e:
        # Exception most probably due to not having json in the response
        safe_log(f"Error while diagnosing the improvement: {e}")
        if max_attempts > 0:
            return diagnose_improvement(
                entry, parent_commit, root_dir, model_patch_file, out_dir, run_id,
                patch_files=patch_files, max_attempts=max_attempts-1,
            )
        else:
            return None
    return improvement_diagnosis

def get_architecture_summary() -> str:
    """Generate a summary of the current system architecture for meta-cognitive analysis.

    Reads key source files and reports their line counts and roles.
    """
    parts = []
    key_files = {
        "coding_agent.py": "Main agent loop",
        "coding_agent_polyglot.py": "Polyglot agent variant",
        "tools/bash.py": "Bash execution tool",
        "tools/edit.py": "File editing tool",
        "prompts/diagnose_improvement_prompt.py": "Improvement diagnosis prompt",
        "prompts/self_improvement_prompt.py": "Self-improvement diagnosis prompts",
        "prompts/tooluse_prompt.py": "Tool usage instructions",
        "prompts/testrepo_prompt.py": "Test execution instructions",
        "llm_withtools.py": "LLM tool-calling interface",
        "llm.py": "LLM client factory",
        "self_improve_step.py": "Self-improvement orchestration",
        "run.py": "Evolution outer loop",
        "utils/docker_utils.py": "Docker sandbox management",
        "utils/evo_utils.py": "Evolution utilities",
        "utils/eval_utils.py": "Evaluation scoring",
        "utils/git_utils.py": "Git operations",
        "utils/common_utils.py": "Common file utilities",
    }
    for fpath, desc in key_files.items():
        full = os.path.join(os.path.abspath('.'), fpath)
        if os.path.exists(full):
            with open(full) as f:
                lines = len(f.readlines())
            parts.append(f"- `{fpath}` ({desc}): {lines} lines")
        else:
            parts.append(f"- `{fpath}` ({desc}): [not present]")
    return "\n".join(parts)


def validate_improvement_proposal(problem_statement: str, patch_file: str | None = None, max_attempts: int = 2) -> tuple:
    """Meta-cognitive validation of an improvement proposal before running the full eval cycle.

    Uses an LLM to assess risk, failure modes, and impact of the proposed change.
    If patch_file is provided, includes the actual diff in the analysis so the
    validator can check whether the patch actually addresses the diagnosed problem.

    Returns (approved, analysis) where approved is False if recommendation is 'reject',
    and analysis is the parsed JSON dict or None on failure.
    """
    patch_diff = ""
    if patch_file and os.path.exists(patch_file):
        try:
            with open(patch_file) as f:
                content = f.read()
                # Truncate very large patches to avoid token overflow
                if len(content) > 50000:
                    content = content[:50000] + "\n... [patch truncated]"
                patch_diff = content
        except Exception as e:
            safe_log(f"Could not read patch file for meta-cognitive validation: {e}")

    arch_summary = get_architecture_summary()
    prompt = META_COGNITIVE_PROMPT.format(
        problem_statement=problem_statement,
        patch_diff=patch_diff,
        architecture_summary=arch_summary,
    )
    client = create_client(diagnose_model)
    for attempt in range(max_attempts):
        try:
            response, _ = get_response_from_llm(
                msg=prompt,
                client=client[0],
                model=client[1],
                system_message="You are a safety-conscious meta-cognitive analyzer for self-improving AI systems. Always prioritize safety and robustness.",
                print_debug=False,
            )
            analysis = extract_json_between_markers(response)
            if analysis and "recommendation" in analysis:
                safe_log(f"Meta-cognitive analysis: risk={analysis.get('risk_level')}, "
                         f"recommendation={analysis.get('recommendation')}, "
                         f"pass@1 impact={analysis.get('impact_pass_at_1')}, "
                         f"alignment={analysis.get('patch_problem_alignment')}")
                approved = analysis.get("recommendation") != "reject"
                return approved, analysis
        except Exception as e:
            safe_log(f"Meta-cognitive analysis attempt {attempt+1} failed: {e}")
    safe_log("Meta-cognitive analysis failed, rejecting proposal (safe default)")
    return False, None


def analyze_patch_quality(patch_file: str) -> dict:
    """Analyze a patch for quality metrics: size, file types modified, etc.

    Returns a dict with size_bytes, files_changed, lines_added, lines_removed.
    Returns all zeros if the file cannot be read.
    """
    try:
        with open(patch_file) as f:
            content = f.read()
    except Exception:
        return {"size_bytes": 0, "files_changed": 0, "lines_added": 0, "lines_removed": 0}

    lines = content.split('\n')
    added = sum(1 for l in lines if l.startswith('+') and not l.startswith('+++'))
    removed = sum(1 for l in lines if l.startswith('-') and not l.startswith('---'))
    files_changed = len([l for l in lines if l.startswith('diff --git')])

    return {
        "size_bytes": len(content.encode()),
        "files_changed": files_changed,
        "lines_added": added,
        "lines_removed": removed,
    }


def save_metadata(metadata, output_dir):
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)

def run_harness_swe(entry, model_name_or_path, patch_files, num_evals, output_dir, metadata, run_id, test_more_threshold, test_task_list, test_task_list_more):
    safe_log('Start harness')
    test_task_list = [entry] if test_task_list is None else test_task_list
    dnames = harness(
        test_task_list=test_task_list,
        num_samples=-1,
        max_workers=min(5, len(test_task_list)),
        model_name_or_path=model_name_or_path,
        model_patch_paths=patch_files,
        num_evals=num_evals,
        num_evals_parallel=5,
        pred_dname=os.path.join(output_dir, "predictions"),
    )
    metadata['swe_dnames'] = [str(dn) for dn in dnames]
    safe_log('Start make_report')
    make_report(
        dnames,
        run_ids=[f"{run_id}_{i}" for i in range(len(dnames))],
        dataset_name="princeton-nlp/SWE-bench_Verified",
        output_dir=output_dir,
        dnames_workers=5,
    )
    safe_log('Start get_performance')
    performances, overall_performance = get_all_performance(model_name_or_path, results_dir=output_dir)
    metadata['overall_performance'] = overall_performance
    safe_log("End of evaluation")

    # Check if additional evaluation should be run
    if (overall_performance and \
        test_more_threshold is not None and test_task_list_more is not None and \
            overall_performance.get('total_resolved_instances', 0) >= len(test_task_list) * test_more_threshold):
        safe_log("Start additional evaluation cycle")
        dnames = harness(
            test_task_list=test_task_list_more,
            num_samples=-1,
            max_workers=min(5, len(test_task_list_more)),
            model_name_or_path=model_name_or_path,
            model_patch_paths=patch_files,
            num_evals=num_evals,
            num_evals_parallel=5,
            pred_dname=os.path.join(output_dir, "predictions"),
        )
        safe_log('Start make_report more')
        make_report(
            dnames,
            run_ids=[f"{run_id}_{i}" for i in range(len(dnames))],
            dataset_name="princeton-nlp/SWE-bench_Verified",
            output_dir=output_dir,
            dnames_workers=5,
        )
        safe_log('Start get_performance')
        performances, overall_performance = get_all_performance(model_name_or_path, results_dir=output_dir)
        metadata['overall_performance'] = overall_performance
        safe_log("End of evaluation more")

def run_harness_polyglot(entry, model_name_or_path, patch_files, num_evals, output_dir, metadata, run_id, test_more_threshold, test_task_list, test_task_list_more):
    safe_log('Start harness')
    test_task_list = [entry] if test_task_list is None else test_task_list
    safe_log(f'workers {min(10, len(test_task_list))}')
    dnames = polyglot_harness(
        test_task_list=test_task_list,
        num_samples=-1,
        max_workers=min(10, len(test_task_list)),
        model_name_or_path=model_name_or_path,
        model_patch_paths=patch_files,
        num_evals=num_evals,
        num_evals_parallel=min(5, num_evals),
        pred_dname=os.path.join(output_dir, "predictions"),
        output_dir=output_dir
    )
    metadata['swe_dnames'] = [str(dn) for dn in dnames]
    safe_log('Start get_performance')
    performances, overall_performance = get_all_performance(model_name_or_path, results_dir=output_dir)
    metadata['overall_performance'] = overall_performance
    safe_log("End of evaluation")

    # Check if additional evaluation should be run
    if (overall_performance and \
        test_more_threshold is not None and test_task_list_more is not None and \
            overall_performance.get('total_resolved_instances', 0) >= len(test_task_list) * test_more_threshold):
        safe_log("Start additional evaluation cycle")
        dnames = polyglot_harness(
            test_task_list=test_task_list_more,
            num_samples=-1,
            max_workers=50,
            model_name_or_path=model_name_or_path,
            model_patch_paths=patch_files,
            num_evals=num_evals,
            num_evals_parallel=min(5, num_evals),
            pred_dname=os.path.join(output_dir, "predictions"),
            output_dir=output_dir
        )
        # metadata['swe_dnames'] = [str(dn) for dn in dnames]
        safe_log('Start get_performance')
        performances, overall_performance = get_all_performance(model_name_or_path, results_dir=output_dir)
        metadata['overall_performance_deep'] = overall_performance
        safe_log("End of evaluation more")

def self_improve(
    parent_commit='initial',
    output_dir='output_selfimprove/',
    force_rebuild=False,
    num_evals=1,
    post_improve_diagnose=True,
    meta_cognitive_validation=True,
    entry=None,
    test_task_list=None,
    test_more_threshold=None,
    test_task_list_more=None,
    full_eval_threshold=None,
    run_baseline=None,
    polyglot=False
):  

    global dataset
    if polyglot:
        with open("polyglot/polyglot_benchmark_metadata.json") as f:
            dataset = json.loads(f.read())
    else:
        from datasets import load_dataset
        dataset = load_dataset("princeton-nlp/SWE-bench_Verified")
        dataset = dataset['test']

    # Variables for this self-improvement attempt
    metadata = {}
    root_dir = os.path.abspath('./')
    run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out_dir_base = output_dir
    output_dir = os.path.join(root_dir, f"{output_dir}/{run_id}/")
    os.makedirs(output_dir, exist_ok=True)
    metadata['run_id'] = run_id
    metadata['parent_commit'] = parent_commit
    metadata['full_eval_threshold'] = full_eval_threshold
    test_task_list_big = load_json_file("./swe_bench/subsets/big.json")

    # Set up logger
    logger = setup_logger(os.path.join(output_dir, "self_improve.log"))

    # Resolve Docker config
    docker_cfg = {
        "image_name": config.get("docker", "image_name", default="godelion"),
        "container_prefix": config.get("docker", "container_prefix", default="godelion"),
        "timeout": config.get("docker", "timeout_seconds", default=1800),
        "network_disabled": config.get("docker", "network_disabled", default=True),
    }
    WORKDIR = "/godelion"

    # Create and start the Docker container
    image_name = docker_cfg["image_name"]
    container_name = f"{docker_cfg['container_prefix']}-container-{run_id}"
    client = docker.from_env()
    remove_existing_container(client, container_name)
    container = build_godelion_container(
        client, root_dir, image_name, container_name,
        force_rebuild=force_rebuild,
    )
    container.start()

    # Configure container network isolation if enabled
    if docker_cfg["network_disabled"]:
        try:
            client.api.update_container(container.id, network_mode="none")
            safe_log("Network disabled for container.")
        except Exception as e:
            safe_log(f"Could not disable network: {e}")

    if polyglot:
        exec_result = container.exec_run(f"rm {WORKDIR}/coding_agent.py", workdir='/')
        log_container_output(exec_result)
        exec_result = container.exec_run(f"mv {WORKDIR}/coding_agent_polyglot.py {WORKDIR}/coding_agent.py", workdir='/')
        log_container_output(exec_result)
        exec_result = container.exec_run(f"rm {WORKDIR}/utils/eval_utils.py", workdir='/')
        log_container_output(exec_result)
        exec_result = container.exec_run(f"rm {WORKDIR}/utils/swe_log_parsers.py", workdir='/')
        log_container_output(exec_result)
    else:
        exec_result = container.exec_run(f"rm {WORKDIR}/coding_agent_polyglot.py", workdir='/')

    # Find all parent patches and apply them
    patch_files = get_model_patch_paths(root_dir, os.path.join(output_dir, '../'), parent_commit)
    if run_baseline not in ['no_selfimprove']:
        for patch_file in patch_files:
            copy_to_container(container, patch_file, f'{WORKDIR}/parent_patch.txt')
            exec_result = container.exec_run(f"/bin/sh -c 'patch -p1 < {WORKDIR}/parent_patch.txt'", workdir=WORKDIR)
            log_container_output(exec_result)
            exec_result = container.exec_run(f"rm {WORKDIR}/parent_patch.txt", workdir=WORKDIR)
            log_container_output(exec_result)

    # Commit this version so that irrelevant changes are not included in the patch
    exec_result = container.exec_run("git add --all", workdir=WORKDIR)
    log_container_output(exec_result)
    exec_result = container.exec_run("git -c user.name='user' -c user.email='you@example.com' commit -m 'baseline commit'", workdir=WORKDIR)
    log_container_output(exec_result)
    commit_output = exec_result.output.decode('utf-8')
    if ']' in commit_output:
        commit_hash = commit_output.split()[1].strip("[]")
    else:
        commit_hash = commit_output.strip()

    # Install requirements again in case of any changes
    exec_result = container.exec_run(f"python -m pip install -r {WORKDIR}/requirements.txt", workdir='/')
    log_container_output(exec_result)

    # Get tasks to improve
    if entry:
        safe_log(f"Task to improve: {entry}")
        problem_statement = diagnose_problem(entry, parent_commit, root_dir, out_dir_base, patch_files=patch_files, polyglot=polyglot)
        safe_log(f"problem_statement: {problem_statement}")
    else:
        safe_log("No entry provided. Exiting.")
        cleanup_container(container)
        save_metadata(metadata, output_dir)
        return metadata

    metadata['entry'] = entry
    metadata['problem_statement'] = problem_statement
    # If problem statement is not found, exit
    if not problem_statement:
        safe_log("Failed to diagnose the problem statement. Exiting.")
        cleanup_container(container)
        save_metadata(metadata, output_dir)
        return metadata

    # Gather environment variables from config
    env_var_names = config.get("docker", "env_vars", default=[
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY",
        "OPENROUTER_API_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION_NAME",
    ])
    env_vars = {}
    for var_name in env_var_names:
        val = os.getenv(var_name)
        if val:
            env_vars[var_name] = val

    # Run self-improvement
    safe_log("Running self-improvement")
    chat_history_file_container = f"{WORKDIR}/self_evo.md"
    test_description = get_test_description(swerepo=False)
    timeout_sec = docker_cfg["timeout"]
    cmd = [
        "timeout", str(timeout_sec),
        "python", f"{WORKDIR}/coding_agent.py",
        "--problem_statement", problem_statement,
        "--git_dir", WORKDIR,
        "--chat_history_file", chat_history_file_container,
        "--base_commit", commit_hash,
        "--outdir", WORKDIR,
        "--test_description", test_description,
        "--self_improve",
    ]
    exec_result = container.exec_run(cmd, environment=env_vars, workdir='/')
    log_container_output(exec_result)

    # Copy output files back to host
    chat_history_file = os.path.join(output_dir, "self_evo.md")
    copy_from_container(container, chat_history_file_container, chat_history_file)
    model_patch_file = os.path.join(output_dir, "model_patch.diff")
    copy_from_container(container, f"{WORKDIR}/model_patch.diff", model_patch_file)

    # Try reading the patch file to validate it
    try:
        # Check if patch file exists and is not empty
        if not os.path.exists(model_patch_file):
            raise Exception("Model patch file is empty or does not exist")
        with open(model_patch_file, 'r') as f:
            patch_content = f.read()
            if not patch_content.strip():
                raise Exception("Model patch file is empty")
    except Exception as e:
        safe_log(f"Failed to read model patch file: {str(e)}")
        save_metadata(metadata, output_dir)
        return metadata

    patch_files.append(model_patch_file)

    # Stop and remove the container
    cleanup_container(container)

    # Analyze patch quality
    patch_quality = analyze_patch_quality(model_patch_file)
    metadata['patch_quality'] = patch_quality

    # Meta-cognitive validation of the improvement proposal (before expensive eval)
    if meta_cognitive_validation and problem_statement:
        safe_log("Running meta-cognitive validation of improvement proposal")
        proposal_ok, meta_analysis = validate_improvement_proposal(problem_statement, patch_file=model_patch_file)
        metadata['meta_cognitive_analysis'] = meta_analysis
        metadata['proposal_validated'] = proposal_ok
        if not proposal_ok:
            safe_log("Proposal rejected by meta-cognitive analysis. Skipping expensive eval.")
            save_metadata(metadata, output_dir)
            return metadata

    # Evaluate the performance of the self-improvement
    model_patch_exists = os.path.exists(model_patch_file)
    metadata['model_patch_exists'] = model_patch_exists
    model_patch_notempty = os.path.getsize(model_patch_file) > 0
    metadata['model_patch_notempty'] = model_patch_notempty
    model_name_or_path = run_id
    if model_patch_exists and model_patch_notempty:
        try:
            if not polyglot:
                run_harness_swe(entry, model_name_or_path, patch_files, num_evals, output_dir, metadata, run_id, test_more_threshold, test_task_list, test_task_list_more)
            else:
                run_harness_polyglot(entry, model_name_or_path, patch_files, num_evals, output_dir, metadata, run_id, test_more_threshold, test_task_list, test_task_list_more)
        except Exception as e:
            safe_log(f"Error while evaluating the self-improvement: {e}")

    # Post-self-improvement diagnosis
    if post_improve_diagnose:
        safe_log("Diagnosing the self-improvement")
        metadata['is_compiled'] = is_compiled_self_improve(metadata)
        if metadata['is_compiled']:
            safe_log("The self-improvement succeed to be complied")
            improvement_diagnosis = diagnose_improvement(
                entry, parent_commit, root_dir,
                model_patch_file, out_dir_base, run_id,
                patch_files=patch_files,
            )
            metadata['improvement_diagnosis'] = improvement_diagnosis
            safe_log(f"Improvement diagnosis: {improvement_diagnosis}")
        else:
            safe_log("The self-improvement fail to be complied")
            metadata['improvement_diagnosis'] = "Fail to complied. Ignore this."

    # Save metadata of this self-improvement attempt
    save_metadata(metadata, output_dir)
    return metadata

def main():
    parser = argparse.ArgumentParser(description="Self-improvement step for the repository.")
    parser.add_argument('--parent_commit', default="initial", type=str, help='Current commit to find the eval results, "initial" if starting from original dgm, else the run_id')
    parser.add_argument('--output_dir', default="./output_selfimprove", type=str, help='Directory to store the output')
    parser.add_argument('--force_rebuild', default=False, action='store_true', help='Force rebuild of the Docker image')
    parser.add_argument('--num_evals', default=1, type=int, help='Repeated number of swe evaluations after self-improvement')
    parser.add_argument('--no_post_improve_diagnose', default=False, action='store_true', help='Skip diagnosing the self-improvement after evaluation')
    parser.add_argument('--no_meta_cognitive', default=False, action='store_true', help='Skip meta-cognitive validation of improvement proposals')
    parser.add_argument('--entry', default="django__django-10999", type=str, help='Task entry to improve')
    parser.add_argument('--test_task_list', default=None, type=str, help='List of tasks to evaluate the self-improvement')
    args = parser.parse_args()

    # Copy cached initial version into experiment dir
    os.system(f"cp -r initial/ {args.output_dir}")

    metadata = self_improve(
        parent_commit=args.parent_commit,
        output_dir=args.output_dir,
        force_rebuild=args.force_rebuild,
        num_evals=args.num_evals,
        post_improve_diagnose=not args.no_post_improve_diagnose,
        meta_cognitive_validation=not args.no_meta_cognitive,
        entry=args.entry,
        test_task_list=args.test_task_list,
    )

if __name__ == "__main__":
    main()

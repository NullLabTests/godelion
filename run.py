#!/usr/bin/env python3
"""
Godelion: Open-Ended Evolution of Self-Improving Coding Agents

Usage:
    python run.py [--config CONFIG_PATH] [OPTIONS]

Options override config.yaml settings. See config.yaml for documentation.
"""
import argparse
import datetime
import json
import math
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from pathlib import Path

from godelion.config import Config, config as global_config
from prompts.self_improvement_prompt import find_selfimprove_eval_logs
from self_improve_step import self_improve
from utils.common_utils import load_json_file
from utils.docker_utils import setup_logger
from utils.evo_utils import load_dgm_metadata, is_compiled_self_improve


def initialize_run(output_dir: str, prevrun_dir: str | None = None, polyglot: bool = False, resume: bool = False) -> tuple:
    """Initialize or resume an evolutionary run.

    Args:
        output_dir: Base output directory.
        prevrun_dir: Previous run directory to continue from (uses metadata.jsonl).
        polyglot: Whether using the polyglot benchmark.
        resume: If True, scan for the latest checkpoint and resume from there.

    Returns:
        Tuple of (archive, start_gen_num, resume_gen_or_None).
    """
    start_gen_num = 0
    archive: list = ['initial']
    resume_gen: int | None = None

    # Auto-resume: find latest checkpoint and use it
    if resume and os.path.isdir(output_dir):
        resume_gen = find_latest_checkpoint(output_dir)
        if resume_gen is not None:
            ckpt = load_checkpoint(output_dir, resume_gen)
            if ckpt:
                archive = ckpt.get('archive', ['initial'])
                start_gen_num = ckpt['generation'] + 1
                print(f"Resuming from generation {resume_gen} (archive: {len(archive)} members)")
                return archive, start_gen_num, resume_gen

    # Continue from previous run directory (old metadata.jsonl method)
    if not resume and prevrun_dir:
        metadata_path = os.path.join(prevrun_dir, "dgm_metadata.jsonl")
        if os.path.exists(metadata_path):
            metadata = load_dgm_metadata(metadata_path, last_only=True)
            archive = metadata['archive']
            start_gen_num = metadata['generation'] + 1
            return archive, start_gen_num, None

    # Fresh start — copy initial evaluation results
    initial_folder_name = 'initial' if not polyglot else 'initial_polyglot'
    if not os.path.exists(f"{output_dir}/{initial_folder_name}"):
        if os.path.exists(initial_folder_name):
            os.system(f"cp -r {initial_folder_name}/ {output_dir}/initial")
        else:
            raise RuntimeError("Need initial evaluation results. Run evaluation first.")
    return archive, start_gen_num, None


def any_exceeding_context_length(output_dir, commit_id, instance_ids):
    for instance_id in instance_ids:
        md_logs, _, _, _ = find_selfimprove_eval_logs(instance_id, output_dir, commit_id, filter=False)
        if not md_logs:
            continue
        md_log = md_logs[0]
        error_str = "Error in get_response_withtools: Error code: 400 - {'message': 'Input is too long for requested model.'}"
        if f'{error_str}\n{error_str}' in md_log:
            return True
    return False


def compute_diversity_scores(output_dir: str, archive: list) -> dict:
    """Compute diversity scores for archive members based on lineage overlap.

    For each archive member, diversity = 1 - average pairwise lineage overlap
    with all other members. Members from distinct evolutionary branches get
    high scores; closely related members get low scores.

    Args:
        output_dir: Base output directory containing run metadata.
        archive: List of run_ids (commit hashes or 'initial').

    Returns:
        dict mapping each commit to its diversity score (0.0–1.0).
    """
    diversity = {}
    if len(archive) < 2:
        return {c: 1.0 for c in archive}

    lineages = {}
    for commit in archive:
        try:
            meta = load_json_file(os.path.join(output_dir, commit, "metadata.json"))
            lineage = [commit]
            parent = meta.get('parent_commit', 'initial')
            while parent != 'initial' and parent in archive:
                lineage.append(parent)
                parent = load_json_file(os.path.join(output_dir, parent, "metadata.json")).get('parent_commit', 'initial')
            lineages[commit] = set(lineage)
        except Exception:
            lineages[commit] = {commit}

    for commit in archive:
        my_lineage = lineages[commit]
        overlap_scores = []
        for other in archive:
            if other == commit:
                continue
            other_lineage = lineages.get(other, {other})
            intersection = len(my_lineage & other_lineage)
            union = len(my_lineage | other_lineage)
            overlap = intersection / max(union, 1)
            overlap_scores.append(overlap)
        avg_overlap = sum(overlap_scores) / max(len(overlap_scores), 1)
        diversity[commit] = 1.0 - avg_overlap

    return diversity


def choose_selfimproves(output_dir, archive, selfimprove_size, method='random', diversity_weight=0.3, run_baseline=None, polyglot=False):
    selfimprove_entries = []
    candidates = {}

    for commit in archive:
        try:
            metadata_path = os.path.join(output_dir, commit, "metadata.json")
            metadata = load_json_file(metadata_path)
            candidates[commit] = {
                'accuracy_score': metadata['overall_performance']['accuracy_score'],
                'total_unresolved_ids': metadata['overall_performance']['total_unresolved_ids'],
                'total_emptypatch_ids': metadata['overall_performance']['total_emptypatch_ids'],
                'total_resolved_ids': metadata['overall_performance']['total_resolved_ids'],
                'children_count': 0,
            }
            if commit != 'initial':
                parent_commit = metadata['parent_commit']
                if parent_commit in candidates:
                    candidates[parent_commit]['children_count'] += 1
        except Exception as e:
            print(f"{commit} not eligible for being a parent: {e}")
            continue

    if not candidates:
        print("No eligible candidates in archive.")
        return []

    if run_baseline == 'no_darwin':
        commits = list(candidates.keys())
        parent_commits = commits[-1:]
        diversity_scores = {c: 1.0 for c in commits}
    else:
        diversity_scores = compute_diversity_scores(output_dir, archive)

    if run_baseline == 'no_darwin':
        parent_commits = list(candidates.keys())[-1:]
    elif method == 'score_prop':
        commits = list(candidates.keys())
        scores = [candidates[commit]['accuracy_score'] for commit in commits]
        scores = [1 / (1 + math.exp(-10 * (score - 0.5))) for score in scores]
        probabilities = [score / sum(scores) for score in scores]
        parent_commits = random.choices(commits, probabilities, k=selfimprove_size)
    elif method == 'score_child_prop':
        commits = list(candidates.keys())
        scores = [candidates[commit]['accuracy_score'] for commit in commits]
        scores = [1 / (1 + math.exp(-10 * (score - 0.5))) for score in scores]
        children_counts = [candidates[commit]['children_count'] for commit in commits]
        children_counts = [1 / (1 + count) for count in children_counts]
        probabilities = [s * c for s, c in zip(scores, children_counts)]
        prob_sum = sum(probabilities)
        if prob_sum > 0:
            probabilities = [p / prob_sum for p in probabilities]
        else:
            probabilities = [1 / len(probabilities)] * len(probabilities)
        parent_commits = random.choices(commits, probabilities, k=selfimprove_size)
    elif method == 'diversity_weighted':
        commits = list(candidates.keys())
        scores = [candidates[commit]['accuracy_score'] for commit in commits]
        div_scores = [diversity_scores.get(commit, 0.5) for commit in commits]
        sig_scores = [1 / (1 + math.exp(-10 * (s - 0.5))) for s in scores]
        acc_weight = 1.0 - diversity_weight
        combined = [acc_weight * s + diversity_weight * d for s, d in zip(sig_scores, div_scores)]
        prob_sum = sum(combined)
        probabilities = [c / prob_sum for c in combined] if prob_sum > 0 else [1 / len(combined)] * len(combined)
        parent_commits = random.choices(commits, probabilities, k=selfimprove_size)
    elif method == 'best':
        sorted_commits = sorted(candidates, key=lambda x: candidates[x]['accuracy_score'], reverse=True)
        parent_commits = sorted_commits[:min(selfimprove_size, len(sorted_commits))]
        if len(parent_commits) < selfimprove_size:
            parent_commits.extend(random.choices(parent_commits, k=selfimprove_size - len(parent_commits)))
    else:
        commit_keys = list(candidates.keys())
        if not commit_keys:
            return []
        parent_commits = random.choices(commit_keys, k=selfimprove_size)

    for parent_commit in parent_commits:
        if parent_commit not in candidates:
            continue
        empty_ids = candidates[parent_commit]['total_emptypatch_ids']
        resolved_ids = candidates[parent_commit]['total_resolved_ids']
        unresolved_ids = candidates[parent_commit]['total_unresolved_ids']

        if polyglot:
            entry_ids = empty_ids + unresolved_ids
            if not entry_ids:
                entry_ids = resolved_ids + empty_ids + unresolved_ids
        else:
            num_total_ids = len(empty_ids) + len(resolved_ids) + len(unresolved_ids)
            if num_total_ids == 0:
                continue
            if len(empty_ids) >= 0.1 * num_total_ids and random.random() < 0.25:
                entry = 'solve_empty_patches'
                selfimprove_entries.append((parent_commit, entry))
                continue
            if random.random() < 0.25:
                entry = 'solve_stochasticity'
                selfimprove_entries.append((parent_commit, entry))
                continue
            if any_exceeding_context_length(output_dir, parent_commit, empty_ids + unresolved_ids) and random.random() < 0.25:
                entry = 'solve_contextlength'
                selfimprove_entries.append((parent_commit, entry))
                continue
            if not unresolved_ids:
                if resolved_ids:
                    entry_ids = resolved_ids
                else:
                    continue
            else:
                entry_ids = unresolved_ids
        if entry_ids:
            entry = random.choice(entry_ids)
            selfimprove_entries.append((parent_commit, entry))

    return selfimprove_entries


def filter_compiled(run_ids, output_dir, num_swe_issues=None, logger=None):
    run_ids_compiled = []
    for run_id in run_ids:
        metadata_path = os.path.join(output_dir, run_id, "metadata.json")
        if not os.path.exists(metadata_path):
            logger.warning(f"Metadata not found for {run_id}, skipping.")
            continue
        metadata = load_json_file(metadata_path)
        if is_compiled_self_improve(metadata, num_swe_issues=num_swe_issues or [], logger=logger):
            run_ids_compiled.append(run_id)
    return run_ids_compiled


def get_original_score(output_dir):
    try:
        metadata = load_json_file(os.path.join(output_dir, "initial", "metadata.json"))
        return metadata["overall_performance"]["accuracy_score"]
    except (FileNotFoundError, KeyError, TypeError):
        return 0.0


def update_archive(output_dir, archive, new_ids, method='keep_all', noise_leeway=0.1, diversity_bonus=0.1):
    if method in ('keep_better', 'keep_diverse'):
        original_score = get_original_score(output_dir) - noise_leeway
    if method == 'keep_better':
        for run_id in new_ids:
            metadata = load_json_file(os.path.join(output_dir, run_id, "metadata.json"))
            score = metadata["overall_performance"]["accuracy_score"]
            if score >= original_score:
                archive.append(run_id)
    elif method == 'keep_diverse':
        diversity = compute_diversity_scores(output_dir, archive + new_ids)
        for run_id in new_ids:
            metadata = load_json_file(os.path.join(output_dir, run_id, "metadata.json"))
            score = metadata["overall_performance"]["accuracy_score"]
            div_bonus = diversity_bonus * diversity.get(run_id, 0.5)
            if score + div_bonus >= original_score:
                archive.append(run_id)
    else:
        archive += new_ids
    archive = list(dict.fromkeys(archive))
    return archive


def get_full_eval_threshold(output_dir, archive):
    archive_scores = []
    num_full_eval = sum(len(load_json_file(f"./swe_bench/subsets/{size}.json")) for size in ['small', 'medium', 'big'])
    original_score = get_original_score(output_dir)
    archive_scores.append(original_score)
    for run_id in archive:
        metadata_path = os.path.join(output_dir, run_id, "metadata.json")
        if not os.path.exists(metadata_path):
            continue
        metadata = load_json_file(metadata_path)
        total_submitted_instances = metadata["overall_performance"]["total_submitted_instances"]
        if total_submitted_instances < num_full_eval * 0.9:
            continue
        score = metadata["overall_performance"]["accuracy_score"]
        archive_scores.append(score)
    threshold = sorted(archive_scores, reverse=True)[1] if len(archive_scores) > 1 else archive_scores[0] if archive_scores else 0.4
    threshold = max(threshold, 0.4)
    return threshold


def save_checkpoint(output_dir: str, gen_num: int, archive: list, selfimprove_entries: list, selfimprove_ids: list, selfimprove_ids_compiled: list) -> None:
    """Save a checkpoint that can be used to resume the run."""
    checkpoint = {
        "generation": gen_num,
        "archive": archive,
        "selfimprove_entries": selfimprove_entries,
        "children": selfimprove_ids,
        "children_compiled": selfimprove_ids_compiled,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    with open(os.path.join(output_dir, f"checkpoint_gen_{gen_num}.json"), "w") as f:
        json.dump(checkpoint, f, indent=2)


def load_checkpoint(output_dir: str, gen_num: int) -> dict | None:
    """Load a checkpoint for a specific generation."""
    path = os.path.join(output_dir, f"checkpoint_gen_{gen_num}.json")
    if os.path.exists(path):
        return load_json_file(path)
    return None


def find_latest_checkpoint(output_dir: str) -> int | None:
    """Scan output_dir for the highest-numbered checkpoint file.

    Returns the generation number of the latest checkpoint, or None if none found.
    """
    import re
    max_gen = None
    if not os.path.isdir(output_dir):
        return None
    for fname in os.listdir(output_dir):
        m = re.match(r'checkpoint_gen_(\d+)\.json', fname)
        if m:
            gen = int(m.group(1))
            if max_gen is None or gen > max_gen:
                max_gen = gen
    return max_gen


def get_archive_diversity_report(output_dir, archive, logger):
    """Log diversity metrics for the current archive."""
    if len(archive) < 2:
        return
    diversity = compute_diversity_scores(output_dir, archive)
    if diversity:
        avg_div = sum(diversity.values()) / len(diversity)
        min_div = min(diversity.values())
        max_div = max(diversity.values())
        logger.info(f"Archive diversity: avg={avg_div:.3f}, min={min_div:.3f}, max={max_div:.3f}")
    return diversity


def main():
    parser = argparse.ArgumentParser(description="Godelion: Open-Ended Evolution of Self-Improving Coding Agents")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--max-generation", type=int, default=None, help="Maximum number of evolution iterations")
    parser.add_argument("--selfimprove-size", type=int, default=None, help="Number of self-improvement attempts per generation")
    parser.add_argument("--selfimprove-workers", type=int, default=None, help="Number of parallel workers")
    parser.add_argument("--selection-method", type=str, default=None, choices=['random', 'score_prop', 'score_child_prop', 'diversity_weighted', 'best'], help="Parent selection method")
    parser.add_argument("--continue-from", type=str, default=None, help="Directory to continue from (uses metadata.jsonl)")
    parser.add_argument("--resume", default=None, action='store_true', help="Auto-resume from latest checkpoint in output dir")
    parser.add_argument("--update-archive", type=str, default=None, choices=['keep_better', 'keep_all', 'keep_diverse'], help="Archive update method")
    parser.add_argument("--num-evals", type=int, default=None, help="Number of repeated evaluations per self-improve")
    parser.add_argument("--post-improve-diagnose", default=None, action='store_true', help="Diagnose after evaluation")
    parser.add_argument("--no-meta-cognitive", default=None, action='store_true', help="Skip meta-cognitive validation of proposals")
    parser.add_argument("--diversity-weight", type=float, default=None, help="Diversity weight for diversity_weighted selection (0.0-1.0)")
    parser.add_argument("--diversity-bonus", type=float, default=None, help="Diversity bonus for keep_diverse archive update")
    parser.add_argument("--shallow-eval", default=None, action='store_true', help="Single shallow evaluation")
    parser.add_argument("--polyglot", default=None, action='store_true', help="Run polyglot benchmark")
    parser.add_argument("--no-full-eval", default=None, action='store_true', help="Skip full evaluation")
    parser.add_argument("--run-baseline", type=str, default=None, choices=['no_selfimprove', 'no_darwin'], help="Baseline to run")
    args = parser.parse_args()

    # Load configuration
    cfg = Config(args.config) if args.config else global_config

    def cfg_or_arg(config_key, arg_value):
        if arg_value is not None:
            return arg_value
        return cfg.get(*config_key.split("."))

    # Resolve settings
    max_generation = cfg_or_arg("evolution.max_generations", args.max_generation) or 80
    selfimprove_size = cfg_or_arg("evolution.self_improve_size", args.selfimprove_size) or 2
    selfimprove_workers = cfg_or_arg("evolution.parallel_workers", args.selfimprove_workers) or 2
    choose_method = cfg_or_arg("evolution.selection_method", args.selection_method) or "score_child_prop"
    archive_method = cfg_or_arg("evolution.archive_update", args.update_archive) or "keep_all"
    num_swe_evals = cfg_or_arg("evaluation.num_evals", args.num_evals) or 1
    post_improve_diagnose_val = cfg_or_arg("evaluation.post_improve_diagnose", args.post_improve_diagnose)
    shallow_eval_val = cfg_or_arg("evaluation.shallow_eval", args.shallow_eval)
    polyglot_val = cfg_or_arg("evaluation.polyglot", args.polyglot)
    no_full_eval_val = cfg_or_arg("evaluation.no_full_eval", args.no_full_eval)
    run_baseline_val = cfg_or_arg("evaluation.run_baseline", args.run_baseline)
    meta_cognitive_val = cfg_or_arg("evaluation.meta_cognitive_validation", args.no_meta_cognitive)
    if args.no_meta_cognitive:
        meta_cognitive_val = False
    else:
        meta_cognitive_val = bool(cfg.get("evaluation", "meta_cognitive_validation", default=True))
    diversity_weight = cfg_or_arg("evolution.selection_weight_diversity", args.diversity_weight) or 0.3
    diversity_bonus = cfg_or_arg("evolution.archive_diversity_bonus", args.diversity_bonus) or 0.1
    eval_noise = cfg.get("evolution.eval_noise", default=0.1)
    output_base_dir = cfg.get("logging.output_dir", default="./output_godelion")

    polyglot_val = bool(polyglot_val)
    shallow_eval_val = bool(shallow_eval_val)
    post_improve_diagnose_val = bool(post_improve_diagnose_val)
    resume_val = bool(args.resume)

    if not args.continue_from and not resume_val:
        run_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S_%f")
    elif resume_val:
        run_id = os.path.basename(output_base_dir) if args.resume else datetime.datetime.now().strftime("%Y%m%d%H%M%S_%f")
    else:
        run_id = os.path.basename(args.continue_from)

    output_dir = os.path.join(output_base_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger(os.path.join(output_dir, "godelion_outer.log"))

    archive, start_gen_num, resume_gen = initialize_run(output_dir, prevrun_dir=args.continue_from, polyglot=polyglot_val, resume=resume_val)
    if resume_gen is not None:
        logger.info(f"Auto-resume detected checkpoint at generation {resume_gen}")
        logger.info(f"Starting from generation {start_gen_num}, archive size: {len(archive)}")

    if not polyglot_val:
        swe_issues_sm = load_json_file("./swe_bench/subsets/small.json")
        swe_issues_med = load_json_file("./swe_bench/subsets/medium.json")
    else:
        swe_issues_sm = load_json_file("./polyglot/subsets/small.json")
        swe_issues_med = load_json_file("./polyglot/subsets/medium.json")
    logger.info(f"Starting Godelion run {run_id}")
    logger.info(f"Config: max_generation={max_generation}, selfimprove_size={selfimprove_size}, workers={selfimprove_workers}")
    logger.info(f"Selection: {choose_method} (diversity_weight={diversity_weight}), Archive: {archive_method} (diversity_bonus={diversity_bonus})")
    logger.info(f"Meta-cognitive validation: {meta_cognitive_val}, Post-diagnose: {post_improve_diagnose_val}")
    logger.info(f"Archive initial: {archive}")

    test_more_threshold = 0.4
    checkpoint_enabled = cfg.get("checkpoint", "enabled", default=True)
    checkpoint_interval = cfg.get("checkpoint", "interval_generations", default=1)

    gen_num = start_gen_num - 1
    for gen_num in range(start_gen_num, max_generation):
        logger.info(f"--- Generation {gen_num} ---")

        # Log diversity metrics at start of generation
        get_archive_diversity_report(output_dir, archive, logger)

        selfimprove_entries = choose_selfimproves(
            output_dir, archive, selfimprove_size,
            method=choose_method,
            diversity_weight=diversity_weight,
            run_baseline=run_baseline_val,
            polyglot=polyglot_val,
        )
        if not selfimprove_entries:
            logger.warning(f"No self-improve entries for generation {gen_num}")
            continue

        logger.info(f"Self-improve entries: {selfimprove_entries}")

        selfimprove_ids = []
        with ThreadPoolExecutor(max_workers=selfimprove_workers) as executor:
            futures = [
                executor.submit(
                    self_improve,
                    parent_commit=parent_commit,
                    output_dir=output_dir,
                    force_rebuild=False,
                    num_evals=num_swe_evals,
                    post_improve_diagnose=post_improve_diagnose_val,
                    meta_cognitive_validation=meta_cognitive_val,
                    entry=entry,
                    test_task_list=swe_issues_sm,
                    test_more_threshold=None if shallow_eval_val else test_more_threshold,
                    test_task_list_more=None if shallow_eval_val else swe_issues_med,
                    polyglot=polyglot_val,
                    full_eval_threshold=None if no_full_eval_val else get_full_eval_threshold(output_dir, archive),
                    run_baseline=run_baseline_val,
                )
                for parent_commit, entry in selfimprove_entries
            ]

            for future in as_completed(futures):
                try:
                    metadata = future.result(timeout=1.5 * 60 * 60)
                    if metadata and 'run_id' in metadata:
                        selfimprove_ids.append(metadata['run_id'])
                except TimeoutError:
                    logger.error("Self-improvement timed out (1.5h).")
                except Exception as e:
                    logger.error(f"Self-improvement failed: {e}")

        logger.info(f"Self-improve IDs: {selfimprove_ids}")
        selfimprove_ids_compiled = filter_compiled(
            selfimprove_ids, output_dir,
            num_swe_issues=[len(swe_issues_sm)] if shallow_eval_val else [len(swe_issues_sm), len(swe_issues_med)],
            logger=logger,
        )
        archive = update_archive(output_dir, archive, selfimprove_ids_compiled, method=archive_method, noise_leeway=eval_noise, diversity_bonus=diversity_bonus)

        gen_record = {
            "generation": gen_num,
            "selfimprove_entries": selfimprove_entries,
            "children": selfimprove_ids,
            "children_compiled": selfimprove_ids_compiled,
            "archive": archive,
            "archive_diversity": compute_diversity_scores(output_dir, archive) if len(archive) > 1 else {},
        }

        with open(os.path.join(output_dir, "dgm_metadata.jsonl"), "a") as f:
            f.write(json.dumps(gen_record) + "\n")

        # Save checkpoint
        if checkpoint_enabled and (gen_num % checkpoint_interval == 0 or gen_num == max_generation - 1):
            save_checkpoint(output_dir, gen_num, archive, selfimprove_entries, selfimprove_ids, selfimprove_ids_compiled)
            logger.info(f"Checkpoint saved for generation {gen_num}")

        logger.info(f"Archive after gen {gen_num}: {archive}")
        logger.info(f"Generation {gen_num} complete.")

    # Write run summary
    summary = {
        "run_id": run_id,
        "output_dir": output_dir,
        "config": {
            "max_generation": max_generation,
            "selfimprove_size": selfimprove_size,
            "workers": selfimprove_workers,
            "selection_method": choose_method,
            "diversity_weight": diversity_weight,
            "archive_method": archive_method,
            "diversity_bonus": diversity_bonus,
            "meta_cognitive_validation": meta_cognitive_val,
            "polyglot": polyglot_val,
        },
        "generations_completed": max(0, gen_num + 1),
        "archive_size": len(archive),
        "archive": archive,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    with open(os.path.join(output_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Run summary saved to {output_dir}/run_summary.json")
    logger.info(f"Archive size: {len(archive)}, Generations: {summary['generations_completed']}")
    logger.info("Godelion run complete.")


if __name__ == "__main__":
    main()

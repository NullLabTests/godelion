"""Smoke tests for v0.2.x features — exercises all new code paths without Docker/LLM.

These tests mock the heavy external dependencies (docker, anthropic, openai, datasets)
so we can verify the new logic (checkpointing, diversity, meta-cognitive plumbing,
patch analysis, CLI wiring, run summaries) actually works end-to-end.

Run separately::

    pytest tests/test_smoke_v020.py -v

Not included in the default ``pytest tests/`` run (see pytest.ini).
"""

import json
import os
import sys
import tempfile
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke


# ===========================================================================
#  Fixture: import project modules with mocked deps in a sandbox
# ===========================================================================


@pytest.fixture(scope="session")
def sm():
    """Return a dict of project functions, imported with mocked Docker/LLM deps.

    The stubs are isolated to this fixture's invocation and cleaned up after
    the session, so no sys.modules pollution leaks to other test files.
    """
    _stub_modules()
    sys.path.insert(0, str(Path(__file__).parent.parent))
    for name in ("self_improve_step", "run"):
        sys.modules.pop(name, None)

    import run as _run
    import self_improve_step as _sis

    yield {
        # self_improve_step
        "analyze_patch_quality": _sis.analyze_patch_quality,
        "validate_improvement_proposal": _sis.validate_improvement_proposal,
        # run
        "save_checkpoint": _run.save_checkpoint,
        "load_checkpoint": _run.load_checkpoint,
        "find_latest_checkpoint": _run.find_latest_checkpoint,
        "compute_diversity_scores": _run.compute_diversity_scores,
        "update_archive": _run.update_archive,
        "get_archive_diversity_report": _run.get_archive_diversity_report,
        "get_full_eval_threshold": _run.get_full_eval_threshold,
        "get_original_score": _run.get_original_score,
        "filter_compiled": _run.filter_compiled,
        "choose_selfimproves": _run.choose_selfimproves,
        "initialize_run": _run.initialize_run,
        "build_parser": _run.build_parser,
        "resolve_config_settings": _run.resolve_config_settings,
        "resolve_run_id": _run.resolve_run_id,
    }

    # Cleanup: remove stubbed modules so other test files aren't affected
    for _name in list(sys.modules):
        if any(
            _name.startswith(p)
            for p in (
                "self_improve_step",
                "run",
                "llm",
                "docker",
                "anthropic",
                "openai",
                "backoff",
                "datasets",
            )
        ):
            sys.modules.pop(_name, None)


def _stub_modules():
    """Register mock modules in sys.modules for all heavy dependencies."""

    def _flat(name, attrs=None):
        mod = types.ModuleType(name)
        mod.__package__ = name
        if attrs:
            for k, v in attrs.items():
                setattr(mod, k, v)
        sys.modules[name] = mod

    def _leaf(path, attrs):
        m = types.ModuleType(path)
        m.__package__ = path
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[path] = m

    # External deps
    _flat("docker", {"from_env": lambda: types.SimpleNamespace()})
    _flat(
        "anthropic",
        {
            "Anthropic": lambda **kw: types.SimpleNamespace(),
            "AnthropicBedrock": lambda **kw: types.SimpleNamespace(),
            "AnthropicVertex": lambda **kw: types.SimpleNamespace(),
            "RateLimitError": type("RateLimitError", (Exception,), {}),
            "APIStatusError": type("APIStatusError", (Exception,), {}),
            "APITimeoutError": type("APITimeoutError", (Exception,), {}),
        },
    )
    _flat(
        "openai",
        {
            "OpenAI": lambda **kw: types.SimpleNamespace(),
            "RateLimitError": type("RateLimitError", (Exception,), {}),
            "APITimeoutError": type("APITimeoutError", (Exception,), {}),
            "APIStatusError": type("APIStatusError", (Exception,), {}),
            "APIConnectionError": type("APIConnectionError", (Exception,), {}),
        },
    )
    _flat(
        "backoff",
        {
            "on_exception": lambda *a, **kw: lambda f: f,
            "expo": lambda **kw: types.SimpleNamespace(),
        },
    )
    _flat("datasets", {"load_dataset": lambda x: types.SimpleNamespace(__getitem__=lambda s, k: [{"patch": "dummy"}])})
    _flat(
        "llm",
        {
            "create_client": lambda model="test": (types.SimpleNamespace(), model),
            "get_response_from_llm": lambda **kw: ('{"recommendation": "approve", "risk_level": "low"}', []),
            "extract_json_between_markers": lambda text: {"recommendation": "approve", "risk_level": "low"},
        },
    )

    # Leaf sub-modules (parent packages exist on filesystem)
    _leaf(
        "utils.docker_utils",
        {
            "build_dgm_container": lambda *a, **kw: types.SimpleNamespace(
                start=lambda: None, exec_run=lambda *a, **kw: types.SimpleNamespace(output=b"abc"), id="abc"
            ),
            "cleanup_container": lambda c: None,
            "copy_from_container": lambda c, s, d: None,
            "copy_to_container": lambda c, s, d: None,
            "log_container_output": lambda r: None,
            "remove_existing_container": lambda c, n: None,
            "setup_logger": lambda p: types.SimpleNamespace(info=lambda m: None),
            "safe_log": lambda m: None,
        },
    )
    _leaf("swe_bench.harness", {"harness": lambda **kw: []})
    _leaf("polyglot.harness", {"harness": lambda **kw: []})
    _leaf("swe_bench.report", {"make_report": lambda **kw: None})


# ===========================================================================
#  Helpers
# ===========================================================================


def _make_metadata(output_dir, run_id, accuracy=0.5, parent="initial", extra=None):
    d = os.path.join(output_dir, run_id)
    os.makedirs(d, exist_ok=True)
    meta = {
        "run_id": run_id,
        "parent_commit": parent,
        "overall_performance": {
            "accuracy_score": accuracy,
            "total_unresolved_ids": ["u1"],
            "total_resolved_ids": ["r1"],
            "total_emptypatch_ids": [],
            "total_submitted_instances": 1,
        },
    }
    if extra:
        meta.update(extra)
    with open(os.path.join(d, "metadata.json"), "w") as f:
        json.dump(meta, f)
    return meta


# ===========================================================================
#  Checkpoint tests
# ===========================================================================


class TestCheckpoint:
    def test_save_and_load_checkpoint(self, sm):
        with tempfile.TemporaryDirectory() as tmp:
            sm["save_checkpoint"](tmp, 0, ["initial"], [("i", "e")], ["c1"], ["c1"])
            ckpt = sm["load_checkpoint"](tmp, 0)
            assert ckpt["generation"] == 0
            assert ckpt["archive"] == ["initial"]
            assert ckpt["children"] == ["c1"]
            assert "timestamp" in ckpt

    def test_load_nonexistent_returns_none(self, sm):
        with tempfile.TemporaryDirectory() as tmp:
            assert sm["load_checkpoint"](tmp, 99) is None

    def test_find_latest_checkpoint(self, sm):
        with tempfile.TemporaryDirectory() as tmp:
            assert sm["find_latest_checkpoint"](tmp) is None
            sm["save_checkpoint"](tmp, 3, [], [], [], [])
            sm["save_checkpoint"](tmp, 1, [], [], [], [])
            sm["save_checkpoint"](tmp, 7, [], [], [], [])
            assert sm["find_latest_checkpoint"](tmp) == 7

    def test_find_latest_ignores_other_files(self, sm):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "dgm_metadata.jsonl").write_text("")
            Path(tmp, "some_other.json").write_text("")
            sm["save_checkpoint"](tmp, 2, [], [], [], [])
            assert sm["find_latest_checkpoint"](tmp) == 2

    def test_find_latest_nonexistent_dir(self, sm):
        assert sm["find_latest_checkpoint"]("/nonexistent_abc123") is None


# ===========================================================================
#  Diversity scoring tests
# ===========================================================================


class TestDiversity:
    def test_single_archive_member(self, sm):
        with tempfile.TemporaryDirectory() as tmp:
            _make_metadata(tmp, "initial", accuracy=0.5)
            assert sm["compute_diversity_scores"](tmp, ["initial"]) == {"initial": 1.0}

    def test_empty_archive(self, sm):
        assert sm["compute_diversity_scores"](tmp := Path(tempfile.mkdtemp()).as_posix(), []) == {}

    def test_two_members_no_lineage_relation(self, sm):
        with tempfile.TemporaryDirectory() as tmp:
            _make_metadata(tmp, "initial", accuracy=0.5)
            _make_metadata(tmp, "ca", accuracy=0.6, parent="initial")
            _make_metadata(tmp, "cb", accuracy=0.7, parent="initial")
            scores = sm["compute_diversity_scores"](tmp, ["ca", "cb"])
            assert scores["ca"] == 1.0 and scores["cb"] == 1.0

    def test_deeper_lineage_has_lower_diversity(self, sm):
        with tempfile.TemporaryDirectory() as tmp:
            _make_metadata(tmp, "initial", accuracy=0.5)
            _make_metadata(tmp, "g1a", accuracy=0.6, parent="initial")
            _make_metadata(tmp, "g1b", accuracy=0.6, parent="initial")
            _make_metadata(tmp, "g2", accuracy=0.7, parent="g1a")
            scores = sm["compute_diversity_scores"](tmp, ["g1a", "g1b", "g2"])
            assert scores["g2"] < scores["g1b"]


# ===========================================================================
#  Patch quality analysis tests
# ===========================================================================


class TestPatchQuality:
    def test_analyze_patch_quality(self, sm):
        with tempfile.TemporaryDirectory() as tmp:
            patch = "diff --git a/f.py b/f.py\n--- /dev/null\n+++ b/f.py\n@@ -0,0 +1,3 @@\n+def f():\n+    return 42\n+"
            p = Path(tmp, "p.diff")
            p.write_text(patch)
            r = sm["analyze_patch_quality"](str(p))
            assert r["files_changed"] == 1 and r["lines_added"] == 3

    def test_analyze_missing_patch(self, sm):
        assert sm["analyze_patch_quality"]("/nonexistent/p.diff") == {"size_bytes": 0, "files_changed": 0, "lines_added": 0, "lines_removed": 0}


# ===========================================================================
#  Archive update tests
# ===========================================================================


class TestArchiveUpdate:
    def test_keep_all(self, sm):
        with tempfile.TemporaryDirectory() as tmp:
            _make_metadata(tmp, "initial")
            assert sm["update_archive"](tmp, ["initial"], ["c1"]) == ["initial", "c1"]

    def test_keep_better_threshold(self, sm):
        with tempfile.TemporaryDirectory() as tmp:
            _make_metadata(tmp, "initial", accuracy=0.4)
            _make_metadata(tmp, "good", accuracy=0.5)
            _make_metadata(tmp, "bad", accuracy=0.2)
            r = sm["update_archive"](tmp, ["initial"], ["good", "bad"], method="keep_better")
            assert "good" in r and "bad" not in r

    def test_keep_diverse_adds_bonus(self, sm):
        with tempfile.TemporaryDirectory() as tmp:
            _make_metadata(tmp, "initial", accuracy=0.4)
            _make_metadata(tmp, "c1", accuracy=0.35)
            r = sm["update_archive"](tmp, ["initial"], ["c1"], method="keep_diverse", diversity_bonus=0.1)
            assert "c1" in r


# ===========================================================================
#  Meta-cognitive validation plumbing tests
# ===========================================================================


class TestMetaCognitive:
    def test_validate_proposal_smoke(self, sm):
        approved, analysis = sm["validate_improvement_proposal"]("Fix something")
        assert approved is True
        assert analysis is not None
        assert analysis.get("recommendation") == "approve"


# ===========================================================================
#  Selection method tests
# ===========================================================================


class TestSelection:
    @staticmethod
    def _with_preds(tmp, run_id):
        os.makedirs(os.path.join(tmp, run_id, "predictions"), exist_ok=True)

    def test_best_method(self, sm):
        with tempfile.TemporaryDirectory() as tmp:
            _make_metadata(tmp, "init", accuracy=0.3)
            _make_metadata(tmp, "ca", accuracy=0.9, parent="init")
            _make_metadata(tmp, "cb", accuracy=0.5, parent="init")
            for r in ("init", "ca", "cb"):
                self._with_preds(tmp, r)
            entries = sm["choose_selfimproves"](tmp, ["init", "ca", "cb"], 1, method="best")
            assert len(entries) >= 1

    def test_diversity_method(self, sm):
        with tempfile.TemporaryDirectory() as tmp:
            _make_metadata(tmp, "init", accuracy=0.3)
            _make_metadata(tmp, "ca", accuracy=0.8, parent="init")
            _make_metadata(tmp, "cb", accuracy=0.6, parent="init")
            for r in ("init", "ca", "cb"):
                self._with_preds(tmp, r)
            entries = sm["choose_selfimproves"](tmp, ["init", "ca", "cb"], 2, method="diversity_weighted", diversity_weight=0.5)
            assert len(entries) >= 1

    def test_score_prop(self, sm):
        with tempfile.TemporaryDirectory() as tmp:
            _make_metadata(tmp, "init", accuracy=0.3)
            self._with_preds(tmp, "init")
            assert len(sm["choose_selfimproves"](tmp, ["init"], 1, method="score_prop")) >= 1

    def test_random(self, sm):
        with tempfile.TemporaryDirectory() as tmp:
            _make_metadata(tmp, "init", accuracy=0.3)
            self._with_preds(tmp, "init")
            assert len(sm["choose_selfimproves"](tmp, ["init"], 1, method="random")) >= 1


# ===========================================================================
#  CLI / config parsing tests
# ===========================================================================


class TestCLIParsing:
    def test_new_flags_accepted(self):
        import argparse

        p = argparse.ArgumentParser()
        p.add_argument("--resume", action="store_true")
        p.add_argument("--diversity-weight", type=float)
        p.add_argument("--diversity-bonus", type=float)
        p.add_argument("--no-meta-cognitive", action="store_true")
        p.add_argument("--selection-method", choices=["random", "score_prop", "score_child_prop", "diversity_weighted", "best"])
        p.add_argument("--update-archive", choices=["keep_better", "keep_all", "keep_diverse"])
        args = p.parse_args(
            [
                "--resume",
                "--diversity-weight",
                "0.4",
                "--diversity-bonus",
                "0.2",
                "--no-meta-cognitive",
                "--selection-method",
                "diversity_weighted",
                "--update-archive",
                "keep_diverse",
            ]
        )
        assert args.resume is True
        assert args.diversity_weight == 0.4
        assert args.diversity_bonus == 0.2
        assert args.no_meta_cognitive is True
        assert args.selection_method == "diversity_weighted"
        assert args.update_archive == "keep_diverse"

    def test_new_flags_at_defaults(self):
        import argparse

        p = argparse.ArgumentParser()
        p.add_argument("--resume", action="store_true")
        p.add_argument("--diversity-weight", type=float, default=None)
        p.add_argument("--diversity-bonus", type=float, default=None)
        p.add_argument("--no-meta-cognitive", action="store_true")
        args = p.parse_args([])
        assert args.resume is False
        assert args.diversity_weight is None
        assert args.diversity_bonus is None
        assert args.no_meta_cognitive is False


# ===========================================================================
#  Initialize run tests
# ===========================================================================


class TestInitializeRun:
    def test_fresh_start_no_initial_dir(self, sm):
        with tempfile.TemporaryDirectory() as tmp:
            orig = os.getcwd()
            os.chdir(tmp)
            try:
                with pytest.raises(RuntimeError, match="Need initial evaluation results"):
                    sm["initialize_run"](tmp, polyglot=False)
            finally:
                os.chdir(orig)

    def test_resume_no_directory(self, sm):
        a, s, r = sm["initialize_run"]("/nonexistent_abc", resume=True)
        assert a == ["initial"]

    def test_continue_from_nonexistent(self, sm):
        with tempfile.TemporaryDirectory() as tmp:
            a, s, r = sm["initialize_run"](tmp, prevrun_dir="/nonexistent")
            assert a == ["initial"] and s == 0 and r is None


# ===========================================================================
#  Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_filter_compiled_empty(self, sm):
        logger = types.SimpleNamespace(warning=lambda m: None)
        with tempfile.TemporaryDirectory() as tmp:
            assert sm["filter_compiled"]([], tmp, logger=logger) == []

    def test_get_original_score(self, sm):
        with tempfile.TemporaryDirectory() as tmp:
            _make_metadata(tmp, "initial", accuracy=0.42)
            assert sm["get_original_score"](tmp) == 0.42

    def test_get_full_eval_threshold(self, sm):
        with tempfile.TemporaryDirectory() as tmp:
            _make_metadata(tmp, "initial", accuracy=0.5)
            _make_metadata(tmp, "c1", accuracy=0.7)
            assert sm["get_full_eval_threshold"](tmp, ["c1"]) >= 0.4

    def test_archive_diversity_report(self, sm):
        with tempfile.TemporaryDirectory() as tmp:
            _make_metadata(tmp, "init", accuracy=0.5)
            _make_metadata(tmp, "ca", accuracy=0.6, parent="init")
            _make_metadata(tmp, "cb", accuracy=0.7, parent="init")
            logger = types.SimpleNamespace(info=lambda m: None)
            report = sm["get_archive_diversity_report"](tmp, ["init", "ca", "cb"], logger)
            assert report is not None and len(report) == 3

    def test_get_original_score_missing_returns_zero(self, sm):
        with tempfile.TemporaryDirectory() as tmp:
            assert sm["get_original_score"](tmp) == 0.0

    def test_choose_selfimproves_empty_archive(self, sm):
        with tempfile.TemporaryDirectory() as tmp:
            entries = sm["choose_selfimproves"](tmp, [], 2, method="random")
            assert entries == []

    def test_choose_selfimproves_all_fail_metadata(self, sm):
        with tempfile.TemporaryDirectory() as tmp:
            entries = sm["choose_selfimproves"](tmp, ["nonexistent"], 2, method="random")
            assert entries == []

    def test_choose_selfimproves_all_fail_metadata_best(self, sm):
        with tempfile.TemporaryDirectory() as tmp:
            entries = sm["choose_selfimproves"](tmp, ["nonexistent"], 2, method="best")
            assert entries == []

    def test_choose_selfimproves_all_fail_metadata_diversity(self, sm):
        with tempfile.TemporaryDirectory() as tmp:
            entries = sm["choose_selfimproves"](tmp, ["nonexistent"], 2, method="diversity_weighted")
            assert entries == []


# ===========================================================================
#  Extracted helpers (v0.2.2+)
# ===========================================================================


class TestConfigExtraction:
    def test_build_parser_returns_parser(self, sm):
        p = sm["build_parser"]()
        ns = p.parse_args(["--max-generation", "10", "--selfimprove-size", "3"])
        assert ns.max_generation == 10
        assert ns.selfimprove_size == 3

    def test_build_parser_defaults(self, sm):
        p = sm["build_parser"]()
        ns = p.parse_args([])
        assert ns.config is None
        assert ns.max_generation is None
        assert ns.resume is None

    def test_build_parser_boolean_flags(self, sm):
        p = sm["build_parser"]()
        ns = p.parse_args(["--polyglot", "--shallow-eval", "--no-meta-cognitive"])
        assert ns.polyglot is True
        assert ns.shallow_eval is True
        assert ns.no_meta_cognitive is True

    def test_build_parser_choices(self, sm):
        p = sm["build_parser"]()
        ns = p.parse_args(
            [
                "--selection-method",
                "best",
                "--run-baseline",
                "no_darwin",
                "--update-archive",
                "keep_diverse",
            ]
        )
        assert ns.selection_method == "best"
        assert ns.run_baseline == "no_darwin"
        assert ns.update_archive == "keep_diverse"

    def test_resolve_run_id_fresh(self, sm):
        import argparse

        ns = argparse.Namespace(continue_from=None, resume=None)
        rid = sm["resolve_run_id"](ns, "/tmp/output", resume_val=False)
        assert rid is not None and len(rid) > 10

    def test_resolve_run_id_continue_from(self, sm):
        import argparse

        ns = argparse.Namespace(continue_from="/tmp/some_run", resume=None)
        rid = sm["resolve_run_id"](ns, "/tmp/output", resume_val=False)
        assert rid == "some_run"

    def test_resolve_run_id_resume(self, sm):
        import argparse

        ns = argparse.Namespace(continue_from=None, resume=True)
        rid = sm["resolve_run_id"](ns, "/tmp/some_output", resume_val=True)
        assert rid == "some_output"

    def test_resolve_config_settings_defaults(self, sm):
        import argparse

        from godelion.config import Config

        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, "test_config.json")
            with open(cfg_path, "w") as f:
                json.dump(
                    {
                        "evolution": {"max_generations": 15, "self_improve_size": 4},
                        "evaluation": {"num_evals": 3},
                    },
                    f,
                )
            test_cfg = Config(cfg_path)
            ns = argparse.Namespace(
                max_generation=None,
                selfimprove_size=None,
                selfimprove_workers=None,
                selection_method=None,
                update_archive=None,
                num_evals=None,
                post_improve_diagnose=None,
                no_meta_cognitive=None,
                diversity_weight=None,
                diversity_bonus=None,
                shallow_eval=None,
                polyglot=None,
                no_full_eval=None,
                run_baseline=None,
                config=None,
                continue_from=None,
                resume=None,
                max_archive_size=None,
            )
            settings = sm["resolve_config_settings"](ns, cfg=test_cfg)
            assert settings["max_generation"] == 15
            assert settings["selfimprove_size"] == 4
            assert settings["num_swe_evals"] == 3
            assert "choose_method" in settings
            assert "meta_cognitive_val" in settings
            assert settings["meta_cognitive_val"] is True  # default

    def test_resolve_config_settings_cli_overrides(self, sm):
        import argparse

        from godelion.config import Config

        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, "test_config.json")
            with open(cfg_path, "w") as f:
                json.dump(
                    {
                        "evolution": {"max_generations": 15, "selection_method": "random"},
                        "evaluation": {"meta_cognitive_validation": True},
                    },
                    f,
                )
            test_cfg = Config(cfg_path)
            ns = argparse.Namespace(
                max_generation=50,
                selfimprove_size=None,
                selfimprove_workers=None,
                selection_method="best",
                update_archive=None,
                num_evals=None,
                post_improve_diagnose=None,
                no_meta_cognitive=True,
                diversity_weight=None,
                diversity_bonus=None,
                shallow_eval=None,
                polyglot=None,
                no_full_eval=None,
                run_baseline=None,
                config=None,
                continue_from=None,
                resume=None,
                max_archive_size=None,
            )
            settings = sm["resolve_config_settings"](ns, cfg=test_cfg)
            assert settings["max_generation"] == 50  # CLI overrides
            assert settings["choose_method"] == "best"  # CLI overrides
            assert settings["meta_cognitive_val"] is False  # --no-meta-cognitive

    def test_resolve_config_settings_meta_cognitive_override(self, sm):
        import argparse

        from godelion.config import Config

        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, "test_config.json")
            with open(cfg_path, "w") as f:
                json.dump({"evaluation": {"meta_cognitive_validation": False}}, f)
            test_cfg = Config(cfg_path)
            ns = argparse.Namespace(
                max_generation=None,
                selfimprove_size=None,
                selfimprove_workers=None,
                selection_method=None,
                update_archive=None,
                num_evals=None,
                post_improve_diagnose=None,
                no_meta_cognitive=None,
                diversity_weight=None,
                diversity_bonus=None,
                shallow_eval=None,
                polyglot=None,
                no_full_eval=None,
                run_baseline=None,
                config=None,
                continue_from=None,
                resume=None,
                max_archive_size=None,
            )
            settings = sm["resolve_config_settings"](ns, cfg=test_cfg)
            assert settings["meta_cognitive_val"] is False

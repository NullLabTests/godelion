"""Tests for Godelion core components."""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConfig:
    def test_default_config_loads(self):
        from godelion.config import Config

        config_path = Path(__file__).parent.parent / "config.yaml"
        cfg = Config(str(config_path))
        assert cfg.get("llm", "coding_model") is not None
        assert cfg.get("evolution", "max_generations") == 80

    def test_env_overrides(self):
        os.environ["GODELION_EVOLUTION__MAX_GENERATIONS"] = "100"
        from godelion.config import Config

        cfg = Config()
        cfg._apply_env_overrides()
        assert cfg.get("evolution", "max_generations") == "100"
        del os.environ["GODELION_EVOLUTION__MAX_GENERATIONS"]

    def test_deep_merge(self):
        from godelion.config import Config

        cfg = Config()
        base = {"a": {"b": 1, "c": 2}}
        override = {"a": {"b": 10, "d": 3}}
        cfg._deep_merge(base, override)
        assert base["a"]["b"] == 10
        assert base["a"]["c"] == 2
        assert base["a"]["d"] == 3

    def test_get_nested_default(self):
        from godelion.config import Config

        cfg = Config()
        assert cfg.get("nonexistent", "key", default="fallback") == "fallback"


class TestLlm:
    def test_extract_json_between_markers(self):
        from llm import extract_json_between_markers

        result = extract_json_between_markers('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

        result = extract_json_between_markers('Some text\n```json\n{"a": 1}\n```\nmore')
        assert result == {"a": 1}

        result = extract_json_between_markers('{"key": "value"}')
        assert result == {"key": "value"}

        result = extract_json_between_markers("no json here")
        assert result is None

    def test_extract_json_with_control_chars(self):
        from llm import extract_json_between_markers

        result = extract_json_between_markers('```json\n{"key": "va\x00lue"}\n```')
        assert result == {"key": "value"}

    def test_create_client_raises_for_unknown(self):
        from llm import create_client

        with pytest.raises(ValueError, match="not supported"):
            create_client("nonexistent-model-12345")


class TestToolLoader:
    def test_load_all_tools(self):
        from tools import load_all_tools

        tools = load_all_tools(logging=lambda x: None)
        assert len(tools) >= 2
        tool_names = [t["name"] for t in tools]
        assert "bash" in tool_names
        assert "edit" in tool_names


class TestEvoUtils:
    def test_is_compiled_self_improve(self):
        from utils.evo_utils import is_compiled_self_improve

        good_metadata = {
            "overall_performance": {
                "accuracy_score": 0.5,
                "total_unresolved_ids": ["id1"],
                "total_resolved_ids": ["id2"],
                "total_emptypatch_ids": [],
                "total_submitted_instances": 2,
            }
        }
        assert is_compiled_self_improve(good_metadata, num_swe_issues=[1])

        bad_metadata = {"overall_performance": {}}
        assert not is_compiled_self_improve(bad_metadata)

        empty_metadata = {
            "overall_performance": {
                "accuracy_score": 0,
                "total_unresolved_ids": [],
                "total_resolved_ids": [],
                "total_emptypatch_ids": [],
                "total_submitted_instances": 0,
            }
        }
        assert not is_compiled_self_improve(empty_metadata)

    def test_get_all_performance(self):
        from utils.evo_utils import get_all_performance

        with tempfile.TemporaryDirectory() as tmpdir:
            eval_file = os.path.join(tmpdir, "test_run_eval.json")
            with open(eval_file, "w") as f:
                json.dump(
                    {
                        "resolved_instances": 5,
                        "submitted_instances": 10,
                        "unresolved_ids": ["a", "b"],
                        "empty_patch_ids": ["c"],
                        "resolved_ids": ["d", "e"],
                    },
                    f,
                )

            results, overall = get_all_performance("test_run", results_dir=tmpdir)
            assert results is not None
            assert overall is not None
            assert overall["accuracy_score"] == 0.5
            assert overall["total_resolved_instances"] == 5
            assert overall["total_submitted_instances"] == 10


class TestCommonUtils:
    def test_load_json_file(self):
        from utils.common_utils import load_json_file

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"test": "data"}, f)
            f.flush()
            result = load_json_file(f.name)
            assert result == {"test": "data"}
            os.unlink(f.name)

    def test_read_file(self):
        from utils.common_utils import read_file

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world\n")
            f.flush()
            content = read_file(f.name)
            assert content == "hello world"
            os.unlink(f.name)


class TestRunEngine:
    def test_initialize_run_no_prev(self):
        import run as run_module

        with tempfile.TemporaryDirectory() as tmpdir:
            archive, start_gen = run_module.initialize_run(tmpdir)
            # Should fail because no initial folder exists
            assert archive == ["initial"]

    def test_choose_selfimproves_empty(self):
        import run as run_module

        result = run_module.choose_selfimproves("/nonexistent", ["initial"], 2, method="random")
        assert result == []

# Development Log

## Session 2026-06-16 (v0.2.2)

### Rationale

After analyzing the codebase, I identified the highest-impact small improvements:

1. **`choose_selfimproves` IndexError crash** — When no archive member has valid metadata (e.g. an empty archive or corrupt files), `random.choices([], k=N)` raises `IndexError`. This kills the entire evolutionary run. Fix: early return `[]` when `candidates` is empty, with a guard for the fallthrough `random.choices` path.

2. **`get_original_score` unhandled crash** — Called by `update_archive` to get the baseline score. If `initial/` or its `metadata.json` is missing, `FileNotFoundError` propagates up and kills the run. Fix: wrap in try/except, return 0.0 as fallback.

3. **`update_archive` redundant calls** — Both `keep_better` and `keep_diverse` branches called `get_original_score()` independently. Moved to a single call before the branch.

4. **Dead code `godelion/llm.py`** — A 364-line file shadowing the root `llm.py` (359 lines). Never imported by anything. The root `llm.py` is the one used across all modules (`from llm import ...`). Removed `godelion/llm.py`.

### Results

- **33 smoke tests pass** (was 28, +5 new edge case tests)
- **31/36 main suite tests pass** (same 5 pre-existing missing-dep failures)
- All pushed to `origin/master` as `bd07a55`

### Refactoring (v0.2.3)

1. **`run.py` `main()` extraction into `build_parser()`, `resolve_config_settings()`, `resolve_run_id()`** — The 200+ line `main()` now delegates CLI argument setup, config resolution, and run_id generation to dedicated testable functions.

2. **10 new smoke tests** for the extracted helpers:
   - `TestConfigExtraction::test_build_parser_returns_parser` / `defaults` / `boolean_flags` / `choices`
   - `TestConfigExtraction::test_resolve_run_id_fresh` / `continue_from` / `resume`
   - `TestConfigExtraction::test_resolve_config_settings_defaults` / `cli_overrides` / `meta_cognitive_override`

3. **Version bumped** from `0.1.0` → `0.2.3`

### Remaining Ideas (future sessions)

- **The `choose_selfimproves` "dead end" problem**: When a parent agent resolves all benchmark issues, it gets skipped entirely (line 230: `if not unresolved_ids: continue`). The system should fall back to resolved IDs or special entries (`solve_stochasticity`, `solve_contextlength`) more aggressively. Without this, the archive converges and evolution stalls.

- **Duplicate `llm.py`**: Now that `godelion/llm.py` is removed, the root `llm.py` remains. It has no type hints and uses hardcoded `MAX_OUTPUT_TOKENS = 4096` instead of reading from config like `godelion/llm.py` did. The improved version was in the deleted file. Should either add config-based settings to the root `llm.py` or reintegrate the better patterns.

- **Integration test for the full generation loop**: The smoke tests test individual functions, but there's no test that exercises `main()` with mocked harnesses. This would catch regressions in the orchestration logic.

- **Config validation at startup**: If config.yaml has typos or missing keys, the system fails at runtime. A `validate_config()` step at startup would catch this early.

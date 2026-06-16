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

### Remaining Ideas (future sessions)

- **The `choose_selfimproves` "dead end" problem**: When a parent agent resolves all benchmark issues, it gets skipped entirely (line 230: `if not unresolved_ids: continue`). The system should fall back to resolved IDs or special entries (`solve_stochasticity`, `solve_contextlength`) more aggressively. Without this, the archive converges and evolution stalls.

- **`run.py` `main()` refactor**: The main function handles arg parsing, config resolution, AND the generation loop. It's 200+ lines. Should split into `parse_args()`, `resolve_config()`, and `evolution_loop()`.

- **Duplicate `llm.py`**: Now that `godelion/llm.py` is removed, the root `llm.py` remains. It has no type hints and uses hardcoded `MAX_OUTPUT_TOKENS = 4096` instead of reading from config like `godelion/llm.py` did. The improved version was in the deleted file. Should either add config-based settings to the root `llm.py` or reintegrate the better patterns.

- **Integration test for the full generation loop**: The smoke tests test individual functions, but there's no test that exercises `main()` with mocked harnesses. This would catch regressions in the orchestration logic.

- **Config validation at startup**: If config.yaml has typos or missing keys, the system fails at runtime. A `validate_config()` step at startup would catch this early.

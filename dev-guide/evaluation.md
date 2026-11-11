# Evaluation internals

## Scope

`tah/evaluate/` loads benchmark rows, renders prompts, invokes backends and persists
graded samples. `script/tah/evaluation/` provides the CLI, protocol orchestration
and analysis. User-facing options live in [the evaluation guide](../guide/evaluation.md).

## Module map

- `datasets.py`: registry loading and standardized rows, including `_original_*` source columns.
- `jobs.py`: per-dataset prompts, indexed callbacks, durable samples, grading and combination.
- `backends.py`: reference, engine, HF and SGLang adapters; engine receives rendered prompt token IDs.
- `progress.py`: background tqdm rendering and recent-window gross token rates.
- `codeeval.py`: original harness code prompts, EvalPlus sanitizer export and unchanged base/plus graders.
- `code_prompts.py`: independent TaH2 reasoning templates and offline block extraction; no TaH2 runtime imports.
- `matheval.py`, `results.py`: math grading and result collection.
- `run_protocol.py`, `compare_runs.py`: protocol run/resume/merge, paired analysis and seed aggregation.

## Validating a change

Run CPU evaluation tests, including existing backend and durable-callback checks:

```bash
mkdir -p data/tmp/evaluation-tests
CUDA_VISIBLE_DEVICES='' TMPDIR="$PWD/data/tmp/evaluation-tests" .venv-tah/bin/python -m pytest -q \
  script/tah/evaluation/tests tah/engine/tests/test_eval_integration.py \
  --basetemp=data/tmp/evaluation-tests/pytest
```

For protocol reader/resume/merge changes, `test_protocol_recovery.py` simulates
host kill → resume → Ctrl-C → resume through `main_run` with a fake subprocess.
It checks same-source CSV/raw selection, preservation of committed rows from
interrupted leaves, pending and graded-but-unflushed raw rejection, outcome-blind
selection, repeated merges, unchanged normal results and strict corruption errors.
Run it together with `test_protocol_tools.py` using the CPU command above.

`test_code_prompts.py` loads only the local checkpoint tokenizer (override its path
with `TAH_CODE_TOKENIZER`). It compares all 164 HumanEval and 378 MBPP prompts to
the actual imported TaH2 prompt function, through both dataset loaders, with Hub
fetches redirected to verified local files. It also checks pre-change harness
prompt digests, chat-template kwargs, reasoning extraction and worker-persisted
solutions for both variants. No model is loaded and no dataset download is needed.
These parity tests intentionally require TaH2's evaluation modules to import in
the test environment; production code never imports them.

`test_generation_progress.py` uses a decorated fake engine and in-flight request
states to check gross counts before any callback, restarts (including the
clear-before-accounting gap), warmup exclusion and a frozen final rate. It also
checks the 60-second window, redirected stdout, 10-second log throttling, final
state on success/failure, and callback independence from a blocked log writer.
Artifact hashes captured from the previous harness guard CSV/raw JSON/JSONL bytes;
the recorded-code tests below also cover real outputs with grading workers.
For a harness-only change also run the engine's CPU suite:

```bash
CUDA_VISIBLE_DEVICES='' TMPDIR="$PWD/data/tmp/evaluation-tests" .venv-tah/bin/python -m pytest -q \
  tah/engine/tests --basetemp=data/tmp/evaluation-tests/pytest-engine
```

Code extraction runs in the grading workers (`_grade_timed`), because EvalPlus
sanitize takes about 12 minutes on one 8192-token length-capped output. The parent
persists finished futures in callback order and stops at the first unfinished one, so
the job CSV and `code_solutions.jsonl` keep callback order. `test_code_extraction_workers.py`
replays recorded R2 seed-42 code results (`data/engine/paper_protocol_seed42`, skipped
if absent) through `run_single_job`, with 0 and 4 workers. It requires byte-identical
`code_solutions.jsonl`, combined CSV and `samples.jsonl`, and checks that a slow extraction
is not overtaken. Its `slow` cases replay whole groups, including capped outputs (about 15 min).

Preflight a protocol without loading a model or launching evaluation:

```bash
CUDA_VISIBLE_DEVICES='' .venv-tah/bin/python script/tah/evaluation/run_protocol.py run \
  --config script/tah/recipes/qwen3_1.7/protocol_paper_code_tah2.yaml \
  --model_path models/TaH-plus-1.7B --backend tah_engine --seed 42 \
  --output_root data/tmp/evaluation-tests/plan-seed42 --plan-only
```

## Internals reference

- `infer.progress_snapshot()`, when present, returns `(generation_elapsed_seconds,
  gross_generated_tokens)` or `None` until measured generation starts. It is display
  data only: never add it to saved result rows or telemetry. Gross counts include
  restart waste, exclude prompt/EOS, and include every request regardless of queue.
- The renderer reads the frozen `generate` frame from the generation thread via
  `sys._current_frames()`, matching the unwrapped code object and engine identity.
  It retains the fixed local `states` list once, then drops frame references; no
  tracing, profiling hooks, engine monkeypatches or engine imports are introduced.
  It reads only Python `wasted_tokens` and `len(generated)` under CPython's GIL;
  no tensor reads, GPU synchronizations or engine locks. Read waste before length:
  `_restart` clears generated before incrementing waste, so a racing sample can
  briefly undercount but cannot double-count. Clamp to the previous gross count;
  reconcile the final count with `last_run.generated_tokens + wasted_tokens`.
  Sampling costs O(number of requests), once per five seconds, not per engine step.
- `SampleProgress` renders one stdout tqdm bar in a daemon thread, with
  `mininterval=10`, `miniters=0` and no per-job cursor positioning (plain `tail -f`).
  Callbacks only advance a Python int and never acquire a rendering lock or write
  the bar. Exit joins the thread after inference and forces the final state, also
  on errors. Token rates use generation-call wall time after warmup, including
  callbacks, and freeze before the grading tail. The 60-second window interpolates
  between five-second observations at the cutoff; before 60 seconds it uses all
  available time. Sample rate and ETA remain tqdm's own estimates.
- `code_prompt` defaults to `harness`; that branch still calls the original prompt
  builder with `reasoning=False` and sanitizes the whole output by entry point.
  Validate `harness | tah2` before backend setup. Math prompts are unaffected.
- `tah2` matches `tah2/evaluate/local_eval.py::process_batch_items`: strip the
  standardized question, leave assertion whitespace intact, render one user turn
  with `tokenize=False, add_generation_prompt=True`, and omit `enable_thinking`.
  This checkpoint's saved template ends at `<|im_start|>assistant\n`; it does not
  insert an open `<think>` prefix. Do not add one independently of the template.
- TaH2's registry points to `TaH-plus/eval_science_code/{humaneval,mbpp}.jsonl`.
  At revision `8737a89ea8d141dae003d33ca1f818de58b237b1`, those files are byte-identical
  to `data/initial_data/eval/` (whole-file SHA256 pinned in the parity test).
  Questions remain the entire `prompt`, including MBPP's triple quotes and embedded
  example. MBPP has a separate `assertion` field on all 378 rows; the loader carries
  it as `_original_assertion`, which `_build_problem` restores. Do not split the
  prompt to synthesize tests: the separate field contains additional test cases.
  HumanEval uses `test` → `_original_test` → assertion, unused by its template.
- Extraction matches the authors' actual offline call `extract_code(output_text)`
  **without** the optional entry point: first complete lowercase `python` fence
  containing a `def`, else last such fence, else empty. Thinking is not removed
  before selection, and no EvalPlus sanitization follows. Preserve this behavior
  even for draft code inside thinking, demos, missing fences or wrong names.
- Code grading remains deferred to EvalPlus, with original task IDs. Callback
  indexes and repeat indexes must survive out-of-order completion. Raw JSON text
  stays intact; only the submitted `solution` uses variant-specific extraction.
- Protocol group `overrides` merge onto the base eval recipe. Put `code_prompt`
  there, so derived configs and their resume hashes record the selection.
- Protocol `load_group_records` is shared by run's missing-pair calculation,
  post-subprocess refresh and merge. A flushed CSV row commits the same-leaf raw
  sample, regardless of the leaf/subprocess success status. Combined CSV/JSONL
  files supersede worker artifacts independently, retaining the legacy reader
  precedence; one-repeat resumes remap local sample 0 to the protocol index.
  Unbacked raw samples may be discarded only in `running`/`failed` leaves of
  unsuccessful or unfinished attempts and only for expected keys. They remain
  missing until a CSV-backed source exists. Duplicate CSV keys (within/across
  attempts), duplicate raw keys within an attempt, missing/pending committed raw
  records, cross-leaf matches and complete/unknown-leaf orphans raise errors.
  Selection never inspects correctness or output text. `merge_provenance.json`
  records every committed CSV/raw location and every discarded raw location with
  its reason and retained source, plus per-group and total counts. Legacy state
  files need no migration; no new state field is required on resume.

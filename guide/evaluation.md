# TaH evaluation

The entrypoint is `script/tah/evaluation/eval.py`. From the repository root,
activate `.venv-tah` with the `tah`, `evaluation` and `engine` extras installed
([installation](installation.md)). Help runs without a GPU:

```bash
export CUDA_VISIBLE_DEVICES=""
source .venv-tah/bin/activate
python script/tah/evaluation/eval.py --help
```

Evaluation commands below are GPU jobs. `CUDA_VISIBLE_DEVICES=0` selects GPU 0
explicitly, even if the shell hid GPUs earlier.

## Quick start

This creates a smoke config from the 1.7B recipe with batch 1, one repeat and a
512-token cap, then evaluates the first five GSM8K problems:

```bash
source .venv-tah/bin/activate
mkdir -p data/eval_configs
python - <<'PY'
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path('script/tah/recipes/qwen3_1.7/eval_tah.yaml').read_text())
cfg.update(batch_size=1, repeat_size=1, max_new_tokens=512)
Path('data/eval_configs/smoke.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
CUDA_VISIBLE_DEVICES=0 python script/tah/evaluation/eval.py \
  --eval_config data/eval_configs/smoke.yaml \
  --model_path nics-efc/TaH-plus-1.7B \
  --dataset_name gsm8k --backend tah_engine \
  --job_nums 1 --tp_size_per_job 1 --data_range 5 \
  --output_dir data/eval_smoke
```

`--model_path` takes a local checkpoint directory or a Hub ID (downloaded on first use).
Recipes: [`qwen3_1.7/eval_tah.yaml`](../script/tah/recipes/qwen3_1.7/eval_tah.yaml)
and [`qwen3_0.6/eval_tah.yaml`](../script/tah/recipes/qwen3_0.6/eval_tah.yaml); use
the 0.6B recipe only with a compatible 0.6B checkpoint. Both use one repeat, batch
16, a 4096-token cap, temperature 0.6, top_p 0.95, top_k 20, min_p 0, `max_iter` 2
and decider threshold 0.9.

## Backends

| `--backend` | Behavior |
| --- | --- |
| `tah_engine` (default) | TaH v1 engine on one GPU: continuous batching, CUDA graphs and host swap. Returns exact generated-token pass counts. See [engine](engine.md). |
| `tah` | Correctness reference: `TaHForCasualLM_generate` on the intact model, SDPA attention, `device_map="auto"`. Returns pass counts. |
| `hf` | Standard `AutoModelForCausalLM.generate`, for ordinary base-model baselines. No TaH wrapper and no pass counts. |
| `sglang` | `sglang.Engine` with TP, dtype and memory settings. For SGLang-supported checkpoints; no TaH wrapper and no pass counts. |

`tah_engine` receives every prompt of its job once and keeps one engine for all
datasets in the job. `hf` and `sglang` run a 100-token warmup; `tah` does not.
Each job loads one backend for its problems across all datasets. The `tah` reference left-pads
each batch and can allocate large prompt × batch × vocabulary buffers, so on a
24 GB GPU lower `batch_size` if it runs out of memory.

## Datasets

Names must appear in
[`dataset_configs.json`](../tah/evaluate/eval_configs/dataset_configs.json).
Hub datasets load through `datasets`; local paths are relative to the repo root.
A missing requested split falls back to `train`, then `test`.

| Name | Source | Configured split | Scoring |
| --- | --- | --- | --- |
| `aime25` | `yentinglin/aime_2025` | `train` | Expression accuracy |
| `aime24` | `Maxwell-Jia/AIME_2024` | `test` | Expression accuracy |
| `gsm8k` | `pss0204/gsm8k_test`, subset `gsm8k_test` | `test` | Expression accuracy |
| `math500` | `HuggingFaceH4/MATH-500` | `test` | LaTeX/math accuracy |
| `amc23` | `math-ai/amc23` | `test` | LaTeX/math accuracy |
| `olympiadbench` | `realtreetune/olympiadbench` | `test` | LaTeX/math accuracy |
| `minerva` | `data/initial_data/eval/minerva.jsonl` | `test` | LaTeX/math accuracy |
| `gpqa` | `data/initial_data/eval/gpqa_diamond.jsonl` | local JSON | Multiple-choice accuracy |
| `arc_e`, `arc_c` | `data/initial_data/eval/arc_{e,c}.jsonl` | local JSON | Multiple-choice accuracy |
| `mmlu_stem`, `mmlu_redux` | `data/initial_data/eval/*.jsonl` | local JSON | Multiple-choice accuracy |
| `humaneval`, `mbpp` | `data/initial_data/eval/{humaneval,mbpp}.jsonl` | local rows in `train` | EvalPlus base and plus pass@k |
| `livecodebench` | Registered only | — | **Unsupported**: rejected before loading |

`brumo25` and `chmath` have math grader entries but no dataset entries, so they
cannot be selected. Unknown or duplicate names fail. LiveCodeBench cannot be
scored by HumanEval/MBPP tests.

Math prompts use the tokenizer's user chat template with a generation prompt;
unextracted answers are incorrect. Code prompts ask for a self-contained Python
script in a markdown block; outputs are sanitized using each task's entry point.
If a code tokenizer has no chat template, the task prompt is used directly.

YAML `code_prompt` selects `harness` (default, existing prompts and sanitization)
or `tah2` for HumanEval/MBPP. `tah2` uses the authors' benchmark-specific reasoning
templates with a generation prompt and no assistant response priming or
`enable_thinking` override. MBPP test cases come from the local row's `assertion`
field, preserved verbatim; the embedded example in `prompt` is retained too.
Outputs use TaH2's Python-block extraction without additional sanitization;
EvalPlus graders are unchanged. Other datasets ignore this setting.

## Selecting rows and repeats

Use comma-separated `--dataset_name` to share one backend load across datasets:

```bash
source .venv-tah/bin/activate
CUDA_VISIBLE_DEVICES=0 python script/tah/evaluation/eval.py \
  --eval_config data/eval_configs/smoke.yaml \
  --model_path nics-efc/TaH-plus-1.7B \
  --dataset_name math500,gpqa --backend tah_engine \
  --dataset_ranges math500=0:8,gpqa=all \
  --job_nums 1 --tp_size_per_job 1 \
  --output_dir data/eval_multi
```

Selection works on each dataset's loaded row order, before job sharding:

- `--data_range 8` selects `[0,8)` **in each dataset**; `--data_range 8 16` selects
  `[8,16)`. The end is exclusive and clamped to the dataset length.
- `--dataset_ranges math500=0:8,gpqa=all` overrides the shared range for the named
  datasets. Unlisted datasets use `--data_range`, or all rows if it is absent.
- `--data_ids gsm8k_0,gsm8k_5` overrides both. IDs are `<dataset>_<original_id>`;
  unmatched IDs are skipped, and no match is an error. HumanEval IDs contain a
  slash: `humaneval_HumanEval/0`.

YAML `repeat_size` produces that many samples per problem, indexed
`0..repeat_size-1`. Repeats are adjacent prompts, so `batch_size` counts
**samples**, not problems. Greedy repeats are identical. Each problem's repeats
stay in one job; problems are distributed round-robin across `--job_nums`.

## CLI options

| Flag | Default | Effect |
| --- | --- | --- |
| `--eval_config PATH` | required | Flat YAML mapping. A copy is saved in each dataset run directory. |
| `--model_path PATH_OR_ID` | required | Local checkpoint directory or Hub ID. |
| `--backend {tah_engine,tah,hf,sglang}` | `tah_engine` | Implementation; see [backends](#backends). |
| `--dataset_name NAMES` | required | One or more comma-separated names. |
| `--engine-options JSON` | none | JSON object merged over the YAML `engine` mapping; keys are the [engine options](engine.md#engine-options). |
| `--engine-max-batch-size INT` | none | Overrides `engine.max_batch_size`. |
| `--engine-host-swap-gib FLOAT` | none | Overrides `engine.host_swap_bytes` in GiB (default 8). |
| `--engine-memory-fraction FLOAT` | none | Overrides `engine.memory_fraction` (default 0.9). |
| `--max-new-tokens INT` | none | Overrides the YAML cap for every request. |
| `--grading-workers INT` | none | Overrides YAML `grading_workers`; 0 grades inline. |
| `--job_nums INT` | 1 | Parallel workers, each with its own model instance. Jobs without problems are skipped. |
| `--tp_size_per_job INT` | 1 | GPUs per worker. `tah_engine` requires 1. |
| `--output_dir PATH` | `<model_path>/eval_results` | Parent of the dataset/backend/timestamp directories. Prefer an explicit path, especially for Hub IDs. |
| `--data_range END` or `--data_range START END` | all rows | Per-dataset slice; end exclusive. |
| `--dataset_ranges SPECS` | none | Per-dataset `name=start:end` or `name=all` overrides. |
| `--data_ids IDS` | none | Standardized IDs; overrides ranges. |
| `--del_job_dir VALUE` | `True` | Delete `job_<id>/` after success. The value is parsed with `bool()`: **any nonempty string is true**, including `False`. Use `--del_job_dir ''` to keep them. Failed runs keep them. |
| `--logger_level LEVEL` | `WARNING` | `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`. |
| `--random_seed INT` | 42 | Seed set in every worker (Python, NumPy, torch, CUDA). Batch composition, backend and sharding can still change outputs. |

## YAML keys

The driver reads a flat mapping. Model path, datasets, job counts, output roots and
the seed come from the CLI, not YAML. Unknown keys are ignored, but a misspelled
required key causes a worker `KeyError`.

| Key | Backends | Notes |
| --- | --- | --- |
| `temperature` | all | Required. 0 is greedy; positive values sample (`tah`, `hf`, `tah_engine`). |
| `top_p` | all | Required. Nucleus mass in [0,1]; 1 disables it. |
| `top_k` | all | `tah`/`tah_engine`: 0 disables (default); positive values keep cutoff ties. Do not pass null. `hf`/`sglang`: omit to use their defaults. |
| `min_p` | all | Keeps tokens with probability ≥ `min_p` × maximum probability. 0 disables. Do not pass null to TaH. |
| `max_new_tokens` | all | Required. Generation cap per sample; EOS can stop earlier. |
| `batch_size` | `tah`, `hf`, `sglang` | Required. Samples per fixed batch. The engine ignores it; use `engine.max_batch_size`. |
| `repeat_size` | all | Required. Samples per selected problem. |
| `dtype` | all | `bfloat16` (the engine requires it). `tah` and `hf` look up `torch.<name>`. |
| `grading_workers` | all | Grading processes. Default 4 for `tah_engine`, 0 otherwise. 0 grades inside the callback. |
| `engine` | `tah_engine` | Mapping of engine constructor options; see [engine settings](#engine-settings). |
| `use_flash_attention` | `hf` | `true` selects `flash_attention_2`; needs flash-attn. Ignored by TaH and SGLang. |
| `mem_fraction_static`, `attention_backend` | `sglang` | Forwarded to SGLang. `mem_fraction_static` is ignored by `hf` and `tah`, so lowering it does not fix reference OOM. |

Reference sampling order is temperature → softmax → min-p → top-k → top-p →
multinomial. Top-p keeps at least one token. Greedy uses argmax and ignores the
filters. HF and SGLang use their own sampling, so tokens are not expected to match
across backends.

### TaH model overrides

`tah` and `tah_engine` also read the model keys of
[`TaHConfig`](../tah/model/tah_config.py), replacing the saved `tah_config.json`
fields that are non-null. Dictionaries are **replaced, not merged**: changing only
`iter_decider_kwargs.threshold` resets the other decider keys to their defaults, so
copy the full recipe dictionary. Keep `max_iter` consistent with the decider's cap, and
avoid architectural overrides that differ from the trained checkpoint (they cause
weight or shape errors).

## Engine settings

The same YAML works with `--backend tah` and `--backend tah_engine`; the reference
ignores the `engine` mapping.

```yaml
engine:
  max_batch_size: 64
  memory_fraction: 0.9
  host_swap_bytes: 8589934592  # 8 GiB
  prefill_chunk_size: 128
  cuda_graphs: true
  compile: false
  overlap: null                # null = auto (on); false = ordinary scheduler
grading_workers: 4
```

Every constructor option in [engine options](engine.md#engine-options) is accepted.
Precedence: the YAML `engine` mapping, then `--engine-options`, then the three
explicit `--engine-*` flags. The harness caps the resident limit to the number of
prompts in the job. `engine.seed` defaults to `--random_seed`. Top-level
`max_new_tokens` governs every evaluation request.

For long generations (up to 16,384 tokens) see [scheduling, memory and long generations](engine.md#scheduling-memory-and-long-generations).

Grading runs alongside generation. With `grading_workers > 0`, a generated raw JSON
appears immediately with `grading_status: pending`; it is atomically replaced once
graded, and the CSV row is flushed. The job waits for every grade before reporting
success. Math-verify runs in each process's main thread so its signal timeouts work.
EvalPlus uses its own process pool.

## GPU and job allocation

Set `CUDA_VISIBLE_DEVICES` to an explicit **comma-separated list of integer
physical GPU indices**. For example, `CUDA_VISIBLE_DEVICES=0,1` with `--job_nums 2
--tp_size_per_job 1` gives each worker one GPU; `--job_nums 1 --tp_size_per_job 2`
gives one job both GPUs. Each worker sees its GPUs renumbered from zero. Do not use
UUID or MIG identifiers.

If visibility is unset **or empty**, the driver assumes GPUs 0..7 and does not check
the count. Empty visibility is safe for `--help` and the collector, but it is not a
way to exclude GPUs from an evaluation: worker startup rewrites the variable from
the driver's pool. `tah_engine` runs on one GPU and needs `--tp_size_per_job 1`.

The driver has no job queue, multi-node mode, CPU device option or automatic retry.

## Crashes and partial results

Each completed sample writes an atomic `sample_<idx>.json` and flushes its worker
CSV row. The `tah`, `hf` and `sglang` backends finish a whole generation batch
before returning, so samples in an interrupted batch are not saved. The engine
backend persists each result as it arrives.

A worker exception writes `job_<id>.error.txt` beside the dataset's run artifacts.
The parent combines whatever completed, checks exact `(problem_id, sample_idx)`
coverage and marks `run_status.json` `complete` or `failed`. A failed worker,
missing report or coverage mismatch exits nonzero and keeps partial results and
worker directories. A failure in one job marks the whole session failed.

Status is updated at combination, not after each sample. Each job shows one tqdm
bar over finished samples, with tqdm's sample rate, elapsed time and ETA. It writes
to stdout about every 10 seconds, including while no samples have finished, and
prints the final state. Redirect stdout and stderr to one log and watch with
`tail -f` (as below).

`tah_engine` adds `865 tok/s (60s), 812 avg`: gross generated tokens per wall-clock
second, including in-flight, swapped, finished and restart-discarded tokens; prompt
tokens and EOS are excluded. The first number covers the last 60 seconds (the
available interval at startup); the second averages this job's generation call.
Both exclude model setup and warmup, include prefills, swaps and callbacks, and stop
when generation returns, before the grading tail. Counts are sampled every five
seconds and displayed on the bar's next refresh. Backends without a live counter show
the same sample bar without the token postfix. For persisted timing and net output
throughput, see [job timing](results.md#job-timing-and-engine-telemetry).
A hard kill can leave `running` with no combined files. Treat runs without
`run_status.json` as unverified.

There is **no resume** for an evaluation: a rerun gets a new timestamp and starts
the selected workload again. To continue after an interruption, keep the old output,
count the saved CSV and raw-output pairs, and run only the missing IDs
(`--data_ids`) into a new output root. The collector needs `--allow_incomplete` for
failed or running results; see [results](results.md).

## Long runs in tmux

Start long jobs in a **detached** tmux session and watch them through a read-only
client. Detaching protects the job from a dropped terminal; an attached client can
still send Ctrl-C into it.

```bash
mkdir -p data/eval_smoke_detached
tmux new-session -d -s tah-eval -c "$PWD" \
  "bash -c 'set -o pipefail; CUDA_VISIBLE_DEVICES=0 .venv-tah/bin/python script/tah/evaluation/eval.py --eval_config data/eval_configs/smoke.yaml --model_path nics-efc/TaH-plus-1.7B --dataset_name gsm8k --backend tah_engine --job_nums 1 --tp_size_per_job 1 --data_range 5 --output_dir data/eval_smoke_detached 2>&1 | tee data/eval_smoke_detached/console.log'"
tmux attach-session -r -t tah-eval
```

To watch the redirected log without attaching:

```bash
tail -f data/eval_smoke_detached/console.log
```

For a snapshot without attaching, use `tmux capture-pane -p -S -30 -t tah-eval`.
The session disappears when the command ends; check `console.log` and
`run_status.json` to tell completion from interruption.

## Report protocol runs

Evaluations for the research report follow [report/PROTOCOL.md](../report/PROTOCOL.md),
which defines the benchmarks, seeds, token caps and statistics. Run them with the
protocol configs below rather than ad-hoc settings. The runner calls `eval.py`
for each group, records every seed, selection, command and derived config, and
merges the groups into ordinary dataset directories:

- [`protocol_wacc.yaml`](../script/tah/recipes/qwen3_1.7/protocol_wacc.yaml): R1,
  the reference (`--backend tah`) or the engine (`--backend tah_engine`).
- [`protocol_paper.yaml`](../script/tah/recipes/qwen3_1.7/protocol_paper.yaml): R2,
  the engine only, with all nine benchmarks and eight samples per problem for AIME25 and AMC23.
- [`protocol_paper_code_tah2.yaml`](../script/tah/recipes/qwen3_1.7/protocol_paper_code_tah2.yaml):
  R2's code-prompt amendment, HumanEval+ and MBPP+ only, one sample at 8192 tokens.

Run each backend and seed into its own output root; `run` and `merge` are separate
commands (the example is the R1 engine run at seed 42):

```bash
CUDA_VISIBLE_DEVICES=0 .venv-tah/bin/python script/tah/evaluation/run_protocol.py run \
  --config script/tah/recipes/qwen3_1.7/protocol_wacc.yaml \
  --model_path models/TaH-plus-1.7B --backend tah_engine --seed 42 \
  --output_root data/eval_protocol/r1_engine_seed42 --logger_level WARNING \
  && CUDA_VISIBLE_DEVICES=0 .venv-tah/bin/python script/tah/evaluation/run_protocol.py merge \
  --output_root data/eval_protocol/r1_engine_seed42
```

Use `--backend tah --seed 1234` for the R1 reference run, and `--config
protocol_paper.yaml` with `--output_root data/eval_protocol/r2_engine_seed<seed>` for R2.
Merged directories are written under `<output_root>/canonical/`.

- **Resume:** repeat the same `run` command; only missing `(problem_id, sample_idx)`
  pairs are regenerated. Rerun `merge` after an interrupted merge.
- **Seeds:** resumed sub-runs and group boundaries can change sampled outputs relative
  to one uninterrupted run, because the harness does not save generator state between
  invocations. The runner records each seed, selection, command and config.

Resume and merge count a pair only from its unique flushed CSV row and matching
raw record in the same dataset directory, including completed samples from failed
or interrupted sub-runs. Raw records without a CSV row in a `running` or `failed`
directory are regenerated; they never replace the CSV-backed copy. Merge lists
every dropped record and its retained source, and writes counts, IDs and source
files to `canonical/merge_provenance.json`. Duplicate CSV-backed pairs, missing
raw records, unexpected pairs and other unexplained mismatches remain errors.

Compare the R1 reference with engine seeds. `--margin` defaults to 0.03 (three points):

```bash
CUDA_VISIBLE_DEVICES='' .venv-tah/bin/python script/tah/evaluation/compare_runs.py equivalence \
  --protocol-config script/tah/recipes/qwen3_1.7/protocol_wacc.yaml \
  --a data/eval_protocol/r1_reference_seed1234/canonical \
  --b data/eval_protocol/r1_engine_seed42/canonical \
  --b data/eval_protocol/r1_engine_seed1234/canonical \
  --b data/eval_protocol/r1_engine_seed2024/canonical \
  --label-a Reference --label-b Engine-seeds --margin 0.03 \
  --output-prefix data/eval_protocol/r1_equivalence
```

The output reports, per dataset and pooled, the candidate-minus-reference accuracy
difference, its paired SE, 95% and 90% intervals, and the passes-per-token ratio.
Equivalence holds only when the 90% interval lies inside the margin. The verdict
`equivalence_not_shown` means that interval does not fit; it does not show a difference.

Aggregate the R2 seed runs against the paper values:

```bash
CUDA_VISIBLE_DEVICES='' .venv-tah/bin/python script/tah/evaluation/compare_runs.py aggregate \
  --protocol-config script/tah/recipes/qwen3_1.7/protocol_paper.yaml \
  --run data/eval_protocol/r2_engine_seed42/canonical \
  --run data/eval_protocol/r2_engine_seed1234/canonical \
  --run data/eval_protocol/r2_engine_seed2024/canonical \
  --output-prefix data/eval_protocol/r2_aggregate
```

Both analysis commands write Markdown and JSON under the output prefix. They reject
failed run statuses, missing or duplicate pairs and incomplete EvalPlus results;
add `--no-paper-values` to omit the paper-value columns.

Source of truth: [CLI](../script/tah/evaluation/eval.py),
[job runner](../tah/evaluate/jobs.py), [backends](../tah/evaluate/backends.py),
[config loading](../tah/model/tah_model.py), [deciders](../tah/model/iter_decider.py),
[sampling](../tah/model/utils.py).

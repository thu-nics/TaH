# Reading and comparing TaH results

The evaluation driver writes one directory per dataset, backend and run under
`--output_dir` (default `<model_path>/eval_results`). Check completion and sample
coverage before interpreting accuracy: a CSV can exist after a failed run.

## Directory layout

```text
<output_root>/
  <dataset>_<backend>/
    YYYYMMDD_HHMMSS/
      [TASK_<start>_<end>/]          # present for a selected row range
        <eval_config_basename>       # copy of the original YAML
        effective_config.yaml        # config sent to workers, with CLI overrides
        run_status.json
        job_<id>.metrics.json        # job timing and engine telemetry
        detailed_results.csv         # combined rows
        evaluation_stats.csv         # combined per-problem and total summaries
        samples.jsonl                # combined raw samples
        code_solutions.jsonl                # HumanEval/MBPP only
        code_solutions.eval_results.json    # HumanEval/MBPP after grading
        job_<id>.error.txt           # present after a caught worker exception
        job_<id>/                    # kept on failure or with --del_job_dir ''
          details/<problem_id>/sample_<sample_idx>.json
  results_summary.csv                # written by the collector only
```

All datasets in one invocation share the timestamp, which uses the host's local
clock. `TASK_0_8` means row indices `[0,8)` after loading, not a problem-ID range.
Full runs and `--data_ids` selections have no `TASK_` directory. Source IDs with a
slash create nested directories (`humaneval_HumanEval/0/`).

Retained worker directories (`job_<id>/`) show progress during a run: the job CSV
is flushed after each sample. Transient `*.tmp` files after a crash are not records.
Console logs need shell redirection.

## Essential CSV columns

`detailed_results.csv` (one row per completed sample, sorted by problem and sample):

| Column | Meaning |
| --- | --- |
| `problem_id` | Standardized ID, `<dataset>_<original_id>`. |
| `sample_idx` | Zero-based repeat index. Together with `problem_id`, the unique key. |
| `correct_answer`, `predicted_answer` | Ground truth and parsed answer as text. Code rows show `pending_code_eval`. |
| `is_correct` | `True`/`False` from the math grader. Code rows are placeholders until EvalPlus runs. |
| `output_tokens` | Retokenized decoded output length. |
| `processing_time` | Backend-reported seconds. Conventions differ by backend (SGLang attaches the batch time to every output; the engine reports request latency including queue time). Do not sum it for throughput. |
| `mean_passes_per_token`, `fraction_tokens_at_least_2_passes` | Pass metrics, blank when unavailable. |
| `pass_stat_tokens` | Generated tokens with pass counts; the weight for pass aggregation. |

`input_tokens` (raw question length) and `has_answer` are also present.

For throughput, measure generation wall time separately: summing `processing_time`
over overlapping requests or jobs is not a throughput measure.

## Statistics and pass metrics

`evaluation_stats.csv` has one row per problem, a blank row and a `Total Accuracy` row.
Use the integer counts for exact accuracy: `correct_count / total_samples`. The stored
`accuracy` is rounded to three decimals. Zero samples means no measurement, even
if a total row shows zero.

Pass count of a generated token = passes at the context position whose final logits
sampled it. This includes the last prompt position's work for the first generated
token. EOS is excluded. The `tah` and `tah_engine` backends export these counts;
`hf` and `sglang` do not.

Aggregate pass metrics over rows with `pass_stat_tokens > 0`:

```text
aggregate metric = sum(row metric * row pass_stat_tokens) / sum(row pass_stat_tokens)
```

This weights by generated tokens, not by problems or output length. With `max_iter` 2,
mean passes is approximately 1 + the fraction of tokens with at least two passes.
Pass counts measure compute, not correctness.

## Raw samples

Each `sample_<idx>.json` contains `problem_id`, `problem`, `output` (decoded text),
`correct_answer`, `predicted_answer` and `is_correct`. Code samples add
`extracted_code` and `entry_point`. Engine samples also include `token_ids`,
`pass_counts`, `prompt_pass_counts`, `finish_reason` and request `timing`.
`samples.jsonl` combines them and adds `id` and `sample`. Raw files are kept after
successful worker cleanup.

## Run status

`run_status.json` contains `status` (`running`, `complete` or `failed`),
`expected_samples`, `completed_samples`, `batch_size`, `repeat_size`, `random_seed`,
`completed_jobs`, `failed_jobs` and, on errors, `error`. `complete` requires worker
success and exact sample coverage. Status is written at combination, not per sample,
so `running` means the run did not finish. Treat runs without status as unverified.
A worker traceback is in `job_<id>.error.txt`.

## Job timing and engine telemetry

`job_<id>.metrics.json` describes one job, which may serve several datasets. Do not
sum the duplicated copies. Key fields:

- `setup_seconds` (load and capture), `inference_call_seconds`, `job_wall_seconds`.
- `grading_work_seconds`, `grading_inline_seconds`, `grading_tail_seconds`. Parallel
  grading overlaps, so their sum is not an elapsed time. With grading workers, code
  extraction (EvalPlus sanitize or the `tah2` block extraction) runs in the workers and is
  counted in `grading_work_seconds`; without workers it runs inline in the callback.
- `engine_options`: the effective options, including the resident cap.
- `memory_budget`: startup KV reservations, graph capture memory and the host swap
  arena (`host_swap`: `arena_bytes`, `page_bytes`, `pages`, `pinned`).
- `engine_run`: `generation_seconds`, `callback_seconds`, `generated_tokens`, `swaps`,
  `restarts`, `wasted_tokens`, `blocked_restart_admission_steps`, `free_rows`,
  `overlap`, `ghost_rows`, `overlap_drains`, `graph_steps`, `eager_steps`,
  `host_swap_arena_bytes` (reserved arena), `host_swap_peak_bytes` (peak arena bytes held
  by swapped requests, in whole pages, over the engine's lifetime), and
  `peak_gpu_allocated_bytes` / `peak_gpu_reserved_bytes` (PyTorch peaks after warmup).
- `run_status.code_grading_seconds` for code grading.

For engine throughput, use `engine_run.generated_tokens / generation_seconds`. It
excludes load, warmup and synchronous callbacks.

## Code evaluation

For HumanEval and MBPP, the driver sanitizes each output into `code_solutions.jsonl`
and scores it with EvalPlus on base and plus tests. Scoring uses only the selected
tasks, so subset runs are not full-benchmark scores. Cached ground truth lives in
EvalPlus's cache directory, outside the output root.

`code_solutions.eval_results.json` contains per-task completions with `base_status`
and `plus_status` (`pass` on success), failing inputs, and `pass_at_k.base` and
`pass_at_k.plus`. Keys `pass@1`, `pass@10` and `pass@100` appear only when every task has
at least that many samples. CSV code placeholders are never rewritten; use this JSON
or the collector's `pass@1 (base)` metric. A missing grade file means ungraded code,
not zero.

## Results collector

Point `--results_dir` at the output root that contains the dataset folders, not at a
dataset or timestamp directory:

```bash
export CUDA_VISIBLE_DEVICES=""
source .venv-tah/bin/activate
python script/tah/evaluation/collect_results.py \
  --results_dir data/eval_smoke --model_name TaH-plus-1.7B
```

This writes `data/eval_smoke/results_summary.csv`, one row per dataset and run. To
include failed or running runs for diagnosis, add `--allow_incomplete` and
`--output_csv`; such rows are not complete benchmarks.

| Flag | Default and behavior |
| --- | --- |
| `--results_dir PATH` | Required. Output root; a missing root is an error. |
| `--output_csv PATH` | `<results_dir>/results_summary.csv`. Overwritten. |
| `--model_name TEXT` | Label; defaults to the parent directory name, so set it for custom roots. |
| `--allow_incomplete` | Off by default: any failed or running run aborts collection. On: included for diagnosis. |

Runs without status metadata are labeled `legacy_unverified`. The collector checks
only totals, so it cannot prove exact coverage or detect every corruption. The Python
API is `tah.evaluate.collect_results(results_dir, output_file=None, model_name=None,
allow_incomplete=False)`.

Summary columns: `model`, `run`, `dataset`, `backend`, `metric` (`accuracy` or
`pass@1 (base)`), `accuracy_or_pass_at_1`, `correct_count`, `total_samples`,
`problem_count`, `avg_output_length`, `mean_passes_per_token`,
`fraction_tokens_at_least_2_passes`, `run_status`, `expected_samples`. Keep the
source CSVs, code JSON, copied YAML and status files with any comparison.

## Compare runs

1. Check status and coverage: each run must have the expected `(problem_id,
   sample_idx)` pairs, from the detailed CSVs. Exclude partial runs.
2. Match everything that affects outputs: checkpoint and tokenizer, dataset
   version and IDs, row slice, repeats, sampling, `max_new_tokens`, dtype, decider
   settings, backend, batch and job configuration, and seed. The copied YAML and
   status cover part of this; keep the exact command line and checkpoint revision too.
3. Compare like datasets and metrics. Do not mix subsets with full datasets. Compare
   code `pass@1 (base)` with base, and inspect plus scores separately.
4. For math, use exact counts. Use output lengths and pass metrics to explain compute
   differences, and measure wall time and GPU memory separately.

Source of truth: [result writers](../tah/evaluate/jobs.py),
[code grader](../tah/evaluate/codeeval.py), [collector](../tah/evaluate/results.py).

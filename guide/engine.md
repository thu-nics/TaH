# TaH inference engine

`TaHEngine` generates from local TaH v1 checkpoints on one CUDA GPU. It runs
prompt passes in bounded chunks, mixes token depths in each decode step and admits
queued requests as others finish. CUDA graph decode is on by default. `tah_engine`
is the default TaH v1 evaluation backend; `--backend tah` runs the intact reference
model instead.

Setup is in [installation](installation.md) and benchmarks are in
[evaluation](evaluation.md). TaH2's `tah2/minisgl/` is a separate engine. Validation
tools, parity fixtures and the cache, attention and KV APIs are documented for
developers in [dev-guide/engine.md](../dev-guide/engine.md).

## Generate with the engine

Run from the repository root in the `.venv-tah` environment:

```bash
source .venv-tah/bin/activate
CUDA_VISIBLE_DEVICES=0 python - <<'PY_ENGINE'
import yaml
from tah.engine import TaHEngine, Request, SamplingParams

config = yaml.safe_load(open('script/tah/recipes/qwen3_1.7/eval_tah.yaml'))
engine = TaHEngine('models/TaH-plus-1.7B', config_override=config,
                   max_batch_size=1, graph_batch_sizes=[1])
requests = [Request('<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n<|im_start|>assistant\n',
                    max_new_tokens=128, seed=42)]
results = engine.generate(requests, SamplingParams(temperature=.6, top_k=20, top_p=.95),
                          on_result=lambda result: print(result.text))
print(results[0].generated_pass_counts)
print(engine.last_run)
PY_ENGINE
```

`generate(requests, sampling=None, on_result=None)` returns `Result` objects in
input order. `on_result` receives each result as it finishes, so completion order
can differ. The call blocks and cannot be nested or run concurrently on one engine.
An exception raised by a callback propagates after the request's KV and host swap
memory are released.

`close()` releases the engine's pinned host swap arena (see [memory notes](#scheduling-memory-and-long-generations));
`TaHEngine` is also a context manager that closes on exit.

A request is a token-ID list, a raw string, or `Request(prompt, max_new_tokens=1024,
seed=0)`. Strings are tokenized with the checkpoint's `tokenizer.json`; apply the
chat template yourself, as the example does. Plain strings and lists use the
engine's `max_new_tokens` and the seed `seed + input index`. A checkpoint without a
tokenizer accepts token IDs and returns empty text.

Each `Result` has `request_id` (input index), `token_ids`, `text`,
`prompt_pass_counts`, `generated_pass_counts`, `finish_reason` (`eos` or `length`)
and `timing`. `timing` holds `total_seconds` (including queue time),
`queue_seconds`, `prefill_seconds`, `swaps`, `restarts` and `wasted_tokens`
(generated tokens discarded by restarts). Pass counts are defined in
[results](results.md#statistics-and-pass-metrics). EOS is never counted.

## Engine options

Keyword arguments of `TaHEngine`. The first argument is the checkpoint path, and
`config_override` (second) uses the same TaH override rules as the evaluation YAML
(see [TaH model overrides](evaluation.md#tah-model-overrides)).

| Option | Default | Effect |
|---|---|---|
| `device` | `"cuda"` | Device. `"cpu"` is for development and needs an explicit `kv_capacity`. |
| `max_batch_size` | 128 | Maximum resident requests; each contributes one row per decode step. The evaluation harness uses 64. |
| `prefill_chunk_size` | 128 | Prompt tokens per prefill chunk; bounds the vocabulary-logit memory. |
| `memory_fraction` | 0.9 | Fraction of free GPU memory (after weights) reserved for KV and workspaces. |
| `kv_capacity` | `None` | KV entries (page size 1). `None` sizes them automatically; a larger-than-safe value is rejected. |
| `admission_headroom` | 0.1 | Fraction of KV slots kept free after a new prompt's worst case. |
| `host_swap_bytes` | 8 GiB | Host RAM for swapped state, allocated and pinned in full at startup (exact, fixed for the engine's lifetime). Overflow restarts a request; 0 disables swapping. |
| `max_new_tokens` | 1024 | Cap for plain string and token-list requests. |
| `seed` | 0 | Base seed for plain requests. |
| `eos_token_id` | `"tokenizer"` | Read EOS from the tokenizer; an integer overrides it; `None` disables EOS stops. |
| `cuda_graphs` | `True` | Capture decode at startup. `False` runs eager decode for debugging. CPU is always eager. |
| `graph_batch_sizes` | `None` | Exact batch sizes to capture; `None` captures 1 to `max_batch_size`. Other sizes run eager; `[]` disables capture. |
| `sampling_topk` | 128 | Largest top-k handled by the bounded shortlist sampler. Larger values, or 0, use the full-vocabulary path. |
| `decider_factory` | `None` | Optional callable receiving the loaded engine model and returning an engine-compatible decider; installed before prefill and graph capture. Keep the returned policy fixed for the engine lifetime. |
| `overlap` | `None` | Pipelined scheduler. `None` is automatic; see [overlap scheduler](#overlap-scheduler). |
| `compile` | `False` | Experimental TorchInductor fusion of the decode step (CUDA only). Keep it off for accuracy work: its end-to-end parity is not established. |
| `compile_cache_dir` | `None` | Persistent TorchInductor cache. Default: `TORCHINDUCTOR_CACHE_DIR`, or `~/.cache/torch/inductor`. |

`swap_chunk_entries` and `profile_steps` are diagnostic knobs and are not needed for normal use.

## Scheduling, memory and long generations

Admission reserves the worst-case passes of a new prompt and the next pass of each
running request. Prefill is chunked and does not pad short prompts to the longest one.
When growth runs out of KV slots, the newest resident request is swapped to pinned host
memory with its KV, cumulative logits, next input and RNG state. Swapped requests
resume first-in first-out, and new admission waits while any request is swapped.

If a swap does not fit the free space of the host swap arena, the newest active request (resident or
swapped) discards its unfinished output and restarts from its original prompt and
seed. Its index is kept and only the final attempt produces a result. A restart
changes the batch composition, so a restarted sample can differ from an unrestarted
run. Retries take priority over new requests. A request that exceeds the model
context, or does not fit the GPU pool by itself, fails with an error.

Memory notes:

- Each KV entry is 112 KiB for the 28-layer 1.7B checkpoint. Prompt and generated
  passes share the GPU pool.
- `engine.memory_budget` reports the free memory measured at startup, the
  reservations (KV, attention workspaces, graphs) and the resulting capacity;
  `memory_budget.host_swap` reports the host swap arena.
- The host swap arena is exactly `host_swap_bytes` of pinned RAM, reserved when the
  engine starts and held until `engine.close()` (or the end of a
  `with TaHEngine(...) as engine:` block, or deleting the engine); swapping allocates
  nothing else. Every engine pins its own arena, so close one before creating the next
  in the same process. A closed engine cannot generate. If
  the RAM cannot be pinned, the engine fails at startup with an error naming the
  budget. It is used in pages of `swap_chunk_entries` KV entries (28 MiB for the 1.7B
  checkpoint); a swapped request takes whole pages, so a budget remainder smaller
  than one page is never used.
- RAM planning: process RSS ≈ baseline + `host_swap_bytes`. Measure the baseline once
  with the same configuration and `host_swap_bytes=0` (`--engine-host-swap-gib 0` in
  the harness), as the peak RSS of the engine process, e.g. `VmHWM` in
  `/proc/<pid>/status`. Grading workers are separate processes with their own RSS.
- Startup captures take time that grows with the resident cap. Create one engine and
  reuse it for all datasets.

For long generations (up to 16,384 tokens) on a machine with about 30 GB RAM, use the
cap 64, `memory_fraction` 0.9, `host_swap_bytes` 8 GiB, CUDA graphs on and compile
off. Restarts are automatic, so the cap need not be lowered to avoid aborts. Restarts
waste work: a smaller resident cap, or a larger host budget if RAM allows, reduces it.
The host budget is pinned RAM for KV and state, separate from operating-system disk
swap. Keep it bounded rather than sizing it for every possible output mix. Lowering
`memory_fraction` shrinks the GPU pool and increases pressure. The counters
`restarts` and `wasted_tokens` appear in `Result.timing`, `engine.last_run` and the
harness metrics (see [results](results.md#job-timing-and-engine-telemetry)).

Grading processes also need host RAM and CPU cores; see
[CPU cores and affinity](#cpu-cores-and-affinity).

## Overlap scheduler

With `overlap` on, the host prepares decode step t+1 while the GPU runs step t, so the
GPU does not wait for host work between steps. It is on by default, except for
checkpoints with dynamic or LongRoPE scaling. `engine.overlap` shows the resolved value.

- Finishes, admissions, swaps and restarts take effect one step late. A request
  that has finished runs one discarded row in the next step, counted as
  `ghost_rows`. Ghost rows never emit tokens and never change any output.
- If a request cannot fit both candidate KV slots that the lag requires, the engine
  waits for the in-flight step (`overlap_drains`) instead of failing.
- Each step still makes one device-to-host status transfer, read one step later.
  CUDA graphs keep their exact batch sizes.
- With the same rows in every step, overlap and non-overlap give identical tokens
  and pass counts. In ordinary runs the one-step lag changes which rows share a
  step, so sampled outputs differ the way two batch compositions do. Accuracy is
  therefore checked end to end.

Use `overlap=False` (or `--engine-options '{"overlap": false}'`, or `engine: {overlap:
false}` in the eval YAML) to debug the scheduler or to reproduce an earlier
non-overlap run with the same seeds. `overlap=True` is rejected for dynamic/LongRoPE
checkpoints. Measured throughput is in the [engine report chapter](../report/1-engine/).

## Sampling

`SamplingParams(temperature=1.0, min_p=0.0, top_k=0, top_p=1.0, do_sample=True)`
(a dictionary also works). `sampling` takes one value shared by all requests, or a
list with one entry per request.

- Temperature 0 or `do_sample=False` selects greedy argmax, which ignores the filters.
- Otherwise the filters run in the reference order described in
  [evaluation](evaluation.md#yaml-keys).
- `top_k` from 1 to `sampling_topk` runs on a bounded shortlist on the GPU. `top_k=0`
  or a larger value uses the full-vocabulary path, which is slower.
- A seed is a signed 64-bit integer. Each request keeps its own draw counter, which
  survives swapping. Repeating a call with the same seeds and batch composition gives
  identical outputs.

Tie-breaking and floating-point cutoff details are in the development reference.

## Playground

`script/tah/playground/inference_example.py` is a single-prompt demo. It uses the
engine with one resident request and graph size 1 by default:

```bash
source .venv-tah/bin/activate
CUDA_VISIBLE_DEVICES=0 python script/tah/playground/inference_example.py
CUDA_VISIBLE_DEVICES=0 python script/tah/playground/inference_example.py --backend tah
CUDA_VISIBLE_DEVICES=0 python script/tah/playground/inference_example.py --max-new-tokens 16384
```

Options: `--model` (Hub ID or local directory; default `nics-efc/TaH-plus-1.7B`),
`--device` (default `cuda:0`), `--backend` (`tah_engine` or `tah`; default the
engine), `--max-new-tokens` (default 512), `--seed` (default 42), `--engine-options`
(a JSON object of constructor options) and `-h`/`--help`.

The demo applies the tokenizer's thinking chat template and prints each token coloured
by its iteration count, then the final text. With the engine the colours appear when
generation completes; with the reference they are printed during the iteration loop.
If the environment check fails, use `--backend tah`.

## CPU cores and affinity

Graph decode needs one CPU core; eager or debug decode needs at least two. Grading
processes need additional cores and RAM. On a small VM, keep graphs on and grade
inline with `--grading-workers 0`.

`taskset` pins the process to CPU IDs from its allowed affinity list. This example
uses the lowest allowed core for graph evaluation of a 16-row run. For eager
debugging, pin two cores and add `--engine-options '{"cuda_graphs": false}'`:

```bash
source .venv-tah/bin/activate
cpu_id=$(python -c 'import os; print(min(os.sched_getaffinity(0)))')
CUDA_VISIBLE_DEVICES=0 taskset -c "$cpu_id" python script/tah/evaluation/eval.py \
  --eval_config script/tah/recipes/qwen3_1.7/eval_tah.yaml \
  --model_path models/TaH-plus-1.7B --dataset_name math500 --data_range 0 16 \
  --backend tah_engine --max-new-tokens 1024 --grading-workers 0 \
  --engine-max-batch-size 16 --output_dir data/eval_one_core
```

Source of truth: [engine package](../tah/engine/engine.py),
[playground](../script/tah/playground/inference_example.py).

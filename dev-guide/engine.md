# TaH v1 engine: developer reference

For engine maintainers. User-facing usage is in [guide/engine.md](../guide/engine.md);
measured results are in [report/1-engine](../report/1-engine/). Run commands from the
repository root with `source .venv-tah/bin/activate`.

## Modules

| Module | Role |
| --- | --- |
| `engine.py` | `TaHEngine`: admission, swap and restart, overlap pipeline, `last_run` metrics. |
| `scheduler.py` | `Request`, `Result`, per-request state, paged `HostSwap`. |
| `host_arena.py` | `HostArena`: the fixed host swap buffer, pinned once with `cudaHostRegister`. |
| `kv_cache.py` | `SlotPool`: page-size-one KV storage, request tables, snapshots, audit. |
| `attention.py`, `attention_kernels.py` | FlashInfer decode/prefill buckets and Triton table kernels. |
| `graphs.py` | CUDA graph capture per exact batch size. |
| `sampling.py` | `SamplingParams`, `GraphSampler`, `RequestRNG`. |
| `model.py`, `cached_model.py`, `layers.py`, `rope.py`, `weights.py` | Engine-owned model: `EagerTaHModel`, cached passes, LoRA, RoPE, checkpoint loading. |
| `decider.py`, `compiled.py` | Iteration decider and the optional TorchInductor decode path. |
| `fixtures.py`, `generate_fixtures.py`, `parity.py`, `reference_capture.py` | Reference fixture suite, comparison and limits. |
| `tests/` | pytest suite and validation CLIs (below). |

## Test suite

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q tah/engine/tests        # CPU; GPU tests are skipped
CUDA_VISIBLE_DEVICES=0 python -m pytest -q tah/engine/tests         # adds GPU tests
CUDA_VISIBLE_DEVICES=0 python -m pytest -q -s -o addopts= -m 'gpu and slow' tah/engine/tests/<file>::<test>
```

`pyproject.toml` sets `addopts = "-m 'not slow'"`. Markers: `gpu` (needs CUDA) and
`slow` (production fixture replay or long validation). Use `-o addopts=` to run slow
cases. `kernel_check.py` is a GPU utility, not a pytest module; see
[installation](../guide/installation.md#kernel-check-gpu).

## Validation CLIs

### engine_check

```bash
CUDA_VISIBLE_DEVICES=0 python -m tah.engine.tests.engine_check --gate <gate> --output <report.json>
```

One gate per process; use a 28-minute `timeout` on GPU runs. Every report records the
runtime, reference and checkpoint hashes.

| `--gate` | What it checks |
| --- | --- |
| `gen` | Parity check against the saved fixture suite for `--checkpoint` (generation mode). |
| `tf` | Earlier fixture-limit teacher-forced diagnostic. Superseded for generated sequences by `generated_gtf`. |
| `smoke` | Math500 `[0,16)`, repeat 1, cap 1024: eager and graph decode at the same capacity and batch composition must match exactly in tokens, finish reasons, pass counts and every decode decision. Needs the real checkpoint and dataset. |
| `perf` | Workload throughput at `--batch-size` (default 128; 64 for reported runs), Math500 prompts repeated twice, cap `--max-new-tokens` (default 4096). Report includes KV audit, live host swap bytes and tokens/s. Build and warmup are excluded. |
| `robust` | Workload with long generations (use `--max-new-tokens 16384`). |
| `long-prompt` | One 16,384-token prompt plus 32 short requests. Use `--max-new-tokens 8`. |
| `swap` | Forced mid-generation swap and immediate restore at unchanged composition; eager and graph outputs, pass counts and decisions must match. Rejects `--overlap`. |
| `profile` | CUDA activity profile over the first eleven decode steps after prefill, with headroom set to zero. Counts kernels once, cross-checks the trace, reports kernel and copy time, and saves `<output>.trace.json` and `.timings.json`. With `--overlap`, one pipeline turn per step; steady turns 2–11 are recorded. |
| `phase-profile` | Four-request run with a forced swap and 48-token cap; splits prefill, swap-out, restore and decode scopes. Asynchronous swap work can finish after its CPU scope, so the residual is not GPU idle time. |
| `overlap` | Overlap-pipeline timing diagnostic alone (`--batch-size`, `--profile-context`); requires graphs. |
| `overlap-identity` | Always overlap on: 24 smoke prompts at resident 16, caps 256–704, a forced swap at step 40 and a forced restart at step 160. Replays the same per-step row sets without overlap (each finished request keeps one discarded row for one step) and requires identical row sets, decisions, tokens, pass counts and finish reasons. A plain non-overlap run is reported as `plain_nonoverlap_first_divergence` (diagnostic only). |

Options: `--checkpoint {real,mlp3,mlp1,label3}` (default `real`); `--batch-size`;
`--max-new-tokens`; `--start`/`--end` select a range of the 128-prompt workload
(defaults 0/128); `--memory-fraction` (default .9); `--profile-context` (default
2048; 0 uses the original prompt lengths); `--model-path` (default
`models/TaH-plus-1.7B`); `--cuda-graphs`/`--no-cuda-graphs` (default on);
`--compile`/`--no-compile` (default off); `--overlap`/`--no-overlap` (default off, so
earlier reports stay reproducible); `--compile-cache-dir`; `--graph-batch-sizes`
(comma-separated exact sizes; other sizes run eager).

Overlap example:

```bash
CUDA_VISIBLE_DEVICES=0 timeout 28m python -m tah.engine.tests.engine_check \
  --gate overlap-identity --output data/engine/checks/overlap-identity.json
CUDA_VISIBLE_DEVICES=0 timeout 28m python -m tah.engine.tests.engine_check \
  --gate perf --batch-size 64 --overlap --output data/engine/checks/perf-overlap.json
```

### Generated-corpus teacher-forced parity (generated_gtf)

```bash
CUDA_VISIBLE_DEVICES=0 python -m tah.engine.tests.generated_gtf --phase generate --checkpoint real --root <new-root>
CUDA_VISIBLE_DEVICES=0 python -m tah.engine.tests.generated_gtf --phase noise    --checkpoint real --root <new-root>
CUDA_VISIBLE_DEVICES='' python -m tah.engine.tests.generated_gtf --phase compare  --checkpoint real --root <new-root>
```

`--phase` runs in order: `generate`, `noise`, `compare`. `--checkpoint` is
`real`, `mlp3`, `mlp1` or `label3`. `--root` should be a new directory for each
independent run; the code's default is a legacy directory name, so pass it explicitly.
`--min-generated-positions` (default 8192) is the minimum processed generated
positions per checkpoint for a comparison. `--overlap` and `--compile` apply to
corpus generation.

- `generate` produces the corpus (manifest, token IDs and engine traces) under
  `gtf-corpus/<checkpoint>/`. Six prompts are repeated in groups with distinct seeds
  at temperature 0.6, top_k 20, top_p 0.95.
- `noise` runs the reference only: it compares two reference layouts (singleton
  prompt chunks of 128 with cached single-token decode, versus a batched left-padded
  prompt with aligned cached decode) and writes `gtf-limits/<checkpoint>.json`
  before any engine comparison.
- `compare` verifies corpus, oracle and source hashes, applies the saved limits and
  writes `tf-<checkpoint>-expanded.json` with PASS/FAIL, decisions, pass counts and
  full and retained-logit metrics.

Neither `noise` nor `compare` overwrites an existing registration. Choose a new root
for an independent run and keep earlier failing reports as evidence.

Limit rules: limits are 1.5× the measured reference-only error or rate. MLP
decision and pass-count reference counts of zero or one use a floor of two.
IterLabel and depth caps stay exact. A flip margin is measured from the effective
threshold on both sides of a reference flip, with a 0.01 minimum.

### Fixture suite and production parity

`generate_fixtures` records teacher-forced decision traces, probabilities, pass
counts, logits and selected hidden inputs, and greedy generations from the reference
model. It is the source of the production parity gates.

| Option | Default and behavior |
| --- | --- |
| `--profile tiny\|full` | `tiny`: CPU development suite (random tiny Qwen3 vocabulary; no production tolerances). `full`: real checkpoint plus three synthetic Qwen3-0.6B-Base checkpoints. |
| `--device cpu\|cuda` | `cpu`. CUDA is explicit and refused when visibility is empty. Full capture requires CUDA. |
| `--prepare-only` | Prepare inputs and synthetic models without capture; marks the manifest prepared. |
| `--output PATH` | Fixture root. Default is a tiny-profile directory; use a separate root for `full`. |
| `--model-output PATH` | Where synthetic models are saved (`<path>/<profile>/{mlp3,mlp1,label3}`). |
| `--real-model PATH` | Local real checkpoint (default `models/TaH-plus-1.7B`). |
| `--base-model ID_OR_PATH` | Pretrained synthetic base (default `Qwen/Qwen3-0.6B-Base`); may download. |
| `--eval-config PATH` | Reference TaH override for the real checkpoint (default: the 1.7B eval recipe). |
| `--checkpoint` | Repeatable; default all checkpoints in the profile. |
| `--seed INTEGER` | Preparation seed (default `20261012`). |
| `--topk INTEGER` | Final-logit values and indices retained per position (default 32). |
| `--chunk-size INTEGER` | Reference cached teacher-forced chunk width (default 256). `0` is unchunked. |
| `--generation-tokens INTEGER` | Greedy cap (default 8 tiny, 256 full). `0` selects the profile default. |
| `--threads INTEGER` | Torch CPU threads for preparation (default 4). |

Preparation on CPU, then capture one checkpoint at a time after the GPU is idle:

```bash
CUDA_VISIBLE_DEVICES='' python -m tah.engine.generate_fixtures --profile full --prepare-only --output <fixture-root>
for checkpoint in real mlp3 mlp1 label3; do
  CUDA_VISIBLE_DEVICES=0 python -m tah.engine.generate_fixtures --profile full --device cuda \
    --checkpoint "$checkpoint" --output <fixture-root> || break
done
```

Full inputs use three lengths (256, 1024, 4096 tokens) from math500, GPQA and
HumanEval. Continuations are teacher-forced: a recorded solution or scaffold repeated
to the target length. Each sequence has five teacher-forced layouts; six prompts per
checkpoint have four greedy layouts. The MLP3 random decider is calibrated to continue
10–40% of active tokens at each nonterminal depth. IterLabel teacher forcing includes
oracle labels, and its greedy cases stop after one pass.

Existing inputs and models are reused after profile, seed and source checks. A
checkpoint's records are regenerated when that checkpoint is rerun. Capture settings
must match an existing suite; choose a new root to change them. Production
eligibility requires all four checkpoint sets and depth coverage; `manifest.json`
records it as `production_gates_eligible`.

Outputs: `inputs.json` (provenance, prompts, token IDs), `manifest.json` (status,
eligibility, configs, hashes, sizes, self-noise), `<checkpoint>/teacher_forced/*.safetensors`
(per-sequence traces; unvisited entries are NaN and masked by `visited`),
`<checkpoint>/greedy/*.safetensors` (prompt and generated IDs, pass counts) and
`<checkpoint>/self_noise.json`. Depth axes are zero-based. Hidden indices follow the
HF hidden-state tuple, with embeddings at index 0 and the normalized output last.
Full-vocabulary self-noise is measured during capture, but only sampled rows are
stored; never compare independent top-k rank lists as if their indices were equal.

Verify on CPU without loading a model:

```bash
CUDA_VISIBLE_DEVICES="" python - <<'PY'
import hashlib
from tah.engine.fixtures import FixtureSuite
suite = FixtureSuite('<fixture-root>')
assert suite.manifest['production_gates_eligible']
for record in suite.records():
    suite.load(record)
    path = suite.root / record['file']
    assert path.stat().st_size == record['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
print(len(suite.manifest['records']))
PY
```

`FixtureSuite(root, allow_partial=False)` rejects incomplete manifests.
`records(kind=None, checkpoint=None)` filters lazily. Use `allow_partial=True` only for
inspection; production gates require `profile='full'`, `status='complete'` and
`production_gates_eligible=true`.

Production teacher-forced parity (`test_gtf.py`, `slow` and `gpu`):

```bash
for checkpoint in real mlp3 mlp1 label3; do
  CUDA_VISIBLE_DEVICES=0 python -m pytest -q -s -o addopts= -m 'gpu and slow' \
    "tah/engine/tests/test_gtf.py::test_gtf_every_wp2_fixture_set[${checkpoint}]" || break
done
```

This checks hashes, replays all five layouts per sequence and compares full-vocabulary
errors against the unchanged reference. Environment variables: `TAH_GTF_FIXTURES`
(fixture root), `TAH_GTF_TOLERANCES` (legacy tolerance export whose fixed-decision
fields are superseded by decision limits derived from the reference trace pairs) and
`TAH_GTF_RESULTS` (report directory). The reports are `<checkpoint>-limits.json`
(written before comparison) and `<checkpoint>.json` (PASS/FAIL, per-layout metrics,
flip counts and rates, pass-count mismatch rate, GPU name and peak memory). Binding
bounds are `noise_limits` and `decision_limits`; `g_tf_*_limit` fields in the summary
are rescaled values, not the bounds. A failed comparison still writes its metrics.

Gate rules: full-logit MAE, RMSE, max error and top-1 disagreement must be within
1.5× the recorded limits with no epsilon floor. Per depth, decision flips count only
positions visited by both implementations; their rate must be within 1.5× the
reference rate, and a zero reference rate requires an exact match. Every flip must lie
within `max(0.01, 1.5 × largest reference flip margin)` of the threshold. Pass-count
mismatch has its own 1.5× limit. Greedy divergence positions are zero-based; `null`
means identical IDs and length. For greedy generation, the candidate median first
divergence must be at least half the reference median.

Cached teacher forcing (`test_cached_gtf.py`) runs the same checks through
`forward_chunk`, with chunk sizes 256 and 1 (token by token):

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest -q -s -o addopts= -m 'gpu and slow' \
  "tah/engine/tests/test_cached_gtf.py::test_cached_production_gtf[256-real]"
```

These take about four minutes per chunked case and 9–13 minutes per token-by-token
case. Compare each chunked report with the reference (binding) and with the full eager
model (diagnostic). The token-by-token dense-row comparison with the full eager forward
has one known diagnostic failure from single-token versus full-sequence execution
layout noise; the binding reference comparison passes in both layouts.

### Performance and metadata tools

`attention_check` (microbenchmark, JSON output):

```bash
CUDA_VISIBLE_DEVICES="" python -m tah.engine.tests.attention_check --device cpu --repeats 10 --output <report.json>
CUDA_VISIBLE_DEVICES=0 python -m tah.engine.tests.attention_check --device cuda --batches 64 \
  --avg-context 2048 --repeats 100 --output <report.json>
CUDA_VISIBLE_DEVICES=0 python -m tah.engine.tests.attention_check --device cuda \
  --model-path models/TaH-plus-1.7B --batches 64 --avg-context 2048 --repeats 20 --output <report.json>
```

`--device cpu|cuda` (default cpu); `--repeats` (default 10); `--batches` (default
1 8 32); `--avg-context` (request lengths vary ±25%; default lengths are `31+17*row`);
`--model-path` (CUDA only) times all-layer cached passes with randomized old KV, sparse
later streams and mixed depths, and fails clearly when KV would use over 85% of free
memory. Times are microseconds. Select CUDA only when the GPU is idle.

`metadata_check` requires empty CUDA visibility and `--mode interpret|compile --output`.
`interpret` checks exact ragged visibility and compaction on CPU. `compile` builds the
Triton kernels offline for sm80, sm86 and sm89 without loading a CUDA module. Reports
carry `production_gates_eligible=false`.

## Validating on a new GPU or machine

Run this when the GPU type, the machine or VM, or the driver or CUDA toolkit changes.
It shows that the test suite passes there and that, on that machine, CUDA graph and
eager decoding, and overlap and non-overlap scheduling, give identical tokens,
decisions and counts. It does not prove bitwise equality with another GPU: different
kernels change bf16 rounding, so outputs can differ at noise level. Use one GPU type
and software stack for any comparison in a report.

Prerequisites (Python 3.10 headers, a C++ compiler, and an `nvcc` matching the torch
CUDA major version) are in [installation](../guide/installation.md#engine-prerequisites).
FlashInfer compiles kernels on first use, so the first run is slow.

From the repository root, in the standard `.venv-tah` environment (created if missing;
the install line syncs it with the current `pyproject.toml`):

```bash
set -o pipefail
export CUDA_VISIBLE_DEVICES=0
out=data/engine/port/$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=name --format=csv,noheader | tr -d '/' | tr ' ' '-')
mkdir -p "$out"
[ -d .venv-tah ] || python3.10 -m venv .venv-tah
source .venv-tah/bin/activate
python -m pip install -e '.[tah,evaluation,engine,dev]' && python -m pip check
nvidia-smi > "$out/nvidia-smi.txt"; nvcc --version > "$out/nvcc.txt"
python - <<'PY' | tee "$out/environment.txt"
import torch, transformers, peft, flashinfer
from tah.engine import check_environment
check_environment()
assert torch.__version__.split('+')[0] == '2.6.0' and transformers.__version__ == '4.52.4' and peft.__version__ == '0.15.2'
print('GPU:', torch.cuda.get_device_name(), 'compute capability:', torch.cuda.get_device_capability())
print('torch/CUDA:', torch.__version__, torch.version.cuda, '| flashinfer:', flashinfer.__version__, '| free/total:', torch.cuda.mem_get_info())
PY
python -c "from huggingface_hub import snapshot_download; snapshot_download('nics-efc/TaH-plus-1.7B', local_dir='models/TaH-plus-1.7B')"
python -m pytest -q tah/engine/tests 2>&1 | tee "$out/default-suite.log"
python -m tah.engine.tests.engine_check --gate smoke --cuda-graphs --batch-size 16 \
  --output "$out/wsmoke-graph.json" 2>&1 | tee "$out/wsmoke-graph.log"
timeout 28m python -m tah.engine.tests.engine_check --gate overlap-identity \
  --output "$out/overlap-identity.json" 2>&1 | tee "$out/overlap-identity.log"
```

Pass criteria:

- No pytest failures. Skips are acceptable only for missing archived development
  fixtures; the log prints their reasons.
- W-smoke exits 0 with `status` `PASS`, `tokens_identical`, `decisions_identical` and
  `pass_counts_identical` true, and `run.graph_steps` equal to `run.steps` and above zero.
- `overlap-identity` exits 0 with `status` `PASS`.

If a gate fails, keep its report and log; do not change registrations or budgets.

## Full-sequence model (EagerTaHModel)

`EagerTaHModel` loads a checkpoint into engine-owned modules and returns logits for a
complete sequence with dense attention and temporary per-pass KV. Use it for
validation; use `TaHEngine.generate` for scheduling.

```python
from tah.engine.model import EagerTaHModel
model = EagerTaHModel.from_pretrained('models/TaH-plus-1.7B', config_override, device='cuda')
output = model([3, 7, 11], hidden_positions=[0, 2])
```

`from_pretrained(model_path, config_override=None, *, device='cpu', dtype=torch.bfloat16)`
accepts a local directory and does not download. Nonempty override fields replace saved
fields. LoRA targets must be internal linear modules; embedding and `lm_head` adapters
are rejected. Default, linear, dynamic NTK, YaRN, LongRoPE and Llama 3 RoPE scaling are
supported. `model(input_ids, *, iter_count_labels=None, position_ids=None,
hidden_positions=None)` takes a nonempty 1D int64 sequence. Unvisited diagnostics are NaN.

## Cache and attention APIs

CUDA attention uses FlashInfer; CPU attention uses dense SDPA for development checks.
Cached teacher forcing (`forward_chunk`) returns logits and traces for one bounded
chunk without sampling or admission.

```python
import torch
from tah.engine.model import EagerTaHModel
from tah.engine.kv_cache import SlotPool

model = EagerTaHModel.from_pretrained('models/TaH-plus-1.7B', device='cuda')
cfg = model.config
free_bytes, _ = torch.cuda.mem_get_info()
capacity = SlotPool.capacity_from_bytes(int(free_bytes * .1), cfg['num_hidden_layers'],
    cfg['num_key_value_heads'], cfg['head_dim'], model.dtype)
pool = SlotPool(capacity, cfg['num_hidden_layers'], cfg['num_key_value_heads'],
    cfg['head_dim'], model.max_iter, device=model.device, dtype=model.dtype)
request = pool.create_request('example', context_limit=cfg['max_position_embeddings'])
try:
    first = model.forward_chunk([3, 7, 11], pool, request)
    second = model.forward_chunk([13, 17], pool, request, start_position=3)
finally:
    pool.release(request)
print(pool.audit())
```

`forward_chunk(input_ids, pool, request, *, start_position=0, iter_count_labels=None,
hidden_positions=None, decode=False)`: positions must increase across chunks;
`decode=True` requires exactly one token. The result has `logits` (`[chunk_tokens,
vocabulary]`), `iter_count`, `visited` and `decision` (`[max_iter, chunk_tokens]`),
`decision_logits`, `continue_probability`, `hidden_positions`, `hidden_layer_indices`
and `decider_hidden`. The cache keeps KV for every executed pass. A capacity failure in
a later pass leaves earlier passes allocated, so reserve growth before executing a chunk
and release the request on failure.

`SlotPool(capacity, num_layers, num_kv_heads, head_dim, max_iter, *, device,
dtype=torch.bfloat16)` allocates page-size-one storage shared by all streams.
`capacity_from_bytes(budget_bytes, num_layers, num_kv_heads, head_dim, dtype)` gives a
slot count; the caller reserves memory for weights, logits and workspaces first.

| Method / property | Behavior |
| --- | --- |
| `create_request(request_id, *, context_limit)` | Empty per-stream tables. IDs are unique among live requests. |
| `allocate(request, stream, positions)` | Device slot IDs for new entries; positions strictly increase per stream and fit the limit. |
| `write(layer, slots, k, v)` | Writes `[rows, kv_heads, head_dim]` into unique live slots. |
| `slot_ids(request)` | Occupied slot IDs in stream and position order. |
| `gather(request, *, out=None)` / `scatter(request, snapshot)` | Contiguous `KVSnapshot` out and back to matching coordinates. |
| `restore(request_id, snapshot)` | Allocates the snapshot's coordinates and scatters the payload into possibly different slots. |
| `release(request)` | Frees all slots; reusing the handle raises. |
| `free_slots`, `audit()` | Free capacity and ownership or leak checks. |
| `visible_count(request, position, depth)` | Host planning count from the ledger; depth is one-based. |

`KVSnapshot.values` has shape `[layers, 2, occupied_entries, 1, kv_heads, head_dim]`.
For pinned host swap, copy `values` to pinned memory with `non_blocking=True`, record a
CUDA event, and wait on it before reusing the source slots. The snapshot covers KV only;
logits, next-pass inputs, RNG and scheduler state are saved separately.

Exhaustion raises `KVCapacityError` before allocation. Invalid coordinates, incompatible
snapshots and released handles raise `ValueError`.

`AttentionRow(request, position, depth)` describes a query with one-based depth. Only
entries with `key_position <= position` and `key_depth <= depth` are visible.
`DecodeBucket(pool, batch_size, max_entries, num_q_heads, *, workspace_bytes=134217728)`
reserves fixed-address metadata; `max_entries` must cover every occupied entry of its
rows. `prepare(rows)` uploads descriptors and plans FlashInfer outside capture; reprepare
after allocation, restore, release or row changes. `build()` compacts visible entries on
the GPU; `run(q, layer, *, out=None)` runs attention. `PrefillAttention(pool, rows,
num_q_heads)` handles prefill groups; prefill planning may synchronize with the host.
Attention objects reject stale cache revisions. Release slots only after in-flight
attention completes.

`model.forward_cached(embeddings, position_ids, depths, slots, pool, attention, *,
rotary=None)` runs one pass per row, across requests and depths. It writes K/V for every
layer and returns pass-local `logits` increments and `hidden_states`. The caller
accumulates logits and handles continuation decisions. Dynamic and LongRoPE scaling need
frequency planning outside capture and precomputed `(cos, sin)` through `rotary`.

## Host swap arena

`HostArena` holds exactly `host_swap_bytes`. It is a plain host tensor pinned in place
with `cudaHostRegister`, not `torch.empty(pin_memory=True)`: PyTorch's caching host
allocator rounds blocks up to powers of two and keeps freed blocks cached, so per-swap
pinned tensors grew RSS far past the budget. The engine creates the arena once; the graph
probe and the final runtime share it. `HostSwap` splits it into pages of
`swap_chunk_entries` KV entries (`page_bytes = chunk × entry bytes`).
`TaHEngine.close()` (also `__exit__`) unpins and frees it at once; otherwise a
`weakref.finalize` does so when the arena is collected. `TaHEngine` has no reference
cycle, so `del engine` releases it without a gc pass, but wrappers that patch closures
onto the engine (`EngineRunner`) do form cycles and must call `close()`. Tests and
gates that do not size the arena on purpose pass small budgets (`TINY_SWAP_BYTES` in
`test_engine_gpu.py`, `BOUNDED_SWAP_BYTES` in `engine_check.py`); keep it that way, since
every engine pins its budget at construction.

- **Layout.** A snapshot is one byte stream over its pages: KV chunk k (contiguous
  `[layers, 2, n, 1, heads, dim]`) fills page k, then the row payload (logits,
  embeddings, seed, counter) follows as raw bytes and may cross into one more page.
  Each copy is contiguous to contiguous, so it is one direct DMA without temporaries.
- **Accounting.** A snapshot needs `ceil((entries × entry_bytes + payload) / page_bytes)`
  pages; `out` raises `HostSwapOverflow` exactly when fewer pages are free, before
  changing any state. Pages are interchangeable, so there is no external
  fragmentation. `used_bytes`/`peak_bytes` count whole pages.
- **Reuse without host sync.** During generation the host never reads or writes arena
  bytes. Pages are written only by swap-out D2H copies and read only by restore H2D
  copies, all on the engine's single CUDA stream. A page freed by `restore` or `discard`
  can still have a copy in flight; the next operation on it is a later swap-out copy
  on the same stream, which runs after every earlier operation of that stream. This
  covers the overlap pipeline (swaps between launch and consume of a step) and discard
  of a swapped request (a restart or a finish during the lag): the `event.synchronize()`
  calls on those discard paths predate the arena and are not needed for page reuse.
  Restore still issues the device-side `wait_event` on the snapshot event.
- **Slot indices.** Gather and scatter indices come from the pool's device stream
  tables (`SlotPool.slot_ids`), already written in stream order by `allocate`/`reserve`;
  there is no per-swap host index upload or staging buffer.
- **Validating a change.** CPU: `python -m pytest -q tah/engine/tests/test_host_swap.py`
  (page edges, fragmentation, bit-exact round trip, budget 0, no pinned or host
  allocation after construction) and `test_engine.py`. GPU: `engine_check --gate swap`
  and `--gate overlap-identity`, `test_engine_gpu.py`, and `--gate phase-profile` for
  swap-out and restore time. Process RSS must stay flat at baseline + budget
  (`VmRSS` in `/proc/<pid>/status`) during a swap-heavy run.

## Overlap internals

The overlap pipeline (`overlap=True`) plans step t+1 before step t's status is read.

- **Device-side row state.** At the end of each step the device updates
  `row_positions` and `row_depths`: a kept row advances one pass; otherwise it moves to
  the next position at pass 1. The next step reads them by row, so rotary values are
  computed on the device. Dynamic and LongRoPE need the exact host context length, so
  overlap is rejected for them.
- **Speculative KV reservation.** For a row at (p, d) in flight, slot A is stream d at
  position p (if d < N) and slot B is stream 0 at p+1 (reused if already reserved).
  A is invisible to row (p+1, 1) and B to row (p, d+1), so the unused slot never enters
  attention. An unused A is popped when status arrives. Unused B is kept as the next
  advance slot. Swap-out pops an unwritten B before snapshotting.
- **Ghost rows.** A request finishing in step t runs once more in step t+1, writing its
  KV to its own B (freed with the request) or to a trash slot on its last allowed token.
  The trash slot is one extra pool entry outside capacity, the free list and tables.
  Ghost sampling outputs are discarded.
- **Stream-ordered frees.** Host frees happen when status is consumed, while the request
  may still be in the in-flight step. Work runs on one CUDA stream, so reuse runs after
  that step. `State.restarts` distinguishes attempts; stale statuses are ignored. A
  status that finishes a request swapped during the lag discards its snapshot after the
  copy event.
- **Accounting.** Admission, restore and growth use the per-row speculative need
  (at most two slots per row). If one row cannot fit, the pipeline drains: it waits for
  the in-flight step, after which the row needs at most one slot (`overlap_drains`).
- **One D2H copy per step.** Status goes to a fresh pinned buffer right after replay,
  before the next replay can overwrite graph-pool memory, and is event-synced a step
  later. Table writes are batched into one host-to-device copy and one kernel per step.
- **Identity gate.** The lag changes composition, so the identity gate compares overlap
  against a non-overlap run that executes the same per-step row sets. The test-only
  `_ghost_lag` flag in the synchronous loop keeps a finished request for one discarded
  row. Forced swap and restart are issued by step hooks at the same step in both modes.
  The overlap hook must not restore in the same turn, because the victim's last status
  is still pending.
- **Profiling.** With overlap, a profile scope is one pipeline turn (consume status,
  schedule, launch). The physical kernel sum excludes launch gaps between graph nodes,
  so bubbles are measured with an unprofiled event probe.

## Compile notes

`compile=True` wraps decode in `torch.compile(fullgraph=True, dynamic=...)`, fusing
projections with LoRA deltas, RMSNorm, RoPE, MLP activations, the decider, bf16 logit
accumulation, input mixing and sampling. Paged FlashInfer attention is an opaque custom
operator, so there are no graph breaks. Compiled code accepts dynamic row counts, but
graph replay stays exact-B. Prefill stays eager. Compilation changes bf16 rounding, so
compiled output must be checked with the generated-corpus rules, not against
compile-disabled logits. Status: experimental. Compile-on teacher-forced and greedy
checks did not pass for the three-pass MLP checkpoint, and capturing all 64 batch sizes
did not finish startup within the time tried; the selected capture set leaves most
sizes on the compiled eager path. Compile with overlap works. The cache location is
`compile_cache_dir`, and `engine.last_run.compile` reports compile time and graph counts.

## Startup capture and memory sizing

Startup captures graphs with a small temporary KV pool, destroys them, sizes the real
KV pool from a fresh free-memory reading, and captures again against the final
addresses. Graph-private storage and releasable executable memory are reserved in KV
sizing. Memory already retained after the probe is in the free-memory reading.
`memory_budget.graph_probe` and `.graphs` report capture times, private bytes and device
deltas. A capture exceeding its reservation plus allocator margin raises
`KVCapacityError`.

KV sizing budgets both 128 MiB attention float workspaces, private integer workspaces
for every possible exact batch size, shared ragged metadata, request tables, logits and
input state, model and sampling intermediates, bounded swap scratch and allocator margin.
Static unsplit plans use compact integer workspaces sized to their real row count.

Graph replay does not consume RNG. `GraphSampler` and `RequestRNG` state move with
their requests, and inactive rows return `-1` without advancing counters.

## Sampling internals

`GraphSampler(vocab_size, max_top_k=...)` implements per-row temperature, min-p, top-k,
top-p and greedy sampling with `RequestRNG`. The bounded sampler selects one top-K
shortlist per large-vocabulary CUDA row and applies the filters to those K candidates.
Sampled rows in that bucket need `1 <= top_k <= max_top_k`; `top_k=0` or larger uses the
full-vocabulary path. Keep these as separate fixed-shape buckets.

- Within the shortlist, equal scores use ascending token-ID order.
- At the outer top-K boundary, tied membership follows the pinned PyTorch 2.6 kernel for
  that shape and dtype. Rerun the tie tests when the environment changes.
- Bounded top-p uses FP64 shortlist arithmetic. A cutoff chosen exactly from a
  full-vocabulary FP32 CDF may select an adjacent nucleus, because reference reductions
  round differently. Such boundaries can resolve either way.
- Non-degenerate distribution tests pass. Outputs across backends need not match.

Create parameter and RNG state before capture. Update parameter values in place so
captured tensor addresses stay fixed:

```python
from tah.engine.sampling import GraphSampler, RequestRNG, SamplingParams

sampler = GraphSampler(vocab_size, max_top_k=20)
batch = sampler.make_batch(per_request_params, device="cuda")
rng = RequestRNG.from_seeds(per_request_seeds, device="cuda")
# Capture sampler.sample(final_logits, batch, rng, active) with fixed buffers.
```

`GraphSampler(vocab_size, greedy_only=True)` gives a separately captured all-greedy bucket.

## External routing policies

`TaHEngine(..., decider_factory=callable)` installs a decider immediately after
loading weights, before hidden indices, prefill and CUDA graphs are initialized.
The returned module follows `forward(logits, depth, hidden, labels=None,
hidden_indices=None)` and returns `(keep, raw, probability)`. It must support
rowwise tensor depths without host reads during capture. Do not replace it after
graph capture. `research.bandit_router.engine.EngineRoutingDecider` implements
the bandit, entropy, fixed-depth and native policies.

The paired collector uses `forward_cached`, with a speculative depth-2 KV entry
removed by `pop_last` whenever frozen imitation stops. It accumulates bf16
logits before computing fp32 losses. It never commits the losing branch.
Validate both branches and future cache isolation on CPU, then replay collected
production histories through unbranched `forward_chunk` on the target GPU:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q research/bandit_router/test_engine.py research/bandit_router/test_core.py tah/engine/tests/test_engine.py
CUDA_VISIBLE_DEVICES=0 python -m research.bandit_router.engine_run verify --cache RUN/cache.pt --max-problems 3
```

Collection is eager; generation uses the production scheduler and CUDA graphs.
Engine generation applies the selected policy to prompt and generated positions;
the legacy reference smoke script used imitation for the prompt prefix. These
are different workloads and their timings must not be compared directly.

# Oracle validation tooling: developer reference

`script/tah/validation/` measures the numerical error of the engine and of the reference
against a high-precision oracle, per generated token and teacher-forced (protocol
[R1-O](../report/PROTOCOL.md#r1-o--numerical-accuracy-against-a-high-precision-oracle)),
and provides the common-random-number primitives for coupled trajectories. It changes
nothing in `tah/engine/` or `tah/model/`: it imports them and patches instances inside
the tool. Run everything from the repository root with `source .venv-tah/bin/activate`.

## Modules

| Module | Role |
| --- | --- |
| `common.py` | Manifest access, position convention, model loading per precision, sampler distributions, compact records, transient logit dumps. |
| `reference_run.py` | `ReferenceRunner`: teacher-forced reference in the `generate` (evaluation) or `chunk` (oracle) layout, decisions observed or forced through a decider forward hook. |
| `engine_run.py` | `EngineRunner`: teacher-forced `TaHEngine.generate` (continuous admission, mixed depths) with forced tokens and, optionally, forced decisions. |
| `capture.py` | CLI: one implementation over a manifest, block by block, writing `records/*.npz` and `run.json`; compares logits against a target. |
| `analyze.py` | CLI: M1–M3, bias, no-large-error check, bootstrap ratios, oracle validity, secondary tables → `r1o1.json`, `r1o1.md`. |
| `fp64_attention.py` | `fp64_attention_guard()`: float64 SDPA workaround for PyTorch #143644, applied by `ReferenceRunner` and the coupled oracle to every float64 run. |
| `select_sequences.py` | CLI: deterministic sequence manifest from evaluation `samples.jsonl` files. |
| `select_o1_manifest.py` | CLI: deterministic nested O1 manifest with stage-1, stage-2, validity and batch-1 subsets. |
| `coupling.py` | `uniform(seed, index)`, `uniforms`, `inverse_cdf_draw`, `multinomial_patch` (R1-O2). |
| `coupled_run.py` | Freeze harness prompts/seeds; resumable production engine, real reference b8, configurable oracle batch and shared harness grading. |
| `coupled_analyze.py` | Paired C1 bootstrap, C2 TOST, trajectory identity/divergence and oracle CDF margins → `r1o2.json`, `r1o2.md`. |
| `tests/test_validation.py` | CPU tests on the tiny fixture checkpoints. |
| `tests/test_fp64_attention.py` | Workaround semantics against native CPU SDPA (masks, causal, GQA, fully masked rows), pass-through, restoration, runner/target guards; one CUDA check against CPU float64. |
| `tests/test_coupled.py` | Batch/stream alignment, swap/restart, EOS, grading, fixed criteria and resumable CPU pipeline. |

## Validating a change

```bash
mkdir -p data/tmp/validation
CUDA_VISIBLE_DEVICES='' TMPDIR="$PWD/data/tmp/validation" python -m pytest -q script/tah/validation/tests --basetemp=data/tmp/validation/pytest
CUDA_VISIBLE_DEVICES='' TMPDIR="$PWD/data/tmp/validation" python -m pytest -q tah/engine/tests --basetemp=data/tmp/validation/engine
```

The tests need `data/engine/fixture-models/tiny/` (`python -m tah.engine.generate_fixtures
--profile tiny --prepare-only`). What they pin:

- `generate` layout = the evaluation loop: tokens drawn by the real
  `TaHForCasualLM_generate` (coupled uniforms), then the teacher-forced batch sees the
  same sampling distributions and pass counts at every step.
- `chunk` layout in bf16 = `tah.engine.reference_capture.capture` (decisions,
  probabilities, logits); a precision change keeps the threshold.
- Forced runs make exactly the oracle's decisions; errors are finite and on identical
  positions; replays of a block are bitwise identical (fingerprints); the engine run has
  32 resident rows with mixed depths.
- Analysis output is byte-identical across runs; `inverse_cdf_draw` reproduces the
  multinomial distribution; the CPU uniform equals the engine's.
- fp32/fp64 oracle `inv_freq` equals the bf16 reference bitwise after casting back;
  identical implementations pass the paired bias criterion, injected bias fails.
- Coupled b1/b8 and engine trajectories coincide on a tiny fixture with identical
  arithmetic and sampler distributions; swap preserves the stream, restart resets
  it. Ordinary bf16/reference and fp64/engine sampler rounding still differs.
- fp32 coupled oracle trajectories and CDF traces at batch 8 and 2 are identical
  on the tiny fixture; resuming with a different oracle batch is refused.

After changing a runner, also rerun a small pilot on the GPU (below, `--subset first:16`)
and check that forced engine replays still reproduce their forced tokens (an assertion).

For changes to the chapter's saved R1-O evidence or analysis, use the CPU checks
in [`report/1-engine/scripts/README.md`](../report/1-engine/scripts/README.md).
`check.py` checks deterministic tables and the amended binding decisions;
`check.py --r1o-sources` additionally compares every new table cell with its raw
saved JSON source. No model run is needed to validate chapter curation.

The saved nested O1 selection can be rebuilt from the recorded evaluation samples
on CPU, then checked against the frozen manifest byte for byte:

```bash
CUDA_VISIBLE_DEVICES='' .venv-tah/bin/python -m script.tah.validation.select_o1_manifest \
  --output data/tmp/report-1-engine/manifest.json
CUDA_VISIBLE_DEVICES='' .venv-tah/bin/python -m pytest -q \
  script/tah/validation/tests/test_select_o1_manifest.py \
  --basetemp=data/tmp/report-1-engine/pytest
```

The check uses the saved evaluation samples and tokenizer; it skips when those
large experiment inputs are absent from the checkout.

## Running R1-O1

```bash
M=models/TaH-plus-1.7B; C=script/tah/recipes/qwen3_1.7/eval_tah.yaml; O=data/engine/oracle/real
cap="python -m script.tah.validation.capture"; common="--manifest data/engine/oracle/manifest.json --model $M --eval-config $C"
python -m script.tah.validation.select_sequences --output data/engine/oracle/manifest.json      # CPU
$cap reference $common --dtype fp32 --layout chunk --chunk 256 --mode free --out $O/oracle-fp32
# oracle validity (subset): float64 oracle, float32 and reference b8 forced to it
$cap reference $common --subset validity --dtype fp64 --layout chunk --chunk 128 --mode free --out $O/oracle-fp64
$cap reference $common --subset validity --dtype fp32 --layout chunk --chunk 256 --mode forced --target $O/oracle-fp64 --out $O/fp32-vs-fp64
$cap reference $common --subset validity --mode forced --target $O/oracle-fp64 --out $O/ref-b8-vs-fp64
# main runs
$cap reference $common --mode forced --target $O/oracle-fp32 --out $O/ref-b8-forced
$cap engine    $common --mode forced --target $O/oracle-fp32 --out $O/engine-forced
$cap reference $common --mode free --out $O/ref-b8-free
$cap engine    $common --mode free --out $O/engine-free
$cap reference $common --subset validity --batch 1 --mode forced --target $O/oracle-fp32 --out $O/ref-b1-forced
python -m script.tah.validation.analyze --manifest data/engine/oracle/manifest.json --oracle $O/oracle-fp32 \
  --reference $O/ref-b8-forced --engine $O/engine-forced --reference-free $O/ref-b8-free --engine-free $O/engine-free \
  --reference-b1 $O/ref-b1-forced --validity-oracle $O/oracle-fp64 --validity-fp32 $O/fp32-vs-fp64 \
  --validity-reference $O/ref-b8-vs-fp64 --output $O/analysis
```

If the validity check fails, the float64 oracle is the target for every forced run
(protocol). The synthetic `mlp3` checkpoint (secondary) uses the same manifest with
`--model data/engine/fixture-models/full/mlp3` and no `--eval-config`; teacher forcing
does not require that the model generated the sequence.

`capture` options beyond `--help` worth knowing: a run directory is resumable (finished
blocks are skipped) and refuses a different configuration; `--buffer-gb` (default 16)
bounds the transient dump in `--buffer-dir` (default `data/tmp/oracle-buffer`), and a
block whose dump does not fit is replayed once per subset; `--keep-target` keeps the
target model resident instead of loading it per comparison.

The target was recorded in free mode and is replayed in that same mode. Only the
implementation being measured has the oracle decisions imposed. Target replay still
requires exact equality of visited passes, decisions, probabilities, top-1 IDs and
gaps (matching NaNs allowed); its error reports each differing field, count, first
index and finite maximum difference. No tolerance or retry-until-pass is used.
A replay failure leaves the current block unwritten; completed blocks can be resumed.

## Float64 attention on PyTorch 2.6 (#143644)

PyTorch < 2.7 computes wrong, run-to-run variable float64 softmax on CUDA for odd
widths 513–1023 not ≡ 31 (mod 32) over more than one row (`cunn_SoftMaxForward`
reduces a negative `int` size modulo an unsigned block extent; fixed upstream in
v2.7.0/v2.8.0). Float64 SDPA reaches it through the math kernel, so the
unprotected float64 oracle has wrong attention. fp32/bf16/fp16 are not affected
(they enter that kernel only above width 1024). The saved source inventory and
run metadata retain the original trigger diagnosis.

Every float64 `ReferenceRunner.run` (oracle capture, target replay, diagnostics)
and the float64 coupled oracle run inside `fp64_attention_guard()`. For an odd
key/value length it adds one zero key/value slot masked with `-inf` (`False` for
boolean masks; `is_causal`/no mask become explicit masks), so the softmax width is
even and the freshly allocated score rows are aligned. The distribution over the
real slots is unchanged; results agree with CPU float64 to rounding, not bitwise
with native kernels. Other dtypes pass through untouched. A float64 run that
never calls `F.scaled_dot_product_attention` raises. Float64 `run.json` configs
record `fp64_attention`, so an old float64 directory cannot be resumed, and
`capture.Target` refuses a float64 target that lacks it: recapture into a fresh directory.

Check it with `tests/test_fp64_attention.py` on CPU, then once on the GPU:
`CUDA_VISIBLE_DEVICES=0 python -m pytest -q script/tah/validation/tests/test_fp64_attention.py -k cuda`
(seconds; also asserts that the native fault is still present on torch < 2.7).
After a PyTorch upgrade to ≥ 2.8 the workaround becomes unnecessary but harmless; keep it until the oracle is recaptured.

## Diagnosing an oracle replay failure

If `capture.py` reports that a target recomputation does not reproduce its record, replay the shortest failing prefix in isolation (same frozen manifest and checkpoint, fresh output directory) and compare it bitwise across repeats and with CPU float64. The 2026-10-15 failure was PyTorch 2.6 bug #143644 (see above).

## Running R1-O2

The workload and margins are fixed in [the protocol](../report/PROTOCOL.md#r1-o2--whole-trajectories-with-common-random-numbers).
`prepare` loads only the tokenizer and harness datasets; it freezes 398 prompts,
answers and seeds. Seed = 42 + problem index in MATH500 [0,200), followed by GPQA
[0,198). Subsets retain those seeds. `run` takes a local checkpoint directory;
CPU execution is restricted to explicitly marked tiny fixtures.

On an available GPU, prepare and run the pilot first (diagnostic, separate outputs):

```bash
M=models/TaH-plus-1.7B
C=script/tah/recipes/qwen3_1.7/eval_tah.yaml
O=data/engine/oracle/coupled
F=$O/manifest.json
OB=8
CUDA_VISIBLE_DEVICES='' python -m script.tah.validation.coupled_run prepare --model "$M" --output "$F"
CUDA_VISIBLE_DEVICES=0 python -m script.tah.validation.coupled_run run engine --model "$M" --eval-config "$C" --manifest "$F" --pilot 32 --max-new-tokens 512 --out "$O/pilot/engine"
CUDA_VISIBLE_DEVICES=0 python -m script.tah.validation.coupled_run run reference --model "$M" --eval-config "$C" --manifest "$F" --pilot 8 --max-new-tokens 512 --out "$O/pilot/reference"
CUDA_VISIBLE_DEVICES=0 python -m script.tah.validation.coupled_run run oracle --model "$M" --eval-config "$C" --manifest "$F" --pilot 8 --max-new-tokens 512 --out "$O/pilot/oracle"
```

Host RAM: `run engine` pins the engine's 8 GiB host swap arena at startup (process
RSS ≈ engine baseline + 8 GiB; see [guide/engine.md](../guide/engine.md)); `capture.py`
pins `--host-swap-gb` per engine replay, one replay at a time. Reference and oracle
runs pin none.

Use `--pilot 8 --max-new-tokens 4096` in **new** directories for a full-context
memory check before the main run. `--oracle-batch N` (positive integer, default 8)
sets only the oracle batch; the bf16 reference remains fixed at batch 8.
PROTOCOL R1-O fixes the oracle's precision, not its layout: fp32 layout noise is
approximately 2^-16 relative versus bf16 approximately 2^-8, negligible for the
oracle. Precision, token cap, sampling and registered criteria stay fixed.

```bash
CUDA_VISIBLE_DEVICES=0 python -m script.tah.validation.coupled_run run oracle --model "$M" --eval-config "$C" --manifest "$F" --pilot 8 --max-new-tokens 4096 --oracle-batch "$OB" --out "$O/pilot-context/oracle-b$OB"
```

If that oracle pilot runs out of memory at `OB=8`, keep its failure artifacts,
set `OB=4` and rerun the same command; if it still fails, set `OB=2` and rerun.
Each batch has a separate output directory. Use the successful `OB` for the main
oracle run below. A reference OOM still requires sufficient memory for batch 8.
Then run the full workload (default token cap 4096; engine blocks 128, resident cap
64, CUDA graphs and overlap on, compile off; reference batch 8, oracle batch `OB`, left padding):

```bash
CUDA_VISIBLE_DEVICES=0 python -m script.tah.validation.coupled_run run engine --model "$M" --eval-config "$C" --manifest "$F" --out "$O/engine"
CUDA_VISIBLE_DEVICES=0 python -m script.tah.validation.coupled_run run reference --model "$M" --eval-config "$C" --manifest "$F" --out "$O/reference"
CUDA_VISIBLE_DEVICES=0 python -m script.tah.validation.coupled_run run oracle --model "$M" --eval-config "$C" --manifest "$F" --oracle-batch "$OB" --out "$O/oracle-fp32-b$OB"
CUDA_VISIBLE_DEVICES='' python -m script.tah.validation.coupled_analyze --manifest "$F" --engine "$O/engine" --reference "$O/reference" --oracle "$O/oracle-fp32-b$OB" --validity-report data/engine/oracle/real/analysis/r1o1.json --output "$O/analysis"
```

Run R1-O1's arithmetic validity check before interpreting the oracle comparison.
If fp32 fails, run the coupled oracle with `--oracle-dtype fp64` into a new
`$O/oracle-fp64-b$OB` directory, retaining `--oracle-batch "$OB"`, and analyze that
directory. Check fp64 memory separately. A missing validity report, pilot or fixture
cannot produce an overall
protocol PASS. C1/C2 diagnostics are still written with their individual verdicts.

Resume with exactly the same command. `run.json` checks checkpoint/config/workload/source/environment
identity, stores completed block indices and engine telemetry. Oracle batch
(`config.oracle_batch`, `config.batch`, `config.block_size`) is
part of resume provenance; a changed oracle batch is refused in the same directory.
Completed blocks are skipped; an interrupted block is replayed completely with the original row
identities (partial sample files are replaced). Change settings only in a new
directory. `samples/<sha256(problem-id)>.json` contains IDs, seeds, returned and
sampled tokens, text, finish reason, pass counts and the shared harness grade.
Oracle `.npz` traces hold adjacent CDF boundaries, total, u·total, boundary distance,
top probability gap and pass count at every draw, including EOS. Analysis refuses
incomplete runs and checks coverage, precision, stream alignment and provenance.

C1 bootstraps the paired problem differences (10,000 resamples, seed 20261013):
lower 95% bound ≥ −0.10 × reference mean prefix. C2 uses the harness comparison's
paired Student t 90% interval (df = problems − 1), strictly inside ±0.01. Pooled
criteria are primary; dataset values are secondary. JSON and Markdown include
full identity fractions, first-divergence histograms, per-problem oracle margins,
and PASS / `not shown`. Margins describe the first differing draw at a common
prefix; they have no added near-tie cutoff. Length censoring has no observed margin.

## Internals reference

- **Positions.** Input = prompt + forced[:-1], where forced = generated tokens plus EOS
  if the run ended with EOS. Generated positions are `len(prompt)-1 .. len(input)-1`;
  every metric is over them. Per-pass fields cover all positions; the pass at the cap
  has no probability (NaN).
- **Records** (`records/<id>.npz`): `visited`, `decision` (applied), `own_decision`
  (the implementation's own decider output, also in forced mode), `prob` (float64 of
  the value in model dtype), and at generated positions `top1`, `gap` (top-1 − top-2
  logit), the sampling distribution as CSR (`ptr`, `dist_ids`, `dist_p`), `sq`
  (Σ_v (x−o)²) and `maxabs` against the target (forced runs only).
- **Sampling distributions.** Reference and oracle: `sample_next_token` in the logits'
  dtype with `torch.multinomial` intercepted (weights renormalized, as multinomial
  does). Engine: the production `GraphSampler` bucket (`max_top_k=sampling_topk`,
  bounded fp64 shortlist on CUDA).
- **Oracle.** The reference loaded with the bf16 checkpoint weights and cast exactly to
  float32/float64 (it differs only in arithmetic), TF32 off, threshold kept at the bf16
  value (0.8984375 for the 1.7B checkpoint). HF RMSNorm and RoPE compute in float32
  inside the model, so the float64 oracle has a float32 floor there. The bf16 reference
  is cast with `.to(bf16)` after loading, as the evaluation backend does (this also
  rounds the RoPE `inv_freq` buffer); every oracle first performs that same bf16
  cast, then casts up. Constants are identical; only arithmetic precision differs.
  Float64 SDPA uses the math kernel: keep the chunk at 128.
- **Logit comparison.** A full-vocabulary oracle–implementation comparison cannot be
  stored (≈ 0.3 MB per position in bf16). The implementation dumps its native-dtype
  logits for a subset of a block to a memmap; the target then replays those sequences
  in its original free mode and chunk layout (it must reproduce its record
  bitwise) and reduces |x − o| per position. Blocks are replayed per subset; every
  replay's fingerprint must equal the first.
- **Reference `generate` layout.** Built with `reference_capture.pack(prompt_lengths=…)`:
  one left-padded prefill, then one column per step. An ended sequence is fed padding
  with mask 0 instead of its post-EOS samples (only its discarded row differs).
- **Engine run.** Eager decode (`cuda_graphs=False`, bitwise identical to graph replay)
  without overlap; `--block` requests per `generate` call with `--max-batch-size`
  resident rows. `run.json` blocks record the per-step row histogram, mixed-depth steps,
  swaps and the KV capacity (reused on every replay and resume).
- **Coupling.** `uniform(seed, j)` is the engine's uniform for the j-th sampled token of
  a request with that seed. `inverse_cdf_draw` orders candidates by probability, then
  token ID (the engine shortlist order) and picks the first whose cumulative weight
  reaches u × total, as `GraphSampler.sample` does.
- **Coupled stream lifecycle.** Prefill samples position 0; active sampling advances
  the counter once, intermediate passes do not. Swap saves seed/counter with the
  other device state. Restart discards the partial output and keeps the seed;
  fresh prefill resets counter to 0. Overlap's discarded ghost draws do not shift
  the retained output stream. Tests exercise both scheduler modes. The reference
  loop keeps all rows, including finished rows; row r always uses its original
  request seed and global generated step j (post-EOS draws are discarded).
- **Paired bias.** Each sequence-bootstrap resample uses the same clusters for
  engine and reference. Compute D = |engine bias| − 1.10·|reference bias| −
  0.1·reference M2; PASS iff its upper 95% percentile ≤ 0 (the protocol amendment).

## Short C1 and the two-stage numerical design

`short_c1.py prepare --model models/TaH-plus-1.7B --output <manifest.json>` freezes
692 eligible MATH500/GPQA questions and a disjoint 512-question GSM8K increment
from the seed-42 canonical R2 files. Only identities, questions and answers are
read; generated answers and correctness do not select cases. Chat-rendered prompts
longer than 675 tokens are excluded before generation. `--source-root` changes the
canonical input root; a different existing manifest is never overwritten. Seeds
are assigned before exclusions and remain fixed across stages. The manifest fixes
the 512-token cap and both named subsets.

Run each implementation with `coupled_run run ... --manifest <manifest.json>
--subset stage1 --max-new-tokens 512` (or `stage2`); oracle rows use
`--oracle-batch 4`, with an OOM retry in a fresh batch-2 directory. Reference stays
b8. These stage subsets are supported only for this protocol. Existing full R1-O2
commands keep their previous coverage/cap checks. Reference/oracle telemetry now
also includes allocated/reserved GPU high-water marks. CPU short-protocol fixtures
are explicitly labelled and cannot establish a confirmatory PASS.

`short_c1 analyze --manifest <manifest.json> --root <run-root> --look 1|2
--validity-report <r1o1.json>` expects `stageN/{engine,reference}/` and an
`stageN/oracle.path` file identifying that stage's successful oracle directory.
It checks complete stage coverage, provenance, production settings and oracle
validity. JSON `lookN.json` includes the paired lower 97.5% bound, administrative
censoring at 512, identical trajectories/EOS censoring, and the next action.
The lower bound is capped at the observed difference; a stage stops for hard
futility if even all remaining prefixes at engine512/reference0 cannot make
the final point margin nonnegative. C2
is withdrawn in this design: no accuracy comparison is made. Stage 2 is exactly
512 new questions and is refused after stage-1 success; saved stage-1 samples and
batch composition are never altered.

`near_tie.py --oracle <dir> --engine <free-dir> --reference <free-dir> --output
<json>` reports both the original maximum-margin rule and an added tail statistic:
the mean of the per-sequence 90th-percentile oracle margins at disagreements
(zero for a sequence with none). Decision and top-1 margins are separate tests.
A studentized paired sequence bootstrap (10,000 resamples, seed 20261013), with a
Student-t critical-value floor, bounds engine minus 1.10 times reference above
at the requested confidence (`--alpha .05` by default for diagnostics, `.025` for
each confirmatory look, giving a 97.5% bound and a t floor with n−1 degrees of freedom).
JSON includes `alpha`, `confidence`, `upper_bound` and `upper95` or `upper975`. Both bounds must be <=0; fewer than 20 disagreement events on either side
are inconclusive. Correlations of sequence maxima, shared-position Jaccard overlap and binary
position phi on jointly eligible positions are descriptive, without an invented correlation threshold. The original maximum
rule is computed in full and reported, without affecting the conclusion or futility.
The calibrated tail criterion is binding; both its addition and the demotion of the
maximum rule were decided after diagnostic pilots and before confirmatory data.

For a free CPU diagnostic, replace `--reference` with `--self-splits 1000`
(`--bootstrap-per-split 1000`): random halves use independent cluster resampling.
The output separates maximum-rule failure, failure to prove non-inferiority,
exact permutation false equality alarms, and false non-inferiority assertions at
the artificial 10% boundary. Only the false-alarm rates are nominally 5%; the
success rate at equal accuracy is power, not a guaranteed 95%.

`capture --sequence-order reverse` reverses the selected sequences and records the
order in provenance; use fresh outputs to compare a different engine batch
composition. It does not change teacher-forced inputs. `paired_look --root <root>
--manifest <manifest> --look 1|2` pools the immutable O1 stage shards, applies the
approved 97.5% bounds, checks exact 64/200 coverage and arithmetic validity, and
retains each criterion's early success permanently. Both forced and free modes start
on 64 sequences; unresolved forced criteria and tail criteria independently trigger
the same frozen +136 IDs, pooled to 200. Each look spends one-sided alpha .025.
`o1/lookN.json` includes `criteria` (forced bounds), `tail` (studentized tail bounds),
`forced_next_action` and `free_next_action`: `stop_for_success`, `continue_fixed_136`,
`stop_at_maximum_not_shown` or `stop_for_futility` (invalid oracle arithmetic).
`original_maximum` records its nonbinding status and both maximum-rule verdicts.
A stopped arm needs no stage-2 records; the other arm can continue alone. Shared
oracle records for the 136 IDs are computed once when either increment is needed.
Validity failure stops both arms and C1 before any further GPU work; maximum-rule
failure never triggers futility. The composition demonstration stays a single
descriptive comparison at its original diagnostic confidence.
The legacy `analyze` conclusion now requires its original 200/500k size and valid
oracle, so its diagnostic 64-sequence output cannot be mistaken for a protocol
PASS; the approved two-stage result lives in `paired_look`'s `o1/lookN.json`.

Validation: `tests/test_short_design.py` covers prompt exclusions/stable seeds,
fixed cap/subset execution, append/resume and fixture guards, quantiles, studentized
paired tails and alpha spending, independent forced/free shell branches, permanent
per-criterion stopping, report-only maxima, self-comparison diagnostic distinctions and mandatory size/validity
qualification. Run the complete CPU validation and engine suites after runner changes.

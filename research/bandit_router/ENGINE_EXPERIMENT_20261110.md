# Engine-backed bandit training experiment

The report index now reflects completed R1-O numerical validation. The new
bandit path uses `tah.engine` for both paired-token collection and generation;
optimizer updates still train only the small router on saved features. The
backbone and native imitation decider remain frozen.

Launched November 10, 2026 at **23:59 KST** (14:59 UTC), PID **240898**.
The parent is detached (session leader, parent PID 1); its engine collector child
was observed running with GPU activity. Initial stage: **collecting**. Training
starts automatically after collection and replay verification.

Selection finished at 500 problems / **401,685 reference tokens** after scanning
40,481 candidates; one lexical benchmark overlap and one duplicate were excluded.
The launch settings and input manifest are saved in
[math500_engine_20261110](math500_engine_20261110/). No training result is available
at launch.

## Registered settings

See the exploratory training section in [the protocol](../../report/PROTOCOL.md).
The run selects 500 complete MoT math solutions, caps prompts at 512 and
references at 1024 tokens, and screens lexical overlap with GSM8K/MATH500.
Seed 42 gives a problem-level 450/50 train/validation split. The existing
five-penalty sweep uses early stopping and selects validation token loss at
imitation's continuation rate. No answer-accuracy result is claimed.

The run is staged in `/root/TaH-bandit-engine-20261110` on the L40S-24Q machine.
It shares the existing checkpoint and Python environment; the old checkout and
200-problem experiment are preserved. `SOURCE_SNAPSHOT.json` records base commit
`93298e5` and the source snapshot SHA256:

```
4a8aeafd518c0b260dbc77bbb9c2e665aad13a11671c5113d8009c9111fd0197
```

## Commands

From the isolated source directory:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export CUDA_VISIBLE_DEVICES=0
export MAX_JOBS=2
export TORCH_CUDA_ARCH_LIST='8.9+PTX'
export OMP_NUM_THREADS=4
.venv-tah/bin/python -u -m research.bandit_router.experiment \
  --backend engine --model models/TaH-plus-0.6B --seed 42 \
  --count 500 --max-reference-tokens 1024 \
  --output-dir research/bandit_router/math500_engine_20261110
```

The detached launch uses `subprocess.Popen(..., start_new_session=True)` with
standard input disconnected and output redirected to `experiment.log`.
`launch.json` records the PID, time, exact argument list and environment.
`status.json` records pipeline stage; `progress.json` records collection counts.
The pipeline verifies the first three complete collected histories, runs the
sweep, audits the selected checkpoint, then runs imitation and bandit generation
checks. Generation uses the selected policy during prefill as well as decoding.

## Checks before launch

```bash
CUDA_VISIBLE_DEVICES='' .venv-tah/bin/python -m pytest -q \
  research/bandit_router/test_engine.py research/bandit_router/test_core.py \
  tah/engine/tests/test_engine.py
```

Output on both development and CUDA hosts: **43 passed, 1 skipped**. The skip is
an existing engine test; no GPU benchmark accuracy claim follows from this suite.
The tests include independent STOP/CONTINUE branches, speculative-cache removal,
feature order, depth caps, engine policy installation, collection resume and
unbranched replay. Ruff's undefined/unused-name checks and `git diff --check`
also pass.

The server uses PyTorch 2.6.0+cu124, FlashInfer 0.2.5 and Triton 3.2.0. FlashInfer,
Ninja, setuptools and pytest were added to the existing environment; the engine
CUDA/toolkit check passed.

Saved engineering outputs are in [engine_validation_20261110](engine_validation_20261110/):

| Check | Output |
| --- | --- |
| Initial real-math pilot, before the graph-indexing fix | 2 complete problems, 1,740 token histories and chosen-action losses replayed successfully (`pilot-replay.json`) |
| Final source, short arithmetic reference | 16 histories/losses replayed; depth-1 hidden features match exactly at storage precision (`final-replay.json`) |
| Saved bandit router on final source | 32 generated tokens; all 33 decode steps used CUDA graphs (`graph-bandit.json`) |
| Native imitation on final source | 32 generated tokens; all 33 decode steps used CUDA graphs (`graph-imitation.json`) |

The 32-token generations both stop at the length cap. They test execution, not
completed-answer accuracy or comparative latency. The initial real-math pilot
is separate from the new training cache. Production collection is eager on the
engine's persistent KV pool; generation uses its scheduler and CUDA graphs.

# Frozen-TaH bandit router prototype

Implements the [CS475 proposal](https://docs.google.com/presentation/d/1_uOVU7VNYI1QoNVYafIdQDxy4w0s1bR42eR02nlC448/edit): freeze TaH+0.6B and its LoRA adapters, cache paired local rewards, train only a small routing MLP, and route output tokens at depth 1 or 2. Uses the original `tah/` model and its existing environment, not `tah2/`.

## Run on the CUDA machine

From `/root/TaH` with the existing `.venv-tah` environment:

```bash
.venv-tah/bin/python -m research.bandit_router.run collect \
  --references research/bandit_router/references.jsonl \
  --max-tokens 32 --output research/bandit_router/artifacts/cache.pt

OMP_NUM_THREADS=2 .venv-tah/bin/python -m research.bandit_router.run train \
  --cache research/bandit_router/artifacts/cache.pt \
  --penalty 0.05 --epochs 100 \
  --output research/bandit_router/artifacts/router.pt

.venv-tah/bin/python -m research.bandit_router.run generate \
  --router research/bandit_router/artifacts/router.pt \
  --prompt 'What is 17 + 25?' --max-tokens 32 \
  --output research/bandit_router/artifacts/generation.json
```

Collection refuses to overwrite an existing cache; choose a new output filename when rerunning. `--model` defaults to `models/TaH-plus-0.6B`, `--device` to `cuda:0`, and the frozen imitation threshold to `0.9`. Training can run on CPU with PyTorch and needs no model load. All three commands accept `--seed`. Run `--help` after a subcommand for options.

Generation supports `--policy stop`, `continue`, `imitation`, `entropy`, or `bandit` on the same checkpoint. Prompt prefill uses imitation for all policies; each policy then controls output-token decisions. Entropy and bandit thresholds default to those calibrated on validation. `--threshold` overrides them. Generation uses greedy decoding and records generated-token depths (including EOS if reached), total latent passes including prefill, continuation fraction, termination, and elapsed time excluding model loading.

## Fixed-history counterfactuals

Each reference problem is teacher-forced token by token:

1. Prefill the prompt except its final token with the frozen imitation router.
2. Clone the imitation KV cache twice, before processing the current input token.
3. Force the current token to STOP in one branch and CONTINUE in the other. Both see identical prior tokens and prior imitation depths.
4. Cache the depth-1 state and the next reference token's two cross-entropy losses. Assert bit-exact equality of depth-1 states and imitation decisions across branches.
5. Commit only the branch corresponding to the frozen imitation decision, then feed the next reference token.

The two branches call the native TaH wrapper, so CONTINUE uses its **additive depth-1 plus depth-2 logits**, with the trained input updater and depth-specific LoRA behavior intact. This is deliberately a slow, easily inspected collector: it reruns depth 1 and deep-copies full caches. It is suitable for a few problems; optimize only after verifying equivalence.

No target labels enter `model.forward`, avoiding the training helper that can force a token to continue. No counterfactual action can enter the next token's history. `core.py` contains the cache isolation, feature extraction, policy adapter, and objective; production model code is untouched.

## State, objective, and split

Features concatenate hidden states from transformer layers `(1, floor(N/2), N)`, next-token entropy, and maximum probability. HF index 0 is embeddings; layer 1 is the first transformer layer. For the 28-layer 0.6B checkpoint this gives `(1, 14, 28)` and 3,074 features. States depend only on the current prefix, not the reference target or its losses.

The router is a 64-unit SiLU MLP. Feature normalization uses **training problems only**. Optimize the exact full-information contextual-bandit objective:

```text
r(STOP)     = 0
r(CONTINUE) = loss_STOP - loss_CONTINUE - lambda
loss_router = -mean(sigmoid(router(state)) * r(CONTINUE))
```

Both rewards are observed offline, so action sampling and a REINFORCE estimator are unnecessary. Losses are stored without lambda, allowing a separate router to be trained for each penalty. Lambda must be nonnegative.

Split by normalized problem text using a fixed seed (90%/10% by default, at least one problem in each split). Duplicate question text stays in one split. Calibrate the hard routing threshold on validation to match its imitation continuation fraction; ties may prevent exact matching. The four synthetic reference problems are smoke-test data, not GSM8K/MATH500 test examples or a research training set. Two references are truncated by the 32-token pilot cap.

The report's always-1/2, entropy, imitation, bandit, and oracle rows are **local actions evaluated on the same fixed imitation histories**. They are not free-running policy benchmarks. The oracle uses the reference target and cannot be used at inference.

## Validation and saved pilot

```bash
.venv-tah/bin/python -m unittest research.bandit_router.test_core -v
.venv-tah/bin/python -m research.bandit_router.verify_cache \
  --cache research/bandit_router/artifacts/cache.pt
```

The CUDA pilot produced 114 paired rows across four problems. All 114 replayed against an unbranched imitation run with bit-exact depth-1 features and matching chosen-action losses. Five unit tests cover cache isolation, imitation-only history commits, reward-gradient direction, feature statistics, and duplicate-safe splitting.

With lambda `0.05`, training used 90 rows from three problems and validation used 24 rows from the remaining problem. The minimized training objective changed from `0.6853` to `-0.6671`. At the same validation average depth `1.0417`, local reference-token NLL was `2.5774` for imitation and `2.6865` for the bandit router. **This pilot does not demonstrate an improvement.** A 16-token bandit generation smoke test reached its token limit at average depth 1.0; it did not produce a complete answer.

Artifacts include `cache.pt`, `cache.verification.json`, `router.pt`, `router.report.json`, and `generation.json`. The cache records exact prompt/reference token IDs, per-token imitation depths, history hashes, selected layer indices, source hashes, and runtime. The router checkpoint includes normalization statistics, policy weights, lambda, threshold, and split/report metadata.

## Larger training experiment

`experiment.py` runs preparation, resumable collection, history replay checks, a five-penalty training sweep, and two generation smoke tests in order. Launch from `/root/TaH`:

```bash
nohup env OMP_NUM_THREADS=4 .venv-tah/bin/python -u \
  -m research.bandit_router.experiment \
  --output-dir research/bandit_router/math200 \
  --count 200 --max-reference-tokens 1024 \
  > research/bandit_router/math200.log 2>&1 < /dev/null &
```

Check progress with:

```bash
tail -f research/bandit_router/math200.log
cat research/bandit_router/math200/status.json
cat research/bandit_router/math200/progress.json
```

The process keeps running after disconnection. A file lock prevents two copies of this experiment from running concurrently. Rerun the same command to resume saved complete-problem shards after a failure. Completed penalties also resume without retraining. Existing toy results are preserved. If settings or collector code change, use a new directory.

Preparation pins the public `open-r1/Mixture-of-Thoughts` math/train revision and uses seed 42. It deduplicates normalized questions and screens against the saved GSM8K/MATH500 test questions using normalized exact matches and five-word-shingle similarity. This is a practical lexical screen, not a guarantee against semantic overlap or contamination in the pretrained checkpoint. Only complete solutions of at most 1,024 tokens with a final response after `</think>` are selected; nothing is truncated. The length filter makes this a pilot of shorter MoT traces.

Collection shares depth 1 between actions, captures its STOP logits and KV snapshot before depth 2, and still commits only the imitation action. `verify_fast.py` checks this optimization against the original two-forward collector. The verified 24-token fixture matched both losses, states, and committed KV tensors bit-for-bit. Shards store features in float16 and losses in float32; a replay check verifies the first three complete problems at that storage precision.

Training splits 180/20 problems, normalizes from training only, and uses mini-batches, AdamW weight decay, gradient clipping, and early stopping. For each lambda in `[0, 0.01, 0.05, 0.1, 0.2]`, it evaluates every epoch and saves the checkpoint with the best **hard-policy validation token loss at imitation's continuation budget**. It then reports continuation budgets from 0% to 100%. Raw policy logits determine calibration and inference decisions, avoiding loss of ranking when sigmoid probabilities round to 1. The best penalty is selected by validation NLL at the same imitation budget.

`math200/training/` contains per-epoch curves, per-penalty checkpoints/reports, `selected_router.pt`, and `summary.json`. Validation is used for model and threshold selection, so these measurements are tuning diagnostics. They require a separate test evaluation before making an accuracy claim. The final two generation checks use 512-token caps on one arithmetic prompt; completed answers and truncations remain visible in their JSON files.

For the next experiment, supply JSONL rows with `id`, `question`, and a complete `reference` assistant continuation. Screen/deduplicate training problems against benchmark test questions before collection. Train on a meaningful math set, tune lambda and thresholds on validation, then generate completed GSM8K/MATH500 answers under all five policies at matched compute budgets. Final-answer accuracy requires that separate generation evaluation.

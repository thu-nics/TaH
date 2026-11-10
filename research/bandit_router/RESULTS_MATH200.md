# Larger bandit-router training results

Audited November 10, 2026. The detached experiment completed successfully in 7.58 hours, including preparation, collection, replay checks, training, and generation checks. The five-penalty training sweep itself took 224 seconds; collection dominated runtime.

The selected router improves the **local validation token-loss surrogate** over imitation at the same continuation budget. This does not establish an improvement in final-answer accuracy.

## Data and checks

- 200 distinct math problems from `open-r1/Mixture-of-Thoughts`, math/train revision `e55fa28006c0d0ec60fb3547520f775dd42d02cd`, seed 42.
- 159,348 paired reference-token states; complete solutions with 55–1,023 tokens, no truncation. Selecting short solutions introduces length bias.
- 180 training problems and 20 validation problems. Validation contains 15,278 tokens.
- Normalized duplicate screening and lexical overlap checks against the saved GSM8K and MATH500 test questions ran before selection. This does not establish absence of semantic contamination or pretrained-model exposure.
- The frozen imitation routing history was preserved for STOP/CONTINUE comparisons. Native accumulated TaH logits determined depth-2 losses.
- Replay verified the first three complete problems: 2,707 tokens, matching depth-1 features at storage precision and matching chosen-action losses. The other 197 problems were not replayed individually.

## Matched-budget comparison

Loss is mean reference-token negative log likelihood, in nats/token; lower is better. Every action is evaluated on the same fixed imitation histories, rather than on separate free-running histories.

| Policy | Continued tokens | Average depth | Validation token loss |
| --- | ---: | ---: | ---: |
| Always STOP | 0% | 1.00000 | 0.644248 |
| Entropy | 1.6887% | 1.01689 | 0.639727 |
| Imitation | 1.6887% | 1.01689 | 0.637412 |
| Bandit, selected checkpoint | 1.6887% | 1.01689 | **0.634048** |
| Always CONTINUE | 100% | 2.00000 | 2.113176 |

Imitation and bandit each continue on exactly 258 of 15,278 validation tokens. Their decisions differ at 308 positions. Bandit reduces token loss by 0.003364 nats/token, or **0.53% relative to imitation**, and improves mean loss on 14 validation problems, worsens it on five, and ties on one.

`training/audit.json` reproduces the saved checkpoint's token loss and continuation rate using the original CPU batch size and thread settings. Changing numerical execution geometry can move a score across the threshold at its boundary; the audit therefore records and reproduces the original evaluation settings. The policy does not enforce an exact continuation quota during generation.

The audit also contains a descriptive bootstrap across validation problems. Because checkpoints, lambda, and thresholds were selected on these same 20 problems, that interval is not an independent test confidence interval and should not support a significance claim.

## Training duration and selection

The sweep used lambda values 0, 0.01, 0.05, 0.1, and 0.2. The selected configuration was **lambda 0.05, epoch 10**, with early stopping after epoch 25. Checkpoints were selected by hard-policy validation token loss at imitation's continuation budget.

| Epoch, lambda 0.05 | Training objective, lower better | Matched-budget validation token loss |
| --- | ---: | ---: |
| 0 | 0.684253 | 0.659898 |
| 10, selected | -0.034059 | **0.634048** |
| 25, stopped | -0.073400 | 0.639027 |

Training reward continued to improve after epoch 10 while the deployed validation policy worsened. More epochs on this dataset would not be justified by the observed curve. The saved checkpoint is the selected epoch, not the final epoch.

The selected model is not strongest at every budget. At approximately 5% continuation, its local loss is 0.631141 versus entropy's 0.633126. At approximately 10%, entropy is better: 0.625177 versus bandit's 0.644050. Separate method selection or a broader claim would require independent evaluation.

## Generation checks

Both checks used greedy decoding on `What is 17 + 25?` with a 512-token limit. Neither emitted a complete final answer before truncation.

| Policy | Generated tokens | Average depth | Latent passes including prefill | Runtime |
| --- | ---: | ---: | ---: | ---: |
| Imitation | 512 | 1.02930 | 545 | 30.82 s |
| Bandit | 512 | 1.01758 | 539 | 29.50 s |

The bandit continuation rate was close to its calibrated reference-token rate on this one prompt. These are execution checks, not an answer-accuracy benchmark or a statistically useful latency comparison. The earlier Standard-versus-TaH+ GSM8K/MATH500 results did not use this learned router.

## Saved artifacts and reproduction

Artifacts are copied locally under `research/bandit_router/math200/`, including `training/selected_router.pt`, per-penalty reports and learning curves, `training/summary.json`, `training/audit.json`, replay verification, and both generation JSON files. The full cache and resumable shards remain on the CUDA machine under `/root/TaH/research/bandit_router/math200/`.

Reproduce the audit on the CUDA machine from `/root/TaH`:

```bash
.venv-tah/bin/python -m research.bandit_router.audit_results \
  --run-dir research/bandit_router/math200
```

The next evaluation should freeze this checkpoint and its settings, then compare completed benchmark answers under bandit, imitation, and entropy routing while measuring their actual generated-token depths. Calibration on reference histories does not guarantee matched compute after each policy generates its own histories.

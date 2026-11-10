# Research report evidence

The [evaluation protocol](PROTOCOL.md) defines confirmatory workloads, seeds and
analysis. Chapters curate the results used in the research report; the
[chapter format](AGENTS.md) defines provenance and reproducibility requirements.

| Chapter | Contents | Status |
| --- | --- | --- |
| [Engine](1-engine/README.md) | Design, reference self-noise, mechanism checks, performance, robustness, confirmatory accuracy (R1) and negative results | R1-O numerical validation complete (PASS within the registered 10% margin); independent R1 accuracy equivalence not shown (underpowered); coupled ±1 pp accuracy test withdrawn before data |
| [Baseline](2-baseline/README.md) | Reproduction of the published TaH+ 1.7B results on nine benchmarks (R2, three seeds), code-prompt analysis, run telemetry at 8192 tokens | Curated summary; R2 complete (8 of 9 benchmarks and the average reproduced with the TaH2 code prompt; GPQA-Diamond below the paper) |

Regenerate the chapters on CPU:

```bash
python report/1-engine/analyze.py
python report/2-baseline/analyze.py
```

GPU reproduction commands and runtime estimates are listed by
`bash report/1-engine/reproduce.sh` and `bash report/2-baseline/reproduce.sh`;
printing the plan runs no experiments.

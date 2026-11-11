# User guide

- [Installation](installation.md): environments, extras, GPU prerequisites, environment check, troubleshooting.
- [Evaluation](evaluation.md): run TaH benchmarks with the harness CLI and YAML recipes.
- [Results](results.md): read saved outputs, check completion, collect summaries and compare runs.
- [Engine](engine.md): generate with the TaH v1 inference engine, memory and scheduling, playground.

These guides cover **TaH v1** (`tah/`). TaH2 has separate entrypoints and
dependencies; see the main [README](../README.md#tah2-usage) for TaH2 training,
evaluation and serving.

Run commands from the repository root in the matching environment. GPU commands
need an available GPU; help and saved-result commands run with
`export CUDA_VISIBLE_DEVICES=""`.

| Task | Read |
| --- | --- |
| Install, check the engine environment, fix import or JIT errors | [installation.md](installation.md) |
| Pick a backend and dataset, select rows, set sampling, repeats and jobs | [evaluation.md](evaluation.md) |
| Run a report protocol | [evaluation.md](evaluation.md#report-protocol-runs) |
| Read CSV/JSON output, check completion, collect and compare runs | [results.md](results.md) |
| Generate with the engine, size memory, choose graphs and sampling | [engine.md](engine.md) |
| Train TaH, prepare labels, run the TaH v1 demo | Main [README](../README.md#tah-usage) and [recipes](../script/tah/recipes/README.md) |
| Train, evaluate or serve TaH2 | Main [README](../README.md#tah2-usage) and [mini-SGLang README](../tah2/minisgl/README.md) |

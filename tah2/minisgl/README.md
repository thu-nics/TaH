# Bundled MiniSGL engine

This directory contains the inference core derived from
[mini-sglang](https://github.com/sgl-project/mini-sglang), adapted for TaH DUO and
uniform checkpoints. It ships with TaH2 and uses the `tah2.minisgl` Python
namespace. The engine source and its CUDA/C++ kernels are included in this repository's wheels.

Install from the repository root into a CUDA-enabled environment:

```bash
python -m pip install -e '.[tah2]'
```

Native kernels need a CUDA toolkit with `nvcc` and a C++20 compiler (GCC 11 or
newer). They compile and cache on first use. Standard `CC`, `CXX`,
`NVCC_PREPEND_FLAGS`, and `LD_LIBRARY_PATH` environment variables can select a
compiler installation and its runtime libraries.

## Generation and serving

```python
from tah2.minisgl.core import SamplingParams
from tah2.minisgl.llm import LLM

llm = LLM(model_path="checkpoint")
try:
    print(llm.generate(["What is 1+1?"], SamplingParams(max_tokens=32)))
finally:
    llm.shutdown()
```

Standard checkpoints load their Qwen3 weights and tokenizer files directly.
TaH2 checkpoints also contain `tah_config.json`, `input_updater.bin`, and
`iter_decider.bin`. DUO uses a trained
`Qwen3MLPIterDecider` and causal recurrent KV from depths up to the query depth.
Uniform uses `AlwaysIterDecider`, current-token recurrence, and `even_mix`.

Both paths read their depth limit from `tah_config.json` (`max_iter`). For
offline generation, `LLM(..., tah_max_iter=12)` overrides the limit. For direct
serving, use `--tah-max-iter 12`. Directory names do not set the depth. DUO's
learned gate can stop before this limit; uniform always uses its fixed depth.

From the repository root:

```bash
python -m tah2.minisgl --model-path checkpoint --port 30080
```

The server exposes `/v1/models`, `/v1/chat/completions`, and `/v1/completions`.
The public model name defaults to `minisgl`; `--served-model-name` changes it.
Responses use that name instead of the local checkpoint path. Completions include
`usage.iter_counts`; DUO also supplies `usage.prompt_iter_counts`.
Chat completions support streaming. Multiple completions, stop sequences, and
presence/frequency penalties are not implemented.

The TaH2 launch and evaluation wrappers select this bundled engine. Offline
evaluation uses `BACKEND=mini_sglang`; `BACKEND=hf` (default) uses Transformers.
Both backends support TaH2 and Standard checkpoints.

Page size is fixed to 1. The offline Python API uses one GPU; use the server's
`--tp-size` option for tensor-parallel inference.

MIT licensed; see [LICENSE](LICENSE). The bundled NCCL header retains NVIDIA's
[license](kernel/csrc/include/minisgl/LICENSE.txt).

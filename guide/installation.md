# Installation and environment checks

Run these commands from the repository root. TaH and TaH2 need separate
environments because their torch, Transformers, Triton and FlashInfer pins
conflict. TaH uses Python 3.10; TaH2 uses Python 3.12 on Linux with CUDA GPUs.

## TaH environment

```bash
python3.10 -m venv .venv-tah
source .venv-tah/bin/activate
python -m pip install -e '.[tah,evaluation,engine]'
python -m pip check
```

Keep the TaH pins (torch 2.6.0, Transformers 4.52.4, PEFT 0.15.2). The CUDA
suffix on torch can vary with the installed build. The `evaluation` extra needs
Git and network access for its EvalPlus dependency, which is pinned to one
commit; do not replace it with another release.

| Extra | Needed for |
| --- | --- |
| `tah` | Reference model, config loading and scripts (torch, Transformers, PEFT, Accelerate, datasets, PyYAML). |
| `evaluation` | Harness backends and graders: pylatexenc, sympy, math_verify, SGLang, pinned EvalPlus. Install with `tah`. |
| `engine` | TaH v1 engine and the `tah_engine` backend: Triton, FlashInfer, Ninja, safetensors, tokenizers. |
| `training` | Training: W&B, DeepSpeed, flash-attn. Install with `tah`. |
| `tah2` | The TaH2 stack, in its own environment. |
| `dev` | pytest and Ruff. |

Install only the extras you use. Engine evaluation needs `.[tah,evaluation,engine]`.
For training, use `python -m pip install -e '.[tah,training,evaluation]'`.
Exact pins are in [pyproject.toml](../pyproject.toml).

## TaH2 environment

```bash
python3.12 -m venv .venv-tah2
source .venv-tah2/bin/activate
python -m pip install -e '.[tah2]'
python -m pip check
```

Use it for `tah2/`, `script/tah2/` and `bash/`, following the main README. Do not
add `tah` or `engine` to this environment.

## Engine prerequisites

The engine needs Linux, a CUDA-enabled torch build, an NVIDIA GPU with compute
capability 8.0 or newer (BF16), and a driver that supports that CUDA build.
The engine's bounded prefill kernel compiles on first use, including with
prebuilt FlashInfer packages. This needs:

- A CUDA Toolkit whose **major version** matches `torch.version.cuda`, including
  `nvcc`. The driver alone is not enough. `nvcc` is searched on `PATH`, then at
  `$CUDA_HOME/bin/nvcc` or `$CUDA_PATH/bin/nvcc`.
- A C++ toolchain and the Python development headers for the active interpreter,
  such as `python3.10-devel` (Fedora) or `python3.10-dev` (Debian/Ubuntu). Header
  search covers the interpreter's include directory, then `CPATH`,
  `C_INCLUDE_PATH` and `CPLUS_INCLUDE_PATH`.

## Environment check

When a GPU is idle, run:

```bash
source .venv-tah/bin/activate
CUDA_VISIBLE_DEVICES=0 python -c 'import tah.engine'
```

Importing `tah.engine` runs `check_environment()` unless `CUDA_VISIBLE_DEVICES`
is exactly the empty string. The check covers imports, CUDA, GPU capability and
the compiler toolchain required by the bounded prefill kernel. Passing it
does not prove kernel correctness or enough free GPU memory.

| Message starts with | Action |
| --- | --- |
| `TaH engine dependencies are missing` | Install `.[tah,evaluation,engine]` in the active environment and read the chained import error. |
| `TaH engine requires a CUDA-enabled PyTorch build` | Use the CUDA torch build on Linux. |
| `... no usable NVIDIA GPU` | Check the driver and `CUDA_VISIBLE_DEVICES`. |
| `... compute capability >= 8.0 (sm80)` | Select a supported GPU. |
| `... nvcc was not found` | Put `nvcc` on `PATH` or set `CUDA_HOME`. |
| `... Python development headers were not found` | Install matching headers or set the include variables above. |
| `Could not run CUDA compiler ...` | Read the reported OS or compiler error. |
| `... nvcc ..., but PyTorch was built with CUDA ...` | Select a toolkit whose major version matches torch. |

Engine environment errors in `eval.py` and the playground suggest
`--backend tah`, the reference backend, which does not need the engine.

## CPU-only commands

```bash
export CUDA_VISIBLE_DEVICES=""
source .venv-tah/bin/activate
python script/tah/evaluation/eval.py --help
python script/tah/evaluation/collect_results.py --help
```

Empty visibility skips the automatic engine check but does not allow inference.
Do not start a full evaluation with empty visibility: the driver then assigns
GPUs 0..7 to its workers (see [GPU and job allocation](evaluation.md#gpu-and-job-allocation)).

## Kernel check (GPU)

With the GPU idle:

```bash
source .venv-tah/bin/activate
CUDA_VISIBLE_DEVICES=0 python -m tah.engine.tests.kernel_check
```

The module has no options and no `--help`. It JIT-compiles as needed, compares
BF16 FlashInfer attention with dense PyTorch SDPA, prints the environment and
per-case timings, and raises on a numerical mismatch or missing CUDA. It writes no files.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Wrong packages or imports | Check `python -m pip --version` and `python -m pip check` in the intended environment (`.venv-tah` or `.venv-tah2`). |
| Missing re-exports after a layout change | Re-run `python -m pip install -e '.[tah]'`; editable installs can cache the old layout. |
| FlashInfer compile fails after the import check passed | Read the compiler output; check headers, the C++ toolchain, toolkit paths and that the JIT cache directory is writable. |
| HF `flash_attention_2` missing | Leave `use_flash_attention` false (default), or install the `training` extra's flash-attn. Affects only `hf`. |
| EvalPlus import mismatch | Install the pinned `evaluation` extra. |
| Local benchmark file missing | Check the [dataset table](evaluation.md#datasets); installation does not fetch `data/`. |
| CUDA OOM during evaluation | For `tah_engine`, lower `engine.max_batch_size` or `prefill_chunk_size`; for `tah`, lower `batch_size`. See [engine memory](engine.md#scheduling-memory-and-long-generations) and [crashes and partial results](evaluation.md#crashes-and-partial-results). |

Source of truth: [dependency extras](../pyproject.toml),
[environment check](../tah/engine/__init__.py),
[kernel check](../tah/engine/tests/kernel_check.py).

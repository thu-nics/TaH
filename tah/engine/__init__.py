"""Runtime checks for the TaH inference engine."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sysconfig


def _find_nvcc() -> str | None:
    candidate = shutil.which("nvcc")
    if candidate:
        return candidate
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        candidate = os.path.join(cuda_home, "bin", "nvcc")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def check_environment() -> None:
    """Fail early with setup guidance when this host cannot run the engine."""
    try:
        import torch
        import flashinfer
    except ImportError as exc:
        raise RuntimeError(
            "TaH engine dependencies are missing. Install them with "
            "`pip install -e '.[tah,evaluation,engine]'`."
        ) from exc

    if torch.version.cuda is None:
        raise RuntimeError(
            "TaH engine requires a CUDA-enabled PyTorch build. Reinstall the project "
            "in a Linux environment with `pip install -e '.[tah,evaluation,engine]'`."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"TaH engine found torch CUDA {torch.version.cuda}, but no usable NVIDIA GPU. "
            "Check the NVIDIA driver and that it supports this PyTorch CUDA build."
        )
    capability = torch.cuda.get_device_capability()
    if capability < (8, 0):
        raise RuntimeError(
            f"TaH engine requires an NVIDIA GPU with compute capability >= 8.0 (sm80); "
            f"found {torch.cuda.get_device_name(0)} with sm_{capability[0]}{capability[1]}."
        )

    # Bounded prefill requires JIT even with prebuilt FlashInfer ops.
    _check_prefill_toolchain(torch.version.cuda)

    # Keep a module reference for diagnostics and make the import check explicit.
    del flashinfer


def _check_prefill_toolchain(torch_cuda_version):
    """The engine-owned prefill variant is compiled locally on first use."""
    nvcc = _find_nvcc()
    if nvcc is None:
        raise RuntimeError(
            "TaH prefill uses runtime JIT compilation, but `nvcc` "
            "was not found. Install the CUDA Toolkit matching the major CUDA version "
            f"reported by PyTorch ({torch_cuda_version}), then put nvcc on PATH or "
            "set CUDA_HOME. The NVIDIA driver alone is not enough for the prefill kernel."
        )
    python_include = sysconfig.get_path("include")
    python_header_dirs = [python_include] if python_include else []
    for variable in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"):
        python_header_dirs.extend(filter(None, os.environ.get(variable, "").split(os.pathsep)))
    if not any(os.path.isfile(os.path.join(path, "Python.h")) for path in python_header_dirs):
        raise RuntimeError(
            "TaH prefill needs runtime JIT compilation, but Python development headers "
            f"were not found at {python_include!r}. Install the development headers "
            "for this Python version (for example, `python3.10-devel` on Fedora or "
            "`python3.10-dev` on Debian/Ubuntu), then retry."
        )
    try:
        output = subprocess.check_output([nvcc, "--version"], text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Could not run CUDA compiler {nvcc!r}: {exc}") from exc
    match = re.search(r"release\s+(\d+)\.(\d+)", output)
    torch_cuda = re.match(r"(\d+)\.(\d+)", torch_cuda_version)
    if match and torch_cuda and match.group(1) != torch_cuda.group(1):
        raise RuntimeError(
            f"FlashInfer JIT found nvcc {match.group(1)}.{match.group(2)}, but PyTorch "
            f"was built with CUDA {torch_cuda_version}. Use a CUDA Toolkit with the same "
            "major version and set CUDA_HOME/PATH to it."
        )


# Reference fixture tools/loaders also live here and run without CUDA. An
# explicitly hidden GPU opts into that mode; normal runtime imports still
# perform the WP0 prerequisite check.
if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    check_environment()

def __getattr__(name):
    if name == 'TaHEngine':
        from .engine import TaHEngine
        return TaHEngine
    if name in ('Request', 'Result'):
        from . import scheduler
        return getattr(scheduler, name)
    if name == 'SamplingParams':
        from .sampling import SamplingParams
        return SamplingParams
    raise AttributeError(name)


__all__ = ["check_environment", "TaHEngine", "Request", "Result", "SamplingParams"]

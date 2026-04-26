"""Test harness shared across component tests.

Each component test follows the same pattern:

  1. ``capture(name, fn, *args)`` runs ``fn(*args)`` against the public TaH at
     ``$TAH_PUBLIC_ROOT`` (default ``/tmp/TaH-pub``) inside a subprocess and
     pickles the inputs+outputs to ``tests/baselines/<name>.pt``.
  2. ``compare(name, fn, *args)`` runs ``fn(*args)`` against the cleaned
     ``tah-release`` in this process and asserts every output tensor matches
     the recorded baseline within ``ACC_TOL``.
  3. ``bench(label, fn, *args, ref_fn)`` measures wall-clock for ``fn`` vs
     ``ref_fn`` over ``WARMUP+ITERS`` calls and prints a one-line speedup.

Snapshots are PyTorch ``.pt`` files containing a dict ``{"args": ..., "out":
...}``. Tensors are saved on CPU; we move them to ``DEVICE`` at compare time.

The split exists because public TaH and the cleaned tah-release share the
package name ``tah``; both cannot live on ``sys.path`` at once.
"""
from __future__ import annotations

import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

REPO = Path(__file__).resolve().parents[1]
BASELINE_DIR = REPO / "tests" / "baselines"
BASELINE_DIR.mkdir(parents=True, exist_ok=True)

PUBLIC_ROOT = Path(os.environ.get("TAH_PUBLIC_ROOT", "/tmp/TaH-pub"))
DEVICE = os.environ.get("TAH_TEST_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
ACC_TOL = float(os.environ.get("TAH_ACC_TOL", "1e-5"))
WARMUP = int(os.environ.get("TAH_BENCH_WARMUP", "5"))
ITERS = int(os.environ.get("TAH_BENCH_ITERS", "30"))


def _to_cpu(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {k: _to_cpu(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = type(x)
        return t(_to_cpu(v) for v in x)
    return x


def _to_device(x: Any, device: str) -> Any:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = type(x)
        return t(_to_device(v, device) for v in x)
    return x


def baseline_path(name: str) -> Path:
    return BASELINE_DIR / f"{name}.pt"


def have_baseline(name: str) -> bool:
    return baseline_path(name).exists()


def capture(name: str, code: str, payload: dict | None = None) -> dict:
    """Run ``code`` (a Python source string) inside a subprocess scoped to
    ``PUBLIC_ROOT``, capture its returned dict, save as a baseline, and return.

    ``code`` MUST define ``def run(payload):`` returning a JSON/pickle-able
    dict. ``payload`` is forwarded as the single argument.
    """
    payload = payload or {}
    payload_pkl = pickle.dumps(payload)
    out_path = baseline_path(name)
    runner = (
        "import os, sys, pickle\n"
        f"sys.path.insert(0, {str(PUBLIC_ROOT)!r})\n"
        "import torch\n"
        f"{code}\n"
        "payload = pickle.loads(sys.stdin.buffer.read())\n"
        "out = run(payload)\n"
        "from tests._harness import _to_cpu  # noqa: E402  -- imported lazily\n"
    )
    # We can't import tests._harness inside the subprocess because the public
    # TaH path is first on sys.path. Inline a CPU mover instead.
    runner = (
        f"import os, sys, pickle\n"
        f"sys.path.insert(0, {str(PUBLIC_ROOT)!r})\n"
        f"import torch\n"
        f"def _to_cpu(x):\n"
        f"    if isinstance(x, torch.Tensor): return x.detach().cpu()\n"
        f"    if isinstance(x, dict): return {{k: _to_cpu(v) for k, v in x.items()}}\n"
        f"    if isinstance(x, (list, tuple)): return type(x)(_to_cpu(v) for v in x)\n"
        f"    return x\n"
        f"{code}\n"
        f"payload = pickle.loads(sys.stdin.buffer.read())\n"
        f"out = run(payload)\n"
        f"sys.stdout.buffer.write(pickle.dumps({{'args': payload, 'out': _to_cpu(out)}}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", runner],
        input=payload_pkl,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(PUBLIC_ROOT)},
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"baseline capture for {name!r} failed:\n"
            f"--- stderr ---\n{proc.stderr.decode(errors='replace')}\n"
            f"--- stdout ---\n{proc.stdout.decode(errors='replace')}"
        )
    snap = pickle.loads(proc.stdout)
    torch.save(snap, out_path)
    return snap


def load_baseline(name: str) -> dict:
    if not have_baseline(name):
        raise FileNotFoundError(
            f"no baseline at {baseline_path(name)} — run capture(...) first"
        )
    return torch.load(baseline_path(name), weights_only=False)


def assert_close(name: str, actual: Any, expected: Any, atol: float = ACC_TOL, rtol: float = 1e-4):
    """Recursively compare actual vs expected; raise if any leaf disagrees."""
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            raise AssertionError(f"{name}: expected Tensor, got {type(actual).__name__}")
        a = actual.detach().cpu()
        e = expected.detach().cpu()
        if a.shape != e.shape:
            raise AssertionError(f"{name}: shape mismatch {tuple(a.shape)} vs {tuple(e.shape)}")
        if a.dtype != e.dtype:
            # Best-effort dtype unify before compare; some bool/long mismatches are OK
            try:
                a = a.to(e.dtype)
            except Exception:
                raise AssertionError(f"{name}: dtype mismatch {a.dtype} vs {e.dtype}") from None
        if e.is_floating_point():
            diff = (a.float() - e.float()).abs()
            max_abs = float(diff.max().item()) if diff.numel() else 0.0
            tol = atol + rtol * float(e.float().abs().max().item() if e.numel() else 0.0)
            if max_abs > tol:
                raise AssertionError(
                    f"{name}: max abs diff {max_abs:.3e} > tol {tol:.3e}"
                )
        else:
            if not torch.equal(a, e):
                n_diff = int((a != e).sum().item())
                raise AssertionError(f"{name}: {n_diff} integer/bool elements differ")
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise AssertionError(f"{name}: expected dict, got {type(actual).__name__}")
        for k in expected:
            if k not in actual:
                raise AssertionError(f"{name}: missing key {k!r}")
            assert_close(f"{name}.{k}", actual[k], expected[k], atol, rtol)
        for k in actual:
            if k not in expected:
                raise AssertionError(f"{name}: unexpected key {k!r}")
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, type(expected)):
            raise AssertionError(f"{name}: container mismatch {type(actual).__name__} vs {type(expected).__name__}")
        if len(expected) != len(actual):
            raise AssertionError(f"{name}: length {len(actual)} vs {len(expected)}")
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_close(f"{name}[{i}]", a, e, atol, rtol)
        return
    if expected != actual:
        raise AssertionError(f"{name}: {actual!r} vs {expected!r}")


def bench(label: str, fn: Callable, ref_fn: Callable | None = None, *, warmup: int = WARMUP, iters: int = ITERS) -> dict:
    """Time ``fn`` (and optionally ``ref_fn``) and print a one-line summary.

    Returns a dict with ``ms`` (cleaned) and ``ref_ms`` (or None) so callers can
    assert non-regression. Caller is responsible for any required CUDA syncing
    inside ``fn`` / ``ref_fn``.
    """
    def _time(f):
        for _ in range(warmup):
            f()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            f()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3

    ms = _time(fn)
    if ref_fn is not None:
        ref_ms = _time(ref_fn)
        speedup = ref_ms / ms if ms > 0 else float("inf")
        print(f"  bench[{label}]: clean={ms:.3f}ms  ref={ref_ms:.3f}ms  speedup={speedup:.2f}x")
        return {"ms": ms, "ref_ms": ref_ms, "speedup": speedup}
    print(f"  bench[{label}]: clean={ms:.3f}ms")
    return {"ms": ms, "ref_ms": None, "speedup": None}

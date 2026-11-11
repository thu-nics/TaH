"""The bounded prefill variant requires JIT even with prebuilt FlashInfer ops."""
import sys
from types import ModuleType

import pytest
import torch

import tah.engine as api


@pytest.mark.parametrize('prebuilt', [False, True])
def test_prefill_requires_compiler_even_with_prebuilt_ops(monkeypatch, prebuilt):
    fake_flashinfer = ModuleType('flashinfer')
    fake_jit = ModuleType('flashinfer.jit')
    fake_jit.has_prebuilt_ops = prebuilt
    fake_flashinfer.jit = fake_jit
    monkeypatch.setitem(sys.modules, 'flashinfer', fake_flashinfer)
    monkeypatch.setitem(sys.modules, 'flashinfer.jit', fake_jit)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda: (8, 6))
    monkeypatch.setattr(api, '_find_nvcc', lambda: None)
    with pytest.raises(RuntimeError, match='TaH prefill uses runtime JIT compilation'):
        api.check_environment()

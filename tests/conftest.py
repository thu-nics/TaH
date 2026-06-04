"""Shared fixtures for tah-release component tests.

We intentionally use *tiny* synthetic shapes so each component test runs in
under a second on CPU, while still exercising the >1 batch / >1 layer / >1
iteration code paths.
"""
from __future__ import annotations

import random

import pytest
import torch


SEED = 4242


@pytest.fixture(autouse=True)
def _deterministic():
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    yield


@pytest.fixture
def device() -> str:
    import os
    return os.environ.get("TAH_TEST_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def shapes() -> dict:
    """Tiny but non-trivial shapes for component tests."""
    return {
        "B": 2,
        "T": 5,
        "V": 64,
        "H": 32,
        "L": 4,           # number of transformer layers exposed for hidden states
        "TOPK": 8,
        "MAX_ITER": 2,
    }

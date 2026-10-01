from __future__ import annotations

import pytest
import torch

from models.reward.pipeline.common import resolve_device


def test_resolve_device_cpu():
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_auto_prefers_mps_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable on this host")
    assert resolve_device("auto") == torch.device("mps")


def test_resolve_device_rejects_unavailable_mps(monkeypatch):
    if not hasattr(torch.backends, "mps"):
        pytest.skip("PyTorch has no MPS backend")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="MPS was requested"):
        resolve_device("mps")

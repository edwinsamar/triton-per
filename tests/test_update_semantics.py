"""Priority-update semantics: block sums stay consistent, duplicates resolve
to a single consistent value, circular overwrite keeps sums correct."""

import pytest
import torch

from triton_per.buffer import PERBuffer, BLOCK_P

CUDA = torch.cuda.is_available()
BACKENDS = [("torch", "cpu")] + ([("triton", "cuda")] if CUDA else [])


def consistent(buf):
    ref = buf.priorities.view(buf.num_blocks, BLOCK_P).sum(dim=1)
    torch.testing.assert_close(buf.block_sums, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("backend,device", BACKENDS)
def test_block_sums_after_updates(backend, device):
    buf = PERBuffer(BLOCK_P * 3, {"x": 2}, device=device, backend=backend)
    buf.add({"x": torch.randn(2000, 2).to(device)})
    consistent(buf)
    idx = torch.randint(0, 2000, (512,), device=device)
    buf.update_priorities(idx, torch.rand(512, device=device) * 5)
    consistent(buf)


@pytest.mark.parametrize("backend,device", BACKENDS)
def test_duplicate_indices(backend, device):
    buf = PERBuffer(BLOCK_P, {"x": 2}, device=device, backend=backend)
    buf.add({"x": torch.randn(100, 2).to(device)})
    idx = torch.zeros(64, dtype=torch.int64, device=device)  # all the same
    buf.update_priorities(idx, torch.linspace(0.1, 5, 64, device=device))
    consistent(buf)
    # final value must be one of the written values
    p0 = buf.priorities[0].item()
    candidates = ((torch.linspace(0.1, 5, 64).abs() + buf.eps) ** buf.alpha)
    assert (candidates - p0).abs().min() < 1e-5


@pytest.mark.parametrize("backend,device", BACKENDS)
def test_circular_overwrite(backend, device):
    buf = PERBuffer(256, {"x": 1}, device=device, backend=backend)
    for _ in range(5):
        buf.add({"x": torch.randn(100, 1).to(device)})
    assert buf.size == 256
    consistent(buf)


@pytest.mark.parametrize("backend,device", BACKENDS)
def test_sample_returns_stored_rows(backend, device):
    buf = PERBuffer(512, {"a": 3, "b": 1}, device=device, backend=backend)
    a = torch.randn(200, 3).to(device)
    b = torch.arange(200, dtype=torch.float32).reshape(-1, 1).to(device)
    buf.add({"a": a, "b": b})
    views, idx, _ = buf.sample(64)
    torch.testing.assert_close(views["a"], a[idx])
    torch.testing.assert_close(views["b"], b[idx])

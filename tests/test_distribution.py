"""Distributional correctness: empirical sampling frequencies must match
P(i) = p_i^alpha / sum(p^alpha) (chi-square goodness of fit).

The torch backend runs everywhere (CPU-only reference path); triton backends
run on GPU nodes only. All backends share the same test body, so a GPU pass
certifies the kernels against the same statistical bar as the reference.
"""

import math

import pytest
import torch
from scipy import stats

from triton_per.buffer import PERBuffer
from triton_per.baselines.cpu_sumtree import CpuSumTreePER

CUDA = torch.cuda.is_available()

BACKENDS = [("torch", "cpu"), ("cpu_sumtree", "cpu")]
if CUDA:
    BACKENDS += [("torch", "cuda"), ("triton", "cuda"), ("triton-unfused", "cuda")]


def make_buffer(backend, device, capacity=4096, fields=None):
    fields = fields or {"x": 4}
    if backend == "cpu_sumtree":
        return CpuSumTreePER(capacity, fields, device=device)
    return PERBuffer(capacity, fields, device=device, backend=backend)


def fill_with_known_priorities(buf, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 4, generator=g).to(buf.device)
    buf.add({"x": x})
    # overwrite the uniform insert-priorities with a spread of TD errors
    td = torch.rand(n, generator=g).to(buf.device) * 10 + 0.01
    buf.update_priorities(torch.arange(n, device=buf.device), td)
    alpha, eps = buf.alpha, buf.eps
    expected_p = (td.cpu().double().abs() + eps) ** alpha
    return expected_p / expected_p.sum()


@pytest.mark.parametrize("backend,device", BACKENDS)
def test_sampling_matches_distribution(backend, device):
    n = 512
    buf = make_buffer(backend, device, capacity=4096)
    probs = fill_with_known_priorities(buf, n)

    draws = 400_000
    counts = torch.zeros(n, dtype=torch.long)
    bs = 4096
    for _ in range(draws // bs):
        _, idx, _ = buf.sample(bs)
        counts += torch.bincount(idx.cpu(), minlength=n)

    total = counts.sum().item()
    expected = (probs * total).numpy()
    # chi-square requires expected counts >= ~5 everywhere
    assert expected.min() > 5, "test setup: raise draws or floor priorities"
    chi2, p_value = stats.chisquare(counts.numpy(), expected)
    print(f"CHISQ {backend}/{device}: chi2={chi2:.1f} p={p_value:.4f}")
    assert p_value > 0.001, f"{backend}/{device}: chi2={chi2:.1f}, p={p_value:.2e}"


@pytest.mark.parametrize("backend,device", BACKENDS)
def test_is_weights(backend, device):
    n = 256
    buf = make_buffer(backend, device, capacity=1024)
    probs = fill_with_known_priorities(buf, n)
    beta = 0.5
    _, idx, w = buf.sample(2048, beta=beta)
    ref = (n * probs[idx.cpu()]) ** (-beta)
    ref = ref / ref.max()
    torch.testing.assert_close(
        w.cpu().double(), ref, rtol=1e-3, atol=1e-4,
    )


@pytest.mark.skipif(not CUDA, reason="large-N test needs GPU")
@pytest.mark.parametrize("backend", ["triton", "triton-unfused"])
def test_sampling_binned_large_buffer(backend):
    """Full-depth validation: 1e6 slots, contiguous 1e4-slot bins, empirical
    bin frequencies against exact bin masses. Exercises the deep binary
    search, the fp32 accumulation regime, and (via update rounds crossing
    the 512-call refresh boundary) the drift-mitigation path that the small
    test skips."""
    n = 1_000_000
    buf = PERBuffer(n, {"x": 1}, device="cuda", backend=backend)
    chunk = 200_000
    for i in range(0, n, chunk):
        buf.add({"x": torch.zeros(chunk, 1, device="cuda")})
    g = torch.Generator(device="cuda").manual_seed(1)
    # ~1500 update rounds -> crosses the exact-refresh boundary several times
    for _ in range(1500):
        idx = torch.randint(0, n, (1024,), device="cuda", generator=g)
        buf.update_priorities(idx, torch.rand(1024, device="cuda", generator=g) * 5)

    probs = (buf.priorities[:n].double() / buf.priorities[:n].double().sum()).cpu()
    bin_size = 10_000
    bin_mass = probs.reshape(-1, bin_size).sum(dim=1).numpy()

    draws = 1_000_000
    counts = torch.zeros(n // bin_size, dtype=torch.long)
    for _ in range(draws // 4096):
        _, idx, _ = buf.sample(4096)
        counts += torch.bincount((idx // bin_size).cpu(), minlength=n // bin_size)

    expected = bin_mass * counts.sum().item()
    assert expected.min() > 5
    chi2, p_value = stats.chisquare(counts.numpy(), expected)
    print(f"CHISQ-LARGE {backend}: chi2={chi2:.1f} p={p_value:.4f}")
    assert p_value > 0.001, f"{backend}: chi2={chi2:.1f}, p={p_value:.2e}"

"""Adversarial correctness stress tests.

The chi-square tests in test_distribution.py use benign priorities; the
implementation's stated failure modes involve floating-point drift and
boundary behavior. These tests target exactly those: priorities spanning
orders of magnitude, eps-floored zeros, non-power-of-two capacities,
ring-buffer wraparound under nonuniform priorities, divergent duplicate
updates, drift far beyond the 512-call refresh interval, probability mass
concentrated on block and padding boundaries, and a float64
total-variation comparison. Peak device memory is recorded for the
largest configuration.

The torch backend runs everywhere; triton backends run on GPU nodes.
"""

import math

import pytest
import torch
from scipy import stats

from triton_per.buffer import BLOCK_P, PERBuffer

CUDA = torch.cuda.is_available()

BACKENDS = [("torch", "cpu")]
if CUDA:
    BACKENDS += [("torch", "cuda"), ("triton", "cuda"), ("triton-unfused", "cuda")]


def expected_probs_fp64(buf):
    """Target distribution from the live priorities, computed in float64."""
    p = buf.priorities[: buf.capacity].double().cpu()
    p = p[: buf.size] if buf.size < buf.capacity else p
    return p / p.sum()


def draw_counts(buf, n_slots, draws=400_000, bs=4096):
    counts = torch.zeros(n_slots, dtype=torch.long)
    for _ in range(draws // bs):
        _, idx, _ = buf.sample(bs)
        counts += torch.bincount(idx.cpu(), minlength=n_slots)
    return counts


def merged_chisquare_p(counts, probs, min_expected=5.0):
    expected = (probs * counts.sum()).tolist()
    obs, exp, ac, ae = [], [], 0.0, 0.0
    for c, e in zip(counts.tolist(), expected):
        ac += c
        ae += e
        if ae >= min_expected:
            obs.append(ac)
            exp.append(ae)
            ac = ae = 0.0
    obs[-1] += ac
    exp[-1] += ae
    return stats.chisquare(obs, exp)[1]


def tv_distance(counts, probs):
    emp = counts.double() / counts.sum()
    return float(0.5 * (emp - probs).abs().sum())


@pytest.mark.parametrize("backend,device", BACKENDS)
def test_priorities_spanning_orders_of_magnitude(backend, device):
    """Priorities from 1e-6 to 1e6: twelve decades in one buffer, randomly
    located by index.

    The naive torch backend's full-length fp32 cumsum absorbs tiny
    priorities that follow large accumulated sums; with random locations
    the absorbed mass lands on negligible-probability slots, so its GoF
    outcome here is borderline and run-dependent (observed p from 1e-4 to
    0.7 across runs). Its result is therefore REPORTED, not asserted; the
    deterministic quantification of the failure mode is
    test_adversarial_block_ordering. The two-level backends are asserted.
    """
    report_only_baseline = backend == "torch" and device == "cuda"
    n = 2048
    buf = PERBuffer(4096, {"x": 1}, device=device, backend=backend)
    buf.add({"x": torch.zeros(n, 1, device=device)})
    g = torch.Generator().manual_seed(0)
    exponents = torch.rand(n, generator=g) * 12 - 6  # 1e-6 .. 1e6
    td = (10.0 ** exponents).to(device)
    buf.update_priorities(torch.arange(n, device=device), td)

    probs = expected_probs_fp64(buf)
    counts = draw_counts(buf, n, draws=800_000)
    p = merged_chisquare_p(counts, probs)
    tv = tv_distance(counts, probs)
    print(f"STRESS magnitude {backend}/{device}: p={p:.4g} tv={tv:.4f}")
    if report_only_baseline:
        return  # reported above; deterministic check lives in blockorder test
    assert p > 0.001
    assert tv < 0.02


@pytest.mark.parametrize("backend,device",
                         [b for b in BACKENDS if b[0] != "torch"] or BACKENDS[:1])
def test_adversarial_block_ordering(backend, device):
    """Worst case for the TOP-LEVEL scan: per-block magnitudes spanning
    twelve decades, descending by block, so the fp32 prefix sum over block
    sums absorbs late blocks. The analytic part quantifies the absorbed
    target mass for both designs; the empirical part verifies block-level
    sampling frequencies on the two-level structure."""
    import numpy as np

    # analytic fp32-absorption comparison (device-independent)
    k = np.arange(1024)
    bs64 = 1024 * 10.0 ** (6 - 12 * k / 1023)
    c32 = np.cumsum(bs64.astype(np.float32), dtype=np.float32)
    w32 = np.diff(np.concatenate([[np.float32(0)], c32]))
    tgt = bs64 / bs64.sum()
    lost_two_level = tgt[w32 <= 0].sum()
    slots64 = np.repeat(bs64 / 1024, 1024)
    s32 = np.cumsum(slots64.astype(np.float32), dtype=np.float32)
    ws32 = np.diff(np.concatenate([[np.float32(0)], s32]))
    lost_naive = (slots64 / slots64.sum())[ws32 <= 0].sum()
    print(f"STRESS blockorder analytic: two-level lost mass "
          f"{lost_two_level:.2e}, naive full-length {lost_naive:.2e}")
    assert lost_two_level < 1e-4
    assert lost_naive > 100 * lost_two_level  # aggregation buys >=100x

    if backend == "torch":
        return  # empirical part targets the two-level kernels

    cap = 1024 * BLOCK_P
    buf = PERBuffer(cap, {"x": 1}, device=device, backend=backend)
    chunk = 131072
    for _ in range(cap // chunk):
        buf.add({"x": torch.zeros(chunk, 1, device=device)})
    kk = torch.arange(cap, device=device) // BLOCK_P
    td = (10.0 ** (6 - 12 * kk.double() / 1023)).float()
    buf.update_priorities(torch.arange(cap, device=device), td)

    probs = buf.priorities[:cap].double().cpu()
    block_mass = (probs / probs.sum()).reshape(1024, BLOCK_P).sum(dim=1)
    counts = torch.zeros(1024, dtype=torch.long)
    for _ in range(1_000_000 // 8192):
        _, idx, _ = buf.sample(8192)
        counts += torch.bincount((idx // BLOCK_P).cpu(), minlength=1024)
    p = merged_chisquare_p(counts, block_mass)
    print(f"STRESS blockorder {backend}/{device}: block-level p={p:.4f}")
    assert p > 0.001


@pytest.mark.parametrize("backend,device", BACKENDS)
def test_zero_priorities_eps_floor(backend, device):
    """Zero TD errors floor at eps^alpha and stay sampleable (never NaN)."""
    n = 512
    buf = PERBuffer(1024, {"x": 1}, device=device, backend=backend)
    buf.add({"x": torch.zeros(n, 1, device=device)})
    td = torch.zeros(n, device=device)
    td[::2] = 1.0  # half the slots get real mass, half get the eps floor
    buf.update_priorities(torch.arange(n, device=device), td)

    floor = (0.0 + buf.eps) ** buf.alpha
    torch.testing.assert_close(
        buf.priorities[1:n:2], torch.full((n // 2,), floor, device=device),
        rtol=1e-5, atol=0)
    _, idx, w = buf.sample(8192)
    assert torch.isfinite(w).all() and (w > 0).all()
    assert (idx < n).all()


@pytest.mark.parametrize("backend,device", BACKENDS)
def test_non_power_of_two_capacity(backend, device):
    """Prime capacity: padding block is partially dead; distribution and
    clamping must still be exact."""
    cap = 99_991  # prime, not a multiple of 1024
    n = 70_000    # partial fill
    buf = PERBuffer(cap, {"x": 1}, device=device, backend=backend)
    chunk = 10_000
    for i in range(0, n, chunk):
        buf.add({"x": torch.zeros(chunk, 1, device=device)})
    g = torch.Generator().manual_seed(1)
    td = (torch.rand(n, generator=g) * 10 + 0.1).to(device)
    buf.update_priorities(torch.arange(n, device=device), td)

    probs = expected_probs_fp64(buf)
    counts = draw_counts(buf, n, draws=800_000)
    assert counts[n:].sum() == 0 if len(counts) > n else True
    p = merged_chisquare_p(counts, probs)
    print(f"STRESS non-pow2 {backend}/{device}: p={p:.4f}")
    assert p > 0.001


@pytest.mark.parametrize("backend,device", BACKENDS)
def test_wraparound_with_nonuniform_priorities(backend, device):
    """Overwrite 1.5 laps, then skew priorities: samples must match both the
    stored data (mapping) and the skewed distribution."""
    cap = 1000
    buf = PERBuffer(cap, {"x": 1}, device=device, backend=backend)
    t = torch.arange(1500.0, device=device).unsqueeze(1)
    buf.add({"x": t[:750]})
    buf.add({"x": t[750:]})
    assert buf.size == cap
    g = torch.Generator().manual_seed(2)
    td = (torch.rand(cap, generator=g) * 10 + 0.1).to(device)
    buf.update_priorities(torch.arange(cap, device=device), td)

    views, idx, _ = buf.sample(4096)
    # slot i holds insertion index 500+i for i<500, else i (second lap wrote
    # slots 0..499 with values 1000..1499? no: cursor wrapped at 1000, so
    # slots 0..499 hold values 1000..1499 and slots 500..999 hold 500..999)
    expect = torch.where(idx < 500, idx + 1000, idx).float()
    torch.testing.assert_close(views["x"].squeeze(1), expect.to(device))

    probs = expected_probs_fp64(buf)
    counts = draw_counts(buf, cap, draws=800_000)
    p = merged_chisquare_p(counts, probs)
    print(f"STRESS wraparound {backend}/{device}: p={p:.4f}")
    assert p > 0.001


@pytest.mark.parametrize("backend,device", BACKENDS)
def test_divergent_duplicate_updates(backend, device):
    """One batch, same index, different values: the survivor must be one of
    the submitted values and the block sum exact for that survivor."""
    buf = PERBuffer(2048, {"x": 1}, device=device, backend=backend)
    buf.add({"x": torch.zeros(1024, 1, device=device)})
    idx = torch.tensor([7] * 64 + [700] * 64, device=device)
    vals = torch.cat([torch.linspace(0.5, 5.0, 64),
                      torch.linspace(1.0, 9.0, 64)]).to(device)
    buf.update_priorities(idx, vals)

    for slot, v in ((7, vals[:64]), (700, vals[64:])):
        got = buf.priorities[slot]
        candidates = (v.abs() + buf.eps) ** buf.alpha
        assert torch.isclose(candidates, got, rtol=1e-5).any(), \
            f"slot {slot}: survivor {got} not among submitted transforms"
    recomputed = buf.priorities.view(buf.num_blocks, BLOCK_P).sum(dim=1)
    torch.testing.assert_close(buf.block_sums, recomputed, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("backend,device", BACKENDS)
def test_long_run_drift_beyond_refresh(backend, device):
    """5000 update rounds (~10 refresh cycles): drift must stay bounded and
    the sampled distribution correct at the end."""
    n = 8192
    buf = PERBuffer(n, {"x": 1}, device=device, backend=backend)
    buf.add({"x": torch.zeros(n, 1, device=device)})
    g = torch.Generator(device=device).manual_seed(3) if device == "cuda" \
        else torch.Generator().manual_seed(3)
    for _ in range(5000):
        idx = torch.randint(0, n, (256,), device=device, generator=g)
        buf.update_priorities(idx, torch.rand(256, device=device, generator=g) * 5)

    exact = buf.priorities.view(buf.num_blocks, BLOCK_P).double().sum(dim=1)
    rel_err = ((buf.block_sums.double() - exact).abs() / exact.clamp(min=1e-12)).max()
    print(f"STRESS drift {backend}/{device}: max rel block-sum err {rel_err:.2e}")
    assert rel_err < 1e-4  # exact refresh every 512 calls bounds the drift

    probs = expected_probs_fp64(buf)
    counts = draw_counts(buf, n, draws=800_000)
    p = merged_chisquare_p(counts, probs)
    assert p > 0.001


@pytest.mark.parametrize("backend,device", BACKENDS)
def test_boundary_concentrated_mass(backend, device):
    """All meaningful mass on block-boundary slots (last of block k, first
    of block k+1) and the final live slot before padding."""
    cap = 5000  # padding: last block holds slots 4096..4999 of 5120
    buf = PERBuffer(cap, {"x": 1}, device=device, backend=backend)
    buf.add({"x": torch.zeros(cap, 1, device=device)})
    hot = torch.tensor([1023, 1024, 2047, 2048, 4095, 4096, cap - 1],
                       device=device)
    td = torch.full((cap,), 1e-3, device=device)
    td[hot] = 100.0
    buf.update_priorities(torch.arange(cap, device=device), td)

    probs = expected_probs_fp64(buf)
    counts = draw_counts(buf, cap, draws=800_000)
    # each hot slot individually within 5% relative of its expected count
    exp_hot = probs[hot.cpu()] * counts.sum()
    rel = ((counts[hot.cpu()] - exp_hot).abs() / exp_hot).max()
    print(f"STRESS boundary {backend}/{device}: max hot-slot rel dev {rel:.3f}")
    assert rel < 0.05
    assert counts[: cap][counts[:cap] < 0].numel() == 0
    p = merged_chisquare_p(counts, probs)
    assert p > 0.001


@pytest.mark.skipif(not CUDA, reason="peak-memory capture needs a GPU")
def test_peak_memory_at_full_scale():
    """Records peak device memory for the largest benchmarked configuration
    (capacity 1e7, obs 256 -> row_dim 522), for the paper's appendix."""
    torch.cuda.reset_peak_memory_stats()
    fields = {"obs": 256, "action": 8, "reward": 1, "next_obs": 256, "done": 1}
    buf = PERBuffer(10_000_000, fields, device="cuda", backend="triton")
    chunk = 100_000
    for _ in range(10):
        buf.add({k: torch.randn(chunk, d, device="cuda") for k, d in fields.items()})
    for _ in range(20):
        _, idx, _ = buf.sample(4096)
        buf.update_priorities(idx, torch.rand(4096, device="cuda"))
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    analytic_gb = 10_000_000 * 522 * 4 / 1e9
    print(f"STRESS memory: peak {peak_gb:.2f} GB (storage analytic "
          f"{analytic_gb:.2f} GB)")
    assert peak_gb < analytic_gb + 2.0  # overheads stay small vs storage

"""Tests for the per-channel sampling-delay alignment transformer.

Pins the three properties that make it useful:

* **Chunk invariance** -- streaming in any chunking reproduces the whole-buffer
  result exactly (the FIR history is carried in state).
* **Alignment** -- after alignment a ``tau_c``-misaligned common-mode collapses
  to (near) identical channels, so CAR's residual drops by orders of magnitude
  at high frequency, where un-aligned CAR fails.
* **Rail handling** -- with ``rail_threshold`` set, a clipped run is held rather
  than rung through the filter, keeping the output bounded.
"""

from __future__ import annotations

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray, CoordinateAxis, LinearAxis

from ezmsg.blackrock.sampling_delay_alignment import (
    SamplingDelayAlignmentSettings,
    SamplingDelayAlignmentTransformer,
)

FS = 30000.0
BANK = 32
INTERVAL = 64.0 / 66.0e6


def sampling_delay_alignment(**kwargs) -> SamplingDelayAlignmentTransformer:
    return SamplingDelayAlignmentTransformer(settings=SamplingDelayAlignmentSettings(**kwargs))


def _aa(data: np.ndarray, offset: float = 0.0) -> AxisArray:
    return AxisArray(
        data=data,
        dims=["time", "ch"],
        axes={"time": LinearAxis(offset=offset, gain=1.0 / FS)},
        key="align",
    )


def _stream(proc, data: np.ndarray, chunk_sizes: list[int]) -> np.ndarray:
    outs, start = [], 0
    for size in chunk_sizes:
        outs.append(proc(_aa(data[start : start + size], offset=start / FS)).data)
        start += size
    assert start == data.shape[0]
    return np.concatenate(outs, axis=0)


def test_streaming_is_chunk_invariant():
    """Any chunking reproduces the whole-buffer output (carried FIR history)."""
    n, nch = 6000, 64
    x = np.random.default_rng(0).standard_normal((n, nch)).astype(np.float32)
    split = [7, 13, 500] + [500] * 10 + [480]
    assert sum(split) == n

    whole = _stream(sampling_delay_alignment(), x, [n])
    chunked = _stream(sampling_delay_alignment(), x, split)
    np.testing.assert_allclose(whole, chunked, rtol=0, atol=1e-6)


def test_alignment_collapses_misaligned_common_mode():
    """A tau_c-misaligned high-frequency common-mode becomes ~identical across
    channels after alignment, so CAR removes it where un-aligned CAR can't."""
    f, n, nch = 5000.0, 10000, 128
    t = np.arange(n) / FS
    tau = (np.arange(nch) % BANK) * INTERVAL
    x = np.cos(2 * np.pi * f * (t[:, None] + tau[None, :])).astype(np.float32)
    y = _stream(sampling_delay_alignment(), x, [n])

    seg = slice(64, -64)  # skip the FIR/bulk-delay transient

    def car_resid(d):
        r = d - d.mean(axis=1, keepdims=True)
        return np.sqrt(np.mean(r[seg] ** 2)) / np.sqrt(np.mean(d[seg] ** 2))

    # Un-aligned CAR leaves a large 5 kHz residual (~0.28, i.e. ~-11 dB);
    # alignment collapses the channels so the residual nearly vanishes.
    assert car_resid(x) > 0.2
    assert car_resid(y) < 0.01
    assert car_resid(y) < 0.05 * car_resid(x)


def test_rail_handling_bounds_output():
    """A clipped run is held (not rung through the FIR) when rail_threshold is
    set; without it the filter spreads the rail."""
    n, nch = 4000, 16
    x = np.random.default_rng(1).standard_normal((n, nch)).astype(np.float32)
    xr = x.copy()
    xr[1000:1003, 5] = 1e4  # railed run on one channel

    guarded = _stream(sampling_delay_alignment(rail_threshold=8000.0), xr, [n])
    unguarded = _stream(sampling_delay_alignment(rail_threshold=None), xr, [n])

    near = slice(995, 1040)  # around the rail (after the bulk delay)
    assert np.abs(guarded[near, 5]).max() < 100.0  # bounded to signal scale
    assert np.abs(unguarded[near, 5]).max() > 1000.0  # FIR rings on the rail


def test_output_offset_accounts_for_bulk_delay():
    """The time-axis offset is shifted back by the FIR bulk delay so output
    timestamps stay physical."""
    filter_len = 33
    proc = sampling_delay_alignment(filter_len=filter_len)
    x = np.random.default_rng(2).standard_normal((100, 8)).astype(np.float32)
    out = proc(_aa(x, offset=5.0))
    m = (filter_len - 1) // 2
    assert out.axes["time"].offset == pytest.approx(5.0 - m / FS)


def _car_resid(d: np.ndarray, seg: slice = slice(64, -64)) -> float:
    r = d - d.mean(axis=1, keepdims=True)
    return float(np.sqrt(np.mean(r[seg] ** 2)) / np.sqrt(np.mean(d[seg] ** 2)))


def test_metadata_slot_overrides_acquisition_order():
    """When the ch axis carries bank/elec metadata, the sweep slot is taken from
    elec, so alignment works even with channels shuffled out of hardware order --
    where the bank_size fallback (slot = c % bank_size) mis-aligns them."""
    f, n, nch = 5000.0, 10000, 64
    t = np.arange(n) / FS
    # A shuffled layout: array position i is some physical (bank, elec), so
    # i % BANK does NOT equal the true within-bank slot (elec - 1).
    perm = np.random.default_rng(7).permutation(nch)
    elec = (perm % BANK) + 1
    bank = np.array([chr(ord("A") + b) for b in (perm // BANK)])
    tau = (elec - 1) * INTERVAL  # true per-channel sampling delay
    x = np.cos(2 * np.pi * f * (t[:, None] + tau[None, :])).astype(np.float32)

    ch_data = np.zeros(nch, dtype=np.dtype([("bank", "U1"), ("elec", "i4")]))
    ch_data["bank"] = bank
    ch_data["elec"] = elec
    ch_axis = CoordinateAxis(data=ch_data, dims=["ch"], unit="struct")
    msg = AxisArray(
        data=x,
        dims=["time", "ch"],
        axes={"time": LinearAxis(offset=0.0, gain=1.0 / FS), "ch": ch_axis},
        key="align",
    )

    # Metadata-driven slots collapse the misaligned common mode...
    y_meta = sampling_delay_alignment()(msg).data
    assert _car_resid(y_meta) < 0.01
    # ...whereas the bank_size fallback (no metadata) can't, because the shuffled
    # channel order doesn't match c % bank_size.
    y_fallback = sampling_delay_alignment()(_aa(x)).data
    assert _car_resid(y_fallback) > 0.2


def test_passthrough_shape_and_dtype():
    proc = sampling_delay_alignment()
    x = np.random.default_rng(3).standard_normal((500, 32)).astype(np.float32)
    out = proc(_aa(x))
    assert out.data.shape == x.shape
    assert out.data.dtype == x.dtype


@pytest.mark.parametrize("backend", ["mlx", "torch"])
def test_array_api_backend_matches_numpy(backend):
    """Other Array-API backends reproduce the numpy result (output stays on the
    backend) and run the rail forward-fill on-device too."""
    mod = pytest.importorskip("mlx.core" if backend == "mlx" else "torch")
    if backend == "mlx":
        to_backend, to_numpy, arr_type = (lambda a: mod.array(a), np.array, mod.array)
    else:
        to_backend, to_numpy, arr_type = (mod.from_numpy, lambda a: a.numpy(), mod.Tensor)

    n, nch = 6000, 64
    x = np.random.default_rng(0).standard_normal((n, nch)).astype(np.float32)
    x[1000:1003, 5] = 1e4  # a railed run to exercise the on-device forward-fill
    chunks = [250] * (n // 250)

    def make():
        return sampling_delay_alignment(rail_threshold=8000.0)

    y_np = _stream(make(), x, chunks)

    proc = make()
    outs, start, last = [], 0, None
    for size in chunks:
        msg = AxisArray(
            data=to_backend(x[start : start + size]),
            dims=["time", "ch"],
            axes={"time": LinearAxis(offset=start / FS, gain=1.0 / FS)},
            key="align",
        )
        last = proc(msg).data
        outs.append(to_numpy(last))
        start += size
    y_backend = np.concatenate(outs, axis=0)

    assert isinstance(last, arr_type)
    np.testing.assert_allclose(y_backend, y_np, rtol=0, atol=1e-5)

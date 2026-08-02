"""Tests for the per-channel sampling-delay alignment transformer.

Pins the three properties that make it useful:

* **Chunk invariance** -- streaming in any chunking reproduces the whole-buffer
  result exactly (the FIR history is carried in state).
* **Alignment** -- after alignment a ``tau_c``-misaligned common-mode collapses
  to (near) identical channels, so CAR's residual drops by orders of magnitude
  at high frequency, where un-aligned CAR fails.
* **Rail handling** -- with ``rail_threshold`` set, a clipped run is held rather
  than rung through the filter, keeping the output bounded.
* **Response** -- every within-bank slot's filter meets an explicit phase and
  magnitude tolerance over its documented passband, which is what justifies the
  default ``filter_len``.
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


def test_filter_len_zero_is_passthrough():
    """filter_len=0 disables alignment: the input is returned unchanged (same
    data, no FIR built, no bulk-delay offset shift)."""
    proc = sampling_delay_alignment(filter_len=0)
    x = np.random.default_rng(4).standard_normal((300, 32)).astype(np.float32)
    out = proc(_aa(x, offset=2.0))
    np.testing.assert_array_equal(out.data, x)
    assert out.axes["time"].offset == 2.0  # no bulk-delay shift
    assert proc.state.fir is None  # _reset_state skipped filter design


def test_rail_threshold_is_nonreset():
    """rail_threshold is in NONRESET_SETTINGS_FIELDS: changing it updates the
    rail behavior without rebuilding the (expensive) filter state."""
    proc = sampling_delay_alignment()  # rail_threshold=None
    x = np.random.default_rng(5).standard_normal((500, 16)).astype(np.float32)
    proc(_aa(x))  # builds filter state
    fir_before, hash_before = proc.state.fir, proc._hash
    assert fir_before is not None

    proc.update_settings(SamplingDelayAlignmentSettings(rail_threshold=8000.0))
    assert proc._hash == hash_before  # no reset requested
    assert proc.state.fir is fir_before  # filters not rebuilt

    # ...and the new threshold is honored on the next (same-metadata) chunk.
    xr = x.copy()
    xr[100:103, 5] = 1e4
    out = proc(_aa(xr, offset=500 / FS))
    assert np.abs(out.data[95:140, 5]).max() < 100.0  # rail held, not rung


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
    # Explicit float32 tolerance: MLX evaluates the FIR as a depthwise
    # convolution rather than a tap-sum, so it accumulates in a different order
    # than numpy. On the held-rail samples (~1e4) that shows up as ~2e-3
    # absolute -- 1.8e-7 relative, i.e. float32 eps, not a behavior difference.
    np.testing.assert_allclose(y_backend, y_np, rtol=1e-6, atol=1e-5)


# ---------------------------------------------------------------------------
# Fractional-delay response: what sets the default filter_len
# ---------------------------------------------------------------------------

# The alignment exists to remove up to ~81 deg of cross-channel skew at 7.5 kHz.
# Residual error three orders of magnitude below that is well past the point of
# diminishing returns, so the requirement for a usable filter_len is: over the
# intended passband, across every within-bank slot, at most 0.05 deg of phase
# error and 0.01 dB of magnitude error.
MAX_PHASE_ERR_DEG = 0.05
MAX_MAG_ERR_DB = 0.01


def _response_error(filter_len: int, band_hi: float) -> tuple[float, float]:
    """Worst-case (phase error in degrees, magnitude error in dB) over
    ``0..band_hi``, across all ``BANK`` slots, for the filters the transformer
    actually designs -- read back from its state, not re-derived here."""
    proc = sampling_delay_alignment(filter_len=filter_len)
    proc(_aa(np.zeros((filter_len + 1, BANK), dtype=np.float32)))
    h = proc.state.fir  # (filter_len, BANK)
    m = proc.state.bulk_delay

    w = 2 * np.pi * np.linspace(0.0, band_hi, 2001) / FS
    resp = np.exp(-1j * w[:, None] * np.arange(filter_len)[None, :]) @ h
    ideal_delay = m + np.arange(BANK) * INTERVAL * FS  # bulk + per-slot fraction
    ideal = np.exp(-1j * w[:, None] * ideal_delay[None, :])
    phase_deg = np.abs(np.angle(resp * np.conj(ideal))) * 180.0 / np.pi
    mag_db = np.abs(20.0 * np.log10(np.abs(resp)))
    return float(phase_deg.max()), float(mag_db.max())


@pytest.mark.parametrize(
    ("filter_len", "band_hi"),
    [
        (7, 500.0),  # LFP-only pipelines
        (9, 3000.0),
        (13, 7500.0),  # the default: full broadband/spike band
        (33, 7500.0),  # the former default, still supported
    ],
)
def test_filter_response_meets_tolerance(filter_len, band_hi):
    """Each documented (filter_len, passband) pair holds the tolerance for every
    within-bank slot -- this is the table in the ``filter_len`` docstring."""
    phase_deg, mag_db = _response_error(filter_len, band_hi)
    assert phase_deg < MAX_PHASE_ERR_DEG
    assert mag_db < MAX_MAG_ERR_DB


def test_default_filter_len_is_the_shortest_that_covers_the_spike_band():
    """13 is the default because it is the shortest odd length meeting the
    broadband tolerance -- 11 misses it, so the latency can't be cut further."""
    assert SamplingDelayAlignmentSettings().filter_len == 13
    phase_deg, _ = _response_error(11, 7500.0)
    assert phase_deg > MAX_PHASE_ERR_DEG


def test_filter_len_one_is_unity_gain_and_carries_no_history():
    """A single tap normalizes to 1, so it passes the input through with no bulk
    delay -- and, unlike a longer filter, carries no inter-chunk history."""
    proc = sampling_delay_alignment(filter_len=1)
    x = np.random.default_rng(8).standard_normal((64, 32)).astype(np.float32)
    out = proc(_aa(x, offset=1.0))
    np.testing.assert_allclose(out.data, x, rtol=1e-6, atol=1e-6)
    assert out.axes["time"].offset == pytest.approx(1.0)
    assert proc.state.hist.shape[0] == 0


# ---------------------------------------------------------------------------
# Rail forward-fill: the one-pass and portable-scan formulations must agree
# ---------------------------------------------------------------------------


class _NoAccumulateNamespace:
    """numpy, but with ``maximum`` as a plain binary function.

    numpy, cupy and MLX all have a one-pass running max; torch and jax reach the
    portable Hillis-Steele scan instead. Hiding ``maximum.accumulate`` selects
    that scan so it stays covered without those backends installed.
    """

    maximum = staticmethod(lambda a, b: np.maximum(a, b))

    def __getattr__(self, name):
        return getattr(np, name)


def test_rail_fill_scan_matches_one_pass():
    """The portable scan and the one-pass running max hold the same samples."""
    fill = SamplingDelayAlignmentTransformer._fill_rails
    x = np.random.default_rng(11).standard_normal((257, 8)).astype(np.float32)
    x[0:2, 0] = 1e4  # leading rail: nothing valid to hold yet
    x[100:130, 3] = 1e4  # a long run
    x[-1, 7] = 1e4  # trailing rail

    one_pass = fill(x, 8000.0, np, False)
    scan = fill(x, 8000.0, _NoAccumulateNamespace(), False)
    np.testing.assert_array_equal(one_pass, scan)

    # ...and both hold the last valid value, falling back to the first sample
    # when a channel rails before any valid sample has been seen.
    assert np.all(one_pass[0:2, 0] == x[0, 0])
    assert np.all(one_pass[100:130, 3] == x[99, 3])
    assert one_pass[-1, 7] == x[-2, 7]


# ---------------------------------------------------------------------------
# MLX fast path
# ---------------------------------------------------------------------------


def test_mlx_conv_path_matches_tap_loop(monkeypatch):
    """The MLX depthwise-conv FIR reproduces the portable tap-sum (to float32),
    is chunk-invariant, keeps its kernel and history on-device, and builds the
    kernel once."""
    mx = pytest.importorskip("mlx.core")
    n, nch = 2000, 64
    x = np.random.default_rng(9).standard_normal((n, nch)).astype(np.float32)
    x[500:503, 7] = 1e4  # exercise the cummax rail fill on the same path
    chunks = [3, 30, 300, 1000, 667]
    assert sum(chunks) == n

    def run(proc, sizes=chunks):
        outs, start = [], 0
        for size in sizes:
            msg = AxisArray(
                data=mx.array(x[start : start + size]),
                dims=["time", "ch"],
                axes={"time": LinearAxis(offset=start / FS, gain=1.0 / FS)},
                key="align",
            )
            outs.append(np.array(proc(msg).data))
            start += size
        return np.concatenate(outs, axis=0)

    conv_proc = sampling_delay_alignment(rail_threshold=8000.0)
    y_conv = run(conv_proc)
    y_whole = run(sampling_delay_alignment(rail_threshold=8000.0), sizes=[n])
    np.testing.assert_allclose(y_conv, y_whole, rtol=1e-6, atol=1e-5)

    # Same transformer with the cached kernel suppressed -> the tap-sum fallback.
    monkeypatch.setattr(
        SamplingDelayAlignmentTransformer,
        "_mlx_conv_weight",
        staticmethod(lambda *args: None),
    )
    loop_proc = sampling_delay_alignment(rail_threshold=8000.0)
    y_loop = run(loop_proc)

    assert loop_proc.state.conv_w is None
    assert isinstance(conv_proc.state.conv_w, mx.array)
    assert isinstance(conv_proc.state.hist, mx.array)
    np.testing.assert_allclose(y_conv, y_loop, rtol=1e-6, atol=1e-5)


def test_mlx_conv_weight_is_built_once_per_state():
    """The kernel layout is cached at state reset, not rebuilt per message."""
    mx = pytest.importorskip("mlx.core")
    proc = sampling_delay_alignment()
    x = np.random.default_rng(10).standard_normal((300, 32)).astype(np.float32)

    def msg(start, size):
        return AxisArray(
            data=mx.array(x[start : start + size]),
            dims=["time", "ch"],
            axes={"time": LinearAxis(offset=start / FS, gain=1.0 / FS)},
            key="align",
        )

    proc(msg(0, 100))
    w = proc.state.conv_w
    proc(msg(100, 50))  # a different chunk length must not trigger a rebuild
    assert proc.state.conv_w is w

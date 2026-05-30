"""Align channels sampled at different instants by a sequential A/D.

The Gemini front-end samples channels in banks of ``bank_size`` (32), one every
``channel_sample_interval`` (~969.7 ns), so channel ``c``'s sample ``n`` is the
signal at ``t_n + tau_c``, ``tau_c = (c % bank_size) * channel_sample_interval``.
For any cross-channel operation (CAR, whitening, beamforming) this misalignment
smears the common-mode at high frequency: the phase spread across a bank is
``2*pi*f*T_bank`` -- negligible at 60 Hz (0.65 deg) but ~81 deg at 7.5 kHz, so
e.g. CAR's common-mode rejection collapses toward Nyquist.

This transformer removes that by delaying each channel by ``tau_c`` with a
per-slot windowed-sinc fractional-delay filter, bringing every channel onto a
common time grid (the bank start). A windowed-sinc is used rather than linear
interpolation on purpose: linear interpolation is a delay-dependent low-pass
that would impose a *different* high-frequency rolloff per channel -- coloring
the band exactly where the misalignment mattered. There are only ``bank_size``
distinct delays, so only that many distinct filters.

The within-bank slot defaults to acquisition order (``c % bank_size``). If the
channel axis carries per-channel ``bank``/``elec`` metadata (e.g. attached by
:class:`~ezmsg.blackrock.ChannelMapUnit`), the slot is taken from ``elec``
(``elec - 1``) instead, so each channel's delay is correct even when channels
are reordered relative to hardware acquisition.

Cost / caveats:
  * **Latency:** the causal FIR has a common bulk delay of ``(filter_len-1)//2``
    samples (the per-channel fractional delays ride on top). The output time
    axis offset is shifted to keep timestamps physically correct.
  * **It resamples the raw data** -- downstream sees interpolated samples. Fine
    for cross-channel cleaning; be deliberate if a step needs raw waveforms.
  * **Railing:** clipped (rail) samples are corrupt and a fractional-delay
    filter would spread that corruption over its support. With
    ``rail_threshold`` set, railed samples are held at the last valid value
    before filtering (a basic mitigation). A production version should also
    emit a reliability mask so downstream can discount the ~``filter_len``
    samples around each rail. FIR (used here) localizes the damage; an IIR
    all-pass (e.g. Thiran) would ring across it.

Array-API compatible: it detects the input's namespace and runs on the working
backend (numpy, MLX, torch, jax, cupy, ...). The sinc taps are designed in numpy
and moved to the backend; everything else -- the FIR tap-sum, concat/state
handling, and the rail forward-fill -- runs on the backend using only standard
Array-API ops (the forward-fill's cumulative max is built from ``maximum`` +
shifts, since the standard lacks one). Only the MLX ``concatenate``-vs-``concat``
spelling is special-cased.
"""

from typing import Any

import ezmsg.core as ez
import numpy as np
import numpy.typing as npt
from array_api_compat import array_namespace
from ezmsg.baseproc import (
    BaseStatefulTransformer,
    BaseTransformerUnit,
    processor_state,
)
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace

try:  # pragma: no cover - exercised only when mlx is installed
    import mlx.core as _mx
except Exception:  # pragma: no cover
    _mx = None


def _is_mlx(arr: object) -> bool:
    return _mx is not None and isinstance(arr, _mx.array)


def _namespace(arr: object) -> tuple[Any, bool]:
    """Return ``(xp, is_mlx)``: the MLX module for MLX arrays, else the array's
    Array-API namespace (numpy, torch, jax, cupy, ...)."""
    if _is_mlx(arr):
        return _mx, True
    return array_namespace(arr), False


def _concat(xp: Any, is_mlx: bool, arrays: list, axis: int = 0) -> Any:
    """Concatenate (MLX spells it ``concatenate``; Array-API uses ``concat``)."""
    return _mx.concatenate(arrays, axis=axis) if is_mlx else xp.concat(arrays, axis=axis)


_DEFAULT_BANK_SIZE = 32
_DEFAULT_CHANNEL_SAMPLE_INTERVAL = 64.0 / 66.0e6


class SamplingDelayAlignmentSettings(ez.Settings):
    """Settings for :class:`SamplingDelayAlignmentTransformer`."""

    bank_size: int = _DEFAULT_BANK_SIZE
    """Channels per simultaneously-started A/D bank. Used to derive each
    channel's sweep slot (``c % bank_size``) only as a fallback, when the channel
    axis carries no ``bank``/``elec`` metadata."""

    channel_sample_interval: float = _DEFAULT_CHANNEL_SAMPLE_INTERVAL
    """Seconds between successive channels within a bank."""

    filter_len: int = 33
    """Sinc FIR length (odd). Bulk delay is ``(filter_len-1)//2`` samples; longer
    = flatter passband / better near Nyquist, at more latency and compute. Set to
    ``0`` to disable alignment entirely -- the transformer becomes a pass-through
    that returns its input unchanged."""

    rail_threshold: float | None = None
    """If set, samples with ``abs(value) >= rail_threshold`` are treated as
    clipped and held at the last valid value before filtering. ``None`` skips
    rail handling. (For Blackrock int16 at 0.25 uV/count, the rail is ~8191 uV.)"""


@processor_state
class SamplingDelayAlignmentState:
    """State for :class:`SamplingDelayAlignmentTransformer`."""

    fir: npt.NDArray | None = None
    """Per-channel sinc FIR taps, shape ``(filter_len, n_ch)``."""

    hist: npt.NDArray | None = None
    """Carried input history, shape ``(filter_len-1, *sample_shape)``."""

    bulk_delay: int = 0
    """Common bulk delay ``(filter_len-1)//2`` samples (for the offset shift)."""


class SamplingDelayAlignmentTransformer(
    BaseStatefulTransformer[
        SamplingDelayAlignmentSettings,
        AxisArray,
        AxisArray,
        SamplingDelayAlignmentState,
    ]
):
    """Per-channel fractional-delay alignment (see module docstring)."""

    # The rail threshold only gates the forward-fill in _process; it doesn't
    # alter the designed filters, so changing it needn't reset the state.
    NONRESET_SETTINGS_FIELDS = frozenset({"rail_threshold"})

    @property
    def _passthrough(self) -> bool:
        """``filter_len <= 0`` disables alignment: the transformer returns its
        input unchanged and skips building the FIR (undefined for ``n_taps`` 0)."""
        return self.settings.filter_len < 1

    def _channel_slots(self, message: AxisArray) -> npt.NDArray:
        """Within-bank A/D sweep position (0-based) for each channel on the
        ``ch`` axis.

        Prefers channel metadata: when the ``ch`` axis carries a structured
        ``.data`` with ``bank`` and ``elec`` fields (as produced by
        :class:`~ezmsg.blackrock.ChannelMapUnit`), the slot is ``elec - 1`` --
        the channel's physical position in its bank's sequential sweep, so the
        delay is correct even when channels are not in hardware-acquisition
        order. Falls back to acquisition-order banks of ``bank_size``
        (``arange(n_ch) % bank_size``) when that metadata is absent.
        """
        n_ch = message.data.shape[message.get_axis_idx("ch")]
        data = getattr(message.axes.get("ch"), "data", None)
        names = getattr(getattr(data, "dtype", None), "names", None)
        if names is not None and "bank" in names and "elec" in names and len(data) == n_ch:
            return data["elec"].astype(np.int64) - 1
        return np.arange(n_ch) % self.settings.bank_size

    def _hash_message(self, message: AxisArray) -> int:
        time_idx = message.get_axis_idx("time")
        sample_shape = message.data.shape[:time_idx] + message.data.shape[time_idx + 1 :]
        # Include the slot layout so a metadata change (e.g. a new channel map)
        # re-designs the filters even when shape/key/gain are unchanged.
        slot = self._channel_slots(message)
        return hash((message.key, message.axes["time"].gain, sample_shape, slot.tobytes()))

    def _reset_state(self, message: AxisArray) -> None:
        if self._passthrough:
            return  # no filters to design; _process returns the input as-is
        time_idx = message.get_axis_idx("time")
        sample_shape = message.data.shape[:time_idx] + message.data.shape[time_idx + 1 :]
        dtype = message.data.dtype
        xp, is_mlx = _namespace(message.data)
        fs = 1.0 / message.axes["time"].gain

        slot = self._channel_slots(message)
        # Fractional-sample delay that brings each channel back to its bank start.
        d = slot * self.settings.channel_sample_interval * fs  # in [0, ~0.9]

        n_taps = int(self.settings.filter_len)
        m = (n_taps - 1) // 2
        self._state.bulk_delay = m

        # Design the per-channel windowed sinc in numpy (total delay m + d_c, DC
        # gain 1), then move the taps onto the working backend.
        k = np.arange(n_taps)[:, None]
        h = np.sinc(k - m - d[None, :]) * np.blackman(n_taps)[:, None]
        h = h / h.sum(axis=0, keepdims=True)
        if is_mlx:
            self._state.fir = _mx.array(h.astype(np.float32))
            self._state.hist = _mx.zeros((n_taps - 1,) + sample_shape, dtype=dtype)
        else:
            # h is numpy; convert to the backend then to its dtype (dtype may be
            # a non-numpy dtype, e.g. torch.float32, that numpy.astype rejects).
            self._state.fir = xp.astype(xp.asarray(h), dtype)
            self._state.hist = xp.zeros((n_taps - 1,) + sample_shape, dtype=dtype)

    @staticmethod
    def _fill_rails(x: npt.NDArray, thresh: float, xp: Any, is_mlx: bool) -> npt.NDArray:
        """Forward-fill (hold last valid) over railed samples, per channel.

        Backend-portable: per (time, channel), find the index of the most recent
        valid sample at or before each position, then gather. Because the
        Array-API standard lacks a cumulative max, it is built from standard ops
        (``maximum`` + shifts) as a Hillis-Steele scan -- valid positions carry
        their (increasing) index and railed ones carry ``-1``, so the running
        max is exactly the last valid index. O(n log n) but fully vectorized,
        and only runs when ``rail_threshold`` is set.
        """
        n = x.shape[0]
        sample_shape = x.shape[1:]
        ar = xp.reshape(xp.arange(n), (n,) + (1,) * (x.ndim - 1))
        idx = xp.where(xp.abs(x) >= thresh, -1, ar)  # index, or -1 where railed
        shift = 1
        while shift < n:
            sentinel = xp.full((shift,) + sample_shape, -1, dtype=idx.dtype)
            shifted = _concat(xp, is_mlx, [sentinel, idx[: n - shift]], axis=0)
            idx = xp.maximum(idx, shifted)
            shift *= 2
        idx = xp.where(idx < 0, 0, idx)  # leading rails -> first sample
        return xp.take_along_axis(x, idx, axis=0)

    def _process(self, message: AxisArray) -> AxisArray:
        if self._passthrough:
            return message
        ax_idx = message.get_axis_idx("time")
        x = message.data
        xp, is_mlx = _namespace(x)
        moved = ax_idx != 0
        if moved:
            x = xp.moveaxis(x, ax_idx, 0)

        if self.settings.rail_threshold is not None:
            x = self._fill_rails(x, self.settings.rail_threshold, xp, is_mlx)

        st = self._state
        fir = st.fir
        n_taps = fir.shape[0]
        n = x.shape[0]

        # FIR via tap-sum, carrying n_taps-1 samples of history across chunks:
        #   y[i] = sum_k fir[k] * xext[(n_taps-1) - k + i],  xext = [hist, x]
        xext = _concat(xp, is_mlx, [st.hist, x], axis=0)
        y = xp.zeros_like(x)
        for k in range(n_taps):
            y = y + fir[k] * xext[n_taps - 1 - k : n_taps - 1 - k + n]
        st.hist = xext[-(n_taps - 1) :]

        if moved:
            y = xp.moveaxis(y, 0, ax_idx)

        # Output sample i carries the bank-start signal delayed by bulk_delay
        # samples; shift the time-axis offset so timestamps stay physical.
        time_axis = message.axes["time"]
        new_axis = replace(
            time_axis,
            offset=time_axis.offset - st.bulk_delay * time_axis.gain,
        )
        return replace(message, data=y, axes={**message.axes, "time": new_axis})


class SamplingDelayAlignment(
    BaseTransformerUnit[
        SamplingDelayAlignmentSettings,
        AxisArray,
        AxisArray,
        SamplingDelayAlignmentTransformer,
    ]
):
    SETTINGS = SamplingDelayAlignmentSettings

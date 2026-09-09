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
and moved to the backend; everything else -- the FIR, concat/state handling, and
the rail forward-fill -- runs on the backend using only standard Array-API ops
(the forward-fill's cumulative max is built from ``maximum`` + shifts, since the
standard lacks one). Only MLX's ``concatenate``-vs-``concat`` spelling needs
special-casing.

Backend-specific fast paths sit on top of that, because live acquisition
delivers chunks of a few dozen samples where per-operation dispatch, not
arithmetic, sets the wall time:
  * on MLX the FIR runs as a single depthwise ``conv_general`` (one group per
    column, kernel cached at state reset) instead of ``filter_len`` multiply-add
    stages -- ~2-3x less time per message, and roughly flat in chunk size;
  * on numpy, if the optional ``numba`` dependency is installed, both the FIR
    and the rail forward-fill run as fused single-pass jitted kernels (see
    :mod:`ezmsg.blackrock._numba_kernels`): several times faster than the tap
    loop at live sizes and, with a threaded FIR for large buffers, ~20x faster
    for offline batch processing. That path also runs out of one reused
    ``[history, chunk]`` buffer rather than concatenating the history onto each
    message, so a steady stream allocates only its output;
  * the forward-fill's running max otherwise uses ``mx.cummax`` on MLX and the
    ``maximum`` ufunc's ``accumulate`` on numpy/cupy, in place of the log scan.
All of these fall back to the portable formulation wherever the op or the
optional dependency is missing.
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
    resolve_stream_dim,
)
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace

try:  # pragma: no cover - exercised only when mlx is installed
    import mlx.core as _mx
except Exception:  # pragma: no cover
    _mx = None

try:  # optional accel extra; the numpy path works without it
    from . import _numba_kernels as _nb
except Exception:  # pragma: no cover - numba not installed
    _nb = None


def _is_mlx(arr: object) -> bool:
    return _mx is not None and isinstance(arr, _mx.array)


def _use_numba(arr: object) -> bool:
    """Whether the jitted kernels apply: numba installed and a plain numpy array
    (torch/jax/cupy arrays aren't ``np.ndarray`` and keep the portable path)."""
    return _nb is not None and isinstance(arr, np.ndarray)


def _namespace(arr: object) -> tuple[Any, bool]:
    """Return ``(xp, is_mlx)``: the MLX module for MLX arrays, else the array's
    Array-API namespace (numpy, torch, jax, cupy, ...)."""
    if _is_mlx(arr):
        return _mx, True
    return array_namespace(arr), False


def _concat(xp: Any, is_mlx: bool, arrays: list, axis: int = 0) -> Any:
    """Concatenate (MLX spells it ``concatenate``; Array-API uses ``concat``)."""
    return _mx.concatenate(arrays, axis=axis) if is_mlx else xp.concat(arrays, axis=axis)


def _own(xp: Any, is_mlx: bool, arr: Any) -> Any:
    """Return a copy of ``arr`` that owns its memory.

    For a slice kept across messages: on every backend here -- numpy, torch,
    cupy and MLX alike -- slicing returns a view that pins its *base* buffer
    alive, so retaining a small tail of a large temporary retains the whole
    temporary (measured: MLX holds 25.6 MB for a 12-row tail; numpy holds a
    312 KiB concat buffer for 12 KiB of history at 300 samples x 256 channels).
    Copying the tail costs far less than pinning the base, and frees the base
    for the allocator to hand straight back for the next message's buffer.
    """
    return _mx.array(arr) if is_mlx else xp.asarray(arr, copy=True)


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

    filter_len: int = 13
    """Sinc FIR length (odd). Bulk delay is ``(filter_len-1)//2`` samples; longer
    = flatter passband / better near Nyquist, at more latency and compute. Set to
    ``0`` to disable alignment entirely -- the transformer becomes a pass-through
    that returns its input unchanged.

    Worst case over all ``bank_size`` fractional delays, at 30 kHz (see
    ``tests/test_sampling_delay_alignment.py`` for the pinned numbers):

    ==========  ==========  ===============  =============  ==========
    Passband    filter_len  Max phase error  Max mag error  Bulk delay
    ==========  ==========  ===============  =============  ==========
    0-500 Hz    7           0.0009 deg       0.00001 dB     3 samples
    0-3 kHz     9           0.0038 deg       0.0001 dB      4 samples
    0-7.5 kHz   13          0.015 deg        0.0015 dB      6 samples
    0-7.5 kHz   33          0.0028 deg       0.0004 dB      16 samples
    ==========  ==========  ===============  =============  ==========

    The default 13 covers the full broadband/spike band with ~3 orders of
    magnitude of margin on the ~81 deg of skew it is correcting, at less than
    half the latency of the former 33-tap default. Use 7 in an LFP-only
    pipeline; 33 buys accuracy that is already far below the noise floor."""

    rail_threshold: float | None = None
    """If set, samples with ``abs(value) >= rail_threshold`` are treated as
    clipped and held at the last valid value before filtering. ``None`` skips
    rail handling. (For Blackrock int16 at 0.25 uV/count, the rail is ~8191 uV.)"""


@processor_state
class SamplingDelayAlignmentState:
    """State for :class:`SamplingDelayAlignmentTransformer`."""

    axis: str = ""
    """The resolved stream dimension, fixed at reset so every later use agrees."""

    fir: npt.NDArray | None = None
    """Per-channel sinc FIR taps, shape ``(filter_len, n_ch)``."""

    conv_w: Any | None = None
    """MLX depthwise-conv kernel, shape ``(n_cols, filter_len, 1)`` -- the same
    taps as :attr:`fir`, reversed and laid out per flattened sample column.
    ``None`` on every other backend (and when the taps don't broadcast over
    ``sample_shape``), which selects the portable tap-sum instead."""

    nb_w: npt.NDArray | None = None
    """numba FIR kernel weights, shape ``(filter_len, n_cols)`` -- the same taps
    as :attr:`fir`, reversed and laid out per flattened sample column, in the
    data dtype. Set only on the numpy backend when numba is installed; ``None``
    otherwise, which selects the portable tap-sum."""

    hist: npt.NDArray | None = None
    """Carried input history, shape ``(filter_len-1, *sample_shape)``. On the
    numba path this is a view of the leading rows of :attr:`scratch`; elsewhere
    it owns its memory (see :func:`_own`)."""

    scratch: npt.NDArray | None = None
    """numba path only: the reused ``(filter_len-1 + capacity, n_cols)`` work
    buffer holding the carried history followed by the current chunk, so the
    per-message concat and its full-chunk temporary don't happen. Grown (never
    shrunk) to the largest chunk seen; ``None`` until the first message."""

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

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Warm the jitted kernels here rather than letting the first message
        # pay for them. A unit builds its transformer from initialize(), which
        # runs in the hosting process before any message flows, so this is the
        # last moment nothing is waiting: on a live source the same cost lands
        # mid-stream, and every millisecond it takes is another chunk queued
        # behind it (see _numba_kernels.warmup).
        #
        # Skipped when alignment is off, since a pass-through never reaches a
        # kernel. Not skipped for a stream that turns out to be MLX or torch --
        # the array type is not knowable until the first message, and paying a
        # warmup those backends will not use is the cheaper mistake.
        if _nb is not None and not self._passthrough:
            _nb.warmup()

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
        # Runs before `_reset_state`, so the axis is resolved from the message
        # rather than read back off state that does not exist yet.
        axis = resolve_stream_dim(message)
        time_idx = message.get_axis_idx(axis)
        sample_shape = message.data.shape[:time_idx] + message.data.shape[time_idx + 1 :]
        # Include the slot layout so a metadata change (e.g. a new channel map)
        # re-designs the filters even when shape/key/gain are unchanged.
        slot = self._channel_slots(message)
        return hash((message.key, message.axes[axis].gain, sample_shape, slot.tobytes()))

    def _reset_state(self, message: AxisArray) -> None:
        if self._passthrough:
            return  # no filters to design; _process returns the input as-is
        self._state.axis = resolve_stream_dim(message)
        time_idx = message.get_axis_idx(self._state.axis)
        sample_shape = message.data.shape[:time_idx] + message.data.shape[time_idx + 1 :]
        dtype = message.data.dtype
        xp, is_mlx = _namespace(message.data)
        fs = 1.0 / message.axes[self._state.axis].gain

        slot = self._channel_slots(message)
        # Fractional-sample delay that brings each channel back to its bank start.
        d = slot * self.settings.channel_sample_interval * fs  # in [0, ~0.9]

        n_taps = int(self.settings.filter_len)
        m = (n_taps - 1) // 2
        self._state.bulk_delay = m
        # The state object survives a reset, so the old buffer (sized for the
        # previous shape/dtype) has to be dropped explicitly.
        self._state.scratch = None

        # Design the per-channel windowed sinc in numpy (total delay m + d_c, DC
        # gain 1), then move the taps onto the working backend.
        k = np.arange(n_taps)[:, None]
        h = np.sinc(k - m - d[None, :]) * np.blackman(n_taps)[:, None]
        h = h / h.sum(axis=0, keepdims=True)
        if is_mlx:
            self._state.fir = _mx.array(h.astype(np.float32))
            self._state.hist = _mx.zeros((n_taps - 1,) + sample_shape, dtype=dtype)
            self._state.conv_w = self._mlx_conv_weight(h, sample_shape, dtype)
        else:
            # h is numpy; convert to the backend then to its dtype (dtype may be
            # a non-numpy dtype, e.g. torch.float32, that numpy.astype rejects).
            self._state.fir = xp.astype(xp.asarray(h), dtype)
            self._state.hist = xp.zeros((n_taps - 1,) + sample_shape, dtype=dtype)
            self._state.conv_w = None
            self._state.nb_w = self._column_taps(h, sample_shape, dtype) if _use_numba(message.data) else None

    @staticmethod
    def _column_taps(h: npt.NDArray, sample_shape: tuple[int, ...], dtype: Any) -> npt.NDArray | None:
        """Per-column, time-reversed taps ``(n_taps, n_cols)`` in ``dtype``.

        Both the MLX conv and the numba FIR consume ``xext`` flattened to
        ``(time, n_cols)`` with ``n_cols = prod(sample_shape)`` in the same
        row-major order ``_process`` uses, and both cross-correlate, so they want
        the taps broadcast over the sample shape and reversed in time. Returns
        ``None`` when the per-channel taps don't broadcast over ``sample_shape``
        (the last sample axis isn't the ``n_ch`` the filters were designed for),
        which keeps the portable tap-sum.
        """
        n_taps, n_ch = h.shape
        if not sample_shape or sample_shape[-1] != n_ch:
            return None
        cols = np.broadcast_to(
            h.reshape((n_taps,) + (1,) * (len(sample_shape) - 1) + (n_ch,)),
            (n_taps,) + tuple(sample_shape),
        ).reshape(n_taps, -1)
        return np.ascontiguousarray(cols[::-1], dtype=dtype)

    @classmethod
    def _mlx_conv_weight(cls, h: npt.NDArray, sample_shape: tuple[int, ...], dtype: Any) -> Any:
        """Lay the designed taps out as an MLX depthwise-conv kernel, once.

        ``mx.conv_general`` cross-correlates over one group per input column, so
        the kernel is :meth:`_column_taps` transposed to ``(n_cols, n_taps, 1)``.
        Returns ``None`` (keep the tap-sum) when the taps don't broadcast.
        """
        cols = cls._column_taps(h, sample_shape, np.float32)  # (n_taps, n_cols)
        if cols is None:
            return None
        w = np.ascontiguousarray(cols.T, dtype=np.float32)[:, :, None]
        # conv_general needs input and kernel in one dtype; use the dtype the
        # tap-sum's float32-taps-times-data product would have promoted to.
        out_dtype = (_mx.zeros(1, dtype=dtype) * _mx.zeros(1, dtype=_mx.float32)).dtype
        return _mx.array(w).astype(out_dtype)

    @staticmethod
    def _fill_rails(x: npt.NDArray, thresh: float, xp: Any, is_mlx: bool) -> npt.NDArray:
        """Forward-fill (hold last valid) over railed samples, per channel.

        Backend-portable: per (time, channel), find the index of the most recent
        valid sample at or before each position, then gather -- valid positions
        carry their (increasing) index and railed ones carry ``-1``, so a running
        max over time is exactly the last valid index.

        The Array-API standard has no cumulative max, so the running max is built
        from ``maximum`` + shifts as a Hillis-Steele scan: O(n log n) backend
        calls, fully vectorized, and correct everywhere. It is also the dominant
        per-message cost once the FIR is fast, so faster forms are taken where
        available -- a fused single left-to-right pass in numba (numpy + accel
        extra), else ``cummax`` on MLX and the ``maximum`` ufunc's ``accumulate``
        on numpy/cupy. Only runs when ``rail_threshold`` is set.

        :meth:`_numba_filter` doesn't come through here -- it runs the same
        jitted fill straight into its shared buffer. The numba branch below
        still covers numpy inputs whose taps don't broadcast over the sample
        shape, which leaves ``nb_w`` (and so that path) unavailable.
        """
        if _use_numba(x):
            flat = np.ascontiguousarray(x).reshape(x.shape[0], -1)
            out = np.empty_like(flat)
            _nb.fill_rails(flat, float(thresh), out)
            return out.reshape(x.shape)
        n = x.shape[0]
        sample_shape = x.shape[1:]
        ar = xp.reshape(xp.arange(n), (n,) + (1,) * (x.ndim - 1))
        idx = xp.where(xp.abs(x) >= thresh, -1, ar)  # index, or -1 where railed
        accumulate = None if is_mlx else getattr(xp.maximum, "accumulate", None)
        if is_mlx:
            idx = _mx.cummax(idx, axis=0)
        elif accumulate is not None:
            idx = accumulate(idx, axis=0)  # numpy/cupy ufunc: one pass
        else:
            shift = 1
            while shift < n:
                sentinel = xp.full((shift,) + sample_shape, -1, dtype=idx.dtype)
                shifted = _concat(xp, is_mlx, [sentinel, idx[: n - shift]], axis=0)
                idx = xp.maximum(idx, shifted)
                shift *= 2
        idx = xp.where(idx < 0, 0, idx)  # leading rails -> first sample
        return xp.take_along_axis(x, idx, axis=0)

    def _scratch(self, n: int, sample_shape: tuple[int, ...], dtype: Any) -> npt.NDArray:
        """The ``[history, chunk]`` work buffer, with room for ``n`` new samples.

        Reallocated only when a chunk outgrows it, and then geometrically: live
        acquisition chunk sizes drift with the backlog, and refitting exactly
        would put an allocation back on the hot path for every small increase.
        The carried history moves into the new buffer, and :attr:`state.hist`
        re-points at it -- a view of a buffer the state means to keep, not of a
        per-message temporary.
        """
        st = self._state
        h = st.nb_w.shape[0] - 1
        n_cols = st.nb_w.shape[1]
        buf = st.scratch
        if buf is None or buf.shape[0] - h < n:
            capacity = 0 if buf is None else buf.shape[0] - h
            buf = np.empty((h + max(n, 2 * capacity), n_cols), dtype=dtype)
            buf[:h] = st.hist.reshape(h, n_cols)
            st.scratch = buf
            st.hist = buf[:h].reshape((h,) + sample_shape)
        return buf

    def _numba_filter(self, x: npt.NDArray, n: int) -> npt.NDArray:
        """The numpy+numba path: rail forward-fill and FIR through :meth:`_scratch`.

        The history the FIR needs is just the tail of the previous chunk, so
        keeping one buffer with that history at the front lets the new chunk be
        written in directly behind it -- no per-message concat, and no
        full-chunk temporary to hold its result. With rail handling on there is
        no extra copy at all, because the forward-fill writes its output into
        that buffer instead of a second one. Only the FIR output is still
        allocated per message, since it is what the message carries downstream.
        """
        st = self._state
        sample_shape = x.shape[1:]
        n_taps, n_cols = st.nb_w.shape
        h = n_taps - 1
        buf = self._scratch(n, sample_shape, x.dtype)

        flat = np.ascontiguousarray(x).reshape(n, n_cols)
        chunk = buf[h : h + n]
        if self.settings.rail_threshold is not None:
            _nb.fill_rails(flat, float(self.settings.rail_threshold), chunk)
        else:
            chunk[...] = flat

        yflat = np.empty((n, n_cols), dtype=buf.dtype)
        _nb.fir(buf[: h + n], st.nb_w, yflat)
        # Carry the last h rows forward as the next message's history (st.hist
        # views these rows). numpy takes a temporary when the two slices overlap,
        # which they do for a chunk shorter than the history.
        buf[:h] = buf[n : n + h]
        return yflat.reshape((n,) + sample_shape)

    def _process(self, message: AxisArray) -> AxisArray:
        if self._passthrough:
            return message
        ax_idx = message.get_axis_idx(self._state.axis)
        x = message.data
        xp, is_mlx = _namespace(x)
        moved = ax_idx != 0
        if moved:
            x = xp.moveaxis(x, ax_idx, 0)

        st = self._state
        fir = st.fir
        n_taps = fir.shape[0]
        n = x.shape[0]

        if st.nb_w is not None:
            # Fused jitted passes over the flattened columns, sharing one
            # preallocated buffer (which is also where the rail fill writes).
            y = self._numba_filter(x, n)
        else:
            if self.settings.rail_threshold is not None:
                x = self._fill_rails(x, self.settings.rail_threshold, xp, is_mlx)

            # FIR, carrying n_taps-1 samples of history across chunks:
            #   y[i] = sum_k fir[k] * xext[(n_taps-1) - k + i],  xext = [hist, x]
            xext = _concat(xp, is_mlx, [st.hist, x], axis=0)
            if st.conv_w is not None:
                # Same sum as below, as one MLX depthwise conv (one group per
                # column) rather than n_taps dispatched multiply-adds -- which is
                # what costs on the short chunks live acquisition delivers.
                n_cols = st.conv_w.shape[0]
                xin = xext if xext.dtype == st.conv_w.dtype else xext.astype(st.conv_w.dtype)
                y = _mx.conv_general(_mx.reshape(xin, (1, xext.shape[0], n_cols)), st.conv_w, groups=n_cols)
                y = _mx.reshape(y, (n,) + x.shape[1:])
            else:
                y = xp.zeros_like(x)
                for k in range(n_taps):
                    y = y + fir[k] * xext[n_taps - 1 - k : n_taps - 1 - k + n]
            st.hist = _own(xp, is_mlx, xext[xext.shape[0] - (n_taps - 1) :])

        if moved:
            y = xp.moveaxis(y, 0, ax_idx)

        # Output sample i carries the bank-start signal delayed by bulk_delay
        # samples; shift the time-axis offset so timestamps stay physical.
        time_axis = message.axes[self._state.axis]
        new_axis = replace(
            time_axis,
            offset=time_axis.offset - st.bulk_delay * time_axis.gain,
        )
        return replace(message, data=y, axes={**message.axes, self._state.axis: new_axis})


class SamplingDelayAlignment(
    BaseTransformerUnit[
        SamplingDelayAlignmentSettings,
        AxisArray,
        AxisArray,
        SamplingDelayAlignmentTransformer,
    ]
):
    SETTINGS = SamplingDelayAlignmentSettings

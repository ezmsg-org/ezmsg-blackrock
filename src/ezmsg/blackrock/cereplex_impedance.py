"""CerePlex impedance measurement pipeline.

The CerePlex headstage injects a 1 kHz, 1 nA sine wave for 100 ms per channel,
cycling sequentially through all channels.  Channels not under test read exactly
zero (filters must be disabled).  Impedance is extracted via single-bin DFT at
1 kHz: Z(kOhm) = V_peak_to_peak(uV) / I_peak_to_peak(nA).

Multiple headstages are tracked independently — each may be at a different point
in its impedance sweep.
"""

import logging
import typing

import ezmsg.core as ez
import numpy as np
import scipy.signal as ss
from ezmsg.baseproc import BaseStatefulTransformer, BaseTransformerUnit, processor_state
from ezmsg.util.messages.axisarray import AxisArray, replace

logger = logging.getLogger(__name__)


class CerePlexImpedanceSettings(ez.Settings):
    headstage_channel_offsets: tuple[int, ...] = (0,)
    """Starting channel index of each CerePlex headstage.  Each headstage's range
    extends from its offset to the next offset (or n_ch for the last).
    Example: two 128-ch headstages → (0, 128)."""

    collect_duration_s: float = 0.1
    """Maximum burst duration to buffer per channel (100 ms for CerePlex)."""

    fft_duration_s: float = 0.09227
    """Duration of data used for FFT, taken from the end of the burst.
    The preceding samples serve as settle time."""

    freq_lo: float = 960.0
    """Lower bound of frequency range for peak extraction (Hz)."""

    freq_hi: float = 1050.0
    """Upper bound of frequency range for peak extraction (Hz)."""

    test_current_nA: float = 1.0
    """Injected test-current peak-to-peak amplitude (nA)."""


class _HeadstageTracker:
    """Per-headstage sequential channel tracker."""

    __slots__ = ("ch_start", "ch_end", "tracking_ch", "buffer", "buf_len")

    def __init__(self, ch_start: int, ch_end: int, buffer: np.ndarray):
        self.ch_start = ch_start
        self.ch_end = ch_end  # exclusive
        self.tracking_ch = -1  # absolute index; -1 = scanning
        self.buffer = buffer
        self.buf_len = 0


@processor_state
class CerePlexImpedanceState:
    trackers: list | None = None  # list[_HeadstageTracker]

    max_buffer_samples: int = 0
    fft_samples: int = 0
    fs: float = 0.0

    impedance: np.ndarray | None = None  # (n_ch,), NaN = unmeasured
    ch_axis: typing.Any = None


def extract_impedance(
    data: np.ndarray,
    fft_samples: int,
    fs: float,
    freq_lo: float,
    freq_hi: float,
    test_current_nA: float,
) -> float | None:
    """Extract impedance (kOhm) from the 1 kHz component via Hann-windowed DFT.

    Detrends the tail of the burst (skipping the settle transient), applies
    a Hann window, computes the FFT, and sums the band-limited energy to
    recover the peak amplitude of the injected test tone. Impedance is
    ``V_p2p / I_p2p``.

    .. important::
       *data* must be in **microvolts**. Passing raw ADC counts will produce
       impedance values that are wrong by the ADC scale factor.
       When using :class:`CereLinkSignalSource`, set ``microvolts=True``.

    Args:
        data: 1-D array of samples for a single channel burst, in microvolts.
        fft_samples: Number of samples (from the end of *data*) to use for the FFT.
        fs: Sampling rate in Hz.
        freq_lo: Lower bound of frequency range for peak extraction (Hz).
        freq_hi: Upper bound of frequency range for peak extraction (Hz).
        test_current_nA: Peak-to-peak amplitude of the injected test current (nA).

    Returns:
        Impedance in kOhm, or ``None`` if *data* is too short or no energy
        is found in the target band.
    """
    if len(data) < fft_samples:
        return None

    # Use the tail of the burst (skip settle transient at the start).
    signal = data[-fft_samples:].astype(np.float64)
    signal = ss.detrend(signal, type="linear")

    # Hann window: concentrates an off-bin tone's energy into a ~4-bin
    # main lobe, so the band-sum amplitude estimate below doesn't under-
    # report by up to 1/√2 for tones that straddle two rectangular bins.
    window = np.hanning(fft_samples)
    spectrum = np.fft.rfft(signal * window)
    freqs = np.fft.rfftfreq(fft_samples, d=1.0 / fs)
    mask = (freqs >= freq_lo) & (freqs <= freq_hi)

    if not np.any(mask):
        return None

    # Power-integrated amplitude. Parseval gives, for a windowed pure tone
    # of peak amplitude A whose main lobe fits inside the mask:
    #
    #     Σ_{rfft bins} |X_w|²  ≈  (N · A² / 4) · Σ w²
    #
    # Solving for A:  A = 2·√(Σ|X|²) / √(N · Σw²).
    #
    # For a rectangular window (Σw² = N) this reduces to the familiar
    # ``(2/N)·√Σ|X|²``; for any other window the ``sum(w)``-style
    # coherent-gain correction would be wrong by ``√(ENBW)``.
    amplitude = (2.0 / np.sqrt(fft_samples * np.sum(window**2))) * np.sqrt(np.sum(np.abs(spectrum[mask]) ** 2))
    p2p = 2 * amplitude
    impedance_kohm = p2p / test_current_nA
    # if impedance_kohm < 8 or impedance_kohm > 900:
    #     import matplotlib.pyplot as plt
    #
    #     plt.subplot(2, 1, 1)
    #     plt.plot(signal)
    #     plt.xlabel("Sample")
    #
    #     plt.subplot(2, 1, 2)
    #     ylim = np.max(np.abs(spectrum[mask]))
    #     plt.semilogx(freqs, spectrum)
    #     plt.xlabel("Frequency (Hz)")
    #     plt.ylim([-ylim, ylim])
    #
    #     plt.tight_layout()
    #     plt.suptitle(f"Impedance = {impedance_kohm:.2f} kOhm")
    #     plt.show()
    return impedance_kohm if impedance_kohm > 0 else None


def _scan_for_active(data: np.ndarray, pos: int, hs: _HeadstageTracker) -> None:
    """Find the currently-active channel and tag the next one for measurement."""
    remaining = data[pos:, hs.ch_start : hs.ch_end]
    has_data = np.any(remaining != 0, axis=0)
    candidates = np.flatnonzero(has_data)
    if len(candidates) == 0:
        return
    # Pick the channel whose first non-zero sample is earliest
    first_nz = np.argmax(remaining[:, candidates] != 0, axis=0)
    active_local = int(candidates[first_nz.argmin()])
    do_next = remaining[0, active_local] != 0  # True if we don't know when active_local started.
    n_hs = hs.ch_end - hs.ch_start
    hs.tracking_ch = hs.ch_start + (active_local + int(do_next)) % n_hs
    hs.buf_len = 0


class CerePlexImpedanceProcessor(
    BaseStatefulTransformer[
        CerePlexImpedanceSettings,
        AxisArray,
        AxisArray | None,
        CerePlexImpedanceState,
    ]
):
    """Stateful transformer that extracts per-channel impedance from a CerePlex sweep.

    Expects a stream of ``AxisArray`` messages with dims ``["time", "ch"]``
    where the data is in **microvolts**. When using :class:`CereLinkSignalSource`,
    set ``microvolts=True``; raw ADC counts will produce incorrect results.

    The processor tracks one or more headstages independently (configured via
    :attr:`CerePlexImpedanceSettings.headstage_channel_offsets`). Each
    headstage's impedance sweep cycles sequentially through its channels:
    exactly one channel is non-zero at a time while the others read zero.
    This relies on the device's internal filtering being disabled — a filter
    produces small non-zero residuals on idle channels, which defeats the
    exact-zero exclusivity checks. Misconfiguration there is a sign that the
    recording chain needs fixing, not that this algorithm should accommodate
    it.

    On each impedance update the processor emits an ``AxisArray`` whose data
    is a ``(1, n_ch)`` array of impedance values in kOhm (``NaN`` for channels
    not yet measured).
    """

    # freq_lo/freq_hi/test_current_nA are read live in extract_impedance().
    # headstage_channel_offsets is handled in-place by update_settings() below
    # to preserve the accumulated state.impedance array across re-layouts.
    NONRESET_SETTINGS_FIELDS = frozenset({"freq_lo", "freq_hi", "test_current_nA", "headstage_channel_offsets"})

    def update_settings(self, new_settings: CerePlexImpedanceSettings) -> None:
        old_offsets = self.settings.headstage_channel_offsets
        super().update_settings(new_settings)
        # If a non-NONRESET field changed, super() armed a full reset (_hash=-1)
        # and _reset_state will rebuild trackers from scratch on the next message.
        # Only patch trackers in place when the offsets-only fast path applies
        # AND state has actually been initialized.
        if (
            self._hash != -1
            and tuple(old_offsets) != tuple(new_settings.headstage_channel_offsets)
            and self.state.impedance is not None
        ):
            self._build_trackers(self.state.impedance.shape[0])

    def _hash_message(self, message: AxisArray) -> int:
        ch_idx = message.dims.index("ch")
        n_ch = message.data.shape[ch_idx]
        time_axis = message.axes.get("time")
        fs = time_axis.gain if hasattr(time_axis, "gain") else 0
        return hash((n_ch, fs))

    def _reset_state(self, message: AxisArray) -> None:
        s = self.state
        s.fs = 1.0 / message.axes["time"].gain
        ch_idx = message.dims.index("ch")
        n_ch = message.data.shape[ch_idx]
        settings = self.settings

        s.max_buffer_samples = int(settings.collect_duration_s * s.fs)
        s.fft_samples = int(settings.fft_duration_s * s.fs)

        self._build_trackers(n_ch)

        s.impedance = np.full(n_ch, np.nan, dtype=np.float64)
        s.ch_axis = message.axes.get("ch")

    def _build_trackers(self, n_ch: int) -> None:
        """Build per-headstage trackers from the current settings.

        Split out from ``_reset_state`` so a settings update that only changes
        ``headstage_channel_offsets`` can rebuild the tracker layout without
        clearing the accumulated ``state.impedance`` array. Any in-flight
        per-tracker burst buffer is discarded; new bursts buffer fresh.
        """
        s = self.state
        offsets = sorted(self.settings.headstage_channel_offsets)
        s.trackers = []
        for i, start in enumerate(offsets):
            end = offsets[i + 1] if i + 1 < len(offsets) else n_ch
            buf = np.zeros(s.max_buffer_samples, dtype=np.float64)
            s.trackers.append(_HeadstageTracker(start, end, buf))

    # --- Per-headstage helpers ---

    def _complete_channel(self, hs: _HeadstageTracker) -> bool:
        """FFT the buffered burst, store impedance, advance to next channel.

        Only updates the stored impedance if the burst contained enough
        samples for a reliable FFT.  Truncated bursts (e.g. from a file-loop
        boundary or impedance mode being disabled mid-sweep) are discarded
        so they don't overwrite a previous good measurement.
        """
        s = self.state
        settings = self.settings
        updated = False
        if hs.buf_len >= s.fft_samples:
            imp = extract_impedance(
                hs.buffer[: hs.buf_len],
                s.fft_samples,
                s.fs,
                settings.freq_lo,
                settings.freq_hi,
                settings.test_current_nA,
            )
            if imp is not None:
                s.impedance[hs.tracking_ch] = imp
                updated = True
        n_hs = hs.ch_end - hs.ch_start
        local = hs.tracking_ch - hs.ch_start
        hs.tracking_ch = hs.ch_start + (local + 1) % n_hs
        hs.buf_len = 0
        return updated

    def _buffer_channel(
        self,
        data: np.ndarray,
        pos: int,
        hs: _HeadstageTracker,
    ) -> tuple[int, bool, bool]:
        """Buffer samples from the tracked channel's column slice.

        Termination: tracked channel is zero AND next channel is non-zero
        (the headstage has handed off to the next channel).

        Returns (samples_consumed, channel_done, impedance_updated).
        """
        s = self.state
        col = data[pos:, hs.tracking_ch]
        n = len(col)
        if n == 0:
            return 0, False, False

        # Next channel in the headstage sequence
        n_hs = hs.ch_end - hs.ch_start
        local = hs.tracking_ch - hs.ch_start
        next_ch = hs.ch_start + (local + 1) % n_hs
        next_col = data[pos:, next_ch]

        # Skip leading zeros if not yet buffering
        start = 0
        if hs.buf_len == 0:
            nz = col != 0
            first_nz = int(np.argmax(nz))
            if not nz[first_nz]:
                return n, False, False  # all zero — channel not active yet
            start = first_nz
            if n_hs > 1 and next_col[start] != 0:
                hs.tracking_ch = -1
                hs.buf_len = 0
                return n, True, False

        tail = col[start:]
        next_tail = next_col[start:]

        # Find end of non-zero run in tracked channel
        zeros = tail == 0.0
        first_zero = len(tail)
        if np.any(zeros):
            first_zero = int(np.argmax(zeros))

        # Exclusivity check: if the next channel in sequence is non-zero
        # while we're buffering, impedance mode was toggled off (or tracking
        # is out of sync). Only meaningful with >1 channel per headstage.
        if n_hs > 1 and first_zero > 0 and np.any(next_tail[:first_zero] != 0):
            hs.tracking_ch = -1
            hs.buf_len = 0
            return start + first_zero, True, False

        # Buffer non-zero portion only
        space = s.max_buffer_samples - hs.buf_len
        n_copy = min(first_zero, space)
        if n_copy > 0:
            hs.buffer[hs.buf_len : hs.buf_len + n_copy] = tail[:n_copy]
            hs.buf_len += n_copy

        # Buffer full → complete regardless
        if hs.buf_len >= s.max_buffer_samples:
            return start + first_zero, True, self._complete_channel(hs)

        # Tracked channel went to zero — determine what happened
        if first_zero < len(tail):
            # 1. Check next channel (expected sequential handoff)
            remainder_next = next_tail[first_zero:]
            if np.any(remainder_next != 0):
                term_pos = first_zero + int(np.argmax(remainder_next != 0))
                consumed = start + term_pos
                if hs.buf_len >= s.fft_samples:
                    return consumed, True, self._complete_channel(hs)
                hs.tracking_ch = -1
                hs.buf_len = 0
                return consumed, True, False

            # 2. Next channel not active — check if ANY headstage channel is
            #    (sequence break: file wrap, channel skip, etc.)
            hs_remainder = data[pos + start + first_zero :, hs.ch_start : hs.ch_end]
            if np.any(hs_remainder != 0):
                consumed = start + first_zero
                updated = False
                if hs.buf_len >= s.fft_samples:
                    updated = self._complete_channel(hs)
                hs.tracking_ch = -1  # force re-scan to re-lock sequence
                hs.buf_len = 0
                return consumed, True, updated

            # 3. No channels active — gap, consume rest and wait
            return start + len(tail), False, False

        # Burst continues to end of chunk
        return start + len(tail), False, False

    # --- Per-headstage processing ---

    def _process_headstage(
        self,
        data: np.ndarray,
        n_time: int,
        hs: _HeadstageTracker,
    ) -> bool:
        any_updated = False
        pos = 0

        while pos < n_time:
            if hs.tracking_ch == -1:
                _scan_for_active(data, pos, hs)
                if hs.tracking_ch == -1:
                    break

            consumed, done, updated = self._buffer_channel(data, pos, hs)
            any_updated |= updated
            pos += consumed
            if not done:
                break  # chunk ended mid-burst, continue next call

        return any_updated

    # --- Main entry point ---

    def _process(self, message: AxisArray) -> AxisArray | None:
        s = self.state
        data = message.data
        n_time = data.shape[0]
        if n_time == 0:
            return None

        # Capture if ch_axis changed for later emission even if no update
        incoming_ch = message.axes.get("ch")
        ch_axis_changed = incoming_ch is not None and incoming_ch is not s.ch_axis
        if ch_axis_changed:
            s.ch_axis = incoming_ch

        any_updated = False
        for hs in s.trackers:
            any_updated |= self._process_headstage(data, n_time, hs)

        if any_updated or ch_axis_changed:
            time_ix = message.get_axis_idx("time")
            new_time_ax = replace(
                message.axes["time"], offset=message.axes["time"].value(message.data.shape[time_ix] - 1)
            )
            return replace(
                message,
                data=s.impedance.copy()[None, :],
                axes={**message.axes, "time": new_time_ax},
            )
        return None


class CerePlexImpedance(
    BaseTransformerUnit[
        CerePlexImpedanceSettings,
        AxisArray,
        AxisArray,
        CerePlexImpedanceProcessor,
    ]
):
    SETTINGS = CerePlexImpedanceSettings

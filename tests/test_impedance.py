"""Unit tests for CerePlexImpedanceProcessor using synthetic data."""

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.blackrock.cereplex_impedance import (
    CerePlexImpedanceProcessor,
    CerePlexImpedanceSettings,
    extract_impedance,
)

FS = 30_000.0
N_CH = 4
FREQ_HZ = 1000.0
TEST_CURRENT_NA = 1.0
BURST_SAMPLES = int(0.1 * FS)  # 100 ms per channel


def _make_axis_array(data: np.ndarray, fs: float = FS, offset: float = 0.0) -> AxisArray:
    """Build an AxisArray with time and ch axes."""
    return AxisArray(
        data,
        dims=["time", "ch"],
        axes={"time": AxisArray.TimeAxis(fs, offset=offset)},
    )


def _sine_burst(n_samples: int, freq: float, amplitude_uv: float, fs: float = FS) -> np.ndarray:
    """Generate a sine burst at the given frequency and amplitude."""
    t = np.arange(n_samples) / fs
    return amplitude_uv * np.sin(2 * np.pi * freq * t)


class TestExtractImpedance:
    """Tests for the standalone extract_impedance function."""

    def test_known_impedance(self):
        """A 100 kOhm channel: 1 nA test current -> 100 uV peak-to-peak sine."""
        expected_kohm = 100.0
        p2p_uv = expected_kohm * TEST_CURRENT_NA  # 100 uV p2p
        amplitude = p2p_uv / 2.0  # 50 uV amplitude
        fft_samples = int(0.09 * FS)
        signal = _sine_burst(fft_samples, FREQ_HZ, amplitude)

        result = extract_impedance(signal, fft_samples, FS, 960.0, 1050.0, TEST_CURRENT_NA)
        assert result is not None
        assert result == pytest.approx(expected_kohm, rel=0.05)

    def test_short_data_returns_none(self):
        fft_samples = int(0.09 * FS)
        signal = np.zeros(fft_samples - 1)
        result = extract_impedance(signal, fft_samples, FS, 960.0, 1050.0, TEST_CURRENT_NA)
        assert result is None


class TestExtractImpedanceRobustness:
    """Off-bin tones, non-integer cycles, and white noise.

    The existing ``test_known_impedance`` uses ``fft_duration_s=0.09`` which
    happens to give exactly 90 cycles of 1 kHz at 30 kHz fs (bin-aligned and
    cycle-aligned — the most favourable case). The production default is
    ``0.09227`` which puts 1 kHz at bin 92.27 with 92.27 cycles, exercising
    the band-summed Hann FFT path. These tests verify the algorithm's
    accuracy under those (and worse) conditions.
    """

    # Production defaults — 1 kHz lands at bin 92.27 (NOT bin-aligned), with
    # 92.27 cycles per window (NOT cycle-aligned).
    FFT_DURATION_S = 0.09227
    FFT_SAMPLES = int(FFT_DURATION_S * FS)  # 2768
    FREQ_LO = 960.0
    FREQ_HI = 1050.0

    @pytest.mark.parametrize(
        "freq_hz",
        [
            980.0,  # off-bin, 90 cycles/window
            1000.0,  # production tone, off-bin (bin 92.27)
            1003.71,  # arbitrary off-bin
            1010.0,  # off-bin, near upper half of band
            1020.0,  # main lobe still inside the mask
        ],
    )
    def test_off_bin_tone_recovery(self, freq_hz):
        """A pure tone at any frequency well inside the mask should be
        recovered to within ~1% by the band-summed Hann estimator —
        regardless of where it falls relative to FFT bins."""
        expected_kohm = 220.0
        amp_uv = expected_kohm * TEST_CURRENT_NA / 2.0
        signal = _sine_burst(self.FFT_SAMPLES, freq_hz, amp_uv)
        result = extract_impedance(signal, self.FFT_SAMPLES, FS, self.FREQ_LO, self.FREQ_HI, TEST_CURRENT_NA)
        assert result is not None
        assert result == pytest.approx(
            expected_kohm, rel=0.01
        ), f"freq={freq_hz} Hz: expected {expected_kohm} kOhm, got {result:.3f}"

    @pytest.mark.parametrize(
        "fft_samples,freq_hz,label",
        [
            (2700, 1000.0, "bin-aligned, integer cycles (favourable)"),
            (2701, 1000.0, "+1 sample → off bin, off cycle"),
            (2730, 1000.0, "91 integer cycles, off bin"),
            (2768, 1000.0, "production default, off bin AND off cycle"),
            (3001, 1000.0, "off bin, off cycle"),
            (2700, 1003.7, "bin width = 11.11 Hz, tone off bin"),
        ],
    )
    def test_non_integer_cycles(self, fft_samples, freq_hz, label):
        """Verifies that fft_samples not being a multiple of one cycle of
        the test tone does not under-report the amplitude. Hann main lobe
        spans ±2 bins, so as long as the band brackets it, Parseval recovers
        the energy regardless of cycle alignment."""
        expected_kohm = 220.0
        amp_uv = expected_kohm * TEST_CURRENT_NA / 2.0
        signal = _sine_burst(fft_samples, freq_hz, amp_uv)
        result = extract_impedance(signal, fft_samples, FS, self.FREQ_LO, self.FREQ_HI, TEST_CURRENT_NA)
        assert result is not None
        assert result == pytest.approx(
            expected_kohm, rel=0.02
        ), f"{label}: N={fft_samples}, f={freq_hz} → got {result:.3f} kOhm"

    def test_220_kohm_off_bin_no_underestimate(self):
        """Specifically reproduces the user's reported 220 kOhm test rig
        condition under production fft window settings, with the tone
        intentionally off-bin. The algorithm itself should NOT explain a
        220 → 200 kOhm under-read."""
        expected_kohm = 220.0
        amp_uv = expected_kohm * TEST_CURRENT_NA / 2.0
        signal = _sine_burst(self.FFT_SAMPLES, 1000.0, amp_uv)
        result = extract_impedance(signal, self.FFT_SAMPLES, FS, self.FREQ_LO, self.FREQ_HI, TEST_CURRENT_NA)
        assert result is not None
        # If the algorithm caused a 9% underestimate this would land at 200 kOhm.
        # Tight tolerance: anything beyond ~1% needs investigation.
        assert result == pytest.approx(expected_kohm, rel=0.01)
        assert result > 217.0, (
            f"Got {result:.2f} kOhm — algorithm underestimates by more than 1.5%, "
            "which would partially explain the 220 → 200 reading."
        )

    @pytest.mark.parametrize("noise_rms_uv", [1.0, 5.0, 20.0])
    def test_white_noise_biases_upward_not_downward(self, noise_rms_uv):
        """White noise integrated in [freq_lo, freq_hi] adds power to the
        sum-of-squares estimate, so it raises the impedance estimate.
        White noise can NOT explain a systematic underestimate."""
        expected_kohm = 220.0
        amp_uv = expected_kohm * TEST_CURRENT_NA / 2.0
        rng = np.random.default_rng(42)

        # Average over trials so single-realisation variance doesn't drown
        # the (small, upward) mean bias.
        n_trials = 200
        results = []
        for _ in range(n_trials):
            sig = _sine_burst(self.FFT_SAMPLES, FREQ_HZ, amp_uv)
            sig = sig + rng.normal(0, noise_rms_uv, self.FFT_SAMPLES)
            r = extract_impedance(sig, self.FFT_SAMPLES, FS, self.FREQ_LO, self.FREQ_HI, TEST_CURRENT_NA)
            assert r is not None
            results.append(r)
        mean_kohm = float(np.mean(results))

        # The estimator's mean is >= true value (with equality at zero noise).
        # Allow a small tolerance (~0.2% of expected) for finite-sample variance.
        assert mean_kohm >= expected_kohm - 0.5, (
            f"σ={noise_rms_uv} uV: noise produced a DOWNWARD bias "
            f"(mean {mean_kohm:.3f} kOhm < {expected_kohm}) — unexpected."
        )
        # Sanity upper bound: noise contribution << signal at these levels.
        assert mean_kohm < expected_kohm * 1.05

    def test_white_noise_only_yields_small_floor(self):
        """With pure white noise (no tone), the estimator returns a small
        positive 'noise floor' value — documents the lower bound the user
        might see on a disconnected channel."""
        rng = np.random.default_rng(0)
        noise_rms_uv = 5.0
        n_trials = 500
        results = []
        for _ in range(n_trials):
            sig = rng.normal(0, noise_rms_uv, self.FFT_SAMPLES)
            r = extract_impedance(sig, self.FFT_SAMPLES, FS, self.FREQ_LO, self.FREQ_HI, TEST_CURRENT_NA)
            if r is not None:
                results.append(r)

        # Theoretical: A_est ≈ 2σ·sqrt(K/N) where K≈8 mask bins, N=2768.
        # → ≈ 2·5·sqrt(8/2768) ≈ 0.54 uV ⇒ ≈ 1.07 kOhm (p2p / 1 nA).
        mean_kohm = float(np.mean(results))
        assert 0.3 < mean_kohm < 3.0, (
            f"Mean noise-floor reading {mean_kohm:.3f} kOhm outside expected " "0.3–3 kOhm window for σ=5 uV."
        )


class TestCerePlexImpedanceProcessor:
    """Tests for the full processor with synthetic sequential impedance sweeps."""

    def _build_sweep(self, impedances_kohm: list[float], n_ch: int = N_CH) -> np.ndarray:
        """Build a time x ch array simulating one full sequential sweep.

        Each channel gets a 100 ms burst of a 1 kHz sine whose amplitude
        encodes the desired impedance, while all other channels read zero.
        A short wrap-around burst on ch0 is appended so the processor can
        detect the handoff from the last channel.
        """
        wrap_samples = int(0.01 * FS)  # 10 ms of ch0 to trigger last-channel completion
        total_samples = BURST_SAMPLES * n_ch + wrap_samples
        data = np.zeros((total_samples, n_ch), dtype=np.float64)
        for ch, z_kohm in enumerate(impedances_kohm):
            p2p = z_kohm * TEST_CURRENT_NA
            amplitude = p2p / 2.0
            start = ch * BURST_SAMPLES
            end = start + BURST_SAMPLES
            data[start:end, ch] = _sine_burst(BURST_SAMPLES, FREQ_HZ, amplitude)
        # Wrap-around: ch0 starts again so the last channel's handoff is detected
        wrap_start = BURST_SAMPLES * n_ch
        data[wrap_start:, 0] = _sine_burst(wrap_samples, FREQ_HZ, impedances_kohm[0] * TEST_CURRENT_NA / 2.0)
        return data

    def test_single_sweep(self):
        """Process one full sweep and verify all channels are measured."""
        impedances = [50.0, 100.0, 200.0, 150.0]
        data = self._build_sweep(impedances)

        proc = CerePlexImpedanceProcessor(
            settings=CerePlexImpedanceSettings(
                headstage_channel_offsets=(0,),
            )
        )

        # Feed in chunks to simulate streaming
        chunk_size = int(0.01 * FS)  # 10 ms chunks
        results = []
        offset = 0.0
        for i in range(0, data.shape[0], chunk_size):
            chunk = data[i : i + chunk_size]
            msg = _make_axis_array(chunk, offset=offset)
            out = proc(msg)
            if out is not None:
                results.append(out.data.copy())
            offset += chunk_size / FS

        assert len(results) > 0, "No impedance updates produced"
        final = results[-1].flatten()
        for ch, expected in enumerate(impedances):
            assert not np.isnan(final[ch]), f"Channel {ch} was not measured"
            assert final[ch] == pytest.approx(
                expected, rel=0.1
            ), f"Channel {ch}: expected {expected} kOhm, got {final[ch]:.1f}"

    def test_two_headstages(self):
        """Two headstages measured independently."""
        n_ch = 4
        imp_hs0 = [80.0, 120.0]
        imp_hs1 = [200.0, 60.0]

        # Build sweep: each headstage cycles through its own channels
        # Add wrap-around so the last channel in each headstage completes
        wrap_samples = int(0.01 * FS)
        total = BURST_SAMPLES * 2 + wrap_samples
        data = np.zeros((total, n_ch), dtype=np.float64)
        # Headstage 0: channels 0-1, then wrap to 0
        for i, z in enumerate(imp_hs0):
            start = i * BURST_SAMPLES
            end = start + BURST_SAMPLES
            data[start:end, i] = _sine_burst(BURST_SAMPLES, FREQ_HZ, z * TEST_CURRENT_NA / 2.0)
        data[BURST_SAMPLES * 2 :, 0] = _sine_burst(wrap_samples, FREQ_HZ, 40.0)
        # Headstage 1: channels 2-3, then wrap to 2
        for i, z in enumerate(imp_hs1):
            start = i * BURST_SAMPLES
            end = start + BURST_SAMPLES
            data[start:end, 2 + i] = _sine_burst(BURST_SAMPLES, FREQ_HZ, z * TEST_CURRENT_NA / 2.0)
        data[BURST_SAMPLES * 2 :, 2] = _sine_burst(wrap_samples, FREQ_HZ, 100.0)

        proc = CerePlexImpedanceProcessor(
            settings=CerePlexImpedanceSettings(
                headstage_channel_offsets=(0, 2),
            )
        )

        chunk_size = int(0.01 * FS)
        results = []
        offset = 0.0
        for i in range(0, data.shape[0], chunk_size):
            chunk = data[i : i + chunk_size]
            msg = _make_axis_array(chunk, offset=offset)
            out = proc(msg)
            if out is not None:
                results.append(out.data.copy())
            offset += chunk_size / FS

        assert len(results) > 0
        final = results[-1].flatten()
        expected = imp_hs0 + imp_hs1
        for ch, exp in enumerate(expected):
            assert not np.isnan(final[ch]), f"Channel {ch} unmeasured"
            assert final[ch] == pytest.approx(exp, rel=0.1)

    def test_offsets_only_update_preserves_impedance(self):
        """Changing only headstage_channel_offsets must not clear measured values."""
        impedances = [50.0, 100.0, 200.0, 150.0]
        data = self._build_sweep(impedances)
        proc = CerePlexImpedanceProcessor(settings=CerePlexImpedanceSettings(headstage_channel_offsets=(0,)))
        chunk_size = int(0.01 * FS)
        offset = 0.0
        for i in range(0, data.shape[0], chunk_size):
            proc(_make_axis_array(data[i : i + chunk_size], offset=offset))
            offset += chunk_size / FS

        before = proc.state.impedance.copy()
        assert not np.isnan(before).any()

        proc.update_settings(CerePlexImpedanceSettings(headstage_channel_offsets=(0, 2)))

        # Trackers re-laid out, but the accumulated impedance array survives.
        assert proc._hash != -1, "offsets-only change must not arm a full reset"
        np.testing.assert_array_equal(proc.state.impedance, before)
        assert len(proc.state.trackers) == 2
        assert proc.state.trackers[0].ch_start == 0 and proc.state.trackers[0].ch_end == 2
        assert proc.state.trackers[1].ch_start == 2 and proc.state.trackers[1].ch_end == N_CH

    def test_non_safe_field_arms_reset(self):
        """A change to a field outside NONRESET_SETTINGS_FIELDS must queue a reset."""
        proc = CerePlexImpedanceProcessor(settings=CerePlexImpedanceSettings())
        # Warm up so _hash is set.
        chunk = np.zeros((int(0.01 * FS), N_CH))
        proc(_make_axis_array(chunk))
        assert proc._hash != -1

        # collect_duration_s is consumed in _reset_state -> requires reset.
        proc.update_settings(CerePlexImpedanceSettings(collect_duration_s=0.2))
        assert proc._hash == -1

    def test_live_safe_field_does_not_reset(self):
        """freq_lo/freq_hi/test_current_nA are read live; updating them must not reset."""
        proc = CerePlexImpedanceProcessor(settings=CerePlexImpedanceSettings())
        chunk = np.zeros((int(0.01 * FS), N_CH))
        proc(_make_axis_array(chunk))
        assert proc._hash != -1

        proc.update_settings(CerePlexImpedanceSettings(freq_lo=900.0, freq_hi=1100.0, test_current_nA=2.0))
        assert proc._hash != -1
        assert proc.settings.test_current_nA == 2.0

    def test_unmeasured_channels_are_nan(self):
        """Channels that haven't been swept should remain NaN."""
        # Activate channel 0, then hand off to channel 1 briefly so ch0 completes,
        # but don't give ch1 a full burst.
        wrap_samples = int(0.01 * FS)
        data = np.zeros((BURST_SAMPLES + wrap_samples, 2), dtype=np.float64)
        data[:BURST_SAMPLES, 0] = _sine_burst(BURST_SAMPLES, FREQ_HZ, 50.0)
        # Brief ch1 activity triggers ch0 completion but isn't enough for a measurement
        data[BURST_SAMPLES:, 1] = _sine_burst(wrap_samples, FREQ_HZ, 50.0)

        proc = CerePlexImpedanceProcessor(
            settings=CerePlexImpedanceSettings(
                headstage_channel_offsets=(0,),
            )
        )

        chunk_size = int(0.01 * FS)
        offset = 0.0
        for i in range(0, data.shape[0], chunk_size):
            chunk = data[i : i + chunk_size]
            proc(_make_axis_array(chunk, offset=offset))
            offset += chunk_size / FS

        imp = proc.state.impedance
        assert imp is not None
        assert not np.isnan(imp[0]), "Channel 0 should be measured"
        assert np.isnan(imp[1]), "Channel 1 should still be NaN (incomplete burst)"

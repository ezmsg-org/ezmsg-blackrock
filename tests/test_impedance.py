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

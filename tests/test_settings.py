"""Unit tests for CereLink settings validation (no hardware needed)."""

import pickle

import pytest
from pycbsdk import ChannelType, SampleRate

from ezmsg.blackrock.cerelink import (
    CcfConfig,
    CereLinkSignalSettings,
    CereLinkSpikeSettings,
    SliceConfig,
)


class TestSignalSettings:
    def test_subscribe_rate_required(self):
        """`SampleRate.NONE` is rejected — `subscribe_rate` must be a real rate."""
        with pytest.raises(ValueError, match="subscribe_rate is required"):
            CereLinkSignalSettings()

    def test_explicit_none_rejected(self):
        with pytest.raises(ValueError, match="subscribe_rate is required"):
            CereLinkSignalSettings(subscribe_rate=SampleRate.NONE)

    def test_real_rate_accepted(self):
        for rate in (
            SampleRate.SR_500,
            SampleRate.SR_1kHz,
            SampleRate.SR_2kHz,
            SampleRate.SR_10kHz,
            SampleRate.SR_30kHz,
            SampleRate.SR_RAW,
        ):
            s = CereLinkSignalSettings(subscribe_rate=rate)
            assert s.subscribe_rate == rate

    def test_idle_defaults(self):
        s = CereLinkSignalSettings(subscribe_rate=SampleRate.SR_2kHz)
        assert s.device_type is None
        assert s.configure is None
        assert s.cbtime is False
        assert s.microvolts is True
        assert s.cont_buffer_dur == pytest.approx(0.5)
        assert s.cmp_configs == ()


class TestSpikeSettings:
    def test_idle_defaults(self):
        s = CereLinkSpikeSettings()
        assert s.device_type is None
        assert s.configure is None
        assert s.cbtime is False
        assert s.microvolts is True
        assert s.spike_buffer_dur == pytest.approx(0.5)


class TestConfigureUnion:
    """`CereLinkSignalSettings.configure` accepts None, CcfConfig, or SliceConfig."""

    def test_none(self):
        s = CereLinkSignalSettings(subscribe_rate=SampleRate.SR_2kHz, configure=None)
        assert s.configure is None

    def test_ccf(self):
        s = CereLinkSignalSettings(
            subscribe_rate=SampleRate.SR_2kHz,
            configure=CcfConfig(path="/tmp/x.ccf"),
        )
        assert isinstance(s.configure, CcfConfig)
        assert s.configure.path == "/tmp/x.ccf"

    def test_slice_defaults(self):
        s = CereLinkSignalSettings(
            subscribe_rate=SampleRate.SR_2kHz,
            configure=SliceConfig(),
        )
        sc = s.configure
        assert isinstance(sc, SliceConfig)
        assert sc.channels is None  # all matching
        assert sc.channel_type == ChannelType.FRONTEND
        assert sc.ac_input_coupling is False
        assert sc.enable_spiking is False

    def test_slice_explicit_channels(self):
        s = CereLinkSignalSettings(
            subscribe_rate=SampleRate.SR_2kHz,
            configure=SliceConfig(channels=[1, 2, 3], ac_input_coupling=True),
        )
        assert s.configure.channels == [1, 2, 3]
        assert s.configure.ac_input_coupling is True


class TestPickleRoundtrip:
    """Settings + their nested configure types must survive pickling so they
    can flow through ezmsg's INPUT_SETTINGS message stream."""

    @pytest.mark.parametrize("cfg", [None, CcfConfig(path="/tmp/x.ccf"), SliceConfig(channels=[1, 2])])
    def test_signal_roundtrip(self, cfg):
        original = CereLinkSignalSettings(subscribe_rate=SampleRate.SR_2kHz, configure=cfg)
        restored = pickle.loads(pickle.dumps(original))
        assert restored.subscribe_rate == SampleRate.SR_2kHz
        assert restored.configure == cfg

    @pytest.mark.parametrize("cfg", [None, CcfConfig(path="/tmp/x.ccf"), SliceConfig(enable_spiking=True)])
    def test_spike_roundtrip(self, cfg):
        original = CereLinkSpikeSettings(configure=cfg)
        restored = pickle.loads(pickle.dumps(original))
        assert restored.configure == cfg

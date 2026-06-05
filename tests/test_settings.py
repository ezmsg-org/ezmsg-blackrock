"""Unit tests for CereLink settings validation (no hardware needed)."""

import pickle
from unittest.mock import MagicMock

import pytest
from pycbsdk import ChannelType, SampleRate

from ezmsg.blackrock.cerelink import (
    CcfConfig,
    CereLinkSignalProducer,
    CereLinkSignalSettings,
    CereLinkSpikeProducer,
    CereLinkSpikeSettings,
    ChannelSelection,
    SliceConfig,
)


class TestSignalSettings:
    def test_default_subscribe_rate_is_sr_raw(self):
        """Omitting ``subscribe_rate`` defaults to ``SR_RAW``."""
        assert CereLinkSignalSettings().subscribe_rate == SampleRate.SR_RAW

    def test_explicit_none_rejected(self):
        with pytest.raises(ValueError, match="SampleRate.NONE is not allowed"):
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
        assert sc.channels is ChannelSelection.ALL  # default: all matching
        assert sc.channel_type == ChannelType.FRONTEND
        assert sc.ac_input_coupling is False
        assert sc.enable_spiking is False

    def test_invalid_channels_type_rejected(self):
        """A bare string/int for channels is rejected (must be list or enum)."""
        with pytest.raises(TypeError, match="must be a list"):
            SliceConfig(channels="all")
        with pytest.raises(TypeError, match="must be a list"):
            SliceConfig(channels=3)

    def test_slice_explicit_channels(self):
        s = CereLinkSignalSettings(
            subscribe_rate=SampleRate.SR_2kHz,
            configure=SliceConfig(channels=[1, 2, 3], ac_input_coupling=True),
        )
        assert s.configure.channels == [1, 2, 3]
        assert s.configure.ac_input_coupling is True


class TestSliceConfigureChannels:
    """``CereLinkSignalProducer._apply_slice_configure`` channel selection,
    driven by a mock pycbsdk Session (no hardware)."""

    @staticmethod
    def _producer_with_session(session) -> CereLinkSignalProducer:
        # Build without ezmsg machinery; we only exercise _apply_slice_configure.
        prod = CereLinkSignalProducer.__new__(CereLinkSignalProducer)
        prod.settings = CereLinkSignalSettings(subscribe_rate=SampleRate.SR_RAW)
        state = MagicMock()
        state.session = session
        prod.state = state
        return prod

    @staticmethod
    def _session_with_enabled(enabled: list[int], total: int = 256):
        """Session where ``enabled`` channels sit in the raw group and the
        device exposes ``total`` FRONTEND channels."""
        sess = MagicMock()
        sess.get_matching_channel_ids.return_value = list(range(1, total + 1))
        sess.get_group_channels.side_effect = lambda g: (list(enabled) if g == int(SampleRate.SR_RAW) else [])
        return sess

    def test_enabled_configures_only_enabled(self):
        sess = self._session_with_enabled(list(range(1, 129)), total=256)
        prod = self._producer_with_session(sess)
        prod._apply_slice_configure(SliceConfig(channel_type=ChannelType.FRONTEND, channels=ChannelSelection.ENABLED))
        chans, ctype, rate = sess.set_sample_group.call_args.args
        assert chans == list(range(1, 129))  # only the enabled bank
        assert rate == SampleRate.SR_RAW
        # disable_others=False → the unused bank is left untouched.
        assert sess.set_sample_group.call_args.kwargs["disable_others"] is False
        # Reads the device state before configuring.
        sess.sync.assert_called_once()

    def test_enabled_unions_continuous_groups(self):
        # Enabled split across two groups → both should be configured.
        sess = MagicMock()
        sess.get_matching_channel_ids.return_value = list(range(1, 257))
        sess.get_group_channels.side_effect = lambda g: {
            int(SampleRate.SR_30kHz): [1, 2],
            int(SampleRate.SR_RAW): [5, 6],
        }.get(g, [])
        prod = self._producer_with_session(sess)
        prod._apply_slice_configure(SliceConfig(channel_type=ChannelType.FRONTEND, channels=ChannelSelection.ENABLED))
        assert sess.set_sample_group.call_args.args[0] == [1, 2, 5, 6]

    def test_enabled_excludes_non_matching_type(self):
        # A channel enabled but not FRONTEND is dropped by the type intersection.
        sess = MagicMock()
        sess.get_matching_channel_ids.return_value = [1, 2, 3]  # FRONTEND only
        sess.get_group_channels.side_effect = lambda g: ([1, 2, 3, 99] if g == int(SampleRate.SR_RAW) else [])
        prod = self._producer_with_session(sess)
        prod._apply_slice_configure(SliceConfig(channel_type=ChannelType.FRONTEND, channels=ChannelSelection.ENABLED))
        assert sess.set_sample_group.call_args.args[0] == [1, 2, 3]

    def test_enabled_empty_warns_but_does_not_raise(self, caplog):
        sess = self._session_with_enabled([], total=256)
        prod = self._producer_with_session(sess)
        prod._apply_slice_configure(SliceConfig(channel_type=ChannelType.FRONTEND, channels=ChannelSelection.ENABLED))
        assert sess.set_sample_group.call_args.args[0] == []
        assert any("no FRONTEND channels are currently enabled" in r.message for r in caplog.records)

    def test_all_configures_all_and_disables_others(self):
        sess = self._session_with_enabled(list(range(1, 129)), total=256)
        prod = self._producer_with_session(sess)
        # Default channels=ChannelSelection.ALL.
        prod._apply_slice_configure(SliceConfig(channel_type=ChannelType.FRONTEND))
        # ALL resolves to every matching channel; others get disabled.
        assert sess.set_sample_group.call_args.args[0] == list(range(1, 257))
        assert sess.set_sample_group.call_args.kwargs["disable_others"] is True
        # ALL does not inspect the enabled set.
        sess.get_group_channels.assert_not_called()

    def test_explicit_list_respected_and_disables_others(self):
        sess = self._session_with_enabled(list(range(1, 129)), total=256)
        prod = self._producer_with_session(sess)
        prod._apply_slice_configure(SliceConfig(channels=[1, 2, 3], channel_type=ChannelType.FRONTEND))
        # Explicit list passed through verbatim; others disabled; enabled set not consulted.
        assert sess.set_sample_group.call_args.args[0] == [1, 2, 3]
        assert sess.set_sample_group.call_args.kwargs["disable_others"] is True
        sess.get_group_channels.assert_not_called()


class TestSpikeSliceConfigure:
    """``CereLinkSpikeProducer`` channel selection / extraction handling,
    driven by a mock pycbsdk Session (no hardware)."""

    @staticmethod
    def _producer_with_session(session) -> CereLinkSpikeProducer:
        prod = CereLinkSpikeProducer.__new__(CereLinkSpikeProducer)
        prod.settings = CereLinkSpikeSettings()
        state = MagicMock()
        state.session = session
        prod.state = state
        return prod

    @staticmethod
    def _session_with_extraction(extracting: list[int], total: int = 8):
        """Session exposing ``total`` FRONTEND channels, of which ``extracting``
        have the SPKOPTS extract bit set."""
        sess = MagicMock()
        ids = list(range(1, total + 1))
        sess.get_matching_channel_ids.return_value = ids
        sess.get_channels_field.return_value = [1 if cid in extracting else 0 for cid in ids]
        return sess

    def test_enabled_leaves_extraction_untouched(self):
        """channels=ENABLED never calls set_spike_extraction, even with enable_spiking."""
        sess = self._session_with_extraction([2, 5])
        prod = self._producer_with_session(sess)
        prod._apply_slice_configure(SliceConfig(channels=ChannelSelection.ENABLED, enable_spiking=True))
        sess.set_spike_extraction.assert_not_called()

    def test_enabled_resolves_to_extraction_enabled_channels(self):
        """The spike-stream meaning of 'enabled' is the SPKOPTS extract bit."""
        sess = self._session_with_extraction([2, 5])
        prod = self._producer_with_session(sess)
        assert prod._enabled_channels(ChannelType.FRONTEND) == [2, 5]

    def test_enable_spiking_with_list_sets_extraction(self):
        sess = self._session_with_extraction([])
        prod = self._producer_with_session(sess)
        prod._apply_slice_configure(SliceConfig(channels=[1, 3], enable_spiking=True))
        chans, ctype, enabled = sess.set_spike_extraction.call_args.args
        assert chans == [1, 3]
        assert enabled is True


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

"""Unit tests for CereLinkSettings validation (no hardware needed)."""

import pytest
from pycbsdk import SampleRate

from ezmsg.blackrock.cerelink import CereLinkSettings


class TestSettingsValidation:
    def test_defaults(self):
        s = CereLinkSettings()
        assert s.config_chans is None
        assert s.config_rate is None
        assert s.subscribe_rate is None
        assert s.ccf_path is None

    def test_config_chans_requires_config_rate(self):
        with pytest.raises(ValueError, match="must be set together"):
            CereLinkSettings(config_chans=8)

    def test_config_rate_requires_config_chans(self):
        """config_rate alone is meaningless (use subscribe_rate to filter)."""
        with pytest.raises(ValueError, match="must be set together"):
            CereLinkSettings(config_rate=SampleRate.SR_30kHz)

    def test_subscribe_rate_alone_is_valid(self):
        """subscribe_rate alone = capture filter (no programmatic setup)."""
        s = CereLinkSettings(subscribe_rate=SampleRate.SR_30kHz)
        assert s.subscribe_rate == SampleRate.SR_30kHz
        assert s.config_chans is None
        assert s.config_rate is None

    def test_programmatic_trio_together(self):
        s = CereLinkSettings(config_chans=8, config_rate=SampleRate.SR_30kHz)
        assert s.config_chans == 8
        assert s.config_rate == SampleRate.SR_30kHz

    def test_huge_config_chans_is_valid(self):
        """A huge config_chans is a valid 'give me all available' shorthand."""
        s = CereLinkSettings(config_chans=int(1e6), config_rate=SampleRate.SR_30kHz)
        assert s.config_chans == int(1e6)

    def test_ccf_excludes_config_chans(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            CereLinkSettings(ccf_path="test.ccf", config_chans=8, config_rate=SampleRate.SR_30kHz)

    def test_ccf_with_subscribe_rate_is_valid(self):
        """CCF + subscribe_rate = CCF configures device, subscribe_rate filters capture group."""
        s = CereLinkSettings(ccf_path="test.ccf", subscribe_rate=SampleRate.SR_30kHz)
        assert s.ccf_path == "test.ccf"
        assert s.subscribe_rate == SampleRate.SR_30kHz

    def test_ccf_alone_is_valid(self):
        s = CereLinkSettings(ccf_path="test.ccf")
        assert s.ccf_path == "test.ccf"
        assert s.config_chans is None
        assert s.config_rate is None
        assert s.subscribe_rate is None

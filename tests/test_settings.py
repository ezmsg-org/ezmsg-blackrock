"""Unit tests for CereLinkSettings validation (no hardware needed)."""

import pytest
from pycbsdk import SampleRate

from ezmsg.blackrock.cerelink import CereLinkSettings


class TestSettingsValidation:
    def test_defaults(self):
        s = CereLinkSettings()
        assert s.n_chans is None
        assert s.sample_rate is None
        assert s.ccf_path is None

    def test_n_chans_requires_sample_rate(self):
        with pytest.raises(ValueError, match="sample_rate is required"):
            CereLinkSettings(n_chans=8)

    def test_sample_rate_without_n_chans_is_valid(self):
        s = CereLinkSettings(sample_rate=SampleRate.SR_30kHz)
        assert s.sample_rate == SampleRate.SR_30kHz
        assert s.n_chans is None

    def test_n_chans_and_sample_rate_together(self):
        s = CereLinkSettings(n_chans=8, sample_rate=SampleRate.SR_30kHz)
        assert s.n_chans == 8
        assert s.sample_rate == SampleRate.SR_30kHz

    def test_ccf_excludes_n_chans(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            CereLinkSettings(ccf_path="test.ccf", n_chans=8, sample_rate=SampleRate.SR_30kHz)

    def test_ccf_excludes_sample_rate(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            CereLinkSettings(ccf_path="test.ccf", sample_rate=SampleRate.SR_30kHz)

    def test_ccf_alone_is_valid(self):
        s = CereLinkSettings(ccf_path="test.ccf")
        assert s.ccf_path == "test.ccf"
        assert s.n_chans is None
        assert s.sample_rate is None

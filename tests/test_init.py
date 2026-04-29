"""Initialization tests using 256-channel data (CCF / CMP)."""

import asyncio

import pytest
from conftest import run_nplayserver
from pycbsdk import DeviceType, SampleRate

from ezmsg.blackrock.cerelink import (
    CcfConfig,
    CereLinkSignalProducer,
    CereLinkSignalSettings,
)
from ezmsg.blackrock.channel_map import ChannelMapSettings

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def nplayserver(nplayserver_binary, ns6_256_path):
    """nPlayServer playing back 256-channel data for this module."""
    with run_nplayserver(nplayserver_binary, ns6_256_path) as proc:
        yield proc


class TestInitialization:
    """`CereLinkSignalProducer` initialization tests."""

    def test_init_with_ccf_and_cmp(self, nplayserver, ccf_256_path, cmp_path):
        """Verify CCF + CMP loading caches non-zero channel positions on the
        producer's state. Rate-agnostic — we don't peek at the per-rate
        template since the CCF determines which rates are active."""
        producer = CereLinkSignalProducer(
            settings=CereLinkSignalSettings(
                device_type=DeviceType.NPLAY,
                # Any rate works; we exercise the CCF + CMP path, not the
                # per-rate buffer/template setup.
                subscribe_rate=SampleRate.SR_30kHz,
                configure=CcfConfig(path=str(ccf_256_path)),
                cmp_configs=(ChannelMapSettings(filepath=str(cmp_path)),),
            )
        )

        async def _open_and_check():
            # Drive the async open path directly — equivalent to what
            # `CereLinkSignalSource.initialize` triggers via __acall__.
            await producer._areset_state()
            assert producer.state.session is not None
            assert producer.state.session.running

            # `_cache_channel_metadata` runs after CCF + CMP load and
            # populates state.ch_positions with (col, row, bank, elec)
            # tuples per FRONTEND channel.
            positions = producer.state.ch_positions
            assert positions, "ch_positions should be populated after CCF+CMP"
            nonzero = [p for p in positions.values() if p[0] != 0 or p[1] != 0]
            assert nonzero, f"CMP positions are all zero across {len(positions)} channels"

        try:
            asyncio.run(_open_and_check())
        finally:
            producer.close()

"""Initialization tests using 256-channel data (CCF / CMP)."""

import numpy as np
import pytest
from conftest import run_nplayserver
from pycbsdk import DeviceType

from ezmsg.blackrock.cerelink import CereLinkProducer, CereLinkSettings

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def nplayserver(nplayserver_binary, ns6_256_path):
    """nPlayServer playing back 256-channel data for this module."""
    with run_nplayserver(nplayserver_binary, ns6_256_path) as proc:
        yield proc


class TestInitialization:
    """CereLinkProducer initialization tests."""

    def test_init_with_ccf_and_cmp(self, nplayserver, ccf_256_path, cmp_path):
        """Verify CereLinkProducer initializes with CCF and CMP files."""
        producer = CereLinkProducer(
            settings=CereLinkSettings(
                device_type=DeviceType.NPLAY,
                ccf_path=str(ccf_256_path),
                cmp_path=str(cmp_path),
            )
        )
        producer.open()
        try:
            assert producer._session is not None
            assert producer._session.running

            # Verify channel axis is a structured array with position data
            for buf in producer._buffers.values():
                ch_ax = buf["template"].axes["ch"]
                ch_data = ch_ax.data
                assert hasattr(ch_data.dtype, "names"), "ch axis should be a structured array"
                assert "x" in ch_data.dtype.names
                assert "y" in ch_data.dtype.names

                # CMP should have populated some non-zero positions
                xs = ch_data["x"]
                ys = ch_data["y"]
                assert np.any(xs != 0) or np.any(ys != 0), "CMP positions are all zero"
        finally:
            producer.close()

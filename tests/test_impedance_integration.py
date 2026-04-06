"""Integration test for CerePlexImpedanceProcessor against real impedance playback data."""

import asyncio
import time

import numpy as np
import pytest
from conftest import run_nplayserver
from pycbsdk import ChannelType, DeviceType, SampleRate

from ezmsg.blackrock.cerelink import CereLinkProducer, CereLinkSettings
from ezmsg.blackrock.cereplex_impedance import CerePlexImpedanceProcessor, CerePlexImpedanceSettings

pytestmark = pytest.mark.integration

# The file starts with ~5s of normal data before impedance mode.
# A full 128-channel sweep takes ~12.8 s (100 ms/channel).
# The file loops, so we also lose ~5s of non-impedance data per loop.
# 30s guarantees at least one complete sweep even in the worst alignment.
COLLECT_DURATION_S = 30.0


@pytest.fixture(scope="module")
def nplayserver(nplayserver_binary, ns6_impedance_path):
    """nPlayServer playing back impedance test data for this module."""
    with run_nplayserver(nplayserver_binary, ns6_impedance_path) as proc:
        yield proc


def test_impedance_from_playback(nplayserver):
    """Play back real impedance data, run through the processor, verify values."""
    producer = CereLinkProducer(
        settings=CereLinkSettings(
            device_type=DeviceType.NPLAY,
            n_chans=128,
            channel_type=ChannelType.FRONTEND,
            sample_rate=SampleRate.SR_RAW,
            ac_input_coupling=False,
            microvolts=True,
            cbtime=False,
        )
    )
    producer.open()

    loop = asyncio.new_event_loop()
    producer._loop = loop

    proc = CerePlexImpedanceProcessor(settings=CerePlexImpedanceSettings(headstage_channel_offsets=(0,)))

    try:
        deadline = time.monotonic() + COLLECT_DURATION_S
        while time.monotonic() < deadline:
            msg = loop.run_until_complete(producer._produce())
            proc(msg)
    finally:
        producer.close()
        loop.close()

    impedance = proc.state.impedance
    assert impedance is not None

    measured = np.flatnonzero(~np.isnan(impedance))
    # Looped playback means channels at the file boundary may not get a
    # complete burst depending on timing.  In real-time use all 128 would
    # be measured; with looped files, allow a few to be missed.
    assert len(measured) >= 120, f"Too few channels measured: {len(measured)}/128"

    vals = impedance[measured]
    assert np.all(vals >= 180), f"Some impedances too low: min={vals.min():.1f} kOhm"
    assert np.all(vals <= 260), f"Some impedances too high: max={vals.max():.1f} kOhm"
    assert np.median(vals) == pytest.approx(205, abs=40), f"Median impedance unexpected: {np.median(vals):.1f} kOhm"

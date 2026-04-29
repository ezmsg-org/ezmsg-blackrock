"""Integration test for CerePlexImpedance against real impedance playback data."""

import shutil

import ezmsg.core as ez
import numpy as np
import pytest
from conftest import read_log, run_nplayserver
from ezmsg.util.messagelogger import MessageLogger
from ezmsg.util.terminate import TerminateOnTotal
from pycbsdk import ChannelType, DeviceType, SampleRate

from ezmsg.blackrock.cerelink import (
    CereLinkSignalSettings,
    CereLinkSignalSource,
    SliceConfig,
)
from ezmsg.blackrock.cereplex_impedance import CerePlexImpedance, CerePlexImpedanceSettings

pytestmark = pytest.mark.integration

# Each channel completion in the sweep produces one impedance message
# (the processor emits ``any_updated or ch_axis_changed``).  A 128-channel
# sweep therefore yields ~128 messages.  Asking for 140 guarantees we cross
# at least one full sweep regardless of where playback started — the file
# loops, so the first partial sweep is followed by complete ones.
N_IMPEDANCE_MESSAGES = 140


@pytest.fixture(scope="module")
def nplayserver(nplayserver_binary, ns6_impedance_path, tmp_path_factory):
    """nPlayServer playing back impedance test data from an isolated dir.

    The shared test-data cache holds multiple ``.ns6`` files in the same
    directory; copying the target file into a fresh temp dir ensures
    nPlayServer only sees the one we asked for.
    """
    isolated_dir = tmp_path_factory.mktemp("impedance_playback")
    isolated_ns6 = isolated_dir / ns6_impedance_path.name
    shutil.copy(ns6_impedance_path, isolated_ns6)
    with run_nplayserver(nplayserver_binary, isolated_ns6) as proc:
        yield proc


def test_impedance_from_playback(nplayserver, tmp_path):
    """Play back real impedance data through the full graph and verify.

    Logging the impedance output (sparse, ~1 message per channel completion)
    rather than the raw signal stream (~150 k tiny messages at SR_RAW)
    keeps the message volume reasonable and lets ``TerminateOnTotal`` end
    the test once we've crossed a full sweep.
    """
    log_path = tmp_path / "impedance_log.jsonl"

    settings = CereLinkSignalSettings(
        device_type=DeviceType.NPLAY,
        subscribe_rate=SampleRate.SR_RAW,
        configure=SliceConfig(
            channels=list(range(1, 129)),
            channel_type=ChannelType.FRONTEND,
            ac_input_coupling=False,
        ),
        microvolts=True,
        cbtime=False,
    )

    comps = {
        "SRC": CereLinkSignalSource(settings),
        "IMP": CerePlexImpedance(settings=CerePlexImpedanceSettings(headstage_channel_offsets=(0,))),
        "LOG": MessageLogger(output=log_path),
        "TERM": TerminateOnTotal(total=N_IMPEDANCE_MESSAGES),
    }
    conns = (
        (comps["SRC"].OUTPUT_SIGNAL, comps["IMP"].INPUT_SIGNAL),
        (comps["IMP"].OUTPUT_SIGNAL, comps["LOG"].INPUT_MESSAGE),
        (comps["LOG"].OUTPUT_MESSAGE, comps["TERM"].INPUT_MESSAGE),
    )
    ez.run(components=comps, connections=conns)

    messages = read_log(log_path)
    assert len(messages) >= N_IMPEDANCE_MESSAGES, f"Only collected {len(messages)} impedance messages"

    # Each impedance message carries the full (1, 128) state. The most-
    # recent message holds the latest reading for every channel.
    final = messages[-1].data.flatten()
    assert final.shape == (128,), f"Unexpected shape {final.shape}"

    measured = np.flatnonzero(~np.isnan(final))
    # Looped playback means a channel near the file boundary may not get a
    # complete burst on a given lap.  In real-time use all 128 would be
    # measured every sweep; with looped files, allow a few to be missed.
    assert len(measured) >= 120, f"Too few channels measured: {len(measured)}/128"

    vals = final[measured]
    assert np.all(vals >= 180), f"Some impedances too low: min={vals.min():.1f} kOhm"
    assert np.all(vals <= 260), f"Some impedances too high: max={vals.max():.1f} kOhm"
    assert np.median(vals) == pytest.approx(205, abs=40), f"Median impedance unexpected: {np.median(vals):.1f} kOhm"

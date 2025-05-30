import json
import os
from pathlib import Path
import pytest
import time
import typing

import ezmsg.core as ez
from ezmsg.util.terminate import TerminateOnTotal
import ezmsg.util.messagelogger as ml
import ezmsg.util.messagecodec as mc
from ezmsg.util.messagecodec import MessageEncoder, LogStart
import numpy as np

from ezmsg.blackrock.nsp import NSPSource


class MyEncoder(mc.MessageEncoder):#json.JSONEncoder):
    def default(self, obj: typing.Any) -> typing.Any:
        if type(obj) is LogStart:
            return {}
        # Assume AxisArray
        return {"ts": float(obj.axes["time"].offset), "samps": obj.data.shape[0]}


# Monkey patch the log_object method to use our custom encoder.
def handle_log_object(obj: typing.Any) -> str:
    """Custom log_object function to use high-resolution timestamps."""
    return json.dumps({"ts": time.time(), "obj": obj}, cls=MyEncoder)
ml.log_object = handle_log_object

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_latency():
    test_path = Path().home() / "test_latency.log"
    comps = {
        "SOURCE": NSPSource(
            inst_addr="192.168.137.200",
            inst_port=51002,
            client_addr="",
            protocol="4.1",
            microvolts=False,
            cbtime=False,
        ),
        "LOGGER": ml.MessageLogger(output=test_path, write_period=1.0),
        "TERM": TerminateOnTotal(total=10_000),
    }
    conns = (
        (comps["SOURCE"].OUTPUT_SIGNAL, comps["LOGGER"].INPUT_MESSAGE),
        (comps["LOGGER"].OUTPUT_MESSAGE, comps["TERM"].INPUT_MESSAGE),
    )
    ez.run(components=comps, connections=conns)

    # Load log file into list of dicts
    log_data = []
    with open(test_path, "r") as f:
        for line in f:
            try:
                log_data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    test_path.unlink(missing_ok=True)
    test_path.unlink(missing_ok=True)

    # Create data array; sub_time, dev_time, samp_count
    data = np.array([[_["ts"], _["obj"]["ts"], _["obj"]["samps"]] for _ in log_data if "obj" in _ and "ts" in _["obj"]])

    sub_time = data[:, 0]  # Time at the subscriber (PC clock)
    dev_time = data[:, 1]  # Estimated time (PC clock) when the first sample in the chunk left the device.
    samp_count = data[:, 2]  # Number of samples in this chunk.
    expected_time = dev_time + samp_count / 30_000
    latency = sub_time - expected_time

    try:
        skip_ix = 3_000
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size': 18})
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.title("Device -> Subscriber Latency")
        plt.hist(latency[skip_ix:] * 1000, bins=100, density=True)
        plt.xlabel("(msec)")
        plt.subplot(1, 2, 2)
        plt.title("Frames Per Chunk\n(256 ch)")
        plt.hist(samp_count[skip_ix:], bins=100, density=True)
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Matplotlib not available, skipping plot.")

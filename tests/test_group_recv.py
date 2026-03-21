import json
import os
import time
import typing
from pathlib import Path

import ezmsg.core as ez
import ezmsg.util.messagecodec as mc
import ezmsg.util.messagelogger as ml
import numpy as np
import pytest
from ezmsg.util.messagecodec import LogStart
from ezmsg.util.terminate import TerminateOnTotal
from pycbsdk import DeviceType

from ezmsg.blackrock.cerelink import CereLinkSettings, CereLinkSource


class MyEncoder(mc.MessageEncoder):
    def default(self, obj: typing.Any) -> typing.Any:
        if type(obj) is LogStart:
            return {}
        # Assume AxisArray
        return {"ts": float(obj.axes["time"].offset), "samps": obj.data.shape[0]}


# Monkey patch the log_object method to use our custom encoder.
def handle_log_object(obj: typing.Any) -> str:
    return json.dumps({"ts": time.monotonic(), "obj": obj}, cls=MyEncoder)


ml.log_object = handle_log_object

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_latency():
    test_path = Path().home() / "test_latency.log"
    comps = {
        "SOURCE": CereLinkSource(
            CereLinkSettings(
                device_type=DeviceType.HUB1,
                microvolts=False,
                cbtime=False,
            )
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

    # Create data array; sub_time, dev_time, samp_count
    data = np.array([[_["ts"], _["obj"]["ts"], _["obj"]["samps"]] for _ in log_data if "obj" in _ and "ts" in _["obj"]])

    sub_time = data[:, 0]
    dev_time = data[:, 1]
    samp_count = data[:, 2]
    expected_time = dev_time + samp_count / 30_000
    latency = sub_time - expected_time

    skip_ix = 3_000
    lat_mean = np.mean(latency[skip_ix:] * 1000)
    lat_std = np.std(latency[skip_ix:] * 1000)
    count_mean = np.mean(samp_count[skip_ix:])
    count_std = np.std(samp_count[skip_ix:])
    print(f"Latency mean: {lat_mean:.2f} +- {lat_std:.2f} msec")
    print(f"Samples per chunk mean: {count_mean:.2f} +- {count_std:.2f} samples")

    try:
        import matplotlib.pyplot as plt

        plt.rcParams.update({"font.size": 18})
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.title("Device -> Subscriber Latency")
        plt.hist(latency[skip_ix:] * 1000, bins=100, density=True)
        plt.xlabel(f"(msec; mean={lat_mean:.2f})")
        plt.axvline(lat_mean, linestyle="-", color="black")
        plt.axvline(lat_mean + lat_std, linestyle="--", color="gray")
        plt.axvline(lat_mean - lat_std, linestyle="--", color="gray")
        plt.xlim([-1.0, 3.0])
        plt.subplot(1, 2, 2)
        plt.title("Frames Per Chunk\n(256 ch)")
        plt.hist(samp_count[skip_ix:], bins=100, density=True)
        plt.axvline(count_mean, linestyle="-", color="black")
        plt.axvline(count_mean + count_std, linestyle="--", color="gray")
        plt.axvline(count_mean - count_std, linestyle="--", color="gray")
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Matplotlib not available, skipping plot.")

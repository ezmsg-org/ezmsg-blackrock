"""Benchmark device-to-subscriber latency for CereLinkSignalSource.

Requires a running Blackrock device or nPlayServer.
Collects latency statistics and optionally plots a histogram.

Usage:
    python examples/latency_bench.py [--device-type NPLAY] [--n-messages 10000]
"""

import json
import tempfile
import time
import typing
from pathlib import Path

import ezmsg.core as ez
import ezmsg.util.messagecodec as mc
import ezmsg.util.messagelogger as ml
import numpy as np
import typer
from ezmsg.util.messagecodec import LogStart
from ezmsg.util.terminate import TerminateOnTotal
from pycbsdk import DeviceType, SampleRate
from typing_extensions import Annotated

from ezmsg.blackrock.cerelink import CereLinkSignalSettings, CereLinkSignalSource, SliceConfig


class _LatencyEncoder(mc.MessageEncoder):
    def default(self, obj: typing.Any) -> typing.Any:
        if type(obj) is LogStart:
            return {}
        return {"ts": float(obj.axes["time"].offset), "samps": obj.data.shape[0]}


def _log_object(obj: typing.Any) -> str:
    return json.dumps({"ts": time.monotonic(), "obj": obj}, cls=_LatencyEncoder)


def main(
    device_type: Annotated[
        str,
        typer.Option(help="Device type: NPLAY, HUB1, HUB2, HUB3, NSP."),
    ] = "NPLAY",
    n_messages: Annotated[
        int,
        typer.Option(help="Number of messages to collect."),
    ] = 10_000,
    skip: Annotated[
        int,
        typer.Option(help="Messages to skip at the start (warm-up)."),
    ] = 3_000,
):
    original_log_object = ml.log_object
    ml.log_object = _log_object

    log_path = Path(tempfile.mkdtemp()) / "latency.log"

    try:
        comps = {
            "SOURCE": CereLinkSignalSource(
                CereLinkSignalSettings(
                    device_type=DeviceType[device_type.upper()],
                    subscribe_rate=SampleRate.SR_30kHz,
                    configure=SliceConfig(),  # all FRONTEND channels
                    microvolts=False,
                    cbtime=False,
                )
            ),
            "LOGGER": ml.MessageLogger(output=log_path, write_period=1.0),
            "TERM": TerminateOnTotal(total=n_messages),
        }
        conns = (
            (comps["SOURCE"].OUTPUT_SIGNAL, comps["LOGGER"].INPUT_MESSAGE),
            (comps["LOGGER"].OUTPUT_MESSAGE, comps["TERM"].INPUT_MESSAGE),
        )
        ez.run(components=comps, connections=conns)

        # Parse log
        log_data = []
        with open(log_path) as f:
            for line in f:
                try:
                    log_data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        data = np.array(
            [[d["ts"], d["obj"]["ts"], d["obj"]["samps"]] for d in log_data if "obj" in d and "ts" in d["obj"]]
        )

        sub_time = data[:, 0]
        dev_time = data[:, 1]
        samp_count = data[:, 2]
        expected_time = dev_time + samp_count / 30_000
        latency = sub_time - expected_time

        lat_mean = np.mean(latency[skip:] * 1000)
        lat_std = np.std(latency[skip:] * 1000)
        count_mean = np.mean(samp_count[skip:])
        count_std = np.std(samp_count[skip:])
        print(f"Latency mean: {lat_mean:.2f} +/- {lat_std:.2f} ms")
        print(f"Samples per chunk: {count_mean:.2f} +/- {count_std:.2f}")

        try:
            import matplotlib.pyplot as plt

            plt.rcParams.update({"font.size": 18})
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

            ax1.set_title("Device -> Subscriber Latency")
            ax1.hist(latency[skip:] * 1000, bins=100, density=True)
            ax1.set_xlabel(f"(ms; mean={lat_mean:.2f})")
            ax1.axvline(lat_mean, linestyle="-", color="black")
            ax1.axvline(lat_mean + lat_std, linestyle="--", color="gray")
            ax1.axvline(lat_mean - lat_std, linestyle="--", color="gray")
            ax1.set_xlim([-1.0, 3.0])

            ax2.set_title("Samples Per Chunk")
            ax2.hist(samp_count[skip:], bins=100, density=True)
            ax2.axvline(count_mean, linestyle="-", color="black")
            ax2.axvline(count_mean + count_std, linestyle="--", color="gray")
            ax2.axvline(count_mean - count_std, linestyle="--", color="gray")

            plt.tight_layout()
            plt.show()
        except ImportError:
            print("Install matplotlib for plots.")
    finally:
        log_path.unlink(missing_ok=True)
        log_path.parent.rmdir()
        ml.log_object = original_log_object


if __name__ == "__main__":
    typer.run(main)

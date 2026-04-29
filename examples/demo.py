import ezmsg.core as ez
import typer
from ezmsg.util.debuglog import DebugLog
from pycbsdk import DeviceType, SampleRate
from typing_extensions import Annotated

from ezmsg.blackrock.cerelink import (
    CcfConfig,
    CereLinkSignalSettings,
    CereLinkSignalSource,
    CereLinkSpikeSettings,
    CereLinkSpikeSource,
    SliceConfig,
)


def main(
    device_type: Annotated[
        str,
        typer.Option(help="Device type: NPLAY, HUB1, HUB2, HUB3, NSP, LEGACY_NSP, CUSTOM."),
    ] = "NPLAY",
    subscribe_rate: Annotated[
        str,
        typer.Option(help="Sample-group rate (SR_500, SR_1kHz, SR_2kHz, SR_10kHz, SR_30kHz, SR_RAW)."),
    ] = "SR_30kHz",
    cbtime: Annotated[
        bool,
        typer.Option(help="Use device timestamps (True) or time.monotonic() (False)."),
    ] = True,
    microvolts: Annotated[
        bool,
        typer.Option(help="Convert continuous data to microvolts (True) or keep raw (False)."),
    ] = True,
    cont_buffer_dur: Annotated[
        float,
        typer.Option(help="Duration of continuous data ring buffer in seconds."),
    ] = 0.5,
    ccf_path: Annotated[
        str,
        typer.Option(help="Path to CCF file to load (or empty for programmatic configure)."),
    ] = "",
):
    # CCF mode: one source loads the CCF (device-wide); the other subscribes only.
    # Programmatic mode: each source's SliceConfig configures its own slice.
    device = DeviceType[device_type.upper()]
    rate = SampleRate[subscribe_rate]

    if ccf_path:
        signal_configure = CcfConfig(path=ccf_path)
        spike_configure = None  # CCF already loaded by the signal source
    else:
        signal_configure = SliceConfig()  # all FRONTEND, default channel_type
        spike_configure = SliceConfig(enable_spiking=True)

    signal_settings = CereLinkSignalSettings(
        device_type=device,
        subscribe_rate=rate,
        configure=signal_configure,
        cbtime=cbtime,
        microvolts=microvolts,
        cont_buffer_dur=cont_buffer_dur,
    )
    spike_settings = CereLinkSpikeSettings(
        device_type=device,
        configure=spike_configure,
        cbtime=cbtime,
    )

    comps = {
        "SIG_SRC": CereLinkSignalSource(signal_settings),
        "SPK_SRC": CereLinkSpikeSource(spike_settings),
        "SIGLOG": DebugLog(name="SIGNAL"),
        "SPKLOG": DebugLog(name="SPIKE"),
    }

    conns = (
        (comps["SIG_SRC"].OUTPUT_SIGNAL, comps["SIGLOG"].INPUT),
        (comps["SPK_SRC"].OUTPUT_SIGNAL, comps["SPKLOG"].INPUT),
    )

    ez.run(components=comps, connections=conns)


if __name__ == "__main__":
    typer.run(main)

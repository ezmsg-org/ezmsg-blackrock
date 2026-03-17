import ezmsg.core as ez
import typer
from ezmsg.util.debuglog import DebugLog
from pycbsdk import DeviceType
from typing_extensions import Annotated

from ezmsg.blackrock.cerelink import CereLinkSettings, CereLinkSource


def main(
    device_type: Annotated[
        str,
        typer.Option(help="Device type: NPLAY, HUB1, HUB2, HUB3, NSP, LEGACY_NSP, CUSTOM."),
    ] = "NPLAY",
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
        typer.Option(help="Path to CCF file to load (or empty to skip)."),
    ] = "",
):
    settings = CereLinkSettings(
        device_type=DeviceType[device_type.upper()],
        cbtime=cbtime,
        microvolts=microvolts,
        cont_buffer_dur=cont_buffer_dur,
        ccf_path=ccf_path or None,
    )

    comps = {
        "SRC": CereLinkSource(settings),
        "SIGLOG": DebugLog(name="SIGNAL"),
        "SPKLOG": DebugLog(name="SPIKE"),
    }

    conns = (
        (comps["SRC"].OUTPUT_SIGNAL, comps["SIGLOG"].INPUT),
        (comps["SRC"].OUTPUT_SPIKE, comps["SPKLOG"].INPUT),
    )

    ez.run(components=comps, connections=conns)


if __name__ == "__main__":
    typer.run(main)

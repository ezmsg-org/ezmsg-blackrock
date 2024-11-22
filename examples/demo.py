import sys

import ezmsg.core as ez
from ezmsg.util.debuglog import DebugLog
from ezmsg.util.messages.key import FilterOnKey
import typer
from typing_extensions import Annotated

from ezmsg.blackrock.nsp import NSPSource, NSPSourceSettings


def main(
    inst_addr: Annotated[
        str,
        typer.Option(
            help="ipv4 address of device. pycbsdk will send control packets to this address. Subnet OK. Use 127.0.0.1 "
            "for use with nPlayServer (non-bcast). The default is 0.0.0.0 (IPADDR_ANY) on Mac and Linux. On "
            "Windows, known IPs will be searched."
        ),
    ] = "192.168.137.128",
    inst_port: Annotated[
        int,
        typer.Option(
            help="Network port to send control packets. Use 51002 for Gemini and 51001 for Legacy NSP."
        ),
    ] = 51001,
    client_addr: Annotated[
        str,
        typer.Option(
            help="ipv4 address of this machine's network adapter we will receive packets on. Defaults to INADDR_ANY. "
            "If address is provided, assumes Cerebus Subnet."
        ),
    ] = "",
    client_port: Annotated[
        int,
        typer.Option(
            help="Network port to receive packets. This should always be 51002."
        ),
    ] = 51002,
    recv_bufsize: Annotated[int, typer.Option(help="UDP socket recv buffer size.")] = (
        8 if sys.platform == "win32" else 6
    )
    * 1024
    * 1024,
    protocol: Annotated[
        str, typer.Option(help="Protocol Version. 3.11, 4.0, or 4.1 supported.")
    ] = "3.11",
    cont_buffer_dur: Annotated[
        float,
        typer.Option(
            help="Duration of buffer for continuous data. Note: buffer may occupy ~15 MB / second."
        ),
    ] = 0.5,
    microvolts: Annotated[
        bool,
        typer.Option(
            help="Convert continuous data to microvolts (True) or keep raw integers (False)."
        ),
    ] = True,
    cbtime: Annotated[
        bool,
        typer.Option(
            help="Use Cerebus time for continuous data (True) or local time.time (False)."
        ),
    ] = True,
):
    source_settings = NSPSourceSettings(
        inst_addr,
        inst_port,
        client_addr,
        client_port,
        recv_bufsize,
        protocol,
        cont_buffer_dur,
        microvolts,
        cbtime,
    )

    comps = {
        "SRC": NSPSource(source_settings),
        "NS5": FilterOnKey(key="ns5"),
        # TODO: SparseResample(fs=1000.0, max_age=0.005),
        # TODO: EventRates(),
        "SPKLOG": DebugLog(name="NEV"),
        "NS5LOG": DebugLog(name="NS5"),
    }

    conns = (
        (comps["SRC"].OUTPUT_SPIKE, comps["SPKLOG"].INPUT),
        (comps["SRC"].OUTPUT_SIGNAL, comps["NS5LOG"].INPUT),
    )

    ez.run(components=comps, connections=conns)


if __name__ == "__main__":
    typer.run(main)

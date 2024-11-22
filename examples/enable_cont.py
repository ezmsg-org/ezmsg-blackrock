import sys
import time
from typing_extensions import Annotated

from pycbsdk import cbsdk
from pycbsdk.cbhw.packet.common import CBChannelType
import typer


def main(
    smp_group: int,
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
):
    params = cbsdk.create_params(
        inst_addr=inst_addr,
        inst_port=inst_port,
        client_addr=client_addr,
        client_port=client_port,
        recv_bufsize=recv_bufsize,
        protocol=protocol,
    )
    device = cbsdk.NSPDevice(params)
    run_level = device.connect(startup_sequence=False)
    if not run_level:
        raise ValueError(f"Failed to connect to NSP; {params=}")
    config = cbsdk.get_config(device, force_refresh=True)
    for chid in [
        k
        for k, v in config["channel_infos"].items()
        if config["channel_types"][k]
        in (CBChannelType.FrontEnd, CBChannelType.AnalogIn)
    ]:
        _ = cbsdk.set_channel_config(device, chid, "smpgroup", smp_group)
    # Refresh config
    time.sleep(0.5)  # Make sure all the config packets have returned.


if __name__ == "__main__":
    typer.run(main)

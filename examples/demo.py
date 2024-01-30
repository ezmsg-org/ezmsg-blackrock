import sys

import ezmsg.core as ez

from ezmsg.blackrock.nsp import NSPSource, NSPSourceSettings
from ezmsg.util.debuglog import DebugLog

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Consume data from NSP")

    parser.add_argument(
        "--inst_addr",
        "-i",
        type=str,
        default="192.168.137.128",
        help="ipv4 address of device. pycbsdk will send control packets to this address. Subnet OK. "
             "Use 127.0.0.1 for use with nPlayServer (non-bcast). "
             "The default is 0.0.0.0 (IPADDR_ANY) on Mac and Linux. On Windows, known IPs will be searched.",
    )

    parser.add_argument(
        "--inst_port",
        type=int,
        default=51001,
        help="Network port to send control packets."
             "Use 51002 for Gemini and 51001 for Legacy NSP.",
    )

    parser.add_argument(
        "--client_addr",
        "-c",
        type=str,
        default="",
        help="ipv4 address of this machine's network adapter we will receive packets on. "
             "Defaults to INADDR_ANY. If address is provided, assumes Cerebus Subnet.",
    )

    parser.add_argument(
        "--client_port",
        "-p",
        type=int,
        default=51002,
        help="Network port to receive packets. This should always be 51002.",
    )

    parser.add_argument(
        "--recv_bufsize",
        "-b",
        type=int,
        help=f"UDP socket recv buffer size. "
             f"Default: {(8 if sys.platform == 'win32' else 6) * 1024 * 1024}.",
    )

    parser.add_argument(
        "--protocol",
        type=str,
        default="3.11",
        help="Protocol Version. 3.11, 4.0, or 4.1 supported.",
    )

    parser.add_argument(
        "--cont_smp_group",
        type=int,
        default=0,
        help="Continuous data Sampling Group (1-6) to publish. Set to 0 to ignore continuous data."
    )

    parser.add_argument(
        "--cont_buffer_dur",
        type=float,
        default=0.5,
        help="Duration of buffer for continuous data. Note: buffer may occupy ~15 MB / second."
    )

    parser.add_argument(
        "--cont_override_config_all",
        action="store_true",
        help="Set this flag to set all analog channels to cont_smp_group (group 0 will disable continuous data)."
    )

    source = NSPSource(NSPSourceSettings(**vars(parser.parse_args())))
    log = DebugLog()

    ez.run(
        SOURCE=source,
        LOG=log,
        connections=(
            (source.OUTPUT_SPIKE, log.INPUT),
        )
    )

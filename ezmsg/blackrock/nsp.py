import typing

import numpy as np
import ezmsg.core as ez

from pycbsdk import cbsdk

from ezmsg.util.messages.axisarray import AxisArray

class NSPSourceSettings(ez.Settings):
    inst_addr: str = ""
    inst_port: int = 51001
    client_addr: str = ""
    client_port: int = 51002
    recv_bufsize: typing.Optional[int] = None
    protocol: str = "3.11"

class NSPSourceState(ez.State):
    device: cbsdk.NSPDevice

class NSPSource(ez.Unit):
    SETTINGS: NSPSourceSettings
    STATE: NSPSourceState
    
    OUTPUT = ez.OutputStream(AxisArray)

    async def initialize(self) -> None:
        
        self.STATE.device = cbsdk.NSPDevice(
            cbsdk.Params(
                inst_addr = self.SETTINGS.inst_addr, 
                inst_port = self.SETTINGS.inst_port, 
                client_addr = self.SETTINGS.client_addr, 
                client_port = self.SETTINGS.client_port, 
                recv_bufsize = self.SETTINGS.recv_bufsize, 
                protocol = self.SETTINGS.protocol
            )
        )
        self.STATE.device.connect(startup_sequence = False)

    @ez.publisher(OUTPUT)
    async def continuous(self) -> typing.AsyncGenerator:
        msg = AxisArray(np.ones(5), dims = ['time'])
        yield self.OUTPUT, msg


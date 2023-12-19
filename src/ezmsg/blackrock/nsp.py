from dataclasses import dataclass
import asyncio
import typing

import numpy as np
from pycbsdk import cbsdk

import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray


# TODO: This probably needs to be part of a larger spike signal processing library
@dataclass
class SpikeEvent:
    channel: str
    timestamp: float
    id: int

class NSPSourceSettings(ez.Settings):
    inst_addr: str = ""
    inst_port: int = 51001
    client_addr: str = ""
    client_port: int = 51002
    recv_bufsize: typing.Optional[int] = None
    protocol: str = "3.11"

class NSPSourceState(ez.State):
    device: cbsdk.NSPDevice
    spike_queue: asyncio.Queue[SpikeEvent]

class NSPSource(ez.Unit):
    SETTINGS: NSPSourceSettings
    STATE: NSPSourceState
    
    OUTPUT_SPIKE = ez.OutputStream(SpikeEvent)

    async def initialize(self) -> None:

        self.STATE.spike_queue = asyncio.Queue()
        
        params = cbsdk.create_params(**vars(self.SETTINGS))
        self.STATE.device = cbsdk.NSPDevice(params)
        run_level = self.STATE.device.connect(startup_sequence = False)

        if not run_level:
            raise ValueError(f"Failed to connect to NSP; {params=}")

        _ = cbsdk.register_spk_callback(self.STATE.device, self.on_spike) # type: ignore

    def shutdown(self) -> None:
        self.STATE.device.disconnect()

    def on_spike(self, spk_pkt):
        self.STATE.spike_queue.put_nowait(
            SpikeEvent(
                channel = spk_pkt.header.chid, 
                timestamp = spk_pkt.header.time, 
                id = 0
            )
        )

    @ez.publisher(OUTPUT_SPIKE)
    async def spikes(self) -> typing.AsyncGenerator:
        while True:
            spike_event = await self.STATE.spike_queue.get()
            yield self.OUTPUT_SPIKE, spike_event

from dataclasses import dataclass, replace
import asyncio
import typing
from ctypes import Structure

import numpy as np
import numpy.typing as npt
from pycbsdk import cbsdk

import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray


grp_fs = {1: 500, 2: 1_000, 3: 2_000, 4: 10_000, 5: 30_000, 6: 30_000}


# TODO: This probably needs to be part of a larger spike signal processing library
@dataclass
class SpikeEvent:
    channel: int  # 0-based index of channel. (pkt.header.chid - 1)
    offset: float  # offset in time (seconds). Might be device time, might be system time.
    id: int  # Unit id. 0=unsorted, 1-5 sorted, 255=noise


class NSPSourceSettings(ez.Settings):
    inst_addr: str = ""
    inst_port: int = 51001
    client_addr: str = ""
    client_port: int = 51002
    recv_bufsize: typing.Optional[int] = None
    protocol: str = "3.11"
    cont_smp_group: typing.Optional[int] = None  # ID of sample group to stream, from 1-6 or None to ignore cont. data.
    cont_buffer_dur: float = 0.5  # Duration of continuous buffer to hold recv packets. Up to ~15 MB / second.
    cont_override_config_all: bool = False  # If True, change all found channels' sample group to cont_smp_group
    # TODO: convert_uV: bool = False  # Returned values are converted to uV (True) or stay raw integers (False)


class NSPSourceState(ez.State):
    device: cbsdk.NSPDevice
    spike_queue: asyncio.Queue[SpikeEvent]
    cont_buffer: typing.Optional[typing.Tuple[npt.NDArray, npt.NDArray]]
    cont_read_idx: int
    cont_write_idx: int
    template_cont: AxisArray
    sysfreq: int


class NSPSource(ez.Unit):
    SETTINGS: NSPSourceSettings
    STATE: NSPSourceState

    OUTPUT_SPIKE = ez.OutputStream(SpikeEvent)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    async def initialize(self) -> None:

        self.STATE.spike_queue = asyncio.Queue()
        params = cbsdk.create_params(
            inst_addr=self.SETTINGS.inst_addr,
            inst_port=self.SETTINGS.inst_port,
            client_addr=self.SETTINGS.client_addr,
            client_port=self.SETTINGS.client_port,
            recv_bufsize=self.SETTINGS.recv_bufsize,
            protocol=self.SETTINGS.protocol
        )
        self.STATE.device = cbsdk.NSPDevice(params)
        run_level = self.STATE.device.connect(startup_sequence=False)
        if not run_level:
            raise ValueError(f"Failed to connect to NSP; {params=}")
        config = cbsdk.get_config(self.STATE.device, force_refresh=True)
        self.STATE.sysfreq = config["sysfreq"]

        if self.SETTINGS.cont_override_config_all:
            from pycbsdk.cbhw.packet.common import CBChannelType
            for chid in [
                k for k, v in config["channel_infos"].items()
                if config["channel_types"][k] in (CBChannelType.FrontEnd, CBChannelType.AnalogIn)
            ]:
                _ = cbsdk.set_channel_config(self.STATE.device, chid, "smpgroup", self.SETTINGS.cont_smp_group)
            # Refresh config
            await asyncio.sleep(0.5)  # Make sure all the config packets have returned.
            config = cbsdk.get_config(self.STATE.device, force_refresh=False)

        _ = cbsdk.register_spk_callback(self.STATE.device, self.on_spike)
        if self.SETTINGS.cont_smp_group is not None and (1 <= self.SETTINGS.cont_smp_group <= 6):
            n_channels = len(config["group_infos"][self.SETTINGS.cont_smp_group])
            self._reset_buffer(n_channels)
            _ = cbsdk.register_group_callback(
                self.STATE.device,
                self.SETTINGS.cont_smp_group,
                self.on_smp_group
            )

    def _reset_buffer(self, n_channels) -> None:
        buff_samples = int(self.SETTINGS.cont_buffer_dur * grp_fs[self.SETTINGS.cont_smp_group])
        self.STATE.cont_buffer = (
            np.zeros((buff_samples,), dtype=int),
            np.zeros((buff_samples, n_channels), dtype=np.int16)
        )
        self.STATE.cont_read_idx = 0
        self.STATE.cont_write_idx = 0
        time_ax = AxisArray.Axis.TimeAxis(grp_fs[self.SETTINGS.cont_smp_group], offset=0.0)
        self.STATE.template_cont = AxisArray(np.zeros((0, 0)), dims=["time", "ch"], axes={"time": time_ax})

    def shutdown(self) -> None:
        self.STATE.device.disconnect()

    def on_smp_group(self, pkt: Structure):
        if len(pkt.data) != self.STATE.cont_buffer[1].shape[1]:
            self._reset_buffer(len(pkt.data))
        self.STATE.cont_buffer[1][self.STATE.cont_write_idx, :] = memoryview(pkt.data)
        self.STATE.cont_buffer[0][self.STATE.cont_write_idx] = pkt.header.time
        self.STATE.cont_write_idx = (self.STATE.cont_write_idx + 1) % len(self.STATE.cont_buffer[0])

    @ez.publisher(OUTPUT_SIGNAL)
    async def pub_cont(self) -> typing.AsyncGenerator:
        while True:
            buff_len = len(self.STATE.cont_buffer[0])
            read_term = self.STATE.cont_write_idx if self.STATE.cont_write_idx >= self.STATE.cont_read_idx else buff_len
            if self.STATE.cont_read_idx == read_term:
                await asyncio.sleep(0.001)
                continue
            read_slice = slice(self.STATE.cont_read_idx, min(buff_len, read_term))
            read_view = self.STATE.cont_buffer[1][read_slice]
            new_offset: float = self.STATE.cont_buffer[0][self.STATE.cont_read_idx] / self.STATE.sysfreq
            new_time_ax = replace(self.STATE.template_cont.axes["time"], offset=new_offset)
            out_msg = replace(
                self.STATE.template_cont,
                data=read_view.copy(),  # TODO: Scale to uV. Needs per-channel scale factor.
                axes={**self.STATE.template_cont.axes, **{"time": new_time_ax}}
            )
            self.STATE.cont_read_idx = read_term % buff_len
            yield self.OUTPUT_SPIKE, out_msg

    def on_spike(self, spk_pkt: Structure):
        self.STATE.spike_queue.put_nowait(
            SpikeEvent(
                channel=spk_pkt.header.chid - 1,
                offset=spk_pkt.header.time / self.STATE.sysfreq,
                id=spk_pkt.unit
            )
        )

    @ez.publisher(OUTPUT_SPIKE)
    async def spikes(self) -> typing.AsyncGenerator:
        while True:
            spike_event = await self.STATE.spike_queue.get()
            yield self.OUTPUT_SPIKE, spike_event

import asyncio
from ctypes import Structure
import functools
import typing

import numpy as np
from pycbsdk import cbsdk
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray, replace
from ezmsg.event.message import EventMessage


grp_fs = {1: 500, 2: 1_000, 3: 2_000, 4: 10_000, 5: 30_000, 6: 30_000}


class NSPSourceSettings(ez.Settings):
    inst_addr: str = ""
    inst_port: int = 51001
    client_addr: str = ""
    client_port: int = 51002
    recv_bufsize: typing.Optional[int] = None
    protocol: str = "3.11"

    cont_buffer_dur: float = 0.5
    """Duration of continuous buffer to hold recv packets. Up to ~15 MB / second."""

    # TODO: convert_uV: bool = False  # Returned values are converted to uV (True) or stay raw integers (False)


class NSPSourceState(ez.State):
    device: cbsdk.NSPDevice
    spike_queue: asyncio.Queue[EventMessage]
    cont_buffer = {
        _: (np.array([], dtype=int), np.array([[]], dtype=np.int16))
        for _ in range(1, 7)
    }
    cont_read_idx = {_: 0 for _ in range(1, 7)}
    cont_write_idx = {_: 0 for _ in range(1, 7)}
    template_cont = {
        _: AxisArray(data=np.array([[]]), dims=["time", "ch"]) for _ in range(1, 7)
    }
    sysfreq: int = 30_000  # Default for pre-Gemini system


class NSPSource(ez.Unit):
    SETTINGS = NSPSourceSettings
    STATE = NSPSourceState

    OUTPUT_SPIKE = ez.OutputStream(EventMessage)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    async def initialize(self) -> None:
        self.STATE.spike_queue = asyncio.Queue()
        params = cbsdk.create_params(
            inst_addr=self.SETTINGS.inst_addr,
            inst_port=self.SETTINGS.inst_port,
            client_addr=self.SETTINGS.client_addr,
            client_port=self.SETTINGS.client_port,
            recv_bufsize=self.SETTINGS.recv_bufsize,
            protocol=self.SETTINGS.protocol,
        )
        self.STATE.device = cbsdk.NSPDevice(params)
        run_level = self.STATE.device.connect(startup_sequence=False)
        if not run_level:
            raise ConnectionError(f"Failed to connect to NSP; {params=}")
        config = cbsdk.get_config(self.STATE.device, force_refresh=True)
        self.STATE.sysfreq = config["sysfreq"]

        _ = cbsdk.register_spk_callback(self.STATE.device, self.on_spike)

        for grp_idx in range(1, 7):
            n_channels = len(config["group_infos"][grp_idx])
            self._reset_buffer(grp_idx, n_channels)
            _ = cbsdk.register_group_callback(
                self.STATE.device,
                grp_idx,
                functools.partial(self.on_smp_group, grp_idx=grp_idx),
            )

    def _reset_buffer(self, grp_idx: int, n_channels: int) -> None:
        buff_samples = int(self.SETTINGS.cont_buffer_dur * grp_fs[grp_idx])
        self.STATE.cont_buffer[grp_idx] = (
            np.zeros((buff_samples,), dtype=int),
            np.zeros((buff_samples, n_channels), dtype=np.int16),
        )
        self.STATE.cont_read_idx[grp_idx] = 0
        self.STATE.cont_write_idx[grp_idx] = 0
        time_ax = AxisArray.Axis.TimeAxis(grp_fs[grp_idx], offset=0.0)
        self.STATE.template_cont[grp_idx] = AxisArray(
            np.zeros((0, 0)),
            dims=["time", "ch"],
            axes={"time": time_ax},  # TODO: Ch CoordinateAxis
            key=f"SMP{grp_idx}" if grp_idx < 6 else "RAW",
        )

    def shutdown(self) -> None:
        if hasattr(self.STATE, "device") and self.STATE.device is not None:
            self.STATE.device.disconnect()

    def on_smp_group(self, pkt: Structure, grp_idx: int = 5):
        _buffer = self.STATE.cont_buffer[grp_idx]
        _write_idx = self.STATE.cont_write_idx[grp_idx]
        if len(pkt.data) != _buffer[1].shape[1]:
            self._reset_buffer(len(pkt.data))
        _buffer[1][_write_idx, :] = memoryview(pkt.data)
        _buffer[0][_write_idx] = pkt.header.time
        self.STATE.cont_write_idx[grp_idx] = (_write_idx + 1) % len(_buffer[0])

    @ez.publisher(OUTPUT_SIGNAL)
    async def pub_cont(self) -> typing.AsyncGenerator:
        while True:
            b_any = False
            for grp_idx in range(1, 7):
                _buff = self.STATE.cont_buffer[grp_idx]
                _read_idx = self.STATE.cont_read_idx[grp_idx]
                _write_idx = self.STATE.cont_write_idx[grp_idx]
                buff_len = len(_buff[0])
                read_term = _write_idx if _write_idx >= _read_idx else buff_len
                if _read_idx == read_term:
                    continue
                else:
                    b_any = True
                read_slice = slice(_read_idx, min(buff_len, read_term))
                read_view = _buff[1][read_slice]
                new_offset: float = _buff[0][_read_idx] / self.STATE.sysfreq
                _templ = self.STATE.template_cont[grp_idx]
                new_time_ax = replace(_templ.axes["time"], offset=new_offset)
                out_msg = replace(
                    _templ,
                    data=read_view.copy(),  # TODO: Scale to uV. Needs per-channel scale factor.
                    axes={**_templ.axes, **{"time": new_time_ax}},
                    key=f"SMP{grp_idx}" if grp_idx < 6 else "RAW",
                )
                self.STATE.cont_read_idx[grp_idx] = read_term % buff_len
                yield self.OUTPUT_SIGNAL, out_msg
            if not b_any:
                await asyncio.sleep(0.001)

    def on_spike(self, spk_pkt: Structure):
        self.STATE.spike_queue.put_nowait(
            EventMessage(
                offset=spk_pkt.header.time / self.STATE.sysfreq,
                ch_idx=spk_pkt.header.chid - 1,
                sub_idx=min(spk_pkt.unit, 6),  # 0=unsorted, 1-5 sorted unit, >5=noise
                value=1,
            )
        )

    @ez.publisher(OUTPUT_SPIKE)
    async def spikes(self) -> typing.AsyncGenerator:
        while True:
            spike_event = await self.STATE.spike_queue.get()
            yield self.OUTPUT_SPIKE, spike_event

import asyncio
from ctypes import Structure
import functools
import typing

import numpy as np
from pycbsdk import cbsdk, cbhw
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray, replace
from ezmsg.event.message import EventMessage

from .util import ClockSync


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

    microvolts: bool = True
    """Convert continuous data to uV (True) or keep raw integers (False)."""

    cbtime: bool = True
    """
    Use Cerebus time for continuous data (True) or local time.time (False).
    Note that time.time is delayed by the network transmission latency relative to Cerebus time.
    """


class NSPSourceState(ez.State):
    device: cbsdk.NSPDevice
    spike_queue: asyncio.Queue[EventMessage]
    cont_buffer = {
        _: (
            np.array([], dtype=int),
            np.array([[]], dtype=np.int16),
        )
        for _ in range(1, 7)
    }
    cont_read_idx = {_: 0 for _ in range(1, 7)}
    cont_write_idx = {_: 0 for _ in range(1, 7)}
    template_cont = {
        _: AxisArray(data=np.array([[]]), dims=["time", "ch"]) for _ in range(1, 7)
    }
    scale_cont = {_: np.array([]) for _ in range(1, 7)}
    sysfreq: int = 30_000  # Default for pre-Gemini system
    n_channels: int = 0


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
        self.STATE.sysfreq = 1e9 if config["b_gemini"] else config["sysfreq"]

        self._clock_sync = ClockSync(alpha=0.1, sysfreq=self.STATE.sysfreq)
        monitor_state = self.STATE.device.get_monitor_state()
        while monitor_state["pkts_received"] < 1:
            await asyncio.sleep(0.1)
            monitor_state = self.STATE.device.get_monitor_state()
        self._clock_sync.add_pair(monitor_state["time"], monitor_state["sys_time"])

        _ = cbsdk.register_spk_callback(self.STATE.device, self.on_spike)

        for grp_idx in range(1, 7):
            self._reset_buffer(grp_idx)
            _ = cbsdk.register_group_callback(
                self.STATE.device,
                grp_idx,
                functools.partial(self.on_smp_group, grp_idx=grp_idx),
            )

    def _reset_buffer(self, grp_idx: int) -> None:
        config: dict = self.STATE.device.config
        chanset = config["group_infos"][grp_idx]
        buff_samples = int(self.SETTINGS.cont_buffer_dur * grp_fs[grp_idx])
        self.STATE.n_channels = len(chanset)
        self.STATE.cont_buffer[grp_idx] = (
            np.zeros((buff_samples,), dtype=int),
            np.zeros((buff_samples, self.STATE.n_channels), dtype=np.int16),
        )
        self.STATE.cont_read_idx[grp_idx] = 0
        self.STATE.cont_write_idx[grp_idx] = 0
        time_ax = AxisArray.TimeAxis(grp_fs[grp_idx], offset=0.0)

        chan_labels = []
        scale_factors = []
        for ch_idx in chanset:
            pkt: cbhw.packet.packets.CBPacketChanInfo = config["channel_infos"][ch_idx]
            chan_labels.append(pkt.label.decode("utf-8"))
            scale_fac = (pkt.scalin.anamax - pkt.scalin.anamin) / (
                pkt.scalin.digmax - pkt.scalin.digmin
            )
            if pkt.scalin.anaunit.decode("utf-8") == "mV":
                scale_fac /= 1000
            scale_factors.append(scale_fac)

        ch_ax = AxisArray.CoordinateAxis(
            data=np.array(chan_labels), dims=["ch"], unit="label"
        )
        self.STATE.template_cont[grp_idx] = AxisArray(
            np.zeros((0, 0)),
            dims=["time", "ch"],
            axes={"time": time_ax, "ch": ch_ax},
            key=f"ns{grp_idx}",
            attrs={"unit": "uV" if self.SETTINGS.microvolts else "raw"},
        )

        self.STATE.scale_cont[grp_idx] = np.array(scale_factors)

    def shutdown(self) -> None:
        if hasattr(self.STATE, "device") and self.STATE.device is not None:
            self.STATE.device.disconnect()

    def on_smp_group(self, pkt: Structure, grp_idx: int = 5):
        _buffer = self.STATE.cont_buffer[grp_idx]
        _write_idx = self.STATE.cont_write_idx[grp_idx]
        if self.STATE.n_channels != len(pkt.data):
            self._reset_buffer(grp_idx)
        _buffer[1][_write_idx, :] = memoryview(pkt.data[:self.STATE.n_channels])
        _buffer[0][_write_idx] = pkt.header.time
        self.STATE.cont_write_idx[grp_idx] = (_write_idx + 1) % len(_buffer[0])

    @ez.task
    async def update_clock(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            if self.STATE.device is not None:
                monitor_state = self.STATE.device.get_monitor_state()
                self._clock_sync.add_pair(
                    monitor_state["time"], monitor_state["sys_time"]
                )

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
                out_dat = _buff[1][read_slice].copy()
                if self.SETTINGS.microvolts:
                    out_dat = out_dat * self.STATE.scale_cont[grp_idx][None, :]
                if self.SETTINGS.cbtime:
                    new_offset: float = _buff[0][_read_idx] / self.STATE.sysfreq
                else:
                    new_offset: float = self._clock_sync.nsp2system(_buff[0][_read_idx])
                _templ = self.STATE.template_cont[grp_idx]
                new_time_ax = replace(_templ.axes["time"], offset=new_offset)
                out_msg = replace(
                    _templ,
                    data=out_dat,
                    axes={**_templ.axes, **{"time": new_time_ax}},
                )
                self.STATE.cont_read_idx[grp_idx] = read_term % buff_len
                yield self.OUTPUT_SIGNAL, out_msg
            if not b_any:
                await asyncio.sleep(0.001)

    def on_spike(self, spk_pkt: Structure):
        self.STATE.spike_queue.put_nowait(
            EventMessage(
                offset=spk_pkt.header.time / self.STATE.sysfreq
                if self.SETTINGS.cbtime
                else self._clock_sync.nsp2system(spk_pkt.header.time),
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

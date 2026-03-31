"""CereLink-based source for ezmsg — streams continuous and spike data from Blackrock devices."""

from __future__ import annotations

import asyncio
import logging
import time
import typing

import ezmsg.core as ez
import numpy as np
from ezmsg.baseproc.processor import BaseProducer
from ezmsg.baseproc.units import BaseProducerUnit
from ezmsg.event.message import EventMessage
from ezmsg.util.messages.axisarray import AxisArray, replace
from pycbsdk import ChannelType, DeviceType, SampleRate, Session

logger = logging.getLogger(__name__)

CHANNEL_DTYPE = np.dtype(
    [
        ("x", "f4"),
        ("y", "f4"),
        ("label", "U16"),
        ("bank", "U1"),
        ("elec", "i4"),
    ]
)


class CereLinkSettings(ez.Settings):
    device_type: DeviceType = DeviceType.NPLAY
    """Device type to connect to."""

    cbtime: bool = True
    """True = raw device nanoseconds, False = time.monotonic() via clock sync."""

    microvolts: bool = True
    """Convert int16 → µV using channel scale factors (when available)."""

    cont_buffer_dur: float = 0.5
    """Ring buffer duration in seconds per sample rate group."""

    ccf_path: str | None = None
    """CCF file to load after connection (or None to skip).
    Mutually exclusive with n_chans/sample_rate."""

    cmp_path: str | None = None
    """CMP file for electrode positions (or None to skip). Bank offset is always 0."""

    n_chans: int | None = None
    """Number of channels to enable (used with channel_type and sample_rate).
    Mutually exclusive with ccf_path."""

    channel_type: ChannelType = ChannelType.FRONTEND
    """Channel type filter for programmatic setup."""

    sample_rate: SampleRate | None = None
    """Sample rate for programmatic setup (e.g. SampleRate.SR_30kHz).
    Mutually exclusive with ccf_path."""

    def __post_init__(self):
        has_manual = self.n_chans is not None or self.sample_rate is not None
        if self.ccf_path is not None and has_manual:
            raise ValueError("ccf_path and n_chans/sample_rate are mutually exclusive")
        if (self.n_chans is None) != (self.sample_rate is None):
            raise ValueError("n_chans and sample_rate must both be set or both be None")


class CereLinkProducer(BaseProducer[CereLinkSettings, AxisArray]):
    """Owns the pycbsdk Session, ring buffers, and spike queue."""

    def __init__(self, *args, settings: CereLinkSettings | None = None, **kwargs):
        super().__init__(*args, settings=settings, **kwargs)
        self._session: Session | None = None
        self._buffers: dict[int, dict] = {}  # rate (int) → buffer info
        self.spike_queue: asyncio.Queue[EventMessage] = asyncio.Queue()
        self._data_event = asyncio.Event()
        self._active_rates: list[int] = []
        self._round_robin_idx: int = 0
        self._loop: asyncio.AbstractEventLoop | None = None

    # -- Lifecycle --

    def open(self) -> None:
        """Create and start the Session, configure channels, register callbacks."""
        self._session = Session(device_type=self.settings.device_type)
        self._session.__enter__()

        deadline = time.monotonic() + 10.0
        while not self._session.running:
            if time.monotonic() > deadline:
                self._session.__exit__(None, None, None)
                self._session = None
                raise TimeoutError("Session did not start within 10 seconds")
            time.sleep(0.1)

        if self.settings.ccf_path:
            self._session.load_ccf(self.settings.ccf_path)
        elif self.settings.n_chans is not None and self.settings.sample_rate is not None:
            self._session.set_channel_sample_group(
                self.settings.n_chans,
                self.settings.channel_type,
                self.settings.sample_rate,
                disable_others=True,
            )

        if self.settings.cmp_path:
            self._session.load_channel_map(self.settings.cmp_path, 0)

        # Pre-fetch channel positions (available after CCF/CMP loading)
        all_ids = self._session.get_matching_channel_ids(ChannelType.FRONTEND)
        all_pos = self._session.get_channels_positions(ChannelType.FRONTEND)
        self._ch_positions: dict[int, tuple] = dict(zip(all_ids, all_pos))

        # Discover active groups and set up ring buffers + callbacks
        for rate in SampleRate:
            if rate == SampleRate.NONE:
                continue
            channels = self._session.get_group_channels(int(rate))
            if not channels:
                continue
            self._setup_group(rate, channels)

        # Spike callback
        @self._session.on_event(ChannelType.FRONTEND)
        def _on_spike(header, data):
            self._handle_spike(header)

    def close(self) -> None:
        """Tear down the Session."""
        if self._session is not None:
            self._session.__exit__(None, None, None)
            self._session = None

    # -- Group setup --

    def _setup_group(self, rate: SampleRate, channels: list[int]) -> None:
        rate_int = int(rate)
        n_ch = len(channels)
        fs = rate.hz
        buff_samples = max(1, int(self.settings.cont_buffer_dur * fs))

        ch_info = np.zeros(n_ch, dtype=CHANNEL_DTYPE)
        scale_factors = []
        for i, ch_id in enumerate(channels):
            label = self._session.get_channel_label(ch_id)
            ch_info[i]["label"] = label or f"ch{ch_id}"

            pos = self._ch_positions.get(ch_id, (0, 0, 0, 0))
            ch_info[i]["x"] = pos[0]
            ch_info[i]["y"] = pos[1]
            ch_info[i]["bank"] = chr(ord("A") + pos[2] - 1) if pos[2] > 0 else ""
            ch_info[i]["elec"] = pos[3]

            scaling = self._session.get_channel_scaling(ch_id)
            if scaling and scaling["digmax"] != scaling["digmin"]:
                sf = (scaling["anamax"] - scaling["anamin"]) / (scaling["digmax"] - scaling["digmin"])
                if scaling["anaunit"] == "mV":
                    sf *= 1000  # mV → µV
                scale_factors.append(sf)
            else:
                scale_factors.append(1.0)

        time_ax = AxisArray.TimeAxis(fs, offset=0.0)
        ch_ax = AxisArray.CoordinateAxis(data=ch_info, dims=["ch"], unit="struct")
        template = AxisArray(
            np.zeros((0, 0)),
            dims=["time", "ch"],
            axes={"time": time_ax, "ch": ch_ax},
            key=rate.name,
            attrs={"unit": "uV" if self.settings.microvolts else "raw"},
        )

        buf = {
            "timestamps": np.zeros(buff_samples, dtype=np.uint64),
            "data": np.zeros((buff_samples, n_ch), dtype=np.int16),
            "write_idx": 0,
            "read_idx": 0,
            "n_channels": n_ch,
            "template": template,
            "scale_factors": np.array(scale_factors, dtype=np.float64),
        }

        self._buffers[rate_int] = buf
        self._active_rates.append(rate_int)

        @self._session.on_group_batch(rate)
        def _on_group_batch(samples, timestamps, _buf=buf):
            self._handle_group_batch(samples, timestamps, _buf)

    # -- Callbacks (run in C/receive thread) --

    def _handle_group_batch(self, samples: np.ndarray, timestamps: np.ndarray, buf: dict) -> None:
        n_ch = buf["n_channels"]
        if samples.shape[1] > n_ch:
            samples = samples[:, :n_ch]  # trim dword-padding columns
        w = buf["write_idx"]
        n = len(timestamps)
        buff_len = len(buf["timestamps"])
        end = w + n
        if end <= buff_len:
            buf["data"][w:end, :] = samples
            buf["timestamps"][w:end] = timestamps
        else:
            first = buff_len - w
            buf["data"][w:buff_len, :] = samples[:first]
            buf["timestamps"][w:buff_len] = timestamps[:first]
            rest = n - first
            buf["data"][:rest, :] = samples[first:]
            buf["timestamps"][:rest] = timestamps[first:]
        buf["write_idx"] = end % buff_len
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._data_event.set)
        else:
            self._data_event.set()

    def _handle_spike(self, header) -> None:
        if self.settings.cbtime:
            offset = header.time / 1e9
        else:
            try:
                offset = self._session.device_to_monotonic(header.time)
            except RuntimeError:
                offset = time.monotonic()

        self.spike_queue.put_nowait(
            EventMessage(
                offset=offset,
                ch_idx=header.chid - 1,
                sub_idx=min(header.type, 6),  # 0=unsorted, 1-5 sorted, >5=noise
                value=1,
            )
        )

    # -- Data production (async, called by BaseProducerUnit loop) --

    async def _produce(self) -> AxisArray:
        if self._loop is None:
            self._loop = asyncio.get_running_loop()

        while True:
            if not self._active_rates:
                await asyncio.sleep(0.1)
                continue

            for _ in range(len(self._active_rates)):
                idx = self._round_robin_idx
                self._round_robin_idx = (idx + 1) % len(self._active_rates)
                rate_int = self._active_rates[idx]
                buf = self._buffers[rate_int]

                read_idx = buf["read_idx"]
                write_idx = buf["write_idx"]
                buff_len = len(buf["timestamps"])

                # Read up to the write pointer (or end of buffer if wrapped)
                read_term = write_idx if write_idx >= read_idx else buff_len
                if read_idx == read_term:
                    continue

                read_slice = slice(read_idx, read_term)
                out_dat = buf["data"][read_slice].copy()

                if self.settings.microvolts:
                    out_dat = out_dat * buf["scale_factors"][None, :]

                first_ts = int(buf["timestamps"][read_idx])
                if self.settings.cbtime:
                    new_offset = first_ts / 1e9
                else:
                    try:
                        new_offset = self._session.device_to_monotonic(first_ts)
                    except RuntimeError:
                        new_offset = time.monotonic()

                template = buf["template"]
                new_time_ax = replace(template.axes["time"], offset=new_offset)
                result = replace(
                    template,
                    data=out_dat,
                    axes={**template.axes, "time": new_time_ax},
                )
                buf["read_idx"] = read_term % buff_len
                return result

            # No data in any group — wait for a callback
            self._data_event.clear()
            await self._data_event.wait()


class CereLinkSource(BaseProducerUnit[CereLinkSettings, AxisArray, CereLinkProducer]):
    """ezmsg Unit that streams continuous and spike data from a Blackrock device."""

    SETTINGS = CereLinkSettings
    OUTPUT_SPIKE = ez.OutputStream(EventMessage)

    async def initialize(self) -> None:
        self.create_producer()
        self.producer._loop = asyncio.get_running_loop()
        self.producer.open()

    def shutdown(self) -> None:
        self.producer.close()

    @ez.publisher(OUTPUT_SPIKE)
    async def spikes(self) -> typing.AsyncGenerator:
        while True:
            spike = await self.producer.spike_queue.get()
            yield self.OUTPUT_SPIKE, spike

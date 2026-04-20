"""CereLink-based source for ezmsg — streams continuous and spike data from Blackrock devices."""

from __future__ import annotations

import asyncio
import logging
import time
import typing
from dataclasses import dataclass

import ezmsg.core as ez
import numpy as np
from ezmsg.baseproc.processor import BaseProducer
from ezmsg.baseproc.units import BaseProducerUnit, get_base_producer_type
from ezmsg.event.message import EventMessage
from ezmsg.util.messages.axisarray import AxisArray, replace
from pycbsdk import ChannelType, DeviceType, SampleRate, Session

logger = logging.getLogger(__name__)


def _device_label(device_type: DeviceType | None) -> str:
    """Human-readable name for a device, or ``"None"`` for the idle case."""
    return device_type.name if device_type is not None else "None"


@dataclass
class DeviceStatus:
    """Result of a settings-driven device switch.

    Emitted on ``CereLinkSource.OUTPUT_DEVICE_STATUS`` after each ``on_settings``
    background ``open()`` attempt. The GUI consumes this to confirm or revert
    its device selection (the snapshot's settings_changed event fires before
    the open completes and so cannot distinguish success from failure).
    """

    device_type: DeviceType | None
    success: bool
    error: str = ""


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
    device_type: DeviceType | None = None
    """Device type to connect to. ``None`` = no Session — the unit produces
    nothing until a real device is selected. This idle-by-default mode lets
    a host app defer the cost of opening pycbsdk until the user picks a
    device, which is useful on macOS where simultaneous Sessions exhaust
    POSIX shared memory."""

    cbtime: bool = False
    """True = raw device nanoseconds/1e9, False = time.monotonic() via clock sync."""

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

    ac_input_coupling: bool = False
    """AC input coupling (highpass filter). Ignored when ccf_path is provided."""

    def __post_init__(self):
        has_manual = self.n_chans is not None or self.sample_rate is not None
        if self.ccf_path is not None and has_manual:
            raise ValueError("ccf_path and n_chans/sample_rate are mutually exclusive")
        if self.n_chans is not None and self.sample_rate is None:
            raise ValueError("sample_rate is required when n_chans is set")


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
        """Create and start the Session, configure channels, register callbacks.

        ``device_type=None`` is an explicit no-op — the producer stays idle
        and ``_produce`` returns ``None`` on every call, which lets the unit
        sit dormant until a real device is selected by the host app.
        """
        if self.settings.device_type is None:
            return
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
            self._session.load_ccf_sync(self.settings.ccf_path)
        elif self.settings.sample_rate is not None:
            n_chans = self.settings.n_chans
            if n_chans is None:
                n_chans = len(self._session.get_matching_channel_ids(self.settings.channel_type))
            self._session.set_channel_sample_group(
                n_chans,
                self.settings.channel_type,
                self.settings.sample_rate,
                disable_others=True,
            )
            self._session.set_ac_input_coupling(
                n_chans,
                self.settings.channel_type,
                self.settings.ac_input_coupling,
            )

        if self.settings.cmp_path:
            self._session.load_channel_map(self.settings.cmp_path, 0)

        self._session.sync()

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
        """Tear down the Session and unblock any in-flight ``_produce``.

        The ``_data_event.set()`` is what frees a coroutine that may be
        awaiting new samples — without it, an in-flight ``_produce`` from
        the previous producer instance would block forever after a
        settings-driven recreate, eventually getting GC'd as a pending task.
        """
        if self._session is not None:
            self._session.__exit__(None, None, None)
            self._session = None
        self._data_event.set()

    # -- Group setup --

    def _build_ch_info(self, channels: list[int]) -> np.ndarray:
        """Build the ``ch_info`` structured array (label/x/y/bank/elec) for one group."""
        n_ch = len(channels)
        ch_info = np.zeros(n_ch, dtype=CHANNEL_DTYPE)
        for i, ch_id in enumerate(channels):
            label = self._session.get_channel_label(ch_id)
            ch_info[i]["label"] = label or f"ch{ch_id}"
            pos = self._ch_positions.get(ch_id, (0, 0, 0, 0))
            ch_info[i]["x"] = pos[0]
            ch_info[i]["y"] = pos[1]
            ch_info[i]["bank"] = chr(ord("A") + pos[2] - 1) if pos[2] > 0 else ""
            ch_info[i]["elec"] = pos[3]
        return ch_info

    def reload_channel_map(self, cmp_path: str) -> None:
        """Apply a new CMP to the existing Session, refresh cached positions,
        and rebuild each active group's template so subsequent AxisArrays
        carry the new (x, y) positions.

        pycbsdk has no API to *clear* an applied CMP — pass ``None`` and
        the call is a no-op with a warning.
        """
        if cmp_path is None:
            logger.warning("CereLinkProducer.reload_channel_map: pycbsdk has no clear API; keeping previous map.")
            return
        self._session.load_channel_map(cmp_path, 0)
        all_ids = self._session.get_matching_channel_ids(ChannelType.FRONTEND)
        all_pos = self._session.get_channels_positions(ChannelType.FRONTEND)
        self._ch_positions = dict(zip(all_ids, all_pos))
        for rate_int, buf in self._buffers.items():
            channels = self._session.get_group_channels(rate_int)
            ch_info = self._build_ch_info(channels)
            new_ch_ax = AxisArray.CoordinateAxis(data=ch_info, dims=["ch"], unit="struct")
            old = buf["template"]
            buf["template"] = replace(old, axes={**old.axes, "ch": new_ch_ax})

    def reload_ccf(self, ccf_path: str) -> None:
        """Apply a new CCF on the existing Session. ``None`` is a no-op."""
        if ccf_path is None:
            logger.warning("CereLinkProducer.reload_ccf: pycbsdk has no clear API; keeping previous CCF.")
            return
        self._session.load_ccf_sync(ccf_path)

    def _setup_group(self, rate: SampleRate, channels: list[int]) -> None:
        rate_int = int(rate)
        n_ch = len(channels)
        fs = rate.hz
        buff_samples = max(1, int(self.settings.cont_buffer_dur * fs))

        ch_info = self._build_ch_info(channels)
        scale_factors = []
        for ch_id in channels:
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

    async def _produce(self) -> AxisArray | None:
        if self._loop is None:
            self._loop = asyncio.get_running_loop()

        while True:
            if self._session is None:
                # No session — either we're idle (device_type=None) or a
                # settings-driven recreate just closed us. Yield back to the
                # event loop so other tasks (subscriber attach handlers, the
                # next on_settings) can run, then return None so the unit's
                # produce loop re-reads ``self.producer``. Without the sleep
                # this becomes a tight loop that starves the asyncio scheduler.
                await asyncio.sleep(0.1)
                return None
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
    OUTPUT_DEVICE_STATUS = ez.OutputStream(DeviceStatus)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Init here (not in ``initialize()``) so the queue exists before any
        # publisher/subscriber coroutine could touch it. asyncio.Queue() in
        # Python 3.10+ doesn't require a running loop.
        self._status_queue: asyncio.Queue[DeviceStatus] = asyncio.Queue()

    def _start_producer(self) -> None:
        # Acquire the device session for the current producer. Called both at
        # startup and after every settings-driven recreate so the new producer
        # has a live Session before produce() runs.
        self.producer._loop = asyncio.get_running_loop()
        self.producer.open()
        s = self.producer.settings
        if s.device_type is None:
            logger.info("CereLinkSource._start_producer: idle (no device selected)")
            return
        logger.info(
            "CereLinkSource._start_producer: opened device=%s n_chans=%s sample_rate=%s",
            s.device_type.name,
            s.n_chans,
            s.sample_rate.name if s.sample_rate is not None else None,
        )

    async def initialize(self) -> None:
        self.create_producer()
        try:
            self._start_producer()
        except Exception as exc:
            logger.exception("CereLinkSource.initialize: startup open() failed")
            self._status_queue.put_nowait(
                DeviceStatus(device_type=self.SETTINGS.device_type, success=False, error=str(exc))
            )
            return
        self._status_queue.put_nowait(DeviceStatus(device_type=self.SETTINGS.device_type, success=True))

    @ez.subscriber(BaseProducerUnit.INPUT_SETTINGS)
    async def on_settings(self, msg: CereLinkSettings) -> None:
        logger.info(
            "CereLinkSource.on_settings: switching to device=%s n_chans=%s sample_rate=%s ccf=%s cmp=%s",
            _device_label(msg.device_type),
            msg.n_chans,
            msg.sample_rate.name if msg.sample_rate is not None else None,
            msg.ccf_path,
            msg.cmp_path,
        )
        prev = self.SETTINGS

        # Recreating a pycbsdk Session against the same physical device (and
        # NPLAY in particular — single-client emulator) corrupts the new
        # session via the old session's ``__exit__``. So if only CMP/CCF
        # changed and we already have a working session, apply the change
        # in-place rather than spawning a fresh producer.
        needs_session_restart = (
            msg.device_type != prev.device_type
            or msg.n_chans != prev.n_chans
            or msg.sample_rate != prev.sample_rate
            or msg.channel_type != prev.channel_type
            or msg.ac_input_coupling != prev.ac_input_coupling
        )
        if not needs_session_restart and getattr(self.producer, "_session", None) is not None:
            asyncio.create_task(self._apply_in_place_settings(msg, prev))
            return

        # Build the new producer eagerly but don't open it yet — open() is
        # synchronous and can block ~2s on a missing device. Doing that here
        # would hold the INPUT_SETTINGS subscriber lease and time out the
        # GUI's update_setting RPC. Defer to a background task so this
        # subscriber returns immediately.
        prev_producer = self.producer
        producer_type = get_base_producer_type(self.__class__)
        new_producer = producer_type(settings=msg)
        new_producer._loop = asyncio.get_running_loop()
        asyncio.create_task(self._finalize_settings_change(msg, prev_producer, new_producer))

    async def _apply_in_place_settings(self, msg: CereLinkSettings, prev: CereLinkSettings) -> None:
        """Apply CMP/CCF changes to the existing producer's Session."""
        try:
            if msg.ccf_path != prev.ccf_path:
                await asyncio.to_thread(self.producer.reload_ccf, msg.ccf_path)
            if msg.cmp_path != prev.cmp_path:
                await asyncio.to_thread(self.producer.reload_channel_map, msg.cmp_path)
        except Exception as exc:
            logger.exception("CereLinkSource: in-place CMP/CCF reload failed")
            await self._status_queue.put(DeviceStatus(device_type=msg.device_type, success=False, error=str(exc)))
            return
        # Update the producer's settings reference so subsequent reads match.
        self.producer.settings = msg
        self.apply_settings(msg)
        await self._status_queue.put(DeviceStatus(device_type=msg.device_type, success=True))

    async def _finalize_settings_change(
        self,
        new_settings: CereLinkSettings,
        prev_producer: CereLinkProducer,
        new_producer: CereLinkProducer,
    ) -> None:
        """Close prev, then open new — never two pycbsdk Sessions at once.

        Overlapping Sessions exhaust POSIX shared memory on macOS when the
        graph subprocess already has multiple ezmsg pubsub channels. We
        accept a brief no-data window during the swap. If ``open()`` fails,
        ``self.producer`` is left pointing at the (now-closed-and-empty) new
        producer, ``self.SETTINGS`` is reset to a no-device config, and the
        GUI receives a failure ``DeviceStatus`` so it can revert.

        Queues a ``DeviceStatus`` on both success and failure — the
        clearing-house's settings_changed event fires before this runs and
        so cannot tell the GUI which way the open went.
        """
        # Park self.producer on the (still-unopened) new instance so the
        # ``produce()`` loop's ``self.producer.__acall__()`` returns ``None``
        # while we close prev / open new (CereLinkProducer._produce returns
        # ``None`` whenever ``_session is None``).
        self.producer = new_producer
        try:
            await asyncio.to_thread(prev_producer.close)
        except Exception:
            logger.exception("CereLinkSource: failed to close previous producer")

        try:
            await asyncio.to_thread(new_producer.open)
        except Exception as exc:
            logger.exception(
                "CereLinkSource: failed to open device=%s",
                _device_label(new_settings.device_type),
            )
            try:
                new_producer.close()
            except Exception:
                logger.exception("CereLinkSource: failed to close half-opened producer")
            # Fall back to a no-device state; the GUI will reconcile via the
            # DeviceStatus failure event. ``new_producer._session`` is None
            # (open failed before assignment) so ``_produce`` already returns
            # None regardless of producer.settings — only self.SETTINGS needs
            # to reflect the rollback for downstream observers.
            self.apply_settings(replace(new_settings, device_type=None))
            await self._status_queue.put(
                DeviceStatus(
                    device_type=new_settings.device_type,
                    success=False,
                    error=str(exc),
                )
            )
            return

        s = new_producer.settings
        if s.device_type is None:
            logger.info("CereLinkSource: now idle (no device)")
        else:
            logger.info(
                "CereLinkSource: opened device=%s n_chans=%s sample_rate=%s",
                s.device_type.name,
                s.n_chans,
                s.sample_rate.name if s.sample_rate is not None else None,
            )
        self.apply_settings(new_settings)
        await self._status_queue.put(DeviceStatus(device_type=new_settings.device_type, success=True))

    def shutdown(self) -> None:
        self.producer.close()

    @ez.publisher(OUTPUT_SPIKE)
    async def spikes(self) -> typing.AsyncGenerator:
        while True:
            spike = await self.producer.spike_queue.get()
            yield self.OUTPUT_SPIKE, spike

    @ez.publisher(OUTPUT_DEVICE_STATUS)
    async def device_status(self) -> typing.AsyncGenerator:
        while True:
            status = await self._status_queue.get()
            yield self.OUTPUT_DEVICE_STATUS, status

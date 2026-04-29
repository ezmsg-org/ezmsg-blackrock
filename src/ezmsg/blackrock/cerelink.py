"""CereLink-based source for ezmsg — streams continuous and spike data from Blackrock devices."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import typing
from dataclasses import dataclass

import ezmsg.core as ez
import numpy as np
from ezmsg.baseproc import processor_state
from ezmsg.baseproc.stateful import BaseStatefulProducer
from ezmsg.baseproc.units import BaseProducerUnit
from ezmsg.util.messages.axisarray import AxisArray, replace
from pycbsdk import ChannelType, DeviceType, SampleRate, Session

from .channel_map import CHANNEL_DTYPE, ChannelMapSettings

logger = logging.getLogger(__name__)


def _device_label(device_type: DeviceType | None) -> str:
    """Human-readable name for a device, or ``"None"`` for the idle case."""
    return device_type.name if device_type is not None else "None"


@dataclass
class DeviceStatus:
    """Result of a settings-driven device switch.

    Emitted on ``CereLinkSignalSource.OUTPUT_DEVICE_STATUS`` (and the
    corresponding spike-source output) after each ``_areset_state`` attempt.
    The GUI consumes this to confirm or revert its device selection — the
    snapshot's settings_changed event fires before the open completes and
    so cannot distinguish success from failure.
    """

    device_type: DeviceType | None
    success: bool
    error: str = ""


# --- Device-configuration types ------------------------------------------


@dataclass
class CcfConfig:
    """Load a CCF file. Device-wide configuration — at most one source per
    graph should carry this. Other sources targeting the same device set
    ``configure=None`` (pure subscriber)."""

    path: str


@dataclass
class SliceConfig:
    """Programmatic per-slice device configuration owned by this source.

    The source applies the device state needed to make its subscribed stream
    produce data on ``channels``: ``set_sample_group`` for signal sources,
    ``set_spike_extraction`` for spike sources, plus AC coupling.

    Multiple sources can carry disjoint slices for the same device; pycbsdk
    handles the merge. Overlap with incompatible settings is the user's
    responsibility.
    """

    channels: list[int] | None = None
    """1-based channel IDs to configure. ``None`` = all matching ``channel_type``."""

    channel_type: ChannelType = ChannelType.FRONTEND

    ac_input_coupling: bool = False
    """Apply AC coupling (highpass filter) on this slice."""

    enable_spiking: bool = False
    """Enable spike extraction on this slice (FRONTEND only). Honored by
    :class:`CereLinkSpikeSource`; ignored by signal sources."""


DeviceConfig = CcfConfig | SliceConfig | None
"""Device-configuration mode for one source. ``None`` means another source
or component owns the device config; this source only subscribes."""


class CereLinkSignalSettings(ez.Settings):
    """Settings for :class:`CereLinkSignalSource` — emits one continuous
    sample-group as :class:`AxisArray`."""

    device_type: DeviceType | None = None
    """Device to connect to. ``None`` = idle (no Session opened)."""

    subscribe_rate: SampleRate = SampleRate.NONE
    """Required. The sample-group rate this source streams.
    ``SampleRate.NONE`` is rejected — pass a real rate."""

    configure: DeviceConfig = None
    """Device configuration this source applies on open."""

    cbtime: bool = False
    """True = raw device nanoseconds/1e9; False = ``time.monotonic()`` via clock sync."""

    microvolts: bool = True
    """Convert int16 → µV using channel scale factors."""

    cont_buffer_dur: float = 0.5
    """Ring buffer duration in seconds."""

    cmp_configs: tuple[ChannelMapSettings, ...] = ()
    """One :class:`ChannelMapSettings` per headstage applied after connection."""

    def __post_init__(self):
        if self.subscribe_rate == SampleRate.NONE:
            raise ValueError(
                "subscribe_rate is required and must be a real SampleRate "
                "(SR_500, SR_1kHz, SR_2kHz, SR_10kHz, SR_30kHz, or SR_RAW)"
            )


class CereLinkSpikeSettings(ez.Settings):
    """Settings for :class:`CereLinkSpikeSource` — emits sparse spike events
    as :class:`AxisArray` of shape ``[time, ch, unit=7]`` at the 30 kHz
    spike clock. Unit indices follow the device convention:
    ``0=unsorted, 1..5=sorted, 6=noise (header.type > 5)``."""

    device_type: DeviceType | None = None
    """Device to connect to. ``None`` = idle."""

    configure: DeviceConfig = None
    """Device configuration this source applies on open."""

    cbtime: bool = False
    """True = raw device nanoseconds/1e9; False = ``time.monotonic()`` via clock sync."""

    microvolts: bool = True
    """Reserved for future spike-waveform emission — int16 → µV scaling."""

    spike_buffer_dur: float = 0.5
    """Ring buffer duration in seconds (at the 30 kHz spike clock)."""

    cmp_configs: tuple[ChannelMapSettings, ...] = ()
    """One :class:`ChannelMapSettings` per headstage applied after connection."""

    def __post_init__(self):
        # Symmetry with CereLinkSignalSettings; nothing to validate today.
        pass


# --- Shared producer base + signal/spike leaves --------------------------
#
# `_CereLinkBaseProducer` owns the pycbsdk Session lifecycle and configure
# dispatch. Concrete leaves (`CereLinkSignalProducer`, `CereLinkSpikeProducer`)
# override the stream-specific bits — callback registration, buffer shape,
# `_produce`. Each leaf has its own Source class.


_SettingsT = typing.TypeVar("_SettingsT")
_StateT = typing.TypeVar("_StateT")


@processor_state
class _CereLinkSharedState:
    """State fields common to signal and spike producers."""

    session: Session | None = None
    ch_positions: dict | None = None  # ch_id -> (col, row, bank, elec)


@processor_state
class CereLinkSignalProducerState(_CereLinkSharedState):
    """Signal-producer ring buffer + emission template."""

    buffer_data: np.ndarray | None = None  # int16 [N_t, n_ch]
    buffer_timestamps: np.ndarray | None = None  # uint64 [N_t]
    write_idx: int = 0
    read_idx: int = 0
    n_channels: int = 0
    template: AxisArray | None = None
    scale_factors: np.ndarray | None = None
    data_event: asyncio.Event | None = None  # set by callback when new samples arrive


class _CereLinkBaseProducer(
    BaseStatefulProducer[_SettingsT, AxisArray, _StateT],
    typing.Generic[_SettingsT, _StateT],
):
    """Shared lifecycle/configure base for one-stream-per-source producers.

    Concrete subclasses override ``_apply_slice_configure``, ``_setup_subscription``,
    and ``_produce``. The async open/close lifecycle is driven by ``_areset_state``
    (run automatically by :class:`BaseStatefulProducer.__acall__` on first call
    and after every :meth:`_request_reset`).
    """

    NONRESET_SETTINGS_FIELDS = frozenset({"cbtime", "microvolts", "cmp_configs"})

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._status_callback: typing.Callable[[DeviceStatus], None] | None = None

    def set_status_callback(self, cb: typing.Callable[[DeviceStatus], None]) -> None:
        """Inject the unit's status emitter — wired up by the Source on construct
        so the producer can publish open/close results without holding a Source ref."""
        self._status_callback = cb

    def _emit_status(self, status: DeviceStatus) -> None:
        if self._status_callback is not None:
            self._status_callback(status)

    def _reset_state(self) -> None:
        """Sync reset hook — no-op. Open/configure is async (see ``_areset_state``)."""
        pass

    async def _areset_state(self) -> None:
        """Close any prior session, open and configure a new one.

        On exception leaves state without a session so ``_produce`` returns
        None and the unit stays idle. Status is emitted on both success and
        failure so the host app can reconcile.
        """
        await self._teardown_state()
        if self.settings.device_type is None:
            return  # idle
        try:
            await self._open_and_configure()
        except Exception as exc:
            logger.exception(
                "CereLink: failed to open device=%s",
                _device_label(self.settings.device_type),
            )
            await self._teardown_state()
            self._emit_status(
                DeviceStatus(
                    device_type=self.settings.device_type,
                    success=False,
                    error=str(exc),
                )
            )
            return
        self._emit_status(DeviceStatus(device_type=self.settings.device_type, success=True))

    async def _teardown_state(self) -> None:
        """Release the Session and wake any waiters."""
        self._on_teardown_pre_close()
        if self.state.session is not None:
            try:
                await asyncio.to_thread(self.state.session.__exit__, None, None, None)
            except Exception:
                logger.exception("CereLink: error during async teardown")
            self.state.session = None

    def _on_teardown_pre_close(self) -> None:
        """Subclass hook called before the Session is closed (e.g., to wake await-ers)."""
        pass

    async def _open_and_configure(self) -> None:
        loop = asyncio.get_running_loop()
        self.state.session = Session(device_type=self.settings.device_type)
        try:
            await asyncio.to_thread(self.state.session.__enter__)
            await self.state.session.wait_until_running(timeout=10.0)
            await asyncio.to_thread(self._apply_configure)
            await asyncio.to_thread(self._apply_channel_maps)
            # Sync so device responses to load_ccf / set_sample_group / load_channel_map
            # are reflected in pycbsdk's local mirror before downstream readers
            # (e.g., get_group_channels) consult it.
            await asyncio.to_thread(self.state.session.sync)
            self._cache_channel_metadata()
            self._setup_subscription(loop)
        except BaseException:
            # Release fds before propagating; outer ``_areset_state`` would also
            # call ``_teardown_state`` on Exception, but doing it here covers
            # BaseException too (KeyboardInterrupt, SystemExit).
            try:
                await asyncio.to_thread(self.state.session.__exit__, None, None, None)
            except Exception:
                logger.exception("CereLink: cleanup-after-failure also failed")
            self.state.session = None
            raise

    def _apply_configure(self) -> None:
        """Apply CcfConfig / SliceConfig device state. Sync — runs in a thread."""
        cfg = self.settings.configure
        if cfg is None:
            return
        if isinstance(cfg, CcfConfig):
            self.state.session.load_ccf_sync(cfg.path)
            return
        # SliceConfig
        self._apply_slice_configure(cfg)
        if cfg.ac_input_coupling:
            self.state.session.set_ac_input_coupling(cfg.channels, cfg.channel_type, True)

    def _apply_slice_configure(self, cfg: SliceConfig) -> None:
        """Subclass hook: stream-specific slice config (sample-group OR spike-extract)."""
        raise NotImplementedError

    def _apply_channel_maps(self) -> None:
        for cmp_cfg in self.settings.cmp_configs:
            if cmp_cfg.filepath:
                self.state.session.load_channel_map(cmp_cfg.filepath, cmp_cfg.start_chan, cmp_cfg.hs_id)

    def _cache_channel_metadata(self) -> None:
        all_ids = self.state.session.get_matching_channel_ids(ChannelType.FRONTEND)
        all_pos = self.state.session.get_channels_positions(ChannelType.FRONTEND)
        self.state.ch_positions = dict(zip(all_ids, all_pos))

    def _build_ch_info(self, channels: list[int]) -> np.ndarray:
        n_ch = len(channels)
        ch_info = np.zeros(n_ch, dtype=CHANNEL_DTYPE)
        for i, ch_id in enumerate(channels):
            label = self.state.session.get_channel_label(ch_id)
            ch_info[i]["label"] = label or f"ch{ch_id}"
            pos = self.state.ch_positions.get(ch_id, (0, 0, 0, 0))
            ch_info[i]["x"] = pos[0]
            ch_info[i]["y"] = pos[1]
            ch_info[i]["bank"] = chr(ord("A") + pos[2] - 1) if pos[2] > 0 else ""
            ch_info[i]["elec"] = pos[3]
        return ch_info

    def _setup_subscription(self, loop: asyncio.AbstractEventLoop) -> None:
        """Subclass hook: register pycbsdk callback, allocate buffer, build template."""
        raise NotImplementedError

    def update_settings(self, new_settings: _SettingsT) -> None:
        """Override to apply ``cmp_configs`` in place when no other RESET field
        changed — preserves the GUI-driven hot-CMP-swap workflow without a
        Session restart."""
        old_cmp = self.settings.cmp_configs
        super().update_settings(new_settings)
        if self._hash != -1 and old_cmp != self.settings.cmp_configs and self.state.session is not None:
            self._reload_channel_maps_in_place()

    def _reload_channel_maps_in_place(self) -> None:
        """Apply current ``cmp_configs`` without restarting the Session.

        ``clear_channel_map()`` resets pycbsdk's overlay so successive applies
        don't accumulate. Subclasses rebuild their template's ``ch`` axis via
        ``_on_channel_maps_reloaded``.
        """
        self.state.session.clear_channel_map()
        for cfg in self.settings.cmp_configs:
            if cfg.filepath:
                self.state.session.load_channel_map(cfg.filepath, cfg.start_chan, cfg.hs_id)
        self._cache_channel_metadata()
        self._on_channel_maps_reloaded()

    def _on_channel_maps_reloaded(self) -> None:
        """Subclass hook: rebuild template's ``ch`` axis with refreshed positions."""
        pass

    def close(self) -> None:
        """Synchronous teardown — for ``Source.shutdown()``. Releases the
        Session and wakes any awaiters."""
        self._on_teardown_pre_close()
        if self.state.session is not None:
            try:
                self.state.session.__exit__(None, None, None)
            except Exception:
                logger.exception("CereLink: error during synchronous close")
            self.state.session = None


class CereLinkSignalProducer(_CereLinkBaseProducer[CereLinkSignalSettings, CereLinkSignalProducerState]):
    """Streams one continuous sample-group as :class:`AxisArray`."""

    def _apply_slice_configure(self, cfg: SliceConfig) -> None:
        rate = self.settings.subscribe_rate
        self.state.session.set_sample_group(cfg.channels, cfg.channel_type, rate, disable_others=True)

    def _setup_subscription(self, loop: asyncio.AbstractEventLoop) -> None:
        rate = self.settings.subscribe_rate
        channels = self.state.session.get_group_channels(int(rate))
        if not channels:
            self.state.n_channels = 0
            return
        n_ch = len(channels)
        fs = rate.hz
        buff_samples = max(1, int(self.settings.cont_buffer_dur * fs))

        scale_factors = self._compute_scale_factors(channels)
        ch_info = self._build_ch_info(channels)
        time_ax = AxisArray.TimeAxis(fs, offset=0.0)
        ch_ax = AxisArray.CoordinateAxis(data=ch_info, dims=["ch"], unit="struct")
        template = AxisArray(
            np.zeros((0, 0)),
            dims=["time", "ch"],
            axes={"time": time_ax, "ch": ch_ax},
            key=rate.name,
            attrs={"unit": "uV" if self.settings.microvolts else "raw"},
        )

        st = self.state
        st.buffer_data = np.zeros((buff_samples, n_ch), dtype=np.int16)
        st.buffer_timestamps = np.zeros(buff_samples, dtype=np.uint64)
        st.write_idx = 0
        st.read_idx = 0
        st.n_channels = n_ch
        st.template = template
        st.scale_factors = scale_factors
        st.data_event = asyncio.Event()

        @st.session.on_group_batch(rate)
        def _on_group_batch(samples, timestamps):
            self._handle_group_batch(samples, timestamps, loop)

    def _compute_scale_factors(self, channels: list[int]) -> np.ndarray:
        sfs = []
        for ch_id in channels:
            scaling = self.state.session.get_channel_scaling(ch_id)
            if scaling and scaling["digmax"] != scaling["digmin"]:
                sf = (scaling["anamax"] - scaling["anamin"]) / (scaling["digmax"] - scaling["digmin"])
                if scaling["anaunit"] == "mV":
                    sf *= 1000  # mV -> uV
                sfs.append(sf)
            else:
                sfs.append(1.0)
        return np.array(sfs, dtype=np.float64)

    def _handle_group_batch(
        self,
        samples: np.ndarray,
        timestamps: np.ndarray,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        st = self.state
        n_ch = st.n_channels
        if samples.shape[1] > n_ch:
            samples = samples[:, :n_ch]  # drop dword-padding columns
        w = st.write_idx
        n = len(timestamps)
        buff_len = len(st.buffer_timestamps)
        end = w + n
        if end <= buff_len:
            st.buffer_data[w:end, :] = samples
            st.buffer_timestamps[w:end] = timestamps
        else:
            first = buff_len - w
            st.buffer_data[w:buff_len, :] = samples[:first]
            st.buffer_timestamps[w:buff_len] = timestamps[:first]
            rest = n - first
            st.buffer_data[:rest, :] = samples[first:]
            st.buffer_timestamps[:rest] = timestamps[first:]
        st.write_idx = end % buff_len
        if loop.is_running():
            loop.call_soon_threadsafe(st.data_event.set)
        else:
            st.data_event.set()

    def _on_teardown_pre_close(self) -> None:
        if self.state.data_event is not None:
            self.state.data_event.set()

    def _on_channel_maps_reloaded(self) -> None:
        rate = self.settings.subscribe_rate
        channels = self.state.session.get_group_channels(int(rate))
        if not channels or self.state.template is None:
            return
        ch_info = self._build_ch_info(channels)
        new_ch_ax = AxisArray.CoordinateAxis(data=ch_info, dims=["ch"], unit="struct")
        old = self.state.template
        self.state.template = replace(old, axes={**old.axes, "ch": new_ch_ax})

    async def _produce(self) -> AxisArray | None:
        st = self.state
        if st.session is None or st.n_channels == 0:
            await asyncio.sleep(0.1)
            return None
        while True:
            read_idx = st.read_idx
            write_idx = st.write_idx
            buff_len = len(st.buffer_timestamps)
            read_term = write_idx if write_idx >= read_idx else buff_len
            if read_idx == read_term:
                st.data_event.clear()
                await st.data_event.wait()
                if st.session is None:  # closed while waiting
                    return None
                continue

            read_slice = slice(read_idx, read_term)
            out_dat = st.buffer_data[read_slice].copy()
            if self.settings.microvolts:
                out_dat = out_dat * st.scale_factors[None, :]

            first_ts = int(st.buffer_timestamps[read_idx])
            if self.settings.cbtime:
                new_offset = first_ts / 1e9
            else:
                try:
                    new_offset = st.session.device_to_monotonic(first_ts)
                except RuntimeError:
                    new_offset = time.monotonic()

            template = st.template
            new_time_ax = replace(template.axes["time"], offset=new_offset)
            result = replace(
                template,
                data=out_dat,
                axes={**template.axes, "time": new_time_ax},
            )
            st.read_idx = read_term % buff_len
            return result


class CereLinkSignalSource(BaseProducerUnit[CereLinkSignalSettings, AxisArray, CereLinkSignalProducer]):
    """ezmsg Unit that streams one continuous sample-group from a Blackrock device."""

    SETTINGS = CereLinkSignalSettings
    OUTPUT_DEVICE_STATUS = ez.OutputStream(DeviceStatus)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Init in __init__ (not initialize) so the queue exists before
        # device_status's coroutine could attach.
        self._status_queue: asyncio.Queue[DeviceStatus] = asyncio.Queue()

    def create_producer(self) -> None:
        super().create_producer()
        self.producer.set_status_callback(self._status_queue.put_nowait)

    def shutdown(self) -> None:
        self.producer.close()

    @ez.publisher(OUTPUT_DEVICE_STATUS)
    async def device_status(self) -> typing.AsyncGenerator:
        while True:
            status = await self._status_queue.get()
            yield self.OUTPUT_DEVICE_STATUS, status


# --- Spike producer/source -----------------------------------------------
#
# Spikes are emitted as `AxisArray[time, ch, unit=7]` on a regular cadence
# (`spike_buffer_dur`), with `time` at the device's 30 kHz spike clock and
# `unit` indexing the device convention: 0=unsorted, 1..5=sorted, 6=noise.


_SPIKE_FS = 30000  # device spike clock — fixed by the protocol
_NS_PER_SECOND = 1_000_000_000
_UNIT_LABELS = np.array(["unsorted", "1", "2", "3", "4", "5", "noise"], dtype="U8")


@processor_state
class CereLinkSpikeProducerState(_CereLinkSharedState):
    """Spike-producer rolling buffer + emission template."""

    buffer: np.ndarray | None = None  # uint8 [N_t, n_ch, 7]
    n_channels: int = 0
    n_t: int = 0
    template: AxisArray | None = None
    data_event: asyncio.Event | None = None
    chid_to_buffer_idx: dict | None = None  # 1-based chid -> column index
    window_origin_ns: int = -1  # device ts (ns) at buffer[0]; -1 = not yet aligned


class CereLinkSpikeProducer(_CereLinkBaseProducer[CereLinkSpikeSettings, CereLinkSpikeProducerState]):
    """Streams spike events as :class:`AxisArray` of shape ``[time, ch, unit=7]``.

    Time axis is the device's 30 kHz spike clock; ``unit`` axis indexes the
    device convention (0=unsorted, 1..5=sorted, 6=noise — values >5 collapse
    into the noise bucket because the unit axis has fixed length 7).

    Emission cadence is regular: every ``spike_buffer_dur`` seconds, one
    window of shape ``[N_t, n_ch, 7]`` is emitted. Empty windows are emitted
    too (downstream wants a steady time axis). The window before the first
    spike is suppressed — there's no device-time anchor until then.

    Spikes whose timestamp lands beyond the current window's range are
    dropped (a downstream-backpressure failure mode); in normal operation
    ``_produce`` runs at the buffer cadence and the buffer is large enough
    to hold one window of spikes.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Guards state.buffer + state.window_origin_ns. Held by both the
        # asyncio loop (in ``_produce``'s emit-and-reset) and the receive
        # thread (in ``_handle_spike``'s sample-index compute + write).
        # Without this lock, the full-buffer copy+zero in ``_produce`` races
        # with per-spike increments — increments landing between copy and
        # zero would be silently lost.
        self._buffer_lock = threading.Lock()

    def _apply_slice_configure(self, cfg: SliceConfig) -> None:
        if cfg.enable_spiking:
            self.state.session.set_spike_extraction(cfg.channels, cfg.channel_type, True)

    def _setup_subscription(self, loop: asyncio.AbstractEventLoop) -> None:
        cfg = self.settings.configure
        if isinstance(cfg, SliceConfig):
            channel_type = cfg.channel_type
            if cfg.channels is not None:
                channels = list(cfg.channels)
            else:
                channels = self.state.session.get_matching_channel_ids(channel_type)
        else:
            # CcfConfig or None: subscribe to all FRONTEND. The device's CCF
            # (or whatever's already configured) decides which of these
            # actually emit spikes; channels with spiking disabled simply
            # never appear in the callback stream.
            channel_type = ChannelType.FRONTEND
            channels = self.state.session.get_matching_channel_ids(channel_type)

        if not channels:
            self.state.n_channels = 0
            return

        n_ch = len(channels)
        n_t = max(1, int(self.settings.spike_buffer_dur * _SPIKE_FS))

        ch_info = self._build_ch_info(channels)
        time_ax = AxisArray.TimeAxis(float(_SPIKE_FS), offset=0.0)
        ch_ax = AxisArray.CoordinateAxis(data=ch_info, dims=["ch"], unit="struct")
        unit_ax = AxisArray.CoordinateAxis(data=_UNIT_LABELS.copy(), dims=["unit"], unit="label")
        template = AxisArray(
            np.zeros((0, 0, 0), dtype=np.uint8),
            dims=["time", "ch", "unit"],
            axes={"time": time_ax, "ch": ch_ax, "unit": unit_ax},
            key="SPIKES",
            attrs={"unit": "count"},
        )

        st = self.state
        st.buffer = np.zeros((n_t, n_ch, 7), dtype=np.uint8)
        st.n_channels = n_ch
        st.n_t = n_t
        st.template = template
        st.data_event = asyncio.Event()
        st.chid_to_buffer_idx = {ch_id: i for i, ch_id in enumerate(channels)}
        st.window_origin_ns = -1

        @st.session.on_event(channel_type)
        def _on_event(header, data):
            self._handle_spike(header, loop)

    def _handle_spike(self, header, loop: asyncio.AbstractEventLoop) -> None:
        st = self.state
        ch_idx = st.chid_to_buffer_idx.get(header.chid)
        if ch_idx is None:
            return  # not subscribed to this channel (e.g., outside our slice)

        # Clamp values >5 into the noise bucket so they fit the length-7 unit axis.
        unit_idx = header.type if header.type < 6 else 6

        spike_ts = header.time  # device ns
        with self._buffer_lock:
            if st.window_origin_ns == -1:
                # Align the window to the first spike's timestamp; subsequent
                # emissions advance by integer multiples of n_t samples.
                st.window_origin_ns = spike_ts
            if spike_ts < st.window_origin_ns:
                return  # late arrival across a window boundary
            # Integer arithmetic: ``sample_idx = (elapsed_ns * 30_000) // 1e9``.
            # Avoids float truncation jitter that would push some spikes one
            # bin too early. Device-side ns/tick rounding still produces ±1
            # sample jitter, but that's intrinsic to pycbsdk's conversion.
            elapsed_ns = spike_ts - st.window_origin_ns
            sample_idx = (elapsed_ns * _SPIKE_FS) // _NS_PER_SECOND
            if sample_idx >= st.n_t:
                # Past the current window; without rotating the buffer here
                # we drop. In normal operation _produce runs at the buffer
                # cadence and this branch shouldn't hit. If it does
                # repeatedly, downstream isn't keeping up.
                return
            st.buffer[sample_idx, ch_idx, unit_idx] += 1

        if loop.is_running():
            loop.call_soon_threadsafe(st.data_event.set)
        else:
            st.data_event.set()

    def _on_teardown_pre_close(self) -> None:
        if self.state.data_event is not None:
            self.state.data_event.set()

    def _on_channel_maps_reloaded(self) -> None:
        st = self.state
        if st.chid_to_buffer_idx is None or st.template is None:
            return
        # chid_to_buffer_idx ordering is preserved across CMP reloads — only
        # positions/labels change, so we rebuild the ch axis in place.
        channels = sorted(st.chid_to_buffer_idx, key=st.chid_to_buffer_idx.get)
        ch_info = self._build_ch_info(channels)
        new_ch_ax = AxisArray.CoordinateAxis(data=ch_info, dims=["ch"], unit="struct")
        old = st.template
        st.template = replace(old, axes={**old.axes, "ch": new_ch_ax})

    async def _produce(self) -> AxisArray | None:
        st = self.state
        if st.session is None or st.n_channels == 0:
            await asyncio.sleep(0.1)
            return None

        if st.window_origin_ns == -1:
            # Block until the first spike sets the window origin. The session
            # close path triggers data_event so this returns cleanly.
            await st.data_event.wait()
            if st.session is None or st.window_origin_ns == -1:
                return None

        # Regular cadence — one window per ``spike_buffer_dur``. The sleep
        # timer is independent of the device clock; window time-axis offsets
        # are computed from ``window_origin_ns`` so they remain aligned to
        # the device regardless of small drifts in _produce's schedule.
        await asyncio.sleep(self.settings.spike_buffer_dur)
        if st.session is None:
            return None

        with self._buffer_lock:
            out_data = st.buffer.copy()
            st.buffer[:] = 0
            emit_origin_ns = st.window_origin_ns
            # Advance origin by one buffer's worth of samples in integer ns.
            # ``ceil_div`` rather than floor avoids overlap between successive
            # windows: window K covers [origin, origin + advance_ns), and
            # window K+1 starts at origin + advance_ns.
            advance_ns = (st.n_t * _NS_PER_SECOND + _SPIKE_FS - 1) // _SPIKE_FS
            st.window_origin_ns += advance_ns

        if self.settings.cbtime:
            new_offset = emit_origin_ns / 1e9
        else:
            try:
                new_offset = st.session.device_to_monotonic(emit_origin_ns)
            except RuntimeError:
                new_offset = time.monotonic()

        template = st.template
        new_time_ax = replace(template.axes["time"], offset=new_offset)
        return replace(template, data=out_data, axes={**template.axes, "time": new_time_ax})


class CereLinkSpikeSource(BaseProducerUnit[CereLinkSpikeSettings, AxisArray, CereLinkSpikeProducer]):
    """ezmsg Unit that streams spike events as `AxisArray[time, ch, unit=7]`."""

    SETTINGS = CereLinkSpikeSettings
    OUTPUT_DEVICE_STATUS = ez.OutputStream(DeviceStatus)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._status_queue: asyncio.Queue[DeviceStatus] = asyncio.Queue()

    def create_producer(self) -> None:
        super().create_producer()
        self.producer.set_status_callback(self._status_queue.put_nowait)

    def shutdown(self) -> None:
        self.producer.close()

    @ez.publisher(OUTPUT_DEVICE_STATUS)
    async def device_status(self) -> typing.AsyncGenerator:
        while True:
            status = await self._status_queue.get()
            yield self.OUTPUT_DEVICE_STATUS, status

"""CereLink-based source for ezmsg — streams continuous and spike data from Blackrock devices."""

from __future__ import annotations

import asyncio
import logging
import time
import typing
from dataclasses import dataclass

import ezmsg.core as ez
import numpy as np
from ezmsg.baseproc import processor_state
from ezmsg.baseproc.stateful import BaseStatefulProducer
from ezmsg.baseproc.units import BaseProducerUnit, get_base_producer_type
from ezmsg.event.message import EventMessage
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

    Emitted on ``CereLinkSource.OUTPUT_DEVICE_STATUS`` after each ``on_settings``
    background ``open()`` attempt. The GUI consumes this to confirm or revert
    its device selection (the snapshot's settings_changed event fires before
    the open completes and so cannot distinguish success from failure).
    """

    device_type: DeviceType | None
    success: bool
    error: str = ""


class CereLinkSettings(ez.Settings):
    # --- Connection ---

    device_type: DeviceType | None = None
    """Device type to connect to. ``None`` = no Session — the unit produces
    nothing until a real device is selected. This idle-by-default mode lets
    a host app defer the cost of opening pycbsdk until the user picks a
    device, which is useful on macOS where simultaneous Sessions exhaust
    POSIX shared memory."""

    # --- Device configuration: ccf_path XOR programmatic (config_chans + config_rate) ---

    ccf_path: str | None = None
    """CCF file to load after connection (or None to skip).
    Mutually exclusive with ``config_chans`` (both touch the device's
    sample-group configuration)."""

    config_chans: int | None = None
    """Programmatic-setup signal:

    * ``None`` (default): leave the device's existing sample-group config
      untouched. Use this when ``ccf_path`` is set, or when relying on
      whatever the device already has loaded.
    * Any integer: call ``set_sample_group`` with this count (clamped to the
      number of channels actually matching ``config_chan_type``, so passing
      ``int(1e6)`` is a valid "give me all available" shorthand). Requires
      ``config_rate`` to be set; mutually exclusive with ``ccf_path``."""

    config_chan_type: ChannelType = ChannelType.FRONTEND
    """Channel type used for programmatic setup, AC coupling, and the
    auto-resolution of ``config_chans``."""

    config_rate: SampleRate | None = None
    """Sample rate for the programmatically-configured group. Co-required
    with ``config_chans`` — set both or neither. Has no effect on its own
    (use ``subscribe_rate`` to filter capture without configuring)."""

    ac_input_coupling: bool = False
    """AC input coupling (highpass filter). Applied whenever ``ccf_path`` is
    None, using ``(config_chans, config_chan_type)`` — both default to "all
    matching channels of ``config_chan_type``" if not specified. Ignored in
    CCF mode (CCF owns coupling)."""

    # --- Capture filter ---

    subscribe_rate: SampleRate | None = None
    """Restricts ``on_group_batch`` registration to this rate. ``None``
    registers for every active group the device reports (the right default
    with a CCF that activates multiple rates). Independent of
    ``config_rate``: a CCF that wires up multiple rates can be filtered
    down to one by setting ``subscribe_rate`` alone."""

    # --- Channel-map overlay ---

    cmp_configs: tuple[ChannelMapSettings, ...] = ()
    """One :class:`ChannelMapSettings` per headstage to apply after connection.
    Each is forwarded to ``session.load_channel_map(filepath, start_chan, hs_id)``.
    Empty tuple skips CMP loading entirely."""

    # --- Data interpretation (no device interaction) ---

    cbtime: bool = False
    """True = raw device nanoseconds/1e9, False = time.monotonic() via clock sync."""

    microvolts: bool = True
    """Convert int16 → µV using channel scale factors (when available)."""

    cont_buffer_dur: float = 0.5
    """Ring buffer duration in seconds per sample rate group."""

    def __post_init__(self):
        if self.ccf_path is not None and self.config_chans is not None:
            raise ValueError("ccf_path and config_chans are mutually exclusive")
        # Programmatic setup is all-or-nothing: config_chans and config_rate
        # are meaningless apart, so reject the partial-config footgun.
        if (self.config_chans is None) != (self.config_rate is None):
            raise ValueError("config_chans and config_rate must be set together (or neither)")


# --- New settings shapes (Phase 1 of the split refactor) -------------------
#
# These types support the one-stream-per-source design. Old `CereLinkSettings`
# above stays in place until Phase 4 cuts the new sources over.


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


# --- End of new settings shapes -------------------------------------------


@processor_state
class CereLinkProducerState:
    """Empty placeholder so :class:`CereLinkProducer` can be a stateful producer.

    The session/buffers/positions live as instance attributes today; moving
    them into this dataclass and routing the open/close lifecycle through
    ``_reset_state`` is the planned follow-up. Until then, this exists only
    so the stateful base-class machinery (``_request_reset``, the
    ``update_settings`` reset gate, etc.) is wired up.
    """


class CereLinkProducer(BaseStatefulProducer[CereLinkSettings, AxisArray, CereLinkProducerState]):
    """Owns the pycbsdk Session, ring buffers, and spike queue."""

    # Fields whose change can be absorbed without closing and re-opening the
    # pycbsdk Session. CereLinkSource.on_settings drives the in-place vs
    # session-restart decision today (it doesn't yet route through
    # update_settings); this list documents the contract.
    NONRESET_SETTINGS_FIELDS = frozenset(
        {"cbtime", "microvolts", "cont_buffer_dur", "cmp_configs", "ac_input_coupling"}
    )

    def __init__(self, *args, settings: CereLinkSettings | None = None, **kwargs):
        super().__init__(*args, settings=settings, **kwargs)
        self._session: Session | None = None
        self._buffers: dict[int, dict] = {}  # rate (int) → buffer info
        self.spike_queue: asyncio.Queue[EventMessage] = asyncio.Queue()
        self._data_event = asyncio.Event()
        self._active_rates: list[int] = []
        self._round_robin_idx: int = 0
        self._loop: asyncio.AbstractEventLoop | None = None

    def _reset_state(self) -> None:
        """No-op: the open/close lifecycle is driven by :class:`CereLinkSource`.

        BaseStatefulProducer calls this once on the first ``__acall__`` and
        again on every ``_request_reset()``. Migrating the session lifecycle
        in here (so settings updates re-run open/close automatically) is the
        planned follow-up; doing it now would conflict with the existing
        async multi-Session dance in ``CereLinkSource.on_settings``.
        """

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

        # Past this point the Session owns ~8 POSIX fds (6 shm + 1 sem + 1 UDP
        # in STANDALONE mode). Any exception below must release them, or we
        # orphan the fds for the life of the process.
        try:
            self._wait_until_running(timeout_s=10.0)
            self._configure_device()
            self._apply_channel_maps()
            # CRITICAL: sync() must precede _apply_ac_coupling(). pycbsdk has no
            # narrowly-scoped AC-coupling packet, so set_ac_input_coupling sends
            # a general-config packet seeded from our local mirror of device
            # state. Until the device's responses to set_sample_group / load_ccf
            # / load_channel_map have come back through sync(), that mirror is
            # stale and the AC packet would replay the stale config — silently
            # reverting whatever set_sample_group / load_ccf just did. Same rule
            # applies to any future call site that touches AC coupling after a
            # config change: sync() in between, every time.
            self._session.sync()
            self._apply_ac_coupling()
            self._cache_channel_positions()
            self._setup_capture_groups()
            self._register_spike_callback()
        except BaseException:
            self._session.__exit__(None, None, None)
            self._session = None
            raise

    def _wait_until_running(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while self._session is None or not self._session.running:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Session did not start within {timeout_s} seconds")
            time.sleep(0.1)

    def _resolve_n_chans(self) -> int:
        """Number of channels to apply settings to — clamped to actually-available."""
        s = self.settings
        available = len(self._session.get_matching_channel_ids(s.config_chan_type))
        if s.config_chans is None:
            return available
        return min(s.config_chans, available)

    def _configure_device(self) -> None:
        """Load CCF or programmatically configure the sample group."""
        s = self.settings
        if s.ccf_path:
            self._session.load_ccf_sync(s.ccf_path)
            return
        if s.config_chans is not None:
            # __post_init__ guarantees config_rate is set when config_chans is.
            self._session.set_sample_group(
                self._resolve_n_chans(),
                s.config_chan_type,
                s.config_rate,
                disable_others=True,
            )
            self._session.sync()

    def _apply_ac_coupling(self) -> None:
        """Apply AC coupling — only meaningful when CCF isn't owning device config.

        .. warning::
           Callers MUST run ``self._session.sync()`` after any preceding
           device-config call (``set_sample_group``, ``load_ccf_sync``,
           ``load_channel_map``, etc.) and BEFORE calling this method.
           ``set_ac_input_coupling`` rides on a general-config packet seeded
           from our local mirror of device state; if device responses to the
           prior config calls haven't been processed yet, the mirror is stale
           and this call will replay the stale config — silently undoing the
           preceding change.
        """
        s = self.settings
        if s.ccf_path:
            return
        self._session.set_ac_input_coupling(
            self._resolve_n_chans(),
            s.config_chan_type,
            s.ac_input_coupling,
        )

    def _apply_channel_maps(self) -> None:
        for cfg in self.settings.cmp_configs:
            if cfg.filepath:
                self._session.load_channel_map(cfg.filepath, cfg.start_chan, cfg.hs_id)

    def _cache_channel_positions(self) -> None:
        """Pre-fetch channel positions (available after CCF/CMP loading)."""
        all_ids = self._session.get_matching_channel_ids(ChannelType.FRONTEND)
        all_pos = self._session.get_channels_positions(ChannelType.FRONTEND)
        self._ch_positions = dict(zip(all_ids, all_pos))

    def _setup_capture_groups(self) -> None:
        """Register ring buffers + on_group_batch callbacks for the configured rate(s).

        ``subscribe_rate=None`` registers every active group the device reports
        (correct default in CCF mode where multiple rates may be active).
        Setting ``subscribe_rate`` restricts capture to that single group.
        """
        if self.settings.subscribe_rate is not None:
            rates = [self.settings.subscribe_rate]
        else:
            rates = [r for r in SampleRate if r != SampleRate.NONE]
        for rate in rates:
            channels = self._session.get_group_channels(int(rate))
            if not channels:
                continue
            self._setup_group(rate, channels)

    def _register_spike_callback(self) -> None:
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

    def reload_channel_maps(self, cmp_configs: tuple[ChannelMapSettings, ...]) -> None:
        """Apply each CMP in *cmp_configs* to the live Session, refresh cached
        positions, and rebuild each active group's template so subsequent
        AxisArrays carry the new (x, y) positions.

        pycbsdk has no API to *clear* applied CMPs — successive calls merge
        into the device's overlay. An empty ``cmp_configs`` is a no-op.
        """
        if not cmp_configs:
            return
        for cfg in cmp_configs:
            if cfg.filepath:
                self._session.load_channel_map(cfg.filepath, cfg.start_chan, cfg.hs_id)
        all_ids = self._session.get_matching_channel_ids(ChannelType.FRONTEND)
        all_pos = self._session.get_channels_positions(ChannelType.FRONTEND)
        self._ch_positions = dict(zip(all_ids, all_pos))
        for rate_int, buf in self._buffers.items():
            channels = self._session.get_group_channels(rate_int)
            ch_info = self._build_ch_info(channels)
            new_ch_ax = AxisArray.CoordinateAxis(data=ch_info, dims=["ch"], unit="struct")
            old = buf["template"]
            buf["template"] = replace(old, axes={**old.axes, "ch": new_ch_ax})

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
            "CereLinkSource._start_producer: opened device=%s config_chans=%s config_rate=%s subscribe_rate=%s",
            s.device_type.name,
            s.config_chans,
            s.config_rate.name if s.config_rate is not None else None,
            s.subscribe_rate.name if s.subscribe_rate is not None else None,
        )

    async def initialize(self) -> None:
        self.create_producer()
        try:
            self._start_producer()
        except Exception as exc:
            logger.exception("CereLinkSource.initialize: startup open() failed")
            try:
                self.producer.close()
            except Exception:
                logger.exception("CereLinkSource.initialize: cleanup close() failed")
            self._status_queue.put_nowait(
                DeviceStatus(device_type=self.SETTINGS.device_type, success=False, error=str(exc))
            )
            return
        self._status_queue.put_nowait(DeviceStatus(device_type=self.SETTINGS.device_type, success=True))

    @ez.subscriber(BaseProducerUnit.INPUT_SETTINGS)
    async def on_settings(self, msg: CereLinkSettings) -> None:
        logger.info(
            "CereLinkSource.on_settings: switching to device=%s config_chans=%s "
            "config_rate=%s subscribe_rate=%s ccf=%s cmp_configs=%d",
            _device_label(msg.device_type),
            msg.config_chans,
            msg.config_rate.name if msg.config_rate is not None else None,
            msg.subscribe_rate.name if msg.subscribe_rate is not None else None,
            msg.ccf_path,
            len(msg.cmp_configs),
        )
        prev = self.SETTINGS

        # Mirrors CereLinkProducer.NONRESET_SETTINGS_FIELDS: anything outside
        # that list requires closing and re-opening the pycbsdk Session.
        # Recreating a Session against the same physical device (especially
        # NPLAY, a single-client emulator) corrupts the new session via the
        # old one's ``__exit__``, so the in-place path is preferred whenever
        # the changed fields permit it.
        # subscribe_rate is conservatively in the restart set — applying it
        # in-place would need a re-register helper for on_group_batch.
        needs_session_restart = (
            msg.device_type != prev.device_type
            or msg.config_chans != prev.config_chans
            or msg.config_rate != prev.config_rate
            or msg.config_chan_type != prev.config_chan_type
            or msg.subscribe_rate != prev.subscribe_rate
            or msg.ccf_path != prev.ccf_path
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
        """Apply NONRESET-eligible settings changes to the live Session.

        ``cont_buffer_dur`` changes are accepted but won't take effect until
        the next session restart — resizing the in-flight ring buffers safely
        is left to the planned ``_reset_state`` migration.
        """
        # Rebind producer.settings up front so the helpers below read the
        # new values when they re-derive things like ``_resolve_n_chans``.
        self.producer.settings = msg
        try:
            if msg.cmp_configs != prev.cmp_configs:
                await asyncio.to_thread(self.producer.reload_channel_maps, msg.cmp_configs)
            if msg.ac_input_coupling != prev.ac_input_coupling:
                # Honor _apply_ac_coupling's contract: any preceding device
                # config (the cmp_configs reload above, or anything that ran
                # since the last sync) must be flushed first, or the general-
                # config packet that carries the AC change will replay a stale
                # local mirror and undo it.
                await asyncio.to_thread(self.producer._session.sync)
                await asyncio.to_thread(self.producer._apply_ac_coupling)
        except Exception as exc:
            logger.exception("CereLinkSource: in-place settings apply failed")
            # Roll producer.settings back to keep state consistent.
            self.producer.settings = prev
            await self._status_queue.put(DeviceStatus(device_type=msg.device_type, success=False, error=str(exc)))
            return
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
                "CereLinkSource: opened device=%s config_chans=%s config_rate=%s subscribe_rate=%s",
                s.device_type.name,
                s.config_chans,
                s.config_rate.name if s.config_rate is not None else None,
                s.subscribe_rate.name if s.subscribe_rate is not None else None,
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


# --- New producer/source classes (Phase 2 of the split refactor) -----------
#
# `_CereLinkBaseProducer` is the shared lifecycle/configure base for one-stream-
# per-source producers. `CereLinkSignalProducer` / `CereLinkSignalSource` is
# the continuous-data leaf. The spike leaf is added in Phase 3. The old
# `CereLinkProducer` / `CereLinkSource` remain in place until Phase 4 cutover.


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

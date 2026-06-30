"""Device-clock → host-clock conversion for CereLink streams.

Two things live here:

* :func:`device_to_monotonic_offset` — the small, reusable conversion that the
  signal and spike producers reach through their ``_produce`` methods. It wraps
  :meth:`pycbsdk.Session.device_to_monotonic` and leaves the no-sync fallback
  policy to the caller.
* :class:`CbtimeToMonotonic` / :class:`CbtimeToMonotonicTransformer` — a
  standalone ezmsg unit that re-stamps a *device-time* :class:`AxisArray` (one
  produced by a source with ``cbtime=True``, whose ``time`` axis offset is
  device-ns / 1e9) onto ``time.monotonic()``. It opens its **own** read-only
  client :class:`~pycbsdk.Session` to the same device — CereLink's shared memory
  lets a second session attach to a device a source already opened (same host).
"""

from __future__ import annotations

import asyncio
import logging

import ezmsg.core as ez
from ezmsg.baseproc import processor_state
from ezmsg.baseproc.stateful import BaseStatefulTransformer
from ezmsg.baseproc.units import BaseTransformerUnit
from ezmsg.util.messages.axisarray import AxisArray, replace
from pycbsdk import DeviceType, Session

logger = logging.getLogger(__name__)


def device_to_monotonic_offset(session: Session, device_ns: int, *, stream_id: int = -1) -> float | None:
    """Convert a device timestamp (ns) to a ``time.monotonic()`` offset in seconds.

    Thin wrapper over :meth:`pycbsdk.Session.device_to_monotonic` that returns
    ``None`` instead of raising when no clock sync is available yet, leaving the
    fallback policy to the caller.

    ``stream_id >= 0`` engages the library's per-stream monotonicity enforcement
    (host time kept non-decreasing, with a floor reset across a genuine clock
    re-sync); ``-1`` is a stateless conversion. Use a stable id per logical
    stream — the scalar call routes through ``device_to_monotonic_batch`` under
    the hood, so a stable id is all that's needed for enforcement.
    """
    try:
        return session.device_to_monotonic(device_ns, stream_id)
    except RuntimeError:
        return None


def device_to_monotonic_batch_offsets(session: Session, device_ns, *, stream_id: int = -1) -> list[float] | None:
    """Convert a whole batch of device timestamps to ``time.monotonic()`` seconds.

    Like :func:`device_to_monotonic_offset` but for an iterable of device
    timestamps, returning ``None`` (instead of raising) when no clock sync is
    available yet.

    Converting the *full* batch (rather than only its first sample) matters for
    monotonicity: the library advances the per-stream floor to the **last**
    converted value, so the next batch's start is clamped to be ``>=`` this
    batch's end. Converting only the first sample would protect batch-start
    offsets but leave the reconstructed per-sample grid free to step backward at
    a batch boundary when the clock offset slews between batches.
    """
    try:
        return session.device_to_monotonic_batch(device_ns, stream_id)
    except RuntimeError:
        return None


class CbtimeToMonotonicSettings(ez.Settings):
    """Settings for :class:`CbtimeToMonotonic`.

    Input contract: the ``time`` axis offset is a device timestamp in seconds
    (device-ns / 1e9), as emitted by ``CereLinkSignalSource`` /
    ``CereLinkSpikeSource`` with ``cbtime=True``.
    """

    device_type: DeviceType | None = None
    """Device whose clock to attach to. ``None`` = idle (pure passthrough)."""

    stream_id: int = 0
    """Per-stream monotonicity key. ``>= 0`` enforces a non-decreasing host
    timeline (each transformer owns its own Session, so the id namespace is
    private); ``-1`` is a stateless conversion."""


@processor_state
class CbtimeToMonotonicState:
    session: Session | None = None


class CbtimeToMonotonicTransformer(
    BaseStatefulTransformer[CbtimeToMonotonicSettings, AxisArray, AxisArray, CbtimeToMonotonicState]
):
    """Re-stamp a device-time :class:`AxisArray` onto ``time.monotonic()``.

    The client Session is opened once, on the first message (default
    ``_hash_message`` returns 0). If it fails to open, or clock sync isn't
    available yet, messages pass through unchanged (still device time).
    """

    def _reset_state(self, message: AxisArray | None = None) -> None:
        """Sync reset hook — no-op. The Session open is async (see ``_areset_state``)."""
        pass

    async def _areset_state(self, message: AxisArray | None = None) -> None:
        await self._teardown()
        if self.settings.device_type is None:
            return  # idle — passthrough
        session = Session(device_type=self.settings.device_type)
        try:
            await asyncio.to_thread(session.__enter__)
            await session.wait_until_running(timeout=10.0)
        except BaseException:
            logger.exception(
                "CbtimeToMonotonic: failed to attach client session to device=%s; "
                "messages will pass through with device time.",
                self.settings.device_type.name,
            )
            try:
                await asyncio.to_thread(session.__exit__, None, None, None)
            except Exception:
                logger.exception("CbtimeToMonotonic: cleanup-after-failure also failed")
            return  # leave state.session=None — passthrough
        self.state.session = session

    async def _teardown(self) -> None:
        if self.state.session is not None:
            try:
                await asyncio.to_thread(self.state.session.__exit__, None, None, None)
            except Exception:
                logger.exception("CbtimeToMonotonic: error during async teardown")
            self.state.session = None

    def _process(self, message: AxisArray) -> AxisArray:
        if self.state.session is None or "time" not in message.axes:
            # TODO: Maybe return None here.
            return message
        time_ax = message.axes["time"]
        n = message.data.shape[message.dims.index("time")] if "time" in message.dims else 1
        gain = float(getattr(time_ax, "gain", 0.0) or 0.0)
        # Convert the first AND last sample (device ns) as one batch so the
        # library's per-stream monotonic floor advances to this message's end.
        first_ns = int(round(time_ax.offset * 1e9))
        last_ns = int(round((time_ax.offset + gain * max(0, n - 1)) * 1e9))
        offsets = device_to_monotonic_batch_offsets(
            self.state.session, (first_ns, last_ns), stream_id=self.settings.stream_id
        )
        if offsets is None:
            return message  # no clock sync yet
        return replace(
            message,
            axes={**message.axes, "time": replace(time_ax, offset=offsets[0])},
        )

    def close(self) -> None:
        """Synchronous teardown — for unit ``shutdown()``."""
        if self.state.session is not None:
            try:
                self.state.session.__exit__(None, None, None)
            except Exception:
                logger.exception("CbtimeToMonotonic: error during synchronous close")
            self.state.session = None


class CbtimeToMonotonic(
    BaseTransformerUnit[CbtimeToMonotonicSettings, AxisArray, AxisArray, CbtimeToMonotonicTransformer]
):
    """ezmsg Unit wrapping :class:`CbtimeToMonotonicTransformer`."""

    SETTINGS = CbtimeToMonotonicSettings

    def shutdown(self) -> None:
        self.processor.close()

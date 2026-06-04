"""Attach Blackrock ``.cmp`` channel-map metadata to an ``AxisArray``'s ``ch`` axis.

The output ``ch`` axis is a structured ``CoordinateAxis`` with fields
``x``, ``y``, ``size``, ``label``, ``bank``, ``elec``, ``headstage`` for every
input channel. ``x``/``y``/``size`` are in micrometers; ``headstage`` is the
1-based headstage id (``0`` = none/auto).

:class:`ChannelMapUnit` takes the *complete* set of per-headstage overlays in
one settings object (:class:`ChannelMapUnitSettings`, a tuple of
:class:`ChannelMapSettings`) and rebuilds the ``ch`` axis from scratch on each
reset. One settings push = the whole map, applied deterministically — there is
no cross-push accumulation that could coalesce if pushes aren't separated by a
data message. An empty tuple clears the map (pure auto-grid).

Each reset proceeds in three phases:

1. **Base layer** — labels are pulled from the incoming ``ch`` axis.
2. **CMP overlays** — for each :class:`ChannelMapSettings` in ``cmp_configs``,
   entries from :func:`pycbsdk.cmp.parse_cmp` are written at their channel
   index. ``parse_cmp`` (CerebusOSS/CereLink#184) returns entries keyed by
   device ``(bank, term)`` with flat ``x``/``y``/``size``/``headstage`` fields
   (``x``/``y`` in micrometers) and verbatim labels; the channel index is
   ``(bank - 1) * 32 + (term - 1)`` — ``start_chan`` is already folded into
   ``bank`` via its ``// 32`` offset. A companion ``cmp_mask`` records which
   indices were set so the auto-grid pass can avoid them.
3. **Auto-grid fill** — positions/bank/elec for indices NOT covered by any
   CMP, laid out below and to the right of the CMP geometry so they don't
   collide with CMP positions. The grid step matches the CMP's electrode
   pitch (inferred from its coordinates), so auto-laid channels share the
   CMP's micrometer scale.

The same :class:`ChannelMapSettings` record is also used as a per-headstage
entry in :attr:`CereLinkSignalSettings.cmp_configs`.
"""

import logging
import math

import ezmsg.core as ez
import numpy as np
from ezmsg.baseproc import BaseStatefulTransformer, BaseTransformerUnit, processor_state
from ezmsg.util.messages.axisarray import AxisArray, CoordinateAxis
from ezmsg.util.messages.util import replace
from pycbsdk.cmp import parse_cmp

logger = logging.getLogger(__name__)

CHANNEL_DTYPE = np.dtype(
    [
        ("x", "i4"),  # electrode x, µm (int32, matching cbPKT_CHANINFO.position)
        ("y", "i4"),  # electrode y, µm
        ("size", "i4"),  # electrode size, µm (0 = unspecified)
        ("label", "U16"),
        ("bank", "U1"),
        ("elec", "i4"),
        ("headstage", "i4"),  # 1-based headstage id (0 = none/auto)
    ]
)


class ChannelMapSettings(ez.Settings):
    filepath: str | None = None
    """Path to the ``.cmp`` file. ``None`` (or an empty path) means no CMP —
    the auto-grid fallback generates coordinates for every channel."""

    start_chan: int = 1
    """1-based channel ID assigned to the first sorted CMP row.
    Mirrors :meth:`pycbsdk.Session.load_channel_map`."""

    hs_id: int = 0
    """Headstage identifier, passed through to :func:`pycbsdk.cmp.parse_cmp`,
    where it sets each entry's ``headstage`` field. Labels are taken verbatim
    (no ``"hs{hs_id}-"`` prefix); ``bank``/``elec`` disambiguate channels that
    reuse a label across headstages. Pass ``0`` for single-headstage rigs."""


class ChannelMapUnitSettings(ez.Settings):
    cmp_configs: tuple[ChannelMapSettings, ...] = ()
    """Per-headstage overlays, applied in order on each reset. Empty (the
    default) means no CMP — the auto-grid lays out every channel."""


@processor_state
class ChannelMapState:
    channel_axis: CoordinateAxis | None = None
    cmp_mask: np.ndarray | None = None  # bool, length == channel_axis.data length


class ChannelMapProcessor(BaseStatefulTransformer[ChannelMapUnitSettings, AxisArray, AxisArray, ChannelMapState]):
    """Stateful transformer that attaches CMP-derived channel metadata.

    Each reset rebuilds the axis in full from ``settings.cmp_configs``: a base
    layer from incoming labels, every CMP overlay applied in order, then the
    auto-grid fills indices no CMP claimed. Reset fires on a channel-count
    change or any ``cmp_configs`` change (the latter via
    :class:`BaseProcessor.update_settings` → ``_request_reset``), so there is
    no cross-push state to coalesce. An empty ``cmp_configs`` yields a pure
    auto-grid.
    """

    def _reset_state(self, message: AxisArray) -> None:
        ch_dim_idx = message.dims.index("ch")
        n_total = message.data.shape[ch_dim_idx]

        # Base layer: labels from incoming; positions filled by the overlays
        # and auto-grid below.
        ch_data = np.zeros(n_total, dtype=CHANNEL_DTYPE)
        for i, label in enumerate(self._incoming_labels(message, n_total)):
            ch_data[i]["label"] = label

        # CMP overlays: write each headstage's entries at chan_id-1 and mark
        # them in cmp_mask so the auto-grid skips them.
        cmp_mask = np.zeros(n_total, dtype=bool)
        for cfg in self.settings.cmp_configs:
            if not cfg.filepath:
                continue
            try:
                parsed = parse_cmp(cfg.filepath, start_chan=cfg.start_chan, hs_id=cfg.hs_id)
            except Exception as exc:
                # _reset_state runs on every message via __acall__ until the
                # hash matches; a re-raise would loop forever. Log and skip.
                logger.warning(
                    "ChannelMapProcessor: could not load %r (start_chan=%d, hs_id=%d): %s; skipping.",
                    cfg.filepath,
                    cfg.start_chan,
                    cfg.hs_id,
                    exc,
                )
                continue
            for (bank, term), entry in parsed.items():
                # parse_cmp keys by device (bank, term); start_chan is already
                # folded into bank via its // 32 offset, so the channel index is
                # a direct (bank, term) → row mapping (32 terminals per bank).
                idx = (bank - 1) * 32 + (term - 1)
                if not (0 <= idx < n_total):
                    continue
                ch_data[idx]["x"] = int(entry.x)
                ch_data[idx]["y"] = int(entry.y)
                ch_data[idx]["size"] = int(entry.size)
                ch_data[idx]["label"] = entry.label  # verbatim (no hs{N}- prefix)
                ch_data[idx]["bank"] = chr(ord("A") + bank - 1)
                ch_data[idx]["elec"] = term
                ch_data[idx]["headstage"] = entry.headstage
                cmp_mask[idx] = True

        self.state.channel_axis = CoordinateAxis(data=ch_data, dims=["ch"], unit="struct")
        self.state.cmp_mask = cmp_mask

        # Auto-grid: position/bank/elec for indices no CMP claimed, offset
        # below the CMP geometry so they don't overlap.
        self._fill_auto_grid()

    def _fill_auto_grid(self) -> None:
        ch_data = self.state.channel_axis.data
        cmp_mask = self.state.cmp_mask
        auto_idx = np.flatnonzero(~cmp_mask)
        if auto_idx.size == 0:
            return

        # Step matches the CMP's electrode pitch (≈400 µm) so the auto-grid
        # sits on the same scale as the CMP geometry. Without this, the µm CMP
        # coordinates would dwarf a unit-spaced auto-grid. Falls back to 1
        # when there is no CMP (pure auto-grid from the origin).
        step = self._cmp_pitch(ch_data, cmp_mask)

        if cmp_mask.any():
            max_row = int(ch_data["y"][cmp_mask].max())
            max_bank_ord = max(
                (ord(str(b)) for b in ch_data["bank"][cmp_mask] if str(b)),
                default=ord("A") - 1,
            )
        else:
            # No CMP yet — start auto-grid at the origin with bank A.
            # max_row = -2*step makes start_row = 0 below.
            max_row = -2 * step
            max_bank_ord = ord("A") - 1

        start_row = max_row + 2 * step
        next_bank_ord = max_bank_ord + 1
        grid_size = max(1, math.ceil(math.sqrt(auto_idx.size)))

        for i, idx in enumerate(auto_idx):
            ch_data[idx]["x"] = (i % grid_size) * step
            ch_data[idx]["y"] = start_row + (i // grid_size) * step
            ch_data[idx]["size"] = step  # synthetic electrodes sized to the grid pitch
            ch_data[idx]["bank"] = chr(next_bank_ord + i // 32)
            ch_data[idx]["elec"] = (i % 32) + 1
            ch_data[idx]["headstage"] = 0  # auto-grid channels have no headstage

    @staticmethod
    def _cmp_pitch(ch_data: np.ndarray, cmp_mask: np.ndarray) -> int:
        """Smallest positive spacing among the CMP's distinct x/y coordinates.

        This is the electrode pitch in micrometers (≈400 for a default Utah
        array). Defaults to ``1`` when there is no CMP or the geometry is
        degenerate, giving the pure auto-grid unit spacing from the origin."""
        if not cmp_mask.any():
            return 1
        deltas: list[int] = []
        for field in ("x", "y"):
            vals = np.unique(ch_data[field][cmp_mask])
            if vals.size > 1:
                deltas.append(int(np.diff(vals).min()))
        positive = [d for d in deltas if d > 0]
        return min(positive) if positive else 1

    @staticmethod
    def _incoming_labels(message: AxisArray, n_total: int) -> list[str]:
        ch_axis = message.axes.get("ch")
        data = getattr(ch_axis, "data", None)
        if data is None:
            return [f"ch{i + 1}" for i in range(n_total)]
        if data.dtype.names is not None and "label" in data.dtype.names:
            labels = [str(x) for x in data["label"][:n_total]]
        else:
            labels = [str(x) for x in data[:n_total]]
        if len(labels) < n_total:
            labels.extend(f"ch{i + 1}" for i in range(len(labels), n_total))
        return labels

    def _hash_message(self, message: AxisArray) -> int:
        ch_dim_idx = message.dims.index("ch")
        return hash(message.data.shape[ch_dim_idx])

    def _process(self, message: AxisArray) -> AxisArray:
        return replace(message, axes={**message.axes, "ch": self.state.channel_axis})


class ChannelMapUnit(BaseTransformerUnit[ChannelMapUnitSettings, AxisArray, AxisArray, ChannelMapProcessor]):
    SETTINGS = ChannelMapUnitSettings

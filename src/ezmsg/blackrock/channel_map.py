"""Attach Blackrock ``.cmp`` channel-map metadata to an ``AxisArray``'s ``ch`` axis.

The output ``ch`` axis is a structured ``CoordinateAxis`` with fields
``x``, ``y``, ``label``, ``bank``, ``elec`` for every input channel.

Each reset proceeds in three phases:

1. **Base layer** — labels are pulled from the incoming ``ch`` axis. The
   axis object is built once per ``n_ch`` and reused across resets so
   successive CMP pushes accumulate into one composite map.
2. **CMP overlay** — entries from :func:`pycbsdk.cmp.parse_cmp` are written
   at indices ``chan_id - 1`` (with ``chan_id`` offset by ``start_chan``).
   A companion ``cmp_mask`` records which indices were set so the auto-grid
   pass can avoid them.
3. **Auto-grid fill** — positions/bank/elec for indices NOT covered by the
   CMP, laid out below and to the right of the CMP geometry so they don't
   collide with CMP positions.

Pushing ``filepath=None`` (or an empty path) clears the accumulated
overlay so the next reset rebuilds the base from scratch.
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
        ("x", "f4"),
        ("y", "f4"),
        ("label", "U16"),
        ("bank", "U1"),
        ("elec", "i4"),
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
    """Headstage identifier; labels are prefixed ``"hs{hs_id}-"`` when nonzero.
    Pass ``0`` (the default) to leave labels un-prefixed."""


@processor_state
class ChannelMapState:
    channel_axis: CoordinateAxis | None = None
    cmp_mask: np.ndarray | None = None  # bool, length == channel_axis.data length


class ChannelMapProcessor(BaseStatefulTransformer[ChannelMapSettings, AxisArray, AxisArray, ChannelMapState]):
    """Stateful transformer that attaches CMP-derived channel metadata.

    The base layer (incoming labels) is rebuilt only when no axis exists yet
    or when ``n_ch`` changes. CMP entries accumulate into ``cmp_mask``;
    auto-grid positions are recomputed each reset for the un-masked indices,
    offset below the CMP geometry to avoid collisions.

    Pushing ``filepath=None`` clears the cumulative overlay (see
    :meth:`update_settings`).
    """

    def update_settings(self, new_settings: ChannelMapSettings) -> None:
        old_filepath = self.settings.filepath
        super().update_settings(new_settings)
        # filepath transitioning from set → unset is the explicit "clear" signal.
        # Drop the cumulative overlay so the next reset rebuilds the base layer
        # from incoming labels.
        if old_filepath and not new_settings.filepath:
            self.state.channel_axis = None
            self.state.cmp_mask = None

    def _reset_state(self, message: AxisArray) -> None:
        ch_dim_idx = message.dims.index("ch")
        n_total = message.data.shape[ch_dim_idx]

        # n_ch changed since the last build → drop cumulative state.
        if self.state.channel_axis is not None and len(self.state.channel_axis.data) != n_total:
            self.state.channel_axis = None
            self.state.cmp_mask = None

        # Base layer: labels from incoming, positions filled below by auto-grid.
        if self.state.channel_axis is None:
            ch_data = np.zeros(n_total, dtype=CHANNEL_DTYPE)
            for i, label in enumerate(self._incoming_labels(message, n_total)):
                ch_data[i]["label"] = label
            self.state.channel_axis = CoordinateAxis(data=ch_data, dims=["ch"], unit="struct")
            self.state.cmp_mask = np.zeros(n_total, dtype=bool)

        # CMP overlay: write entries at chan_id-1 and mark them in cmp_mask.
        path = self.settings.filepath
        if path:
            try:
                parsed = parse_cmp(
                    path,
                    start_chan=self.settings.start_chan,
                    hs_id=self.settings.hs_id,
                )
            except Exception as exc:
                # _reset_state runs on every message via __acall__ until the
                # hash matches; a re-raise would loop forever. Log and skip.
                logger.warning(
                    "ChannelMapProcessor: could not load %r (%s); keeping current channel axis.",
                    path,
                    exc,
                )
                parsed = {}
            ch_data = self.state.channel_axis.data
            for chan_id, entry in parsed.items():
                idx = chan_id - 1
                if not (0 <= idx < n_total):
                    continue
                col, row, bank_idx, elec = entry.position
                ch_data[idx]["x"] = float(col)
                ch_data[idx]["y"] = float(row)
                ch_data[idx]["label"] = entry.label
                ch_data[idx]["bank"] = chr(ord("A") + bank_idx - 1)
                ch_data[idx]["elec"] = elec
                self.state.cmp_mask[idx] = True

        # Auto-grid: position/bank/elec for indices the CMP didn't claim,
        # offset below the CMP's geometry so they don't overlap.
        self._fill_auto_grid()

    def _fill_auto_grid(self) -> None:
        ch_data = self.state.channel_axis.data
        cmp_mask = self.state.cmp_mask
        auto_idx = np.flatnonzero(~cmp_mask)
        if auto_idx.size == 0:
            return

        if cmp_mask.any():
            max_row = float(ch_data["y"][cmp_mask].max())
            max_bank_ord = max(
                (ord(str(b)) for b in ch_data["bank"][cmp_mask] if str(b)),
                default=ord("A") - 1,
            )
        else:
            # No CMP yet — start auto-grid at the origin with bank A.
            # max_row = -2 makes start_row = 0 below.
            max_row = -2.0
            max_bank_ord = ord("A") - 1

        start_row = max_row + 2
        next_bank_ord = max_bank_ord + 1
        grid_size = max(1, math.ceil(math.sqrt(auto_idx.size)))

        for i, idx in enumerate(auto_idx):
            ch_data[idx]["x"] = float(i % grid_size)
            ch_data[idx]["y"] = float(start_row + i // grid_size)
            ch_data[idx]["bank"] = chr(next_bank_ord + i // 32)
            ch_data[idx]["elec"] = (i % 32) + 1

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


class ChannelMapUnit(BaseTransformerUnit[ChannelMapSettings, AxisArray, AxisArray, ChannelMapProcessor]):
    SETTINGS = ChannelMapSettings

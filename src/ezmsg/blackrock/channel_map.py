"""Attach Blackrock ``.cmp`` channel-map metadata to an ``AxisArray``'s ``ch`` axis.

Reads a single Blackrock Neurotech ``.cmp`` file and writes a structured
``CoordinateAxis`` (``x``, ``y``, ``label``, ``bank``, ``elec``) onto the
incoming message's ``ch`` axis. Any channels beyond the ones the CMP
defines — e.g. a 256-channel stream played through a 128-channel CMP —
get an auto-generated grid appended so downstream widgets always have a
full set of coordinates to plot against.

One CMP per device: if you have a multi-headstage device, compose a
single CMP file for the whole device rather than stitching multiple maps
inside this unit.
"""

import logging
import math

import ezmsg.core as ez
import numpy as np
from ezmsg.baseproc import BaseStatefulTransformer, BaseTransformerUnit, processor_state
from ezmsg.util.messages.axisarray import AxisArray, CoordinateAxis
from ezmsg.util.messages.util import replace

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


def parse_cmp(path: str) -> list[tuple[int, int, str, int, str]]:
    """Parse a Blackrock ``.cmp`` file.

    Returns a list of ``(col, row, bank, electrode, label)`` tuples, one
    per channel, in file order. *bank* is a single letter (typically
    ``A`` – ``D``) drawn from the file itself.
    """
    entries: list[tuple[int, int, str, int, str]] = []
    description_seen = False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            if not description_seen:
                description_seen = True  # first non-comment line is description
                continue
            parts = line.split()
            entries.append(
                (
                    int(parts[0]),  # col
                    int(parts[1]),  # row
                    parts[2],  # bank letter
                    int(parts[3]),  # electrode
                    parts[4],  # label
                )
            )
    return entries


class ChannelMapSettings(ez.Settings):
    channel_map: str | None = None
    """Path to the CMP file for this device. ``None`` (or an empty path)
    means no CMP — the auto-grid fallback generates coordinates for every
    channel."""


@processor_state
class ChannelMapState:
    channel_axis: CoordinateAxis | None = None


class ChannelMapProcessor(BaseStatefulTransformer[ChannelMapSettings, AxisArray, AxisArray, ChannelMapState]):
    """Stateful transformer that attaches CMP-derived channel metadata.

    State is re-built whenever the channel-axis length changes (detected
    via :meth:`_hash_message`), so a mid-stream device swap that changes
    ``n_ch`` triggers a fresh parse and auto-grid rebuild.
    """

    def _reset_state(self, message: AxisArray) -> None:
        records: list[tuple[float, float, str, str, int]] = []
        path = self.settings.channel_map
        if path:
            # A bad path or a malformed CMP must not take the pipeline
            # down — the failure would otherwise loop on every incoming
            # message (each triggers a fresh _reset_state via __acall__)
            # and downstream would never see a message again. Log the
            # problem and fall back to the auto-grid branch so the app
            # keeps rendering with generic labels until the user picks a
            # valid file.
            try:
                parsed = parse_cmp(path)
            except Exception as exc:
                logger.warning(
                    "ChannelMapProcessor: could not load %r (%s); " "falling back to auto-grid labels.",
                    path,
                    exc,
                )
                parsed = []
            # Sort by (bank, elec) so the CMP rows line up with the
            # device's native channel ordering instead of whatever order
            # the file happened to list them in.
            parsed = sorted(parsed, key=lambda e: (e[2], e[3]))
            for col, row, bank_letter, elec, label in parsed:
                records.append((float(col), float(row), label, bank_letter, elec))

        n_mapped = len(records)
        ch_dim_idx = message.dims.index("ch")
        n_total = message.data.shape[ch_dim_idx]

        if n_total > n_mapped:
            # Auto-grid for channels the CMP doesn't cover. Tile them in a
            # square-ish block below the last mapped row, using bank letters
            # that start one past the CMP's highest.
            n_remaining = n_total - n_mapped
            grid_size = math.ceil(math.sqrt(n_remaining))

            if records:
                max_row = max(r[1] for r in records)
                max_bank_ord = max(ord(r[3]) for r in records)
            else:
                max_row = -2.0
                max_bank_ord = ord("A") - 1

            start_row = max_row + 2
            next_bank_ord = max_bank_ord + 1

            for i in range(n_remaining):
                label = f"auto{n_mapped + i + 1}"
                bank = chr(next_bank_ord + i // 32)
                elec = (i % 32) + 1
                col = float(i % grid_size)
                row = start_row + i // grid_size
                records.append((col, row, label, bank, elec))

        ch_data = np.array(records, dtype=CHANNEL_DTYPE)
        self.state.channel_axis = CoordinateAxis(data=ch_data[:n_total], dims=["ch"], unit="struct")

    def _hash_message(self, message: AxisArray) -> int:
        ch_dim_idx = message.dims.index("ch")
        return hash(message.data.shape[ch_dim_idx])

    def _process(self, message: AxisArray) -> AxisArray:
        return replace(message, axes={**message.axes, "ch": self.state.channel_axis})


class ChannelMapUnit(BaseTransformerUnit[ChannelMapSettings, AxisArray, AxisArray, ChannelMapProcessor]):
    SETTINGS = ChannelMapSettings

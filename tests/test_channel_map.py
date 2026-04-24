"""Tests for ezmsg.blackrock.channel_map."""

import math
import pathlib

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray, CoordinateAxis, LinearAxis

from ezmsg.blackrock.channel_map import (
    CHANNEL_DTYPE,
    ChannelMapProcessor,
    ChannelMapSettings,
    parse_cmp,
)

CMP_FILE = str(pathlib.Path(__file__).resolve().parent / "128ChannelDefaultMapping.cmp")


def _make_processor(channel_map: str | None = None) -> ChannelMapProcessor:
    return ChannelMapProcessor(settings=ChannelMapSettings(channel_map=channel_map))


def _make_message(n_channels: int, n_time: int = 5) -> AxisArray:
    return AxisArray(
        data=np.zeros((n_time, n_channels)),
        dims=["time", "ch"],
        axes={
            "time": LinearAxis(offset=0.0, gain=0.001),
            "ch": CoordinateAxis(data=np.arange(n_channels), dims=["ch"]),
        },
    )


# ---------------------------------------------------------------------------
# parse_cmp
# ---------------------------------------------------------------------------


class TestParseCmp:
    def test_parse_128_channel_file(self):
        entries = parse_cmp(CMP_FILE)
        assert len(entries) == 128

    def test_first_entry(self):
        entries = parse_cmp(CMP_FILE)
        col, row, bank, elec, label = entries[0]
        assert col == 0
        assert row == 7
        assert bank == "A"
        assert elec == 1
        assert label == "chan1"

    def test_last_entry(self):
        entries = parse_cmp(CMP_FILE)
        col, row, bank, elec, label = entries[-1]
        assert col == 15
        assert row == 0
        assert bank == "D"
        assert elec == 32
        assert label == "chan128"

    def test_bank_boundaries(self):
        entries = parse_cmp(CMP_FILE)
        # Entry 32 → start of bank B
        _, _, bank, elec, label = entries[32]
        assert bank == "B"
        assert elec == 1
        assert label == "chan33"
        # Entry 64 → start of bank C
        _, _, bank, elec, label = entries[64]
        assert bank == "C"
        assert elec == 1
        assert label == "chan65"
        # Entry 96 → start of bank D
        _, _, bank, elec, label = entries[96]
        assert bank == "D"
        assert elec == 1
        assert label == "chan97"

    def test_all_banks_present(self):
        entries = parse_cmp(CMP_FILE)
        banks = {e[2] for e in entries}
        assert banks == {"A", "B", "C", "D"}

    def test_electrode_range(self):
        entries = parse_cmp(CMP_FILE)
        electrodes = [e[3] for e in entries]
        assert min(electrodes) == 1
        assert max(electrodes) == 32


# ---------------------------------------------------------------------------
# ChannelMapProcessor
# ---------------------------------------------------------------------------


class TestChannelMapProcessor:
    def test_single_file(self):
        proc = _make_processor(CMP_FILE)
        out = proc(_make_message(128))
        ch_ax = out.axes["ch"]
        assert isinstance(ch_ax, CoordinateAxis)
        assert ch_ax.data.dtype == CHANNEL_DTYPE
        assert len(ch_ax.data) == 128

    def test_structured_fields(self):
        """First and last channels have correct label, bank, elec, x, y."""
        proc = _make_processor(CMP_FILE)
        out = proc(_make_message(128))
        data = out.axes["ch"].data
        first = data[0]
        assert first["label"] == "chan1"
        assert first["bank"] == "A"
        assert first["elec"] == 1
        assert first["x"] == pytest.approx(0.0)
        assert first["y"] == pytest.approx(7.0)
        last = data[-1]
        assert last["label"] == "chan128"
        assert last["bank"] == "D"
        assert last["elec"] == 32
        assert last["x"] == pytest.approx(15.0)
        assert last["y"] == pytest.approx(0.0)

    def test_original_data_preserved(self):
        proc = _make_processor(CMP_FILE)
        msg = _make_message(128)
        original_data = msg.data.copy()
        out = proc(msg)
        np.testing.assert_array_equal(out.data, original_data)

    def test_time_axis_preserved(self):
        proc = _make_processor(CMP_FILE)
        msg = _make_message(128)
        out = proc(msg)
        assert out.axes["time"] == msg.axes["time"]

    def test_idempotent_on_same_shape(self):
        """Same-shaped input reuses cached state."""
        proc = _make_processor(CMP_FILE)
        out1 = proc(_make_message(128))
        out2 = proc(_make_message(128))
        assert np.array_equal(out1.axes["ch"].data, out2.axes["ch"].data)

    def test_field_access_patterns(self):
        """Can slice structured array by field name across all channels."""
        proc = _make_processor(CMP_FILE)
        out = proc(_make_message(128))
        data = out.axes["ch"].data
        assert data["label"].shape == (128,)
        assert data["bank"].shape == (128,)
        assert data["elec"].shape == (128,)
        assert data["x"].shape == (128,)
        assert data["y"].shape == (128,)

    def test_electrode_values(self):
        """Electrode IDs cycle 1-32 within each bank."""
        proc = _make_processor(CMP_FILE)
        out = proc(_make_message(128))
        data = out.axes["ch"].data
        for start in range(0, 128, 32):
            np.testing.assert_array_equal(data[start : start + 32]["elec"], np.arange(1, 33))


# ---------------------------------------------------------------------------
# Auto-grid for unmapped channels
# ---------------------------------------------------------------------------


class TestAutoGrid:
    def test_extra_channels_appended(self):
        """256 data channels with 128-ch CMP → 256 coordinate rows."""
        proc = _make_processor(CMP_FILE)
        out = proc(_make_message(256))
        assert len(out.axes["ch"].data) == 256

    def test_mapped_channels_unchanged(self):
        """First 128 channels should be identical to the CMP-only case."""
        exact = _make_processor(CMP_FILE)(_make_message(128)).axes["ch"].data
        extra = _make_processor(CMP_FILE)(_make_message(256)).axes["ch"].data
        assert np.array_equal(extra[:128], exact)

    def test_grid_size(self):
        """Auto-grid side length is ceil(sqrt(n_remaining))."""
        proc = _make_processor(CMP_FILE)
        out = proc(_make_message(256))
        data = out.axes["ch"].data
        grid_size = math.ceil(math.sqrt(128))  # 12
        assert data[128:]["x"].max() < grid_size

    def test_row_offset(self):
        """First auto-grid row is max_row + 2."""
        proc = _make_processor(CMP_FILE)
        out = proc(_make_message(256))
        data = out.axes["ch"].data
        # Max row in CMP is 7, so first auto row should be 9.
        assert data[128]["y"] == pytest.approx(9.0)

    def test_first_extra_channel(self):
        """Channel 129 (idx 128): label auto129, bank E, elec 1, col 0, row 9."""
        proc = _make_processor(CMP_FILE)
        out = proc(_make_message(256))
        rec = out.axes["ch"].data[128]
        assert rec["label"] == "auto129"
        assert rec["bank"] == "E"
        assert rec["elec"] == 1
        assert rec["x"] == pytest.approx(0.0)
        assert rec["y"] == pytest.approx(9.0)

    def test_bank_increments_every_32(self):
        proc = _make_processor(CMP_FILE)
        out = proc(_make_message(256))
        data = out.axes["ch"].data
        assert np.all(data[128:160]["bank"] == "E")
        assert np.all(data[160:192]["bank"] == "F")
        assert np.all(data[192:224]["bank"] == "G")
        assert np.all(data[224:256]["bank"] == "H")

    def test_grid_wraps_columns(self):
        proc = _make_processor(CMP_FILE)
        out = proc(_make_message(256))
        data = out.axes["ch"].data
        # 13th extra channel (idx 12) should wrap to col 0, row 10.
        assert data[128 + 12]["x"] == pytest.approx(0.0)
        assert data[128 + 12]["y"] == pytest.approx(10.0)

    def test_auto_labels_sequential(self):
        proc = _make_processor(CMP_FILE)
        out = proc(_make_message(131))
        data = out.axes["ch"].data
        assert data[128]["label"] == "auto129"
        assert data[129]["label"] == "auto130"
        assert data[130]["label"] == "auto131"

    def test_auto_electrode_resets_per_bank(self):
        proc = _make_processor(CMP_FILE)
        out = proc(_make_message(256))
        data = out.axes["ch"].data
        for start in range(128, 256, 32):
            np.testing.assert_array_equal(data[start : start + 32]["elec"], np.arange(1, 33))

    def test_no_cmp_file_all_auto(self):
        """With no CMP file, all channels are auto-generated."""
        proc = _make_processor(None)
        out = proc(_make_message(64))
        data = out.axes["ch"].data
        assert len(data) == 64
        assert data[0]["label"] == "auto1"
        assert data[0]["bank"] == "A"
        assert data[0]["elec"] == 1
        assert data[0]["x"] == pytest.approx(0.0)
        assert data[0]["y"] == pytest.approx(0.0)

    def test_exact_match_no_auto(self):
        """When channel count equals mapped count, no auto-grid is generated."""
        proc = _make_processor(CMP_FILE)
        out = proc(_make_message(128))
        data = out.axes["ch"].data
        assert not any(str(label).startswith("auto") for label in data["label"])

    def test_one_extra_channel(self):
        """Single extra channel gets a 1x1 grid."""
        proc = _make_processor(CMP_FILE)
        out = proc(_make_message(129))
        data = out.axes["ch"].data
        assert len(data) == 129
        rec = data[128]
        assert rec["label"] == "auto129"
        assert rec["bank"] == "E"
        assert rec["x"] == pytest.approx(0.0)
        assert rec["y"] == pytest.approx(9.0)

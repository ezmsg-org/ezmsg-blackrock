"""Tests for ezmsg.blackrock.channel_map."""

import pathlib

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray, CoordinateAxis, LinearAxis

from ezmsg.blackrock.channel_map import (
    CHANNEL_DTYPE,
    ChannelMapProcessor,
    ChannelMapSettings,
)

CMP_FILE = str(pathlib.Path(__file__).resolve().parent / "128ChannelDefaultMapping.cmp")


def _make_processor(
    filepath: str | None = None,
    start_chan: int = 1,
    hs_id: int = 0,
) -> ChannelMapProcessor:
    return ChannelMapProcessor(settings=ChannelMapSettings(filepath=filepath, start_chan=start_chan, hs_id=hs_id))


def _make_message(n_channels: int, n_time: int = 5, ch_data: np.ndarray | None = None) -> AxisArray:
    if ch_data is None:
        ch_data = np.arange(n_channels)
    return AxisArray(
        data=np.zeros((n_time, n_channels)),
        dims=["time", "ch"],
        axes={
            "time": LinearAxis(offset=0.0, gain=0.001),
            "ch": CoordinateAxis(data=ch_data, dims=["ch"]),
        },
    )


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
# Base auto-grid layer (no CMP)
# ---------------------------------------------------------------------------


class TestBaseLayer:
    def test_no_cmp_uses_incoming_labels(self):
        """With no CMP, labels are pulled from the incoming ch axis."""
        labels = np.array([f"in{i}" for i in range(64)])
        proc = _make_processor(None)
        out = proc(_make_message(64, ch_data=labels))
        data = out.axes["ch"].data
        assert data[0]["label"] == "in0"
        assert data[63]["label"] == "in63"

    def test_no_cmp_grid_starts_at_origin(self):
        """Base auto-grid lays channels out from (0, 0) regardless of CMP."""
        proc = _make_processor(None)
        out = proc(_make_message(64))
        data = out.axes["ch"].data
        assert data[0]["x"] == pytest.approx(0.0)
        assert data[0]["y"] == pytest.approx(0.0)
        assert data[0]["bank"] == "A"
        assert data[0]["elec"] == 1

    def test_no_cmp_falls_back_to_chN_labels_when_axis_missing(self):
        """If incoming has no ch axis, labels use ``ch{i+1}``."""
        msg = AxisArray(
            data=np.zeros((5, 4)),
            dims=["time", "ch"],
            axes={"time": LinearAxis(offset=0.0, gain=0.001)},
        )
        out = _make_processor(None)(msg)
        labels = out.axes["ch"].data["label"]
        assert list(labels) == ["ch1", "ch2", "ch3", "ch4"]

    def test_bank_increments_every_32(self):
        proc = _make_processor(None)
        out = proc(_make_message(256))
        data = out.axes["ch"].data
        for k, letter in enumerate("ABCDEFGH"):
            assert np.all(data[k * 32 : (k + 1) * 32]["bank"] == letter)

    def test_electrode_resets_per_bank(self):
        proc = _make_processor(None)
        out = proc(_make_message(256))
        data = out.axes["ch"].data
        for start in range(0, 256, 32):
            np.testing.assert_array_equal(data[start : start + 32]["elec"], np.arange(1, 33))


# ---------------------------------------------------------------------------
# CMP overlay
# ---------------------------------------------------------------------------


class TestCmpOverlay:
    def test_overlay_lands_at_chan_id_minus_one_with_default_start_chan(self):
        """start_chan=1 (default) → CMP entries land at indices 0..127."""
        proc = _make_processor(CMP_FILE)
        out = proc(_make_message(256))
        data = out.axes["ch"].data
        assert data[0]["label"] == "chan1"
        assert data[127]["label"] == "chan128"

    def test_overlay_with_start_chan_offset(self):
        """start_chan=129 → CMP describes the upper half; lower half stays base."""
        proc = _make_processor(CMP_FILE, start_chan=129)
        labels = np.array([f"in{i}" for i in range(256)])
        out = proc(_make_message(256, ch_data=labels))
        data = out.axes["ch"].data
        # Lower half: untouched base layer (labels from incoming axis).
        assert data[0]["label"] == "in0"
        assert data[127]["label"] == "in127"
        # Upper half: CMP entries landed here.
        assert data[128]["label"] == "chan1"
        assert data[255]["label"] == "chan128"

    def test_overlay_preserves_n_total(self):
        """Output axis still has one row per input channel."""
        out = _make_processor(CMP_FILE)(_make_message(256))
        assert len(out.axes["ch"].data) == 256

    def test_overlay_out_of_range_chan_id_skipped(self):
        """CMP entries whose chan_id-1 ≥ n_total are ignored, not appended."""
        proc = _make_processor(CMP_FILE, start_chan=200)  # would write 200..327
        out = proc(_make_message(256))
        data = out.axes["ch"].data
        assert len(data) == 256
        # Indices 199..255 carry the CMP's first 57 entries; rest skipped.
        assert data[199]["label"] == "chan1"
        assert data[255]["label"] == "chan57"

    def test_hs_id_prefixes_labels(self):
        """hs_id != 0 prefixes labels with 'hs{hs_id}-'."""
        proc = _make_processor(CMP_FILE, hs_id=2)
        out = proc(_make_message(128))
        labels = out.axes["ch"].data["label"]
        assert labels[0] == "hs2-chan1"
        assert labels[127] == "hs2-chan128"

    def test_missing_cmp_keeps_base_layer(self):
        """A bad CMP path warns and leaves the existing axis intact."""
        proc = _make_processor("/nonexistent/path.cmp")
        out = proc(_make_message(8))
        labels = out.axes["ch"].data["label"]
        # Base layer survived: labels from incoming (np.arange) → "0".."7".
        assert list(labels) == [str(i) for i in range(8)]


# ---------------------------------------------------------------------------
# Cross-reset state preservation
# ---------------------------------------------------------------------------


class TestStatePreservation:
    def test_axis_preserved_across_reset_with_same_n_ch(self):
        """Reset triggered by a setting change keeps the same axis object."""
        proc = _make_processor(CMP_FILE)
        proc(_make_message(256))
        base_axis = proc.state.channel_axis
        # Push a CMP-related setting change (start_chan) — arms a reset, but
        # n_ch is unchanged so the axis object should be reused in place.
        proc.update_settings(ChannelMapSettings(filepath=CMP_FILE, start_chan=129))
        proc(_make_message(256))
        assert proc.state.channel_axis is base_axis

    def test_n_ch_change_rebuilds_base(self):
        """A different n_ch forces base-layer rebuild on next message."""
        labels_a = np.array([f"a{i}" for i in range(4)])
        labels_b = np.array([f"b{i}" for i in range(8)])
        proc = _make_processor(None)
        proc(_make_message(4, ch_data=labels_a))
        proc(_make_message(8, ch_data=labels_b))
        labels_out = proc.state.channel_axis.data["label"]
        assert list(labels_out) == ["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"]

    def test_successive_cmps_accumulate(self):
        """Pushing a second CMP at a disjoint start_chan adds to the existing overlay."""
        proc = _make_processor(CMP_FILE)  # writes 0..127
        proc(_make_message(256))
        # Second push: same CMP, but offset to indices 128..255.
        proc.update_settings(ChannelMapSettings(filepath=CMP_FILE, start_chan=129))
        out = proc(_make_message(256))
        data = out.axes["ch"].data
        # Both halves now carry CMP labels.
        assert data[0]["label"] == "chan1"
        assert data[127]["label"] == "chan128"
        assert data[128]["label"] == "chan1"
        assert data[255]["label"] == "chan128"

    def test_filepath_cleared_rebuilds_base(self):
        """Pushing filepath=None drops the cumulative overlay and rebuilds the base."""
        proc = _make_processor(CMP_FILE)
        labels = np.array([f"in{i}" for i in range(256)])
        proc(_make_message(256, ch_data=labels))
        # Sanity: CMP overlay populated indices 0..127.
        assert proc.state.cmp_mask[:128].all()
        # Clear signal.
        proc.update_settings(ChannelMapSettings(filepath=None))
        out = proc(_make_message(256, ch_data=labels))
        data = out.axes["ch"].data
        # Overlay gone — every label comes from the incoming axis again.
        assert list(data["label"][:5]) == ["in0", "in1", "in2", "in3", "in4"]
        assert not proc.state.cmp_mask.any()


# ---------------------------------------------------------------------------
# Auto-grid placement avoids CMP positions
# ---------------------------------------------------------------------------


class TestAutoGridPlacement:
    def test_auto_grid_y_offset_below_cmp(self):
        """Auto-grid rows start below the CMP's max y (gap of +2)."""
        proc = _make_processor(CMP_FILE)
        out = proc(_make_message(256))
        data = out.axes["ch"].data
        cmp_y_max = float(data["y"][:128].max())
        auto_y_min = float(data["y"][128:].min())
        assert auto_y_min >= cmp_y_max + 2

    def test_auto_grid_bank_starts_past_cmp(self):
        """Auto-grid banks start one letter past the CMP's highest bank."""
        proc = _make_processor(CMP_FILE)  # CMP uses banks A-D
        out = proc(_make_message(256))
        data = out.axes["ch"].data
        cmp_banks = set(data["bank"][:128].tolist())
        auto_banks = set(data["bank"][128:].tolist())
        assert max(cmp_banks) == "D"
        assert min(auto_banks) == "E"

    def test_auto_grid_skips_cmp_indices_with_offset_start_chan(self):
        """With start_chan=129 the CMP fills 128..255; auto-grid lays out 0..127 below."""
        proc = _make_processor(CMP_FILE, start_chan=129)
        out = proc(_make_message(256))
        data = out.axes["ch"].data
        # CMP positions on the upper half.
        assert data[128]["bank"] == "A"  # CMP's first sorted bank
        # Auto-grid (lower half) is offset below CMP's max y (=7) → y >= 9.
        assert data[0]["y"] >= 9.0
        # And uses banks past CMP's max (D) → starts at E.
        assert data[0]["bank"] == "E"

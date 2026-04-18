"""File-playback integration tests for CerePlexImpedanceProcessor.

Unlike :mod:`test_impedance_integration`, these tests don't need
``nPlayServer`` — they read the ``.ns6`` fixtures directly via
:class:`ezmsg.neo.NeoIterator` and feed 20 ms chunks through the processor.
This keeps the critical path (sequential-sweep tracking + DFT extraction)
under test without the timing / networking complexity of real-time playback.

Two fixtures exercise the matrix:

* ``cplxe_dnss_impedance/Hub1-datafile002.ns6`` — single 128-ch CerePlex.
  Ships with a ``Central_impedances`` reference; tolerated deviation is
  generous because Central over-estimates slightly (typ. ~20 kOhm).  This
  file is the one the algorithm was originally developed against and must
  continue to work.
* ``cplxe_dnss_impedance/Hub1-128002.ns6`` — a 96-ch CerePlex headstage
  (ch 0-95), a 128-ch CerePlex headstage (ch 96-223), and 32 disconnected
  channels (ch 224-255) on one hub.  Requires
  ``headstage_channel_offsets=(0, 96, 224)``; the last "headstage" is a
  dummy range that should produce all-NaN impedance since those channels
  never enter a valid sweep.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.blackrock.cereplex_impedance import (
    CerePlexImpedanceProcessor,
    CerePlexImpedanceSettings,
)

_CHUNK_DUR_S = 0.02  # 20 ms chunks, well under a 100 ms impedance burst


def _run_processor(path: Path, headstage_channel_offsets: tuple[int, ...] = (0,)) -> np.ndarray:
    """Stream ``path`` through the processor in 20 ms chunks.

    Only the analog-signal stream is fed to the processor; event and
    spike messages (if any) are skipped.  Returns the final per-channel
    impedance vector (NaN = unmeasured).
    """
    from ezmsg.neo.source import NeoIterator, NeoIteratorSettings

    proc = CerePlexImpedanceProcessor(
        settings=CerePlexImpedanceSettings(
            headstage_channel_offsets=headstage_channel_offsets,
        )
    )
    it = NeoIterator(NeoIteratorSettings(filepath=path, chunk_dur=_CHUNK_DUR_S))
    for msg in it:
        if not isinstance(msg, AxisArray):
            continue
        if msg.dims != ["time", "ch"]:
            continue
        proc(msg)
    return np.asarray(proc.state.impedance, dtype=np.float64).copy()


def _parse_central_impedances(path: Path) -> np.ndarray:
    """Parse a Central-Suite ``Central_impedances`` report.

    Each data row looks like ``Hub1-chanN\\tNNN kOhm`` or ``Hub1-chanN\\t<= 15kOhm``.
    Returns a float array with NaN for ``<=`` entries (treated as
    disconnected / not comparable).
    """
    values: list[float] = []
    for line in path.read_text().splitlines():
        if "Hub1-chan" not in line:
            continue
        _, _, right = line.partition("\t")
        right = right.strip()
        if right.startswith("<="):
            values.append(np.nan)
            continue
        try:
            values.append(float(right.split()[0]))
        except (ValueError, IndexError):
            values.append(np.nan)
    return np.array(values, dtype=np.float64)


# ---------------------------------------------------------------------------
# Single 128-ch headstage — the original development fixture.
# ---------------------------------------------------------------------------


class TestSingleHeadstage128:
    def test_measures_connected_channels(self, ns6_impedance_path: Path):
        imp = _run_processor(ns6_impedance_path, headstage_channel_offsets=(0,))
        assert imp.shape[0] >= 128
        connected = imp[:128]
        measured_frac = np.mean(~np.isnan(connected))
        assert measured_frac >= 0.95, f"Expected >=95% of channels 1-128 to be measured; got {measured_frac:.2%}"
        measured = connected[~np.isnan(connected)]
        # Central reports 200-240 kOhm on this file; our Parseval-summed
        # p2p estimate runs ~20 kOhm lower.
        assert np.median(measured) == pytest.approx(200.0, abs=30.0)
        assert measured.min() > 100.0
        assert measured.max() < 300.0

    def test_matches_central_impedances_reference(self, test_data_impedance: Path, ns6_impedance_path: Path):
        ref_path = next(test_data_impedance.rglob("Central_impedances"), None)
        if ref_path is None:
            pytest.skip("Central_impedances reference not in dataset")
        ref = _parse_central_impedances(ref_path)
        imp = _run_processor(ns6_impedance_path, headstage_channel_offsets=(0,))
        n = min(128, len(ref), len(imp))
        both = ~np.isnan(ref[:n]) & ~np.isnan(imp[:n])
        assert both.sum() >= 100
        # Central over-estimates by roughly a constant offset; per-channel
        # deviation should stay within ~30 kOhm for the vast majority.
        err = np.abs(imp[:n][both] - ref[:n][both])
        assert np.median(err) < 25.0
        assert np.mean(err < 40.0) >= 0.9


# ---------------------------------------------------------------------------
# Hub1-128002.ns6 — 96-ch + 128-ch headstages + 32 disconnected channels.
# ---------------------------------------------------------------------------


class TestMixedHeadstages:
    def test_measures_populated_headstages_and_skips_disconnected(self, ns6_impedance_mixed_path: Path):
        imp = _run_processor(ns6_impedance_mixed_path, headstage_channel_offsets=(0, 96, 224))
        assert imp.shape[0] == 256

        # Both populated headstages should measure essentially every channel.
        frac_a = np.mean(~np.isnan(imp[:96]))
        frac_b = np.mean(~np.isnan(imp[96:224]))
        assert frac_a >= 0.95, f"96-ch headstage coverage: {frac_a:.2%}"
        assert frac_b >= 0.95, f"128-ch headstage coverage: {frac_b:.2%}"

        # Disconnected tail must never get an impedance value assigned.
        assert np.all(np.isnan(imp[224:])), f"Disconnected channels produced values: {imp[224:][~np.isnan(imp[224:])]}"

        # Measured values should sit in the same ~200 kOhm band as the
        # single-headstage file.
        measured_a = imp[:96][~np.isnan(imp[:96])]
        assert np.median(measured_a) == pytest.approx(200.0, abs=40.0)
        assert np.all(np.isfinite(measured_a))
        measured_b = imp[96:224][~np.isnan(imp[96:224])]
        assert np.median(measured_b) == pytest.approx(200.0, abs=40.0)
        assert np.all(np.isfinite(measured_b))

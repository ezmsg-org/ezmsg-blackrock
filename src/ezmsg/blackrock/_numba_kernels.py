"""Optional numba-jitted kernels for SamplingDelayAlignment's numpy path.

Imported best-effort by :mod:`ezmsg.blackrock.sampling_delay_alignment`; when
numba is not installed the transformer falls back to its portable Array-API
implementation, so nothing here is required. Install the ``accel`` extra to get
it (``pip install ezmsg-blackrock[accel]``).

Both operations the numpy path spends time in become a single fused pass here:

* **FIR** (:func:`fir`) -- ``y[i, c] = sum_k w[k, c] * xext[i + k, c]`` with the
  per-column taps already reversed into ``w``. The tap loop is ``filter_len``
  vectorized multiply-adds, each a full read+write of the whole buffer; this
  reads each input once, several times faster at live chunk sizes and, with the
  threaded variant, ~20x faster on the large buffers offline batch processing
  feeds through. The column (channel) loop is innermost so it stays contiguous
  and auto-vectorizes; accumulation is in the array dtype, so the result matches
  the tap sum within the float32 tolerance the tests pin.
* **Rail forward-fill** (:func:`fill_rails`) -- the single left-to-right pass the
  operation actually is, rather than the O(n log n) Hillis-Steele scan the
  Array-API standard forces (it has no cumulative max). This is the numpy path's
  dominant per-message cost once the FIR is fast.

Both take and return 2D ``(time, column)`` arrays; the caller flattens the
sample shape to columns (matching the MLX conv layout) and reshapes back.
"""

from __future__ import annotations

import numpy.typing as npt
from numba import njit, prange

# Below this many samples the parallel FIR's thread fork-join (~0.1 ms) costs
# more than it saves, and -- more importantly -- spawning a thread pool per
# message inside a live ezmsg graph invites latency jitter and oversubscription.
# Live acquisition chunks are far smaller than this, so they always take the
# serial kernel; only the large buffers of offline batch processing parallelize.
PARALLEL_MIN_SAMPLES = 4096


@njit(cache=True, fastmath=True)
def _fir_serial(xext: npt.NDArray, w: npt.NDArray, out: npt.NDArray) -> None:
    n, n_cols = out.shape
    n_taps = w.shape[0]
    for i in range(n):
        for c in range(n_cols):  # tap 0 seeds the row (avoids a separate zeroing pass)
            out[i, c] = w[0, c] * xext[i, c]
        for k in range(1, n_taps):
            row = i + k
            for c in range(n_cols):  # innermost + contiguous -> auto-vectorized
                out[i, c] += w[k, c] * xext[row, c]


@njit(cache=True, fastmath=True, parallel=True)
def _fir_parallel(xext: npt.NDArray, w: npt.NDArray, out: npt.NDArray) -> None:
    n, n_cols = out.shape
    n_taps = w.shape[0]
    for i in prange(n):
        for c in range(n_cols):
            out[i, c] = w[0, c] * xext[i, c]
        for k in range(1, n_taps):
            row = i + k
            for c in range(n_cols):
                out[i, c] += w[k, c] * xext[row, c]


def fir(xext: npt.NDArray, w: npt.NDArray, out: npt.NDArray) -> None:
    """Fill ``out[i, c] = sum_k w[k, c] * xext[i + k, c]`` (reversed taps in w).

    ``xext`` has ``out.shape[0] + w.shape[0] - 1`` rows (the carried history
    prepended). Picks the threaded kernel only for large buffers; see
    :data:`PARALLEL_MIN_SAMPLES`.
    """
    if out.shape[0] >= PARALLEL_MIN_SAMPLES:
        _fir_parallel(xext, w, out)
    else:
        _fir_serial(xext, w, out)


@njit(cache=True)
def _fill_rails_serial(x: npt.NDArray, thresh: float, out: npt.NDArray) -> None:
    for c in range(x.shape[1]):
        _fill_rails_column(x, thresh, out, c)


@njit(cache=True, parallel=True)
def _fill_rails_parallel(x: npt.NDArray, thresh: float, out: npt.NDArray) -> None:
    # Columns are independent, so the forward-fill's per-column time recurrence
    # parallelizes cleanly across channels -- which the vectorized-across-columns
    # scan it replaces cannot do (its recurrence runs along time).
    for c in prange(x.shape[1]):
        _fill_rails_column(x, thresh, out, c)


@njit(cache=True, inline="always")
def _fill_rails_column(x: npt.NDArray, thresh: float, out: npt.NDArray, c: int) -> None:
    last_valid = x[0, c]  # leading-rail fallback: the first sample, as-is
    seen = False
    for i in range(x.shape[0]):
        v = x[i, c]
        if v >= thresh or v <= -thresh:
            out[i, c] = last_valid if seen else x[0, c]
        else:
            out[i, c] = v
            last_valid = v
            seen = True


def fill_rails(x: npt.NDArray, thresh: float, out: npt.NDArray) -> None:
    """Forward-fill (hold last valid) over railed samples, per column, in place
    of the portable scan. Matches its semantics exactly: a run of ``|x| >=
    thresh`` holds the last valid value, and a leading rail (nothing valid seen
    yet) falls back to the column's first sample. Parallelizes over columns only
    for large buffers; see :data:`PARALLEL_MIN_SAMPLES`."""
    if x.shape[0] >= PARALLEL_MIN_SAMPLES:
        _fill_rails_parallel(x, thresh, out)
    else:
        _fill_rails_serial(x, thresh, out)

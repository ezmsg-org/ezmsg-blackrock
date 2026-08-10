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
  dominant per-message cost once the FIR is fast, so it takes the same loop
  order: time outer, columns innermost and contiguous, with the running
  ``last_valid`` carried as a vector. Walking one column at a time -- the
  obvious shape for a per-column recurrence -- instead strides ``n_cols`` per
  step and streams the buffer once per column, which costs superlinearly in
  channel count (measured 13x slower at 1024 channels x 300 samples, and 20x at
  3000 samples).

Both take and return 2D ``(time, column)`` arrays; the caller flattens the
sample shape to columns (matching the MLX conv layout) and reshapes back.
"""

from __future__ import annotations

import numpy.typing as npt
from numba import get_num_threads, njit, prange

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


@njit(cache=True, inline="always")
def _fill_rails_block(x: npt.NDArray, thresh: float, out: npt.NDArray, c0: int, c1: int) -> None:
    """Forward-fill columns ``[c0, c1)`` in one time-outer pass.

    The recurrence runs along time and the columns are independent, so time is
    the outer loop and the column loop is innermost -- contiguous, and carrying
    ``last_valid`` as a vector rather than a scalar. Walking a column at a time
    instead would stride ``n_cols`` per step and stream the whole buffer once
    per column, which costs superlinearly in channel count.

    ``last_valid`` seeds from row 0, which is also the leading-rail fallback, so
    a rail before any valid sample holds the column's first sample with no
    "have we seen one yet" flag: the seed is only overwritten by a valid sample.
    """
    last_valid = x[0, c0:c1].copy()
    for i in range(x.shape[0]):
        for c in range(c1 - c0):
            v = x[i, c0 + c]
            if v >= thresh or v <= -thresh:
                v = last_valid[c]
            else:
                last_valid[c] = v
            out[i, c0 + c] = v


@njit(cache=True)
def _fill_rails_serial(x: npt.NDArray, thresh: float, out: npt.NDArray) -> None:
    _fill_rails_block(x, thresh, out, 0, x.shape[1])


@njit(cache=True, parallel=True)
def _fill_rails_parallel(x: npt.NDArray, thresh: float, out: npt.NDArray, block: int) -> None:
    # Time carries the recurrence, so the split is over *blocks of columns*:
    # each thread keeps the contiguous inner loop, just over a narrower slice.
    n_cols = x.shape[1]
    n_blocks = (n_cols + block - 1) // block
    for b in prange(n_blocks):
        c0 = b * block
        _fill_rails_block(x, thresh, out, c0, min(c0 + block, n_cols))


# Narrower than this and a thread's inner loop stops filling a cache line, which
# costs more than the extra parallelism buys.
MIN_PARALLEL_BLOCK = 16


def fill_rails(x: npt.NDArray, thresh: float, out: npt.NDArray) -> None:
    """Forward-fill (hold last valid) over railed samples, per column, in place
    of the portable scan. Matches its semantics exactly: a run of ``|x| >=
    thresh`` holds the last valid value, and a leading rail (nothing valid seen
    yet) falls back to the column's first sample. Splits across column blocks
    only for large buffers; see :data:`PARALLEL_MIN_SAMPLES`."""
    if x.shape[0] >= PARALLEL_MIN_SAMPLES:
        n_cols = x.shape[1]
        block = max(MIN_PARALLEL_BLOCK, -(-n_cols // get_num_threads()))
        _fill_rails_parallel(x, thresh, out, block)
    else:
        _fill_rails_serial(x, thresh, out)

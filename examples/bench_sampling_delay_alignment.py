"""Benchmark SamplingDelayAlignment on short, live-acquisition-sized chunks.

The transformer runs once per incoming message, so what matters is per-message
wall time at the chunk sizes an NSP actually delivers (a few samples to a few
hundred), not throughput on a long buffer. This script measures that, with each
message evaluated to completion so MLX's lazy graph construction can't hide in
the timing.

It reports, per configuration:

* **fir** -- alignment only (``rail_threshold=None``).
* **rail+fir** -- with the rail forward-fill enabled, so the fill's share of the
  message cost is visible next to the FIR it feeds.
* **portable** -- the same transformer with its backend fast path suppressed:
  the per-tap multiply-add loop and the log-depth rail scan. This is the
  before/after -- the MLX depthwise conv on the ``mlx`` backend, the numba
  kernels on ``numpy`` (shown only when numba is installed).

To see the parallel FIR kernel, pass large ``--sizes`` (e.g. ``30000,300000``);
below a few thousand samples the numba path runs its serial kernel.

Usage:
    python examples/bench_sampling_delay_alignment.py
    python examples/bench_sampling_delay_alignment.py --channels 256 --reps 500
    python examples/bench_sampling_delay_alignment.py --backends numpy --sizes 300,30000,300000
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from ezmsg.util.messages.axisarray import AxisArray, LinearAxis

import ezmsg.blackrock.sampling_delay_alignment as sda
from ezmsg.blackrock.sampling_delay_alignment import (
    SamplingDelayAlignmentSettings,
    SamplingDelayAlignmentTransformer,
)

FS = 30000.0
RAIL = 8000.0

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover - MLX is an optional, Apple-silicon dep
    mx = None


def _message(data: np.ndarray, backend: str, offset: float) -> AxisArray:
    payload = mx.array(data) if backend == "mlx" else data
    return AxisArray(
        data=payload,
        dims=["time", "ch"],
        axes={"time": LinearAxis(offset=offset, gain=1.0 / FS)},
        key="bench",
    )


def _time_stream(
    backend: str,
    n_ch: int,
    sizes: list[int],
    filter_len: int,
    rail: bool,
    reps: int,
    force_portable: bool = False,
) -> float:
    """Mean milliseconds per message, cycling through ``sizes``.

    ``force_portable`` suppresses the backend fast path (numba kernels on numpy,
    depthwise conv on MLX) so the tap loop / log-scan baseline can be timed. Each
    message is synchronized before the clock is read, so MLX timings measure
    execution rather than graph construction.
    """
    rng = np.random.default_rng(0)
    # Hiding the numba kernels must happen before the first message (state reset
    # is where nb_w is chosen); restore afterward so other configs still use it.
    nb_saved = sda._nb
    if force_portable:
        sda._nb = None
    try:
        proc = SamplingDelayAlignmentTransformer(
            settings=SamplingDelayAlignmentSettings(
                filter_len=filter_len,
                rail_threshold=RAIL if rail else None,
            )
        )
        chunks = [
            _message(rng.standard_normal((size, n_ch)).astype(np.float32), backend, i / FS)
            for i, size in enumerate(sizes)
        ]

        def run_once(i: int) -> None:
            out = proc(chunks[i % len(chunks)])
            if backend == "mlx":
                mx.eval(out.data)

        run_once(0)
        if force_portable and backend == "mlx":
            # The sample shape is fixed across this stream, so the state (and its
            # cached kernel) is not rebuilt -- dropping it here selects the
            # tap-sum for the rest of the run.
            proc.state.conv_w = None
        for i in range(min(reps, 20)):  # warm up
            run_once(i)

        start = time.perf_counter()
        for i in range(reps):
            run_once(i)
        return (time.perf_counter() - start) / reps * 1e3
    finally:
        sda._nb = nb_saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", default="96,256", help="Comma-separated channel counts.")
    parser.add_argument("--sizes", default="3,30,300,1000", help="Comma-separated samples/message.")
    parser.add_argument("--filter-len", type=int, default=13, help="FIR length.")
    parser.add_argument("--reps", type=int, default=300, help="Messages timed per configuration.")
    parser.add_argument(
        "--backends",
        default="numpy,mlx",
        help="Comma-separated backends to time (numpy, mlx).",
    )
    args = parser.parse_args()

    channels = [int(c) for c in args.channels.split(",")]
    sizes = [int(s) for s in args.sizes.split(",")]
    backends = [b.strip() for b in args.backends.split(",")]
    if "mlx" in backends and mx is None:
        print("mlx not installed; skipping the mlx backend.")
        backends = [b for b in backends if b != "mlx"]

    numba_on = sda._nb is not None
    print(f"filter_len={args.filter_len}  fs={FS:.0f} Hz  reps={args.reps}  (ms/message)")
    print(f"numba: {'installed' if numba_on else 'not installed'}\n")
    for backend in backends:
        # A portable-baseline column wherever a fast path exists to compare to:
        # always on MLX, on numpy only when numba is actually installed.
        show_portable = backend == "mlx" or numba_on
        for n_ch in channels:
            print(f"--- {backend}, {n_ch} channels ---")
            header = f"{'samples/msg':>12}  {'fir':>9}  {'rail+fir':>9}"
            if show_portable:
                header += f"  {'portable':>9}  {'speedup':>8}"
            print(header)
            # Each fixed size, then the mixed stream a live pipeline really sees.
            for label, stream in [(str(s), [s]) for s in sizes] + [("mixed", sizes)]:
                fir = _time_stream(backend, n_ch, stream, args.filter_len, False, args.reps)
                both = _time_stream(backend, n_ch, stream, args.filter_len, True, args.reps)
                row = f"{label:>12}  {fir:9.3f}  {both:9.3f}"
                if show_portable:
                    port = _time_stream(backend, n_ch, stream, args.filter_len, False, args.reps, force_portable=True)
                    row += f"  {port:9.3f}  {port / fir:7.2f}x"
                print(row)
            print()


if __name__ == "__main__":
    main()

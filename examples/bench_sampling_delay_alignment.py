"""Benchmark SamplingDelayAlignment on short, live-acquisition-sized chunks.

The transformer runs once per incoming message, so what matters is per-message
wall time at the chunk sizes an NSP actually delivers (a few samples to a few
hundred), not throughput on a long buffer. This script measures that, with each
message evaluated to completion so MLX's lazy graph construction can't hide in
the timing.

It reports three things per configuration:

* **fir** -- alignment only (``rail_threshold=None``).
* **rail+fir** -- with the rail forward-fill enabled, so the fill's share of the
  message cost is visible next to the FIR it feeds.
* **tap-loop** (MLX only) -- the same transformer with its cached depthwise-conv
  kernel suppressed, which falls back to the portable per-tap multiply-add. This
  is the before/after for the MLX fast path.

Usage:
    python examples/bench_sampling_delay_alignment.py
    python examples/bench_sampling_delay_alignment.py --channels 256 --reps 500
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from ezmsg.util.messages.axisarray import AxisArray, LinearAxis

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
    force_tap_loop: bool = False,
) -> float:
    """Mean milliseconds per message, cycling through ``sizes``.

    Each message is synchronized before the clock is read, so MLX timings
    measure execution rather than graph construction.
    """
    rng = np.random.default_rng(0)
    proc = SamplingDelayAlignmentTransformer(
        settings=SamplingDelayAlignmentSettings(
            filter_len=filter_len,
            rail_threshold=RAIL if rail else None,
        )
    )
    chunks = [
        _message(rng.standard_normal((size, n_ch)).astype(np.float32), backend, i / FS) for i, size in enumerate(sizes)
    ]

    def run_once(i: int) -> None:
        out = proc(chunks[i % len(chunks)])
        if backend == "mlx":
            mx.eval(out.data)

    run_once(0)
    if force_tap_loop:
        # The sample shape is fixed across this stream, so the state (and with it
        # the cached kernel) is not rebuilt -- dropping it here selects the
        # portable tap-sum for the rest of the run.
        proc.state.conv_w = None
    for i in range(min(reps, 20)):  # warm up
        run_once(i)

    start = time.perf_counter()
    for i in range(reps):
        run_once(i)
    return (time.perf_counter() - start) / reps * 1e3


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

    print(f"filter_len={args.filter_len}  fs={FS:.0f} Hz  reps={args.reps}  (ms/message)\n")
    for backend in backends:
        for n_ch in channels:
            print(f"--- {backend}, {n_ch} channels ---")
            header = f"{'samples/msg':>12}  {'fir':>9}  {'rail+fir':>9}"
            if backend == "mlx":
                header += f"  {'tap-loop':>9}  {'speedup':>8}"
            print(header)
            # Each fixed size, then the mixed stream a live pipeline really sees.
            for label, stream in [(str(s), [s]) for s in sizes] + [("mixed", sizes)]:
                fir = _time_stream(backend, n_ch, stream, args.filter_len, False, args.reps)
                both = _time_stream(backend, n_ch, stream, args.filter_len, True, args.reps)
                row = f"{label:>12}  {fir:9.3f}  {both:9.3f}"
                if backend == "mlx":
                    loop = _time_stream(backend, n_ch, stream, args.filter_len, False, args.reps, force_tap_loop=True)
                    row += f"  {loop:9.3f}  {loop / fir:7.2f}x"
                print(row)
            print()


if __name__ == "__main__":
    main()

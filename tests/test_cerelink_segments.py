"""Gaps in device time cut a batch and start a new ``AxisArray.segment``. No hardware needed."""

import asyncio

import numpy as np
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.blackrock.cerelink import CereLinkSignalProducer, CereLinkSignalSettings, DeviceType

FS = 30_000
PERIOD_NS = 1_000_000_000 // FS
T0_NS = 1_700_000_000_000_000_000  # epoch-scale, like a PTP device clock


def _producer(**settings) -> CereLinkSignalProducer:
    """A signal producer with its buffers set up as ``_setup_subscription`` would, minus the device."""
    producer = CereLinkSignalProducer(
        settings=CereLinkSignalSettings(device_type=DeviceType.NPLAY, cbtime=True, microvolts=False, **settings)
    )
    st = producer.state
    st.session = object()
    st.n_channels = 2
    st.buffer_data = np.zeros((FS, 2), dtype=np.int16)
    st.buffer_timestamps = np.zeros(FS, dtype=np.uint64)
    st.template = AxisArray(
        np.zeros((0, 2)), dims=["time", "ch"], axes={"time": AxisArray.TimeAxis(FS, offset=0.0)}, stream_dim="time"
    )
    st.data_event = asyncio.Event()
    producer._segment = 0
    return producer


def _timestamps(n: int, gaps: dict[int, float] | None = None) -> np.ndarray:
    """``n`` device timestamps at 30 kHz; ``gaps`` maps a sample index to seconds lost just before it."""
    ts = T0_NS + np.arange(n, dtype=np.int64) * PERIOD_NS
    for idx, dur in (gaps or {}).items():
        ts[idx:] += int(dur * 1e9)
    return ts.astype(np.uint64)


def _run(producer, batches) -> list[AxisArray]:
    async def go():
        loop = asyncio.get_running_loop()
        out = []
        for ts in batches:
            producer._handle_group_batch(np.zeros((len(ts), 2), dtype=np.int16), ts, loop)
            while producer.state.read_idx != producer.state.write_idx:
                out.append(await producer._produce())
        return out

    return asyncio.run(go())


def test_continuous_batches_stay_in_one_segment():
    ts = _timestamps(900)
    msgs = _run(_producer(), [ts[:300], ts[300:600], ts[600:]])
    assert [m.data.shape[0] for m in msgs] == [300, 300, 300]
    assert [m.segment for m in msgs] == [0, 0, 0]


def test_gap_inside_a_batch_cuts_it_and_starts_a_segment():
    ts = _timestamps(600, {200: 0.5})
    msgs = _run(_producer(), [ts])
    assert [m.data.shape[0] for m in msgs] == [200, 400]
    assert [m.segment for m in msgs] == [0, 1]
    # Each part is stamped with its own first sample's device time.
    assert msgs[1].axes["time"].offset == int(ts[200]) / 1e9
    assert msgs[1].axes["time"].offset - msgs[0].axes["time"].offset > 0.5


def test_gap_between_batches_starts_a_segment_without_a_cut():
    ts = _timestamps(600, {300: 0.5})
    msgs = _run(_producer(), [ts[:300], ts[300:]])
    assert [m.data.shape[0] for m in msgs] == [300, 300]
    assert [m.segment for m in msgs] == [0, 1]


def test_gap_under_the_tolerance_is_not_a_gap():
    ts = _timestamps(600, {200: 0.005, 400: 0.012})  # 5 ms: tolerated; 12 ms: not
    msgs = _run(_producer(), [ts])
    assert [m.data.shape[0] for m in msgs] == [400, 200]
    assert [m.segment for m in msgs] == [0, 1]


def test_a_step_backward_is_a_gap():
    ts = _timestamps(600)
    ts[300:] -= np.uint64(int(0.5 * 1e9))
    msgs = _run(_producer(), [ts])
    assert [m.segment for m in msgs] == [0, 1]


def test_infinite_tolerance_turns_detection_off():
    ts = _timestamps(600, {200: 0.5})
    msgs = _run(_producer(gap_tolerance=float("inf")), [ts])
    assert [m.data.shape[0] for m in msgs] == [600]
    assert [m.segment for m in msgs] == [0]


def test_tolerance_can_change_without_reopening_the_device():
    producer = _producer()
    producer._hash = 0  # as if already running
    producer.update_settings(CereLinkSignalSettings(device_type=DeviceType.NPLAY, cbtime=True, gap_tolerance=1.0))
    assert producer._hash == 0
    assert producer.settings.gap_tolerance == 1.0

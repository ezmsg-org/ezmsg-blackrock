"""Every message declares the ``config_id`` of its stream, digested once per (re)build of the
template, and ``segment`` numbers start at 0 on the first open and move on only on a reopen or a
gap in device time. Driven by a mock pycbsdk Session (no hardware)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
from ezmsg.util.messages.axisarray import chain_config_id, compute_config_id

from ezmsg.blackrock import cerelink
from ezmsg.blackrock.cerelink import (
    CereLinkSignalProducer,
    CereLinkSignalSettings,
    CereLinkSpikeProducer,
    CereLinkSpikeSettings,
    DeviceType,
    SampleRate,
)
from ezmsg.blackrock.channel_map import ChannelMapSettings

FS = 30_000
PERIOD_NS = 1_000_000_000 // FS
T0_NS = 1_700_000_000_000_000_000


def _session(channels=(1, 2), labels=None, scale=0.25) -> MagicMock:
    """The few Session calls `_setup_subscription` and `_build_ch_info` make."""
    sess = MagicMock()
    sess.get_group_channels.return_value = list(channels)
    sess.get_matching_channel_ids.return_value = list(channels)
    sess.get_channel_conversion.return_value = {"scale": scale, "offset": 0.0, "unit": "uV"}
    sess.get_channel_label.side_effect = labels or (lambda ch_id: f"elec{ch_id}")
    return sess


def _signal(sess=None, **settings) -> CereLinkSignalProducer:
    producer = CereLinkSignalProducer(
        settings=CereLinkSignalSettings(device_type=DeviceType.NPLAY, cbtime=True, **settings)
    )
    producer.state.session = sess or _session()
    producer.state.ch_positions = {}
    producer._setup_subscription(None)
    return producer


def _stream(producer, n_batches=3, n=300, gap_after=None) -> list:
    """Feed ``n_batches`` contiguous batches (a 0.5 s gap before batch ``gap_after``) and produce."""

    async def go():
        loop = asyncio.get_running_loop()
        out, t = [], T0_NS
        for i in range(n_batches):
            if i == gap_after:
                t += 500_000_000
            ts = (t + np.arange(n, dtype=np.int64) * PERIOD_NS).astype(np.uint64)
            t += n * PERIOD_NS
            producer._handle_group_batch(np.ones((n, producer.state.n_channels), dtype=np.int16), ts, loop)
            while producer.state.read_idx != producer.state.write_idx:
                out.append(await producer._produce())
        return out

    return asyncio.run(go())


def expected_config_id(m) -> int:
    """The layout digest, with a raw stream's per-channel scaling folded in."""
    if "conversion" not in m.attrs:
        return compute_config_id(m)
    n_ch = m.data.shape[1]
    scaling = (tuple(np.broadcast_to(m.attrs[k], n_ch).tolist()) for k in ("conversion", "offset"))
    return chain_config_id(compute_config_id(m), *scaling)


def _declares_its_config(msgs) -> int:
    ids = {m.config_id for m in msgs}
    assert len(ids) == 1
    for m in msgs:
        assert m.config_id != 0
        assert m.config_id == expected_config_id(m)
    return ids.pop()


def test_signal_messages_declare_their_config_id_raw_and_microvolts():
    raw = _declares_its_config(_stream(_signal(microvolts=False)))
    uv = _declares_its_config(_stream(_signal(microvolts=True)))
    assert raw != uv  # int16 vs float64


def test_config_id_is_computed_once_not_per_message():
    with patch.object(cerelink, "compute_config_id", wraps=compute_config_id) as spy:
        producer = _signal()
        msgs = _stream(producer, n_batches=5)
    assert len(msgs) == 5
    assert spy.call_count == 1


def test_a_gap_changes_the_segment_but_not_the_config_id():
    msgs = _stream(_signal(), n_batches=4, gap_after=2)
    _declares_its_config(msgs)
    s0 = msgs[0].segment
    assert [m.segment for m in msgs] == [s0, s0, s0 + 1, s0 + 1]


def test_reloading_channel_maps_restamps_but_continues_the_segment():
    labels = {"suffix": "a"}
    producer = _signal(_session(labels=lambda ch_id: f"elec{ch_id}{labels['suffix']}"))
    (before,) = _stream(producer, n_batches=1)
    labels["suffix"] = "b"
    producer.settings = CereLinkSignalSettings(
        device_type=DeviceType.NPLAY, cbtime=True, cmp_configs=(ChannelMapSettings(filepath="x.cmp"),)
    )
    producer._reload_channel_maps_in_place()
    (after,) = _stream(producer, n_batches=1)
    assert after.config_id == compute_config_id(after) != before.config_id
    assert after.segment == before.segment  # no reopen, no data lost


def test_toggling_microvolts_restamps_but_continues_the_segment():
    producer = _signal(microvolts=False)
    producer._hash = 0  # as if running
    (before,) = _stream(producer, n_batches=1)
    producer.update_settings(CereLinkSignalSettings(device_type=DeviceType.NPLAY, cbtime=True, microvolts=True))
    assert producer._hash == 0  # not reopened
    (after,) = _stream(producer, n_batches=1)
    assert after.data.dtype == np.float64
    assert after.config_id == compute_config_id(after) != before.config_id
    assert after.segment == before.segment  # no reopen, no data lost
    # A measured gap after it still starts a segment, in the same config.
    gapped = _stream(producer, n_batches=2, gap_after=1)
    assert [m.segment for m in gapped] == [before.segment, before.segment + 1]
    assert {m.config_id for m in gapped} == {after.config_id}


def test_a_raw_stream_folds_its_scaling_into_config_id():
    """attrs are not in the layout digest, so a new scale at the same int16 dtype needs folding in."""
    quarter = _declares_its_config(_stream(_signal(_session(scale=0.25), microvolts=False), n_batches=1))
    half = _declares_its_config(_stream(_signal(_session(scale=0.5), microvolts=False), n_batches=1))
    assert quarter != half
    # Microvolt values are already scaled, so the scale is not part of their config.
    uv = [_declares_its_config(_stream(_signal(_session(scale=s)), n_batches=1)) for s in (0.25, 0.5)]
    assert uv[0] == uv[1]


def _open(producer, sess):
    sess.wait_until_running = AsyncMock()
    with patch.object(cerelink, "Session", return_value=sess):
        asyncio.run(producer._open_and_configure())


def test_two_fresh_sources_both_start_at_segment_0():
    for _ in range(2):
        producer = CereLinkSignalProducer(settings=CereLinkSignalSettings(device_type=DeviceType.NPLAY, cbtime=True))
        _open(producer, _session())
        assert [m.segment for m in _stream(producer, n_batches=2)] == [0, 0]


def test_reopening_a_reconfigured_device_restamps_and_starts_a_segment():
    producer = CereLinkSignalProducer(settings=CereLinkSignalSettings(device_type=DeviceType.NPLAY, cbtime=True))
    _open(producer, _session(channels=(1, 2)))
    (first,) = _stream(producer, n_batches=1)
    _open(producer, _session(channels=(1, 2, 3)))
    (second,) = _stream(producer, n_batches=1)
    assert second.segment == first.segment + 1
    assert second.data.shape[1] == 3
    assert second.config_id == compute_config_id(second) != first.config_id
    # The same configuration reopened is the same config, in a new segment.
    _open(producer, _session(channels=(1, 2, 3)))
    (third,) = _stream(producer, n_batches=1)
    assert third.segment == second.segment + 1
    assert third.config_id == second.config_id


def _spike_windows(producer, n=3) -> list:
    async def go():
        loop = asyncio.get_running_loop()
        out = []
        for _ in range(n):
            producer._handle_spike(MagicMock(chid=1, type=0, time=producer.state.window_origin_ns), loop)
            out.append(await producer._produce())
        return out

    producer.state.window_origin_ns = T0_NS
    return asyncio.run(go())


def test_spike_windows_declare_one_config_id_distinct_from_the_signal():
    with patch.object(cerelink, "compute_config_id", wraps=compute_config_id) as spy:
        producer = CereLinkSpikeProducer(
            settings=CereLinkSpikeSettings(device_type=DeviceType.NPLAY, cbtime=True, spike_buffer_dur=0.01)
        )
        producer.state.session = _session()
        producer.state.ch_positions = {}
        producer._setup_subscription(None)
        msgs = _spike_windows(producer)
    assert spy.call_count == 1
    spikes = _declares_its_config(msgs)
    assert msgs[0].data.shape[1:] == (2, 7)
    assert len({m.segment for m in msgs}) == 1
    signal = _declares_its_config(_stream(_signal(subscribe_rate=SampleRate.SR_RAW), n_batches=1))
    assert spikes != signal

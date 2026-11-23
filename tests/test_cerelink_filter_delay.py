"""Filter-delay compensation: the low-pass group delay is subtracted from the time axis. No hardware needed."""

import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.blackrock.cerelink import (
    CereLinkSignalProducer,
    CereLinkSignalSettings,
    DeviceType,
    SampleRate,
    butterworth_dc_group_delay,
)

T0_NS = 1_700_000_000_000_000_000
HUB_DELAY = 2 / (2 * np.pi * 7500)


@pytest.mark.parametrize("order, factor", [(1, 1.0), (2, np.sqrt(2)), (3, 2.0)])
def test_butterworth_group_delay_closed_forms(order, factor):
    assert butterworth_dc_group_delay(order, 250.0) == pytest.approx(factor / (2 * np.pi * 250.0))


def test_no_lowpass_has_no_delay():
    assert butterworth_dc_group_delay(0, 250.0) == 0.0
    assert butterworth_dc_group_delay(3, 0.0) == 0.0


def _offset(device_type, rate=SampleRate.SR_RAW, filt=None, **settings) -> tuple[float, dict]:
    producer = CereLinkSignalProducer(
        settings=CereLinkSignalSettings(
            device_type=device_type, subscribe_rate=rate, cbtime=True, microvolts=False, **settings
        )
    )
    st = producer.state
    st.session = MagicMock()
    st.session.get_channel_smpfilter.return_value = 1
    st.session.get_filter_info.return_value = filt
    st.n_channels = 1
    st.buffer_data = np.zeros((100, 1), dtype=np.int16)
    st.buffer_timestamps = np.zeros(100, dtype=np.uint64)
    st.conversion = np.ones(1)
    st.conversion_offset = np.zeros(1)
    st.filter_delay = producer._compute_filter_delay([1])
    st.template = AxisArray(
        np.zeros((0, 1)),
        dims=["time", "ch"],
        axes={"time": AxisArray.TimeAxis(rate.hz, offset=0.0)},
        attrs=producer._signal_attrs(),
        stream_dim="time",
    )
    st.data_event = asyncio.Event()
    producer._segment = 0

    async def go():
        ts = (T0_NS + np.arange(10) * 33_333).astype(np.uint64)
        producer._handle_group_batch(np.zeros((10, 1), dtype=np.int16), ts, asyncio.get_running_loop())
        return await producer._produce()

    msg = asyncio.run(go())
    return msg.axes["time"].offset - T0_NS / 1e9, msg.attrs


def test_hub_raw_is_shifted_by_the_front_end_delay():
    offset, attrs = _offset(DeviceType.HUB1)
    assert offset == pytest.approx(-HUB_DELAY, abs=1e-6)
    assert attrs["filter_delay"] == pytest.approx(HUB_DELAY)


def test_nsp_raw_is_not_shifted():
    offset, attrs = _offset(DeviceType.NSP)
    assert offset == pytest.approx(0.0, abs=1e-6)
    assert "filter_delay" not in attrs


def test_compensation_can_be_turned_off():
    offset, attrs = _offset(DeviceType.HUB1, compensate_filter_delay=False)
    assert offset == pytest.approx(0.0, abs=1e-6)
    assert "filter_delay" not in attrs


def test_filtered_group_adds_the_digital_low_pass():
    filt = {"lpfreq": 250_000, "lporder": 4, "hpfreq": 0, "hporder": 0, "label": ""}
    expected = butterworth_dc_group_delay(4, 250.0)
    offset, _ = _offset(DeviceType.NSP, SampleRate.SR_1kHz, filt)
    assert offset == pytest.approx(-expected, abs=1e-6)

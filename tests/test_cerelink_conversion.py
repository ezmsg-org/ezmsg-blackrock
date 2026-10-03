"""Raw-to-microvolt conversion derived from pycbsdk channel scaling.

``CereLinkSignalSource`` multiplies by it when ``microvolts`` is set, and
otherwise records it as ``attrs["conversion"]`` / ``attrs["offset"]``
(``uV = sample * conversion + offset``).
"""

import numpy as np
import pytest

from ezmsg.blackrock.cerelink import _channel_conversion, _per_channel_or_scalar


def _scaling(digmin, digmax, anamin, anamax, anaunit="uV"):
    return {"digmin": digmin, "digmax": digmax, "anamin": anamin, "anamax": anamax, "anaunit": anaunit}


def test_symmetric_front_end_range():
    conversion, offset = _channel_conversion(_scaling(-32764, 32764, -8191, 8191))
    assert conversion == pytest.approx(0.25)
    assert offset == pytest.approx(0.0)


def test_millivolt_channels_convert_to_microvolts():
    conversion, offset = _channel_conversion(_scaling(-32764, 32764, -5000, 5000, "mV"))
    assert conversion == pytest.approx(5000 * 1000 / 32764)
    assert offset == pytest.approx(0.0)


def test_asymmetric_range_has_an_offset():
    scaling = _scaling(0, 1000, -10, 90)
    conversion, offset = _channel_conversion(scaling)
    # Both ends of the digital range map onto the analog range.
    assert 0 * conversion + offset == pytest.approx(-10)
    assert 1000 * conversion + offset == pytest.approx(90)


@pytest.mark.parametrize("scaling", [None, {}, _scaling(5, 5, 0, 1)])
def test_unusable_scaling_passes_raw_values_through(scaling):
    assert _channel_conversion(scaling) == (1.0, 0.0)


def test_shared_values_collapse_to_a_scalar():
    value = _per_channel_or_scalar(np.array([0.25, 0.25, 0.25]))
    assert isinstance(value, float) and value == 0.25


def test_mixed_values_stay_per_channel():
    values = np.array([0.25, 152.6])
    out = _per_channel_or_scalar(values)
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, values)
    assert out is not values


def test_toggling_microvolts_relabels_the_template_once():
    """``microvolts`` is a non-reset setting: the template's attrs follow it on
    the settings change, not per message."""
    from ezmsg.util.messages.axisarray import AxisArray

    from ezmsg.blackrock.cerelink import CereLinkSignalProducer, CereLinkSignalSettings, DeviceType

    producer = CereLinkSignalProducer(settings=CereLinkSignalSettings(device_type=DeviceType.NPLAY, microvolts=True))
    st = producer.state
    st.conversion = np.array([0.25, 0.25])
    st.conversion_offset = np.zeros(2)
    st.template = AxisArray(np.zeros((0, 2)), dims=["time", "ch"], attrs=producer._signal_attrs())
    assert st.template.attrs == {"unit": "uV", "manufacturer": "CereLink", "device": "NPLAY"}

    producer.update_settings(CereLinkSignalSettings(device_type=DeviceType.NPLAY, microvolts=False))
    assert st.template.attrs["conversion"] == 0.25
    assert st.template.attrs["offset"] == 0.0
    assert st.template.attrs["unit"] == "uV"

    producer.update_settings(CereLinkSignalSettings(device_type=DeviceType.NPLAY, microvolts=True))
    assert "conversion" not in st.template.attrs and "offset" not in st.template.attrs

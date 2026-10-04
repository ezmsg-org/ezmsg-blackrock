"""Raw-to-microvolt conversion from pycbsdk's per-channel physical conversion.

``CereLinkSignalSource`` multiplies by it when ``microvolts`` is set, and
otherwise records it as ``attrs["conversion"]`` / ``attrs["offset"]``
(``uV = sample * conversion + offset``).
"""

import numpy as np
import pytest

from ezmsg.blackrock.cerelink import _channel_conversion, _per_channel_or_scalar


def _conversion(scale, offset=0.0, unit="uV"):
    """What pycbsdk's ``Session.get_channel_conversion`` returns."""
    return {"scale": scale, "offset": offset, "unit": unit}


def test_microvolt_channels_pass_through():
    assert _channel_conversion(_conversion(0.25)) == (0.25, 0.0)


def test_millivolt_channels_convert_to_microvolts():
    conversion, offset = _channel_conversion(_conversion(5000 / 32764, -0.5, "mV"))
    assert conversion == pytest.approx(5000 * 1000 / 32764)
    assert offset == pytest.approx(-500.0)


@pytest.mark.parametrize("value", [None, _conversion(2.0, 0.0, "degC")], ids=["no usable map", "not a voltage"])
def test_unusable_channels_pass_raw_values_through(value):
    assert _channel_conversion(value) == (1.0, 0.0)


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

"""Integration tests requiring a running nPlayServer instance (4-channel data)."""

import ast
import base64
import json
import time
from pathlib import Path

import ezmsg.core as ez
import numpy as np
import pytest
from conftest import run_nplayserver
from ezmsg.util.messagecodec import NDARRAY_TYPE, PICKLE_TYPE, TYPE, LogStart, import_type
from ezmsg.util.messagelogger import MessageLogger
from ezmsg.util.terminate import TerminateOnTotal
from pycbsdk import ChannelType, DeviceType, SampleRate

from ezmsg.blackrock.cerelink import CereLinkSettings, CereLinkSource

pytestmark = pytest.mark.integration

N_MESSAGES = 50


def _object_hook(obj: dict) -> object:
    """Like ezmsg's MessageDecoder hook, but handles structured numpy dtypes."""
    obj_type = obj.get(TYPE)
    if obj_type is None:
        return obj
    if obj_type == NDARRAY_TYPE:
        dtype_str = obj.get("dtype", "")
        try:
            dtype = np.dtype(dtype_str)
        except TypeError:
            dtype = np.dtype(ast.literal_eval(dtype_str))
        buf = base64.b64decode(obj["data"].encode("ascii"))
        return np.frombuffer(buf, dtype=dtype).reshape(obj["shape"])
    if obj_type == PICKLE_TYPE:
        import pickle

        buf = base64.b64decode(obj["data"].encode("ascii"))
        return pickle.loads(buf)  # noqa: S301
    cls = import_type(obj_type)
    del obj[TYPE]
    return cls(**obj)


def _read_log(path: Path) -> list:
    """Read a MessageLogger file, returning deserialized message objects."""
    messages = []
    with open(path) as f:
        for line in f:
            entry = json.loads(line, object_hook=_object_hook)
            if isinstance(entry["obj"], LogStart):
                continue
            messages.append(entry["obj"])
    return messages


def _run_source(settings: CereLinkSettings, log_path: Path, n_messages: int = N_MESSAGES) -> list:
    """Run CereLinkSource, log messages, return deserialized AxisArray list."""
    comps = {
        "SRC": CereLinkSource(settings),
        "LOG": MessageLogger(output=log_path),
        "TERM": TerminateOnTotal(total=n_messages),
    }
    conns = (
        (comps["SRC"].OUTPUT_SIGNAL, comps["LOG"].INPUT_MESSAGE),
        (comps["LOG"].OUTPUT_MESSAGE, comps["TERM"].INPUT_MESSAGE),
    )
    ez.run(components=comps, connections=conns)
    return _read_log(log_path)


@pytest.fixture(scope="module")
def nplayserver(nplayserver_binary, ns6_path):
    """nPlayServer playing back 4-channel data for this module."""
    with run_nplayserver(nplayserver_binary, ns6_path) as proc:
        yield proc


class TestCereLinkSource:
    """CereLinkSource end-to-end tests (4-channel data, max n_chans=4)."""

    def test_receive_data(self, nplayserver, tmp_path):
        n_ch = 2
        messages = _run_source(
            CereLinkSettings(
                device_type=DeviceType.NPLAY,
                n_chans=n_ch,
                channel_type=ChannelType.FRONTEND,
                sample_rate=SampleRate.SR_30kHz,
                microvolts=False,
                cbtime=True,
            ),
            tmp_path / "log.jsonl",
        )
        assert len(messages) >= N_MESSAGES
        for msg in messages:
            assert msg.data.shape[0] > 0, "empty time dimension"
            assert msg.data.shape[1] == n_ch
            assert msg.data.dtype == np.int16
            assert 1.0 / msg.axes["time"].gain == pytest.approx(30_000.0)
            assert msg.key == "SR_30kHz"
            assert msg.attrs["unit"] == "raw"

    def test_microvolts(self, nplayserver, tmp_path):
        n_ch = 3
        messages = _run_source(
            CereLinkSettings(
                device_type=DeviceType.NPLAY,
                n_chans=n_ch,
                channel_type=ChannelType.FRONTEND,
                sample_rate=SampleRate.SR_30kHz,
                microvolts=True,
                cbtime=True,
            ),
            tmp_path / "log.jsonl",
        )
        assert len(messages) >= N_MESSAGES
        for msg in messages:
            assert msg.data.shape[1] == n_ch
            assert msg.data.dtype == np.float64
            assert msg.attrs["unit"] == "uV"

    def test_monotonic_timestamps(self, nplayserver, tmp_path):
        messages = _run_source(
            CereLinkSettings(
                device_type=DeviceType.NPLAY,
                n_chans=1,
                channel_type=ChannelType.FRONTEND,
                sample_rate=SampleRate.SR_30kHz,
                microvolts=False,
                cbtime=False,
            ),
            tmp_path / "log.jsonl",
        )
        assert len(messages) >= N_MESSAGES
        offsets = [msg.axes["time"].offset for msg in messages]
        assert all(offsets[i] <= offsets[i + 1] for i in range(len(offsets) - 1))

    def test_all_channels(self, nplayserver, tmp_path):
        n_ch = 4
        messages = _run_source(
            CereLinkSettings(
                device_type=DeviceType.NPLAY,
                n_chans=n_ch,
                channel_type=ChannelType.FRONTEND,
                sample_rate=SampleRate.SR_30kHz,
                microvolts=False,
                cbtime=True,
            ),
            tmp_path / "log.jsonl",
        )
        assert len(messages) >= N_MESSAGES
        for msg in messages:
            assert msg.data.shape[1] == n_ch

    def test_offsets_near_monotonic(self, nplayserver, tmp_path):
        """Verify time offsets are close to time.monotonic() (cbtime=False)."""
        t_before = time.monotonic()
        messages = _run_source(
            CereLinkSettings(
                device_type=DeviceType.NPLAY,
                n_chans=2,
                channel_type=ChannelType.FRONTEND,
                sample_rate=SampleRate.SR_30kHz,
                microvolts=False,
                cbtime=False,
            ),
            tmp_path / "log.jsonl",
        )
        t_after = time.monotonic()
        assert len(messages) >= N_MESSAGES
        offsets = [msg.axes["time"].offset for msg in messages]
        for offset in offsets:
            assert t_before <= offset <= t_after, f"offset {offset:.3f} outside [{t_before:.3f}, {t_after:.3f}]"

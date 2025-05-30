import threading
import typing

import numpy.typing as npt


class ClockSync:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, alpha: float = 0.1, sysfreq: float = 1.0):
        if not hasattr(self, "_initialized"):
            self._alpha = alpha
            self._initialized = True
            self._offset: typing.Optional[float] = None
            self._last_pair: typing.Optional[typing.Tuple[int, float]] = None
            self._sysfreq = sysfreq

    def add_pair(self, nsp_time, sys_time):
        # The protocol monitor packet arrived in the system 560 usec after
        #  it left the Gemini device. We can get the Gemini clock in our
        #  system time by subtracting this number:
        sys_time = sys_time - 0.00056
        offset = sys_time - nsp_time / self.sysfreq
        if self.offset is None or (
            self._last_pair is not None
            and (nsp_time < self._last_pair[0] or sys_time < self._last_pair[1])
        ):
            self._offset = offset
        self._offset = (1 - self._alpha) * self._offset + self._alpha * offset
        self._last_pair = (nsp_time, sys_time)

    @property
    def offset(self) -> float:
        with self._lock:
            return self._offset

    @property
    def sysfreq(self) -> float:
        with self._lock:
            return self._sysfreq

    @typing.overload
    def nsp2system(self, nsp_timestamp: int) -> float: ...

    @typing.overload
    def nsp2system(self, nsp_timestamp: npt.NDArray[int]) -> npt.NDArray[float]: ...

    def nsp2system(self, nsp_timestamp):
        # offset = system - nsp / sysfreq --> system = nsp / sysfreq + offset
        with self._lock:
            return nsp_timestamp / self._sysfreq + self._offset

    @typing.overload
    def system2nsp(self, system_timestamp: float) -> int: ...

    @typing.overload
    def system2nsp(self, system_timestamp: npt.NDArray[int]) -> npt.NDArray[float]: ...

    def system2nsp(self, system_timestamp):
        # offset = system - nsp / sysfreq --> nsp = (system - offset) * sysfreq
        with self._lock:
            return int((system_timestamp - self._offset) * self._sysfreq)

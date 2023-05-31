import typing

import numpy as np
import ezmsg.core as ez

from ezmsg.util.messages.axisarray import AxisArray

class CerebusSourceSettings(ez.Settings):
    ...

class CerebusSourceState(ez.State):
    ...

class CerebusSource(ez.Unit):
    SETTINGS: CerebusSourceSettings
    STATE: CerebusSourceState


    OUTPUT = ez.OutputStream(AxisArray)

    async def initialize(self) -> None:
        ...

    @ez.publisher(OUTPUT)
    async def continuous(self) -> typing.AsyncGenerator:
        msg = AxisArray(np.ones(5), dims = ['time'])
        yield self.OUTPUT, msg


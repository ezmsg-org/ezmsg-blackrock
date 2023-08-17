import ezmsg.core as ez

from ezmsg.blackrock.nsp import NSPSource, NSPSourceSettings
from ezmsg.util.debuglog import DebugLog

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    settings = NSPSourceSettings()
    source = NSPSource(settings)
    log = DebugLog()

    ez.run(
        SOURCE = source,
        LOG = log,
        connections = (
            (source.OUTPUT, log.INPUT),
        )
    )
import ezmsg.core as ez

from ezmsg.blackrock.cerebus import CerebusSource, CerebusSourceSettings

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    settings = CerebusSourceSettings()
    system = CerebusSource(settings)

    ez.run(SYSTEM = system)
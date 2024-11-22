# ezmsg.blackrock

Interface for Blackrock Cerebus ecosystem (incl. Neuroport) using `pycbsdk`

## Installation
`pip install git+https://github.com/CerebusOSS/ezmsg-blackrock`

## Dependencies

* `python` >=3.9
* `pycbsdk` 
* `ezmsg-event`

## Setup (Development)

1. Install `ezmsg` either using `pip install ezmsg` or set up the repo for development as described in the `ezmsg` readme.
2. `cd` to this directory and run `pip install -e .`
3. Blackrock components are available under `import ezmsg.blackrock`

## Setup Notes

__IMPORTANT NOTE:__ _`pycbsdk` is only compatible with Central Neuroport/Cerebus Suite/NSP Firmware `7.0.5+`, built using `cbhwlib`/hardware library/network protocol `3.11+`.  Anything older is not supported by `pycbsdk`_.  Check this requirement in Central by navigating to `Help > About Central`

Blackrock Neuroport/Cerebus uses UDP multicast traffic to deliver upwards of 30000 packets/sec of data with low latency.  There are better ways to do this now, but back when this hardware was made, the engineers were dealing with the limitations of the hardware they had.  UDP comes with no packet delivery guarantee, but dropped packets can be catastrophic when interfacing with the Blackrock hardware.  As such, do your best to locate the highest quality networking equipment you can find, and have a high tolerance for blaming ethernet adapters and/or your router/switch for errors during troubleshooting.

The NSP startup procedure involves blasting several thousand UDP packets from the NSP to the interfacing computer in a few milliseconds.  Many low-quality adapters and network switches will simply drop some of these UDP packets, but this will result in errors during startup.

### Hardware Setup

This setup describes a setup with a Legacy NSP, a Windows PC (called "Central PC") running Central Suite that can configure the NSP, and a third machine (Windows, MacOS, or Linux) that will be running `ezmsg-blackrock`, henceforth forward referred to as the "Client PC". The documentation highlights the differences when using Blockrock's newer ("Gemini") NSP and Hubs.

Central cannot run on the same PC simultaneously as `ezmsg-blackrock` because `pycbsdk` exclusively binds the port that central would use to communicate with the NSP.  The NSP has too many settings to implement in `ezmsg-blackrock`, so in practice, we use Central to configure the NSP then acquire live data thereafter.  If you want to, you _can_ use `ezmsg-blackrock` on the Central PC once you've configured the NSP and closed Central.  When you do this, the Central PC and the Client PC are the same PC/IP address.  Its assumed that this PC is running windows because Central Suite is only available on Windows.

Find the highest quality ethernet (yes, wired) network switch or router that you can.  Look for 1+ Gigabit speed ratings (2.5 Gigabit better, 10 Gigabit is best), and features like QoS (Quality of Service) that can guarantee packet delivery. Ensure both the Central PC and the Client PC have high quality ethernet network adapters.  Not every ethernet adapter will work, due to packet buffering requirements. High quality ethernet cables are also important; aim for Cat6E or better.

1. Attach the NSP, Central PC, and Client PC to the router or switch using ethernet cables.
1. Blackrock NSP and Gemini Hubs use hard-coded static IP addresses.
    * If using a router (recommended), configure the router to sit at `192.168.137.254` and configure static DHCP assignments (mapping adapter MAC addresses to preset IP addresses) for:
        * NSP (Legacy or Gemini) -- `192.168.137.128`
        * Gemini Hub1 -- `192.168.137.200`
        * Gemini Hub2 -- `192.168.137.201`
        * Central PC -- `192.168.137.1` (Central likes to run on IP addresses below `192.168.137.16`)
        * Client PC -- `192.168.137.32` (Arbitrary address within subnet)
    * If using a network switch, configure the network adapters on the Central PC and Client PC to the aforementioned IP addresses
1. At this point, it's handy to make sure you can ping the NSP and Client PC from the Central PC and that you can ping the NSP and the Central PC from the Client PC.  Adjust firewall settings accordingly, enabling multicast UDP traffic on the ethernet adapters, and all TCP/UDP traffic on ports `51001` and `51002`.
1. Download and install the Central Suite on the Central PC from the [Blackrock Support](https://blackrockneurotech.com/support/) site: 
    * The public "Cerebus Central Suite" will not function with the FDA approved Neuroport system, which will instead require the "Neuroport Central Suite".  Acquiring this installer will require you to submit a ticket to Blackrock Support because the download is NOT public.

#### Emulate an NSP (Useful for development without an NSP)

It is possible to emulate Blackrock hardware while playing back previously recorded data using a software tool called "nPlayServer".
The publicly-available and supported nPlayServer is only available on Windows but unsupported Mac and Linux binaries (x86 or ARM) are available. Open an issue to ask.  

1. Power down the NSP if you already have it on the switch/router.
1. Download and extract the "Real Sample Data" from the [Blackrock Support](https://blackrockneurotech.com/support/) site. Extract the `*.ns6` file somewhere memorable in the filesystem (like `Documents`)
1. Run nPlayServer. On Windows, use the nPlayServer shortcut on the desktop to start Central and use the nPlay control window to navigate to the `*.ns6` file directory, select the file to replay, and start replaying it.  Then close Central/nPlayServer. This is important for using the `-L` flag with `nPlayServer.exe` later.
1. Navigate to the installation directory (e.g., "C:\Program Files\Blackrock Microsystems\Cerebus Central Suite") and locate the `nPlayServer.exe` executable.  Right click it and create a shortcut on the desktop.
1. Rename the shortcut to "Simulate NSP" or another friendly name.

If your intention is to only test on 'localhost' (i.e., ezmsg-blackrock and nPlayServer on the same machine), then there is nothing more to be done.
However, if your intention is for the emulation PC to be a different machine than the client PC, nPlayServer must be configured to work over the network.

1. Reconfigure the emulation host PC to have an IP address matching the emulated device (see addresses listed above). 
1. Right click the shortcut and go to "Properties".  Set the "Target" line to: `"C:\Program Files (x86)\Blackrock Microsystems\Neuroport Central Suite\nPlayServer.exe" --network bcast=192.168.137.255:51002 --network inst=192.168.137.128:51001 -L`, adjusting for the actual install path of your Central suite.
    * `bcast` address is `.255` because that's the UDP multicast address.
    * `inst` address is `.128` to emulate Legacy NSP. See the list above for Gemini hardware. Also, Gemini hardware requires the instrument port to be set to `51002`.
    * `-L` uses the last `.nsX` file for replay.  You can also manually specify an `.nsX` file for replay, but be aware nothing will replay when using `-L` if you've never replayed a file using nPlayServer, or if that file has moved or no longer exists.

### Software Setup

* Use Central to set up the NSP hardware.
* If running Central and `ezmsg-blackrock` on the same PC, shut down Central
* Assuming `pycbsdk` is installed in your python environment (which it should be, considering it's a dependency of `ezmsg-blackrock` that automatically gets installed with a `pip install -e .`), you should have the `pycbsdk-rates` command on your path.  Test the connection and hardware/software setup using `pycbsdk-rates --inst_addr 192.168.137.128 --inst_port 51001 --protocol 3.11`
    * Note that this command line is for Legacy NSP with hard-coded address `192.168.137.128` running on port 51001
    * Also note that network protocol is not automatically detected.  Our old hardware can only be updated to firmware `7.0.6` which uses hardware library/protocol `3.11`.  This can be checked by opening Central and navigating to "About Central" which will tell you which hardware library the firmware is running (assuming your NSP's firmware and Central version match). As of this writing, most Gemini hardware should be using protocol 4.1 (default).

### Troubleshooting

* When in doubt, restart equipment.
* Try software multiple times.  If you get different results every time you run, you might have ethernet adapters/network equipment of insufficient quality (it might be dropping packets, especially during configuration).
* Make sure the Central PC and the Client PC can ping each other and the NSP.
* Ensure NSP Firmware matches Central version.
* Ensure the protocol version you specify matches the firmware's protocol version.
* `pycbsdk-rates` and Wireshark + [a dissector](https://github.com/CerebusOSS/CerebusWireshark) are your friends. Godspeed.

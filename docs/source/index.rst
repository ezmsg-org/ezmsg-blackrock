ezmsg.blackrock
===============

Interface for Blackrock Cerebus ecosystem (incl. Neuroport) using ``pycbsdk``.

Overview
--------

``ezmsg-blackrock`` provides an interface to the Blackrock Cerebus ecosystem, including Neuroport devices, using the ``pycbsdk`` library. This package enables real-time data acquisition from Blackrock neural recording systems within the ezmsg framework.

Key features:

* **Real-time neural data acquisition** - Stream neural data from Blackrock NSP/Neuroport systems
* **Event handling** - Integration with ezmsg-event for spike events and other neural events
* **Multi-channel support** - Handle multiple channels of neural data simultaneously
* **Hardware compatibility** - Supports Central Neuroport/Cerebus Suite/NSP Firmware 7.0.5+ with cbhwlib 3.11+

.. note::
   This package requires ``pycbsdk`` and is only compatible with Central Neuroport/Cerebus Suite/NSP Firmware 7.0.5+, built using cbhwlib/hardware library/network protocol 3.11+.

Installation
------------

Install from PyPI:

.. code-block:: bash

   pip install ezmsg-blackrock

Or install the latest development version:

.. code-block:: bash

   pip install git+https://github.com/ezmsg-org/ezmsg-blackrock@main

Dependencies
^^^^^^^^^^^^

Core dependencies:

* ``ezmsg`` - Core messaging framework
* ``pycbsdk`` - Blackrock Cerebus SDK Python interface
* ``ezmsg-event`` - Event handling for neural spikes and events
* ``numpy`` - Numerical computing

Setup Notes
-----------

Hardware Setup
^^^^^^^^^^^^^^

Blackrock Neuroport/Cerebus uses UDP multicast traffic to deliver upwards of 30000 packets/sec of data with low latency. Due to the high packet rate and UDP's lack of delivery guarantees, high-quality networking equipment is essential:

* Use high-quality Gigabit (or faster) ethernet switches or routers
* Look for QoS (Quality of Service) features
* Use Cat6E or better ethernet cables
* Ensure network adapters can handle high packet rates

Network Configuration
^^^^^^^^^^^^^^^^^^^^^

The Blackrock NSP uses hard-coded static IP addresses. Configure your network as follows:

* **NSP (Legacy or Gemini)**: ``192.168.137.128``
* **Gemini Hub1**: ``192.168.137.200``
* **Gemini Hub2**: ``192.168.137.201``
* **Central PC**: ``192.168.137.1`` (or below ``192.168.137.16``)
* **Client PC**: Any address within the ``192.168.137.x`` subnet

Enable multicast UDP traffic and TCP/UDP traffic on ports 51001 and 51002 in your firewall settings.

Quick Start
-----------

For general ezmsg tutorials and guides, visit `ezmsg.org <https://www.ezmsg.org>`_.

For package-specific documentation:

* **API Reference** - See :doc:`api/index` for complete API documentation
* **README** - See the `GitHub repository <https://github.com/ezmsg-org/ezmsg-blackrock>`_ for detailed setup instructions

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/index


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`

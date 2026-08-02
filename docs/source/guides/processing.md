# Processing modules

Beyond acquisition, `ezmsg-blackrock` ships a few stateful transformers that
operate on the `AxisArray` stream coming off the device. Each is available both
as an ezmsg `Unit` (for use in a graph) and as a plain transformer/processor
(for use in a script or test). Full signatures and settings live in the
{doc}`API reference </api/index>`; this page covers what each does and when to
reach for it.

## Channel mapping

{class}`~ezmsg.blackrock.ChannelMapUnit` attaches Blackrock `.cmp` channel-map
metadata to the `ch` axis of the stream. The incoming `ch` axis is replaced with
a structured `CoordinateAxis` carrying, per channel, the fields `x`, `y`,
`label`, `bank`, `elec`, and `device` — so downstream components (plots, spatial
filters, electrode-geometry-aware steps) can address channels by physical
position and label rather than by raw index.

The unit takes the *complete* set of per-headstage overlays in a single settings
object: {class}`~ezmsg.blackrock.ChannelMapUnitSettings` holds a tuple of
{class}`~ezmsg.blackrock.ChannelMapSettings`, one per headstage. On each reset it
rebuilds the `ch` axis from scratch in three phases:

1. **Base layer** — labels are taken from the incoming `ch` axis (falling back to
   `ch1`, `ch2`, … when none are present), and pass-through fields such as
   `device` are carried forward.
2. **CMP overlays** — for each configured `.cmp` file, the parsed entries are
   written at their channel index (`chan_id - 1`, with `chan_id` offset by
   `start_chan`), filling in position, bank, electrode, and label. When `hs_id`
   is nonzero, labels are prefixed `hs{hs_id}-`. A mask records which channels a
   CMP claimed.
3. **Auto-grid fill** — any channel no CMP claimed is laid out on a generated
   grid, positioned below and to the right of the CMP geometry so the synthetic
   coordinates never collide with mapped electrodes.

Pushing an empty `cmp_configs` tuple clears the map and yields a pure auto-grid.
The same {class}`~ezmsg.blackrock.ChannelMapSettings` record doubles as the
per-headstage entry in `CereLinkSignalSettings.cmp_configs`.

## CerePlex impedance measurement

{class}`~ezmsg.blackrock.CerePlexImpedance` extracts per-channel electrode
impedance from a CerePlex headstage's built-in impedance sweep. During the sweep
the headstage injects a 1 kHz, 1 nA sine wave for ~100 ms into one channel at a
time, cycling sequentially through the array; every channel not under test reads
*exactly* zero. The processor watches that exclusivity to lock onto the active
channel, buffers its burst, and recovers the impedance from a single-bin DFT at
1 kHz — impedance in kOhm is the peak-to-peak voltage (µV) divided by the
peak-to-peak test current (nA): `Z = V_pp / I_pp`.

Each impedance update emits an `AxisArray` whose data is a `(1, n_ch)` row of
values in kOhm, with `NaN` for channels not yet measured. Multiple headstages
are tracked independently via
{attr}`~ezmsg.blackrock.CerePlexImpedanceSettings.headstage_channel_offsets`,
each free to be at a different point in its own sweep.

Two requirements matter for correct results:

- **The input must be in microvolts.** Passing raw ADC counts scales every
  impedance by the ADC factor. When the data comes from
  `CereLinkSignalSource`, set `microvolts=True`.
- **Device-side filtering must be disabled.** A filter leaves small non-zero
  residuals on idle channels, which defeats the exact-zero exclusivity checks
  the tracker relies on.

## Sampling-delay alignment

{class}`~ezmsg.blackrock.SamplingDelayAlignment` corrects the per-channel timing
skew introduced by the front-end's sequential analog-to-digital converter. The
CerePlex headstages sample channels in banks of 32, one every ~969.7 ns, so within
a bank channel *c*'s sample is the signal delayed by `c × channel_sample_interval`
relative to the bank start. For any cross-channel operation — common-average
referencing, whitening, beamforming — this skew smears the common mode at high
frequency: negligible at 60 Hz (~0.65°) but ~81° across a bank near 7.5 kHz, so
CAR's common-mode rejection collapses toward Nyquist.

The transformer removes the skew by delaying each channel back onto a common
time grid (the bank start) with a per-slot windowed-sinc fractional-delay FIR
filter. There are only `bank_size` distinct delays, so only that many distinct
filters. A windowed sinc is used deliberately instead of linear interpolation:
linear interpolation is a delay-dependent low-pass that would impose a
*different* high-frequency rolloff per channel — coloring the band exactly where
the misalignment mattered.

The per-channel delay defaults to acquisition order (slot `c % bank_size`). If
the `ch` axis carries `bank`/`elec` metadata — for example after
{class}`~ezmsg.blackrock.ChannelMapUnit` has run — the slot is taken from `elec`
instead, so each channel's delay stays correct even when channels have been
reordered relative to hardware acquisition.

Set `filter_len` to `0` to disable alignment entirely — the transformer becomes
a pass-through that returns its input unchanged, handy for A/B comparisons or for
leaving the unit wired in but inert.

### Choosing `filter_len`

Longer filters flatten the response nearer Nyquist but add latency: the bulk
delay is `(filter_len-1)//2` samples. The table below is the worst case over all
32 within-bank fractional delays at 30 kHz, and is pinned by
`tests/test_sampling_delay_alignment.py`:

| Passband | `filter_len` | Max phase error | Max magnitude error | Bulk delay |
|---|---|---|---|---|
| 0–500 Hz (LFP) | 7 | 0.0009° | 0.00001 dB | 3 samples (100 µs) |
| 0–3 kHz | 9 | 0.0038° | 0.0001 dB | 4 samples (133 µs) |
| 0–7.5 kHz (broadband/spike) | 13 *(default)* | 0.015° | 0.0015 dB | 6 samples (200 µs) |
| 0–7.5 kHz | 33 | 0.0028° | 0.0004 dB | 16 samples (533 µs) |

The requirement the default is chosen against is 0.05° of phase error and 0.01 dB
of magnitude error over the intended passband — roughly three orders of magnitude
below the ~81° of skew being corrected. 13 taps is the shortest odd length that
holds that over the full broadband band (11 taps misses it at 0.09°), so it is
the default. Drop to 7 in an LFP-only pipeline to halve the latency again; 33
was the previous default and still works, but buys accuracy that is already far
below the noise floor at three times the latency.

### Performance

The transformer runs once per message, so per-message cost at live chunk sizes —
tens of samples, not thousands — is what matters. On MLX the FIR is evaluated as
a single depthwise convolution (one group per channel, kernel laid out once at
state reset) rather than one dispatched multiply-add per tap, which is about
2–3× faster per message and roughly flat from 3 to 1000 samples. The rail
forward-fill's running max likewise uses `mx.cummax` on MLX and the `maximum`
ufunc's `accumulate` on numpy/cupy, falling back to the portable log-depth scan
elsewhere. `examples/bench_sampling_delay_alignment.py` reports all of this
across channel counts, message sizes, and rail handling on/off.

A few things to keep in mind:

- **Latency.** The causal FIR adds a common bulk delay of `(filter_len-1)//2`
  samples (the per-channel fractional delays ride on top). The output time-axis
  offset is shifted so timestamps stay physically correct.
- **It resamples the raw data.** Downstream sees interpolated samples — fine for
  cross-channel cleaning, but be deliberate if a later step needs raw waveforms.
- **Railing.** Clipped samples are corrupt and a fractional-delay filter would
  spread that corruption across its support. Setting `rail_threshold` holds
  railed samples at the last valid value before filtering as a basic mitigation.
- **Backend portability.** The module is Array-API compatible: it detects the
  input's namespace and runs on numpy, MLX, torch, jax, cupy, and friends.

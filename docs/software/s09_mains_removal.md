<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# Mains removal in EMG

Surface EMG is microvolts; the mains field around it is millivolts. Where the
two electrodes of a pair differ in impedance, that common-mode field turns into
differential signal the amplifier cannot reject, and it lands squarely in the
20–450 Hz EMG band. This page describes what the code does about it, and — just
as important — what it cannot do.

## What it cannot do

Filtering is **salvage, not a fix**. Two failure modes survive any filter:

- **Interference that is simply too large.** Removal is proportional: a
  channel carrying 3.2 mV of hum on a 61 kΩ contact came down 19× on
  `s01_f_30` and was still 168 µV, against 5–20 µV for normal resting EMG. The
  fit also follows the interference only as fast as its block length lets it
  (see the trade-off below); modulation faster than that stays.
- **Clipping.** Once the front end has railed, the samples no longer contain the
  EMG. Nothing recovers it.

So treat a large `mains_share` as a hardware finding first. `jtsmm emg check`
reports the bias node's AC level (informational, n/a while CH8 is railed); a
node swinging tens of mV means fix the coupling rather than the filter. See `docs/operations/op03_troubleshooting.md`.

## The two implementations

Both remove the line frequency *and its harmonics* (50, 100, … 450 Hz): the
interference is rarely a clean sine, so the harmonics carry real power.

| | `jtsmm.emg.mains` (offline) | `jtsmm.viewer.processing` (live) |
|---|---|---|
| method | block-wise least-squares fit | quadrature tracker + frequency lock |
| causal | no (fits whole blocks) | yes, state carries across blocks |
| needs | the sample timestamps (`t_us`) | the sample timestamps, via `process_timed` |
| use | analysis, plots (`plot_emg` default) | the viewer; `plot_emg --mains track` reproduces it |

Both need the timestamps, for the same reason — see *The dropout trap* below.
On `s01_f_30` take1 they agree: resting RMS after removal is 11 / 168 / 36 / 26 /
5 / 7 µV offline and 11 / 152 / 34 / 24 / 5 / 7 µV through the live chain.

Both **fit and subtract** the interference rather than notching a band out. That
is the key difference from the old biquad bank: a 5 Hz notch × 8 harmonics
discards 40 Hz of the EMG band, whereas fitting removes only the coherent tone
and leaves broadband EMG at the same frequencies intact.

### Why the frequency has to be measured

The grid is never exactly 50.000 Hz. Two sessions an hour apart measured
**50.016** and **49.986** Hz. A notch fixed at the nominal frequency degrades as
the grid wanders; measured on a 50.2 Hz excursion it leaves 8.4% of the tone
against the tracker's 0.02%. So:

- offline, `estimate_line_hz()` refines the FFT peak by parabolic interpolation
  to ~0.001 Hz;
- live, `HarmonicCanceller` reads the rotation of its own I/Q phasor — which is
  exactly the frequency error — and pulls its oscillators onto the grid. The
  correction is clamped (`lock_hz`, default ±0.5 Hz) so it can never walk onto
  the EMG band, and it re-locks on a fixed amount of data so the result does not
  depend on how the stream was chunked.

## The dropout trap — read this before filtering any recording

**Row index is not time.** A stream with dropped samples is packed contiguously,
so the sample index compresses wherever samples are missing, and every filter
ever written assumes uniform sampling. On `s01_f_30` take1, 2.92% of samples
were missing; index-as-time put the line frequency at **50.224 Hz** against a
true **50.020**. Worse, the gaps are unevenly spread, so the apparent frequency
*wanders* and no single correction fixes it. Filtering that array barely works.

Always pass the timestamps:

```python
from jtsmm.emg.mains import remove_mains

cleaned, f0 = remove_mains(uv, fs_hz, t_us=t_master_us)   # <- t_us, always
```

`remove_mains` then fits on a uniform grid and maps the result back onto your
original rows, so the shape you pass in is the shape you get back. Without
`t_us` it assumes the samples are already uniform.

The live tracker has the same problem in a different costume. It assumes
consecutive samples are consecutive in time, so at every gap the mains moves on
and its oscillator does not: a median 11 ms gap is **198°** of 50 Hz, and with
gaps arriving 2.2× a second the tracker never re-converged. On take1 it removed
*nothing* (rest 402 µV against 403 raw) while the old notch bank — memoryless
enough to recover in ~60 ms — still halved it. Not the algorithm: the same
tracker on a gap-free copy of the signal reached 19–65 µV. So the hub now feeds
the filter through `CausalEmgFilter.process_timed(t_us, samples)`, which splits
a block at every gap and advances the oscillator phase over the missing samples;
the estimate stays valid, only the reference has to catch up. Two details had to
follow:

- the canceller moved **ahead of** the 20 Hz high-pass. A second-order HPF rings
  at ~20 Hz for ~50 ms when the tone jumps phase at a gap — in-band, and not a
  sinusoid the canceller can model (20 µV of residual on a 300 µV tone with 3%
  dropouts; 7 µV with the canceller first);
- but the canceller must not see raw DC either: a 25 mV electrode offset times a
  reference whose phase skips at every gap does not average to zero, and it
  leaked ~600 µV of phantom tone — the offline intercept bug again. A
  **first-order** DC block far below 50 Hz goes first: it cannot ring at a phase
  jump, and a slow offset cannot leak into I/Q. Chain: DC-block → canceller →
  HPF → LPF.

The underlying cause is fixed (`drain_records(limit=…)`), so new recordings
should not have gaps — but check `jtsmm integrity` rather than assume, and keep
passing `t_us` regardless. `plot_emg --mains track` honours the timestamps too,
so a replay shows what the live viewer would have shown.

## Using it

Plots, via the existing tool:

```bash
python -m jtsmm.tools.plot_emg session.h5 --mains offline    # default: the fit, at the recorded timestamps
python -m jtsmm.tools.plot_emg session.h5 --mains track      # what the live viewer shows
python -m jtsmm.tools.plot_emg session.h5 --mains notch      # the old biquad bank
python -m jtsmm.tools.plot_emg session.h5 --mains off        # band-pass only
```

Your own analysis:

```python
import h5py, numpy as np
from jtsmm.emg.mains import remove_mains, mains_share, estimate_line_hz

with h5py.File("session.h5") as f:
    uv = np.asarray(f["emg/samples_uv"])[:, :6]
    t_us = np.asarray(f["emg/t_master_us"])
    fs = float(f["emg"].attrs["sample_rate_sps"])

clean, f0 = remove_mains(uv, fs, t_us=t_us)
print(f"grid {f0:.3f} Hz, share {mains_share(uv[:, 0], fs, f0):.2f} "
      f"-> {mains_share(clean[:, 0], fs, f0):.2f}")
```

`mains_share()` is the number to report, and report it **after** removal too:
the fraction of 20–450 Hz power still sitting at the line harmonics. Above ~0.5
the channel is interference with EMG buried in it; below ~0.1 it is clean. A
share that stays high after `remove_mains` means the interference is modulated
and no filter will fix it.

A falling RMS is **not** proof that filtering worked — signal destruction lowers
RMS just as well. Judge with `mains_share` and with rest-versus-activity
(below), never with amplitude alone.

The viewer needs no code change — `track` is the default. To go back:

```yaml
host:
  emg_filter:
    mains_mode: notch     # or: off
```

## Judging the result

Do not judge by how the trace looks; judge by a number that cannot flatter you.
The best one is **rest versus activity on the same channel**, because both
contain the same interference:

```python
snr = clean[chewing].std() / clean[rest].std()
```

Resting RMS on `s01_f_30` take1, before → after, measured with `_bandpass_rms`
on the `rest` phase (grid at 50.018 Hz, 1073 dropout gaps handled via `t_us`):

| channel | before | after | residual mains share | chewing / rest |
|---|---|---|---|---|
| `masseter_l` | 139 µV | **5 µV** | 0.13 | 7.3 |
| `temporalis_l` | 126 µV | **7 µV** | 0.06 | 6.5 |
| `masseter_r` | 403 µV | **11 µV** | 0.14 | 5.8 |
| `digastric_r` | 434 µV | 26 µV | 0.27 | 6.7 |
| `digastric_l` | 132 µV | 36 µV | 0.27 | 6.9 |
| `temporalis_r` | 3232 µV | 168 µV | 0.16 | 3.3 |

Before removal, chewing was indistinguishable from rest on five of six channels
(ratio 0.99–1.06). After, three channels sit inside the normal resting band and
two more are close; `temporalis_r` is 19× better and still not usable.

**What it costs.** Measured by injecting a known 30 µV band-limited signal and
checking how much survives: the default 1 s blocks remove ~2% of broadband EMG
energy (18 sinusoid parameters per 2000 samples, measured at 2 kSPS). That is the number to weigh
against the block length:

| `block_s` | tracks modulation up to | EMG energy removed | rest RMS (median) |
|---|---|---|---|
| 1.0 s (default) | ~0.5 Hz | ~2% | 29 µV |
| 0.25 s | ~2 Hz | ~8% | 15 µV |

Shorter blocks follow a swinging bias loop better and cost more EMG. There is
no free setting; 1 s is the default because the residual is already at the
normal resting floor on most channels.

### A bug that looked like physics

An earlier version of this page said the residual was *modulated* interference
that "no harmonic method removes", because in-band RMS went **up** on two
channels after fitting (132 → 288 µV). Least squares cannot raise energy inside
its own block, so that was a bug: the design matrix had no constant column. An
Ag/AgCl pair sits at 10–25 mV of DC offset; on a uniform grid whole cycles of a
sinusoid are orthogonal to a constant, but at the real timestamps with dropped
samples punching holes in the cycles they are not, and ~2% of a 25 mV offset
leaked into the fitted 50 Hz amplitude — ~500 µV of tone subtracted that was
never there. The zero-mean synthetic test never saw it. With intercept and trend
columns the in-band RMS falls in every block, at the same EMG cost. The lesson
is general: when a linear method makes a signal *worse*, suspect the model
before the data.

### What else was tried

- **Bias-node (DRL) reference canceller.** CH8 reads the bias amplifier output,
  a live reference of the interference. A 32-tap Wiener filter fitted on the
  rest phase preserves EMG best of anything here (96% kept) but only cancels
  ~2.7×; the channel-to-bias path is evidently not a fixed linear filter
  (2% of the reference is clipped, and part of the pickup is lead-loop, not
  common-mode). Chaining it before the harmonic fit gives the lowest residual
  share (0.13) but no better floor. Not shipped.
- **Spatial common-mode removal** (regress each channel on the other five).
  Resting RMS looks excellent — and the injection test returns a *negative*
  preservation score: it mixes EMG between channels and destroys the signal.
  This is exactly why a falling RMS proves nothing. Rejected.
- **Zero-phase demodulation tracker** (the viewer's algorithm run offline,
  bandwidth `B`). Equivalent to the block fit: 22–24 µV at `B` = 0.5–1 Hz for the
  same EMG cost; `B` = 0.25 Hz is too narrow to follow the grid and fails.
- **Half-overlapping Hann overlap-add** of the block models: neighbouring blocks
  fit slightly different phases, so averaging attenuates what is being
  subtracted (6.5 → 158 µV against a 6.55 µV ground truth). Rejected.

Against a synthetic 2561 µV of hum over a known 6.55 µV of EMG with 3% of
samples dropped, the shipped fit recovers 6.50 µV; with a 25 mV offset added it
still does. Both are tests.

## What `jtsmm emg check` does with it

The three stages need different treatment, and only one of them wanted filtering:

| stage | mains removal? | why |
|---|---|---|
| impedance (`|Z|`) | **no** | fitted jointly instead: the 31.25 Hz tone is estimated by least squares together with the mains harmonics, so hum cannot leak into it. |
| line peak | **no** | it *is* the mains measurement. Filtering first would make it read zero. |
| baseline noise | **yes** | the 10 µV limit asks "is this electrode quiet enough to see 5–20 µV of resting EMG". With hum in the same band it measured the room instead. |

So `noise_rms_uv` is now the mains-removed figure and `noise_ok` judges it, while
`noise_total_uv` reports the raw in-band RMS for context and `line_peak_uv` keeps
the verdict on the mains. A channel can now fail one, the other, or both — and
they have different fixes: re-prep the electrode, or deal with the coupling.

Two related corrections landed with this:

- **`_bandpass_rms` was returning `RMS × √n`.** Parseval for a one-sided FFT
  needs `|X|²/n²`, not `/n`, and it zero-padded to a power of two on top. At a
  5 s, 2 kHz capture that is a factor of **128**, so the 10 µV limit was really
  0.08 µV, every channel of every session failed the noise check, and the
  readings were six-figure microvolts that looked like a railed front end
  (`s01_f_30` reported `noise 232259.6 µV` for a true ~1.8 mV). Fixed and pinned
  by calibration tests.
- **The `floating_open` heuristic** now needs broadband noise *and* hum. A
  dangling input has both; a connected electrode in a hummy room has only the
  hum, and used to trip. Genuine opens are caught more reliably by the lead-off
  comparator and by `|Z|` anyway, and `floating_open` never gates recording.

## One more trap: normalisation

If your MVC reference was recorded under the same interference, it is inflated,
and every %MVC computed from it is understated. On `s01_f_30` the digastric MVC
amplitudes fell **87%** once mains was removed — they were mostly hum. **Filter
the MVC with the same settings before deriving any reference amplitude.**

## Where things live

| | |
|---|---|
| offline fit, `estimate_line_hz`, `uniform_grid`, `mains_share` | `software/jtsmm/jtsmm/emg/mains.py` |
| live canceller, `CausalEmgFilter(mains_mode=…)` | `software/jtsmm/jtsmm/viewer/processing.py` |
| plot tool `--mains` | `software/jtsmm/jtsmm/tools/plot_emg.py` |
| config key | `host.emg_filter.mains_mode` |
| tests | `software/jtsmm/tests/test_emg_mains.py` |

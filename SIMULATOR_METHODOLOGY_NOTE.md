# Simulator Methodology Note: `simulate_training_pairs.py`

**Status: PAUSED, two confirmed negative results.** This document exists
so a future session doesn't have to re-derive five rounds of debugging
from WEEK20_LOG.txt alone, the same role NOVEL_METHODOLOGY_PROPOSAL.md
served for the hotspot-map work. It records what was tried, what was
ruled out and why, and what genuinely remains open. Nothing here claims
the simulator is fixed — as of Week 20 it is not, and this
sub-investigation is explicitly paused (§6), not resolved.

## 1. What this script is for

Track B's blueprint architecture (4ypblueprint.pdf, Paper 2 / Dumont Le
Brazidec 2024 — U-Net segmentation -> CNN regression) needs labeled
(XCO2 tile, plume mask, true Q) training pairs. No real labeled dataset
exists at this project's scale, so `simulate_training_pairs.py` generates
them synthetically, reusing `plume_model.py` (this project's own
validated Briggs/Pasquill-Gifford Gaussian plume physics) as the
simulation engine — never an external transport model, never a spatial
field fit or solved to trivially match a target Q.

## 2. The core tension this note is about

The simulator produces a genuinely physical 2D concentration field. The
open problem is entirely about **reading a single "how much CO2 is this"
number back out of that field for calibration** — and every readout
method tried so far disagrees with real satellite-observed enhancement
values by an order of magnitude, in *opposite directions* depending on
which readout is used. That is the central, unresolved finding.

## 3. Fixes established as correct (not in question)

| Fix | What it does | Verified how |
|---|---|---|
| **Near-field guard** (Bug 1) | Clips downwind distance to `max(3*stack_height, 300m)` before evaluating concentration; zeroes true-upwind pixels | `plume_model.ground_level_concentration()`'s own div-by-zero guard doesn't stop the physically-unrealistic near-source spike; this does |
| **Wind-speed floor** (Bug 2) | Samples wind >= 1.2 m/s, not down to 0.5 | Matches the real per-overpass minimum (1.226 m/s) across all 24 facilities in `data/emission_estimates.json` — not an arbitrary "stop it exploding" clip |
| **Resolution match** (Bug 3) | Training tiles are 60km/64px (~937.5 m/px), matching `export_facility_tiles.py` | A finer grid would under-sample the spatial averaging a real satellite pixel naturally provides |
| **Area-averaging** (Task 2) | Each pixel is evaluated on a 5x5 subgrid spanning its own footprint and averaged, not sampled once at its center | Convergence test: peak value stable from N=9 subgrid density upward — not a discretization artifact, a real correctness fix |

None of these are in question. They stay as implemented.

## 4. What was ruled out, and why

### 4.1 Q source (IME-derived vs. CEA ground truth) — ruled out (Task 3)

**Hypothesis tested**: maybe the unrealistic tail comes from using
`physics_ime.py`'s own (known-imperfect) Q estimates as the simulator's
input, rather than a genuinely independent ground-truth Q.

**Check before running anything**: compared the two Q distributions'
*shape*, not just scale — P95/median ratio, matched N=24 facilities:

| Source | P95/median |
|---|---|
| IME (`data/emission_estimates.json`) | 5.549 |
| CEA (`data/cea_ground_truth_2020_21.json`) | 2.576 |

IME's distribution is ~2.2x more right-skewed than CEA's — a real shape
difference, so this was a legitimate test, not a doomed rescale.

**Result**: capping Q to CEA's real range (2.06e6-3.32e7 t/yr) made the
peak-pixel tail **worse**, not better — median enhancement rose from 4.65
to 9.84 ppm (real max is 3.70 ppm), and the fraction of tiles within the
real range fell from 44.7% to 16.3%. Cause: CEA's range floor is ~6x
IME's — even CEA's smallest real facility is still a large point source,
so the narrower-but-higher CEA range pushes the typical (geometric-mean)
sampled Q up, not down.

**Conclusion**: Q source is not the fixable variable. Ruled out.

### 4.2 Readout scale (peak pixel vs. IME-consistent disk integration) — tested, both fail, in opposite directions (Task 4)

**Hypothesis tested**: maybe comparing a single hottest-pixel value to a
Q estimated by IME's completely different method (spatial mass-balance
over a ~28km-radius disk vs. a ~44-100km background annulus) was never a
fair comparison in the first place — replace the readout with the exact
same near-plant/background-annulus geometry `physics_ime.py` itself uses
(`NEAR=0.25 deg`, `BG_IN/BG_OUT=0.4/0.9 deg`, `~27.75/44.4/99.9 km` at the
`~111 km/deg` conversion this project's own comments already use),
computing `mean(near-zone) - mean(background-annulus)` — exactly
`process_plant.py`'s `co2_enhancement_ppm` definition. Because that
background annulus (out to ~100 km) is far larger than Track A's 60km
training tile, this readout is evaluated on a **separate, larger
calibration-only grid** (`READOUT_HALF_EXTENT_KM` ≈ 110 km,
`READOUT_PX_SIZE_KM` = 1.5 km, matching `physics_ime.py`'s own
`FOOTPRINT_AREA_M2` sounding-footprint scale) — never saved as a
training tile, and it does not change the training tile's own geometry
or physics at all.

**Result** (600 positive tiles, CEA-range Q, same seed=42):

| | min | p10 | median | mean | p90 | p99 | max |
|---|---|---|---|---|---|---|---|
| Simulated (IME-consistent readout) | -0.040 | 0.003 | 0.054 | 0.072 | 0.163 | 0.327 | 0.562 |
| Real (`plant_results.json`, N=24) | -1.277 | -0.866 | 0.619 | 0.620 | 1.839 | 3.409 | 3.698 |

100% of simulated tiles fall within the real [-1.277, 3.698] min/max
envelope — but that is a misleading pass. The real median is 0.619 ppm;
the simulated median is 0.054 ppm, over 10x smaller, and the simulated
**max** (0.562) doesn't even reach the real **median**. The distribution
collapsed toward zero; it did not come to resemble the real one. Reported
as a failure, not a fix, despite the flattering min/max number.

**Cause, verified analytically**: the Gaussian plume's crosswind
1-sigma width (`sigma_y`) at the near-zone's outer radius (27.75 km,
stability B) is ~2285 m. A disk of that radius has circumference ~174 km;
the plume's ~2-sigma core width there is ~4.6 km — about **2.6%** of the
disk's circumference. For a single fixed wind direction (every synthetic
tile samples exactly one), over 97% of the near-zone disk sits at pure
background, diluting the disk mean by roughly two orders of magnitude
relative to the plume's own peak.

Real near-plant soundings are not one snapshot. They are whatever OCO-3
soundings fell inside that disk across **many overpass days over up to a
year**, each potentially carrying a *different* real wind direction —
different days sweep different wedges of the disk, so the real
aggregated "near" sample captures elevated readings from several
directions at once. A single synthetic tile with one sampled wind
direction cannot reproduce that by construction — it compares a
single-day snapshot's spatial mean to a multi-day aggregate, which is not
a like-for-like readout no matter how the zone geometry is defined.

### 4.3 Multi-day aggregation (Task 5) — implemented, SECOND confirmed negative result

**Hypothesis tested**: §4.2's own diagnosis suggested the fix — pool
several single-direction days' near/background samples the way
`physics_ime.py` pools real per-overpass soundings (concatenate raw
samples across days, then take one mean), rather than reading out one
day at a time.

**Scoping constraint, strictly enforced**: this changes ONLY the
calibration/verification readout. Every individual training tile remains
the exact single-snapshot mechanism of Tasks 1-4 — one wind direction,
one day, its own exact mask, saved unaggregated. `make_tile()` gained
optional `q`/`wind_speed`/`stack_height`/`stability` arguments so several
tiles can share one synthetic facility's physical characteristics, but
`wind_from_deg` is always resampled fresh inside the function regardless.
The only disclosed structural change is that tiles are now generated in
facility-grouped batches (sharing Q etc.) rather than fully independently
— not a change to what any individual tile represents or how it's made.

`n_days` per facility was bootstrap-sampled from the real per-facility
`hit_days` values in `data/plant_results.json` (N=30: min 1, max 25,
median 8, mean 9.93) — not an arbitrary constant. 200 synthetic
facilities were simulated, yielding 1936 total single-snapshot positive
training tiles (well over the 500-tile standard) plus 300 negative tiles.

**Result** (200 pooled facility readouts vs. the single-day distribution
from the same run, vs. real):

| | min | p10 | median | mean | p90 | p99 | max |
|---|---|---|---|---|---|---|---|
| Single-day (Task 4, this run) | -0.086 | 0.000 | 0.057 | 0.075 | 0.180 | 0.293 | 0.437 |
| Pooled multi-day (Task 5) | -0.048 | 0.010 | 0.058 | 0.076 | 0.174 | 0.255 | 0.410 |
| Real (N=24) | -1.277 | -0.866 | 0.619 | 0.620 | 1.839 | 3.409 | 3.698 |

Pooling did not close the gap. The pooled median (0.058) is
indistinguishable from the single-day median (0.057) — both remain over
10x smaller than the real median (0.619). What DID change: the spread
narrowed slightly (max 0.437→0.410, p99 0.293→0.255) while the center
stayed flat. Checked directly: correlation between a facility's `n_days`
(1–25, the real range) and its pooled readout is **-0.018** —
statistically zero. Pooling more days gives no systematic increase,
regardless of how many.

**Cause, verified analytically**: this implementation's per-day sample is
the FULL near-zone/background-annulus grid — thousands of pixels,
exhaustively covering every crosswind position — not a sparse
satellite-track sample. Because a single day's grid already spatially
integrates over the *entire* disk, its on-plume fraction is already a
stable expected value from one day; pooling more full-disk days is
statistically close to averaging several i.i.d. draws of an
already-converged population mean — it reduces the *noise* of the pooled
estimate (explaining the narrower spread) but cannot shift its *expected
value*. A real per-overpass "near" sample is not an exhaustive disk — it
is wherever OCO-3's actual, narrow ground track happened to cross that
day, so a day whose track crosses the plume contributes a
disproportionately plume-heavy sample (few points, many on-plume), and
pooling across many such days lets favorable-track days pull the
aggregate up in a way an exhaustive per-day grid mean structurally
cannot. The missing ingredient in this implementation isn't "more days"
— it's *sparse, orbital-track-shaped* per-day sampling, which was not
simulated.

## 5. Current best explanation (confirmed across two independent tests)

Peak-pixel readout over-shoots the real range by 1-2 orders of magnitude
(Tasks 1-3). IME-style disk-integration under-shoots it by about the same
amount in the *other* direction (Task 4), and pooling that disk-based
readout across a realistic number of days does not close the gap either
(Task 5) — for the **identical** underlying physical field and the
**identical** Q throughout. `physics_ime.py`'s Q and this simulator's
forward Gaussian dispersion, read out via an exhaustive spatial grid at
any scale or day-count tried, are not mathematical inverses of each
other. The Task 5 diagnosis narrows *why* one level further: the missing
ingredient is not "single day vs. many days" but "exhaustive spatial
coverage vs. sparse, orbital-track-shaped sampling" — a materially
different kind of fix than anything tried in Tasks 1-5.

## 6. Recommendation: PAUSE this sub-investigation

Five rounds have now been run against this gap — near-field/wind-floor/
resolution fixes and area-averaging (Tasks 1-2, correct and kept, but
insufficient alone), Q-source falsification (Task 3, ruled out, made
things worse), IME-consistent readout geometry (Task 4, confirmed the
mismatch but flipped the error direction), and multi-day pooling
(Task 5, confirmed again, near-zero effect, now with a specific
analytical cause). This matches this project's own diminishing-returns
discipline for when to stop iterating and report plainly — see
WEEK13_LOG.txt's Rihand investigation, paused after four independently
rejected explanations. This sub-investigation is PAUSED here on the same
basis, not resolved.

### Paths forward — not attempted, listed so they aren't re-derived from scratch

1. **Simulate real orbital sampling geometry.** The one path Task 5's
   diagnosis points to directly: replace the exhaustive per-day readout
   grid with a sparse sample matching OCO-3's actual ground-track width
   and sounding density, so a day's near-zone sample can be
   disproportionately plume-heavy when the track crosses the plume and
   near-empty otherwise. This is effectively building a second,
   independent satellite-sampling simulator, not a readout-geometry
   adjustment — a materially larger scope change than anything attempted
   in this note so far.
2. **Different label-generation methodology entirely.** Abandon
   single-snapshot forward Gaussian simulation as the training-label
   source and generate segmentation masks/Q labels through a method
   structurally consistent with IME's own multi-day aggregation
   assumption (e.g. synthesize the near/background statistics directly
   rather than a spatial field, then derive a plausible plume shape for
   the segmentation mask separately).
3. **Re-scope what the U-Net is meant to learn.** If neither of the above
   converges, an option not explored here is training the segmentation
   stage on the spatial *shape* of the plume only (mask quality), with Q
   regression handled by a separate mechanism not dependent on this
   simulator's ppm calibration at all.

## 7. What is NOT open

- The Gaussian plume physics itself (`plume_model.py`) is validated and
  unchanged throughout all of this — see `tests/test_plume_model.py`.
- The near-field guard, wind floor, resolution match, and area-averaging
  fixes (§3) are all independently verified and not implicated in the
  Task 3/4/5 findings.
- The multi-day pooling MECHANISM itself (§4.3) is verified correct — it
  reduces variance as expected; it simply doesn't address a bias that
  turned out not to be a sampling-count problem.
- The U-Net (blueprint Paper 2 stage 1) has not been started, and per
  this note's recommendation (§6) should not be started until either the
  sparse-sampling path (§6.1) is tried and verified, or a different
  label-generation approach (§6.2/§6.3) is adopted and independently
  verified against the real distribution — the same standard this note
  itself was held to.

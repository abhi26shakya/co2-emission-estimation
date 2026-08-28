# Simulator Methodology Note: `simulate_training_pairs.py`

**Status: OPEN, PAUSED negative result.** This document exists so a future
session doesn't have to re-derive four rounds of debugging from
WEEK20_LOG.txt alone, the same role NOVEL_METHODOLOGY_PROPOSAL.md served
for the hotspot-map work. It records what was tried, what was ruled out
and why, and what genuinely remains open. Nothing here claims the
simulator is fixed — as of Week 20 it is not.

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

## 5. Current best explanation (confirmed, not just hypothesized)

Peak-pixel readout over-shoots the real range by 1-2 orders of magnitude
(Tasks 1-3). IME-style disk-integration under-shoots it by about the same
amount in the *other* direction (Task 4) — for the **identical**
underlying physical field and the **identical** Q. Both failure
directions trace to the same root cause: a forward model evaluated at
**one** wind direction cannot be read out, at **any** single spatial
scale, to match a real-world quantity whose definition is implicitly
averaged over **many** wind directions across a year of overpasses.
`physics_ime.py`'s Q and this simulator's forward Gaussian dispersion are
not mathematical inverses of each other, and no single-snapshot readout
geometry closes that gap.

## 6. Paths forward — not attempted this session, listed so they aren't re-derived from scratch

1. **Multi-direction aggregation.** Simulate several wind-direction draws
   per plant (representing several overpass days) and aggregate their
   near/background means before computing the readout, directly testing
   §5's diagnosis. Materially larger scope than a readout-geometry change
   — not attempted, flagged as the most direct next test.
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
  Task 3/4 findings.
- The U-Net (blueprint Paper 2 stage 1) has not been started. It should
  not be started until this note's open question is resolved or a
  different label-generation approach (§6.2/§6.3) is adopted and
  independently verified against the real distribution — the same
  standard this note itself was held to.

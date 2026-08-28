# Simulator Methodology Note: `simulate_training_pairs.py`

**Status: PAUSED. Two confirmed negative results (Tasks 3, 5) plus one
genuinely mixed result (Task 6) — not a clean success, not a clean
failure.** This document exists so a future session doesn't have to
re-derive six rounds of debugging from WEEK20_LOG.txt alone, the same
role NOVEL_METHODOLOGY_PROPOSAL.md served for the hotspot-map work. It
records what was tried, what was ruled out and why, and what genuinely
remains open. Nothing here claims the simulator is fixed — as of Week 20
it is not, and this sub-investigation is explicitly paused (§6), not
resolved.

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

### 4.4 Real OCO-3 SAM orbital sampling geometry (Task 6) — a genuinely mixed result, not a repeat failure

**What was implemented**: real OCO-3 Snapshot Area Mapping (SAM)
instrument/orbital geometry, replacing the exhaustive readout grid with
sparse simulated footprints. Individual footprints 1.6km (cross-track) x
2.2km (along-track), approximated as a rectangle rather than a true
rhombus (the corner-area difference is small relative to the
footprint-averaging step itself — a stated simplification). Frames of 8
footprints span a 12.8km swath, spaced contiguously along-track at the
2.2km footprint dimension (a standard pushbroom-design assumption, used
instead of an independently-sourced ISS ground-track speed). SAM mode
covers an 80x80km box via 7 parallel swaths x 37 along-track frames x 8
footprints = **2072 raw theoretical footprints per scan**, before
cloud/data-quality loss. The background annulus (44.4-99.9km, too far
for an 80km box) is modeled separately as a sparser Poisson scatter,
density calibrated as a fraction of the near-zone's — the less
rigorously derived half of this fix, disclosed as such.

Both the retention fraction (cloud loss on the 2072 raw footprints) and
the background/near density ratio were calibrated from the same 5 real
facilities used throughout this note (`data/plant_results.json`
"soundings"/hit_days, `data/emission_estimates.json`
"n_bg_before_month_filter"/hit_days): `RETENTION_FRAC_RANGE = (0.132,
0.534)`, `BG_DENSITY_RATIO_RANGE = (0.0352, 0.0600)` — real ranges, not
invented constants.

**The key implementation choice**: Q is read back out using
`physics_ime.py`'s own, unmodified `estimate_emission_rate_from_arrays()`
applied directly to the sparse simulated soundings — not a hand-rolled
mean-difference scalar (Tasks 4-5's approach). This is mathematically
different, not just a sampling-geometry change: `IME_kg` is a SUM of
positive excess (soundings below background contribute exactly zero, not
a diluting pull toward zero the way an unconditional mean does), and
`L_eff = sqrt(n_used * FOOTPRINT_AREA_M2)` scales with how many soundings
actually sit on-plume, not with the near-zone disk's fixed full radius.

**Required validation** (run before trusting anything downstream): each
of the 5 real facilities' own (Q, wind speed, hit_days) replayed through
the simulator, 5 repeats each:

| Facility | near-count ratio (sim/real) | n_used ratio (sim/real) | recovered Q / true Q |
|---|---|---|---|
| Sasan | 1.71x | 2.52x | 0.763 |
| Vindhyachal | 1.66x | 2.15x | 0.720 |
| Talcher | 6.37x | **25.55x** | 3.383 |
| Rihand | 2.14x | 2.17x | 0.690 |
| Tamnar | 5.22x | 9.80x | 2.702 |

Four of five land within a reasonable 1.7-2.5x band — same order of
magnitude as real. Talcher's n_used ratio (25.55x) does exceed an order
of magnitude. Diagnosed: Talcher and Tamnar have the lowest real
retention of the 5 (0.132, 0.154), at the extreme low end of the shared
range these validation repeats draw from — random redraws frequently
land much higher than their true value, inflating their counts. This is
a disclosed limitation of using ONE shared retention range across
facilities with genuinely different real cloud/data-quality conditions
(plausibly regional), not a bug in the raw footprint geometry — the
2072-footprint theoretical count is validated correct in scale for all 5
facilities *before* retention is applied.

**Full result, 200 facilities / 2038 positive tiles**:

| | min | p10 | median | mean | p90 | p99 | max |
|---|---|---|---|---|---|---|---|
| Exhaustive-grid readout (Task 4, carryover) | -0.067 | 0.002 | 0.055 | 0.081 | 0.191 | 0.451 | 0.549 |
| SAM-sparse simple ppm readout | -0.128 | -0.006 | 0.055 | 0.080 | 0.202 | 0.313 | 0.515 |
| Real (N=24, ppm) | -1.277 | -0.866 | 0.619 | 0.620 | 1.839 | 3.409 | 3.698 |

The naive ppm mean-difference metric is **unchanged** by sparse sampling
— median 0.055 either way, still ~11x below real. Confirms Task 5's
diagnosis a fourth time: sparse-but-still-representative sampling doesn't
change the *expected* on-plume fraction, only the noise around it.

`physics_ime.py`'s actual Q-recovery formula tells a different story:

| Metric | This simulator (N=200) | Real IME vs. real CEA (Week 12, N=24) |
|---|---|---|
| Median bias | 2.112x | 0.69x |
| sd(log ratio) | 0.901 | 1.33 |
| Within 2x | 43.5% (87/200) | 41.7% (10/24) |

The simulated forward-model-plus-real-IME pipeline's accuracy *profile*
lands in the **same scale** as how well the real method performs against
real ground truth — qualitatively different from Tasks 3-5, which were
off by 1-2 full orders of magnitude. This is not full closure (a
genuinely reconciled pipeline would show median near 1.0x with tighter
scatter), but it is not a repeat of the prior two rounds' flat failures.

**Residual bias, diagnosed not left unexplained**: correlations across
the 200 facilities between log(recovered/true) and each facility's
parameters — strongest by far, `corr(log_ratio, true_Q) = -0.691`:
facilities with SMALLER true Q are over-estimated MORE.
`corr(log_ratio, n_days) = 0.509`; `corr(log_ratio, wind_speed) = 0.287`
(weaker, real but secondary). The true-Q correlation is the dominant,
most defensible explanation: `SOUNDING_NOISE_STD_PPM = 0.8ppm` is a fixed
per-sounding noise floor regardless of the true plume's strength — for
weaker plumes this floor becomes proportionally more significant, so
more near-zone soundings register spurious `excess > 0` from noise alone
(`clip(near - bg_mean, 0, None)` can't distinguish a small real signal
from a small positive noise fluctuation), inflating both `IME_kg` and
`n_used` without a correspondingly larger true signal to divide back
down by. This mirrors, structurally, this project's own Week 13
signal-to-noise reasoning for Rihand.

## 5. Current best explanation (three consistent findings, not one)

1. A naive mean-difference readout cannot be fixed by sampling geometry
   alone — confirmed four times now (Tasks 4, 5, and twice within
   Task 6's own comparison).
2. `physics_ime.py`'s actual Q-recovery formula, applied to realistically
   sparse SAM-mode soundings, substantially closes the SCALE of the gap
   (from 10-100x off to ~2x median off) — the first approach in this
   investigation to land in the same range as the real method's own
   real-world accuracy, rather than off by orders of magnitude.
3. That improvement is not complete: a diagnosed, systematic weak-signal
   bias (driven by a fixed per-sounding noise floor becoming
   proportionally larger for smaller true-Q facilities) keeps only 43.5%
   of simulated facilities within 2x, and the validation step surfaced a
   real, disclosed sampling-density limitation (shared retention range
   vs. facility-specific real conditions) for facilities at the range's
   extremes.

## 6. Recommendation: PAUSE this sub-investigation

Six rounds have now been run against this gap — near-field/wind-floor/
resolution fixes and area-averaging (Tasks 1-2, correct and kept, but
insufficient alone), Q-source falsification (Task 3, ruled out, made
things worse), IME-consistent readout geometry (Task 4, confirmed the
mismatch but flipped the error direction), multi-day pooling (Task 5,
confirmed again, near-zero effect, specific analytical cause), and real
orbital sampling geometry with native IME Q-recovery (Task 6, the first
genuinely mixed result — substantially improved scale, still not fully
closed, with a diagnosed residual bias). This matches this project's own
diminishing-returns discipline — see WEEK13_LOG.txt's Rihand
investigation, paused after four independently rejected explanations.
Task 6 is NOT a rejected explanation the way Tasks 3 and 5 were; it is
paused because a partially-improved, partially-still-biased result is
not, on its own, a basis to proceed to the U-Net without further
targeted work — not because the approach failed outright.

### Paths forward — not attempted, listed so they aren't re-derived from scratch

1. **Per-facility (or per-region) retention calibration**, rather than
   one shared range across all synthetic facilities. Task 6's own
   validation showed this is where the largest remaining count mismatch
   (Talcher, >10x) comes from — a smaller, more targeted fix than
   anything else on this list.
2. **Model or correct the weak-signal noise-floor bias directly**, now
   that it's diagnosed (§4.4): e.g. a Q-dependent correction informed by
   the measured `corr(log_ratio, true_Q) = -0.691` relationship, or
   revisiting whether `SOUNDING_NOISE_STD_PPM` should scale with
   local background variability rather than being a single fixed
   constant.
3. **Different label-generation methodology entirely.** Abandon
   single-snapshot forward Gaussian simulation as the training-label
   source and generate segmentation masks/Q labels through a method
   structurally consistent with IME's own multi-day aggregation
   assumption.
4. **Re-scope what the U-Net is meant to learn.** If none of the above
   converges, train the segmentation stage on the spatial *shape* of the
   plume only (mask quality), with Q regression handled by a separate
   mechanism not dependent on this simulator's ppm calibration at all.

## 7. What is NOT open

- The Gaussian plume physics itself (`plume_model.py`) is validated and
  unchanged throughout all of this — see `tests/test_plume_model.py`.
- The near-field guard, wind floor, resolution match, and area-averaging
  fixes (§3) are all independently verified and not implicated in the
  Task 3/4/5/6 findings.
- The multi-day pooling MECHANISM itself (§4.3) is verified correct — it
  reduces variance as expected; it simply doesn't address a bias that
  turned out not to be a sampling-count problem.
- The SAM raw footprint GEOMETRY itself (§4.4) is validated correct in
  scale against 4 of 5 real facilities before retention calibration is
  even applied — the residual mismatch traces to the shared-range
  calibration choice and the weak-signal noise bias, not the orbital
  geometry parameters.
- The U-Net (blueprint Paper 2 stage 1) has not been started, and per
  this note's recommendation (§6) should not be started until either the
  sparse-sampling path (§6.1) is tried and verified, or a different
  label-generation approach (§6.2/§6.3) is adopted and independently
  verified against the real distribution — the same standard this note
  itself was held to.

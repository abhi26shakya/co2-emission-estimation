# Simulator Methodology Note: `simulate_training_pairs.py`

**Status: CONCLUDED (Task 8), not paused.** Eight tasks, eight
mechanistically distinct attempts. Q regression as originally scoped in
the blueprint is a documented deviation — this investigation could not
produce a Q label this project's own diminishing-returns discipline
would trust. The segmentation stage is NOT blocked by any of this and
remains a defensible, separately-buildable capability — see §6 for the
full verdict. This document exists so a future session doesn't have to
re-derive eight rounds of debugging from WEEK20_LOG.txt alone, the same
role NOVEL_METHODOLOGY_PROPOSAL.md served for the hotspot-map work.

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

### 4.5 Per-facility retention calibration (Task 7) — clean win on sounding counts, plus a real bug found and fixed; Q-bias moved for a diagnosed, non-obvious reason

**Scoping, per explicit instruction**: this task targets ONLY the
sounding-COUNT mismatch §4.4's validation surfaced (Talcher's `n_used`
off by >10x, traced to Task 6's single shared `RETENTION_FRAC_RANGE`/
`BG_DENSITY_RATIO_RANGE` applied uniformly across facilities with
genuinely different real retention). This is explicitly a SEPARATE issue
from the median Q-bias (2.1x, §4.4's weak-signal noise-floor diagnosis)
— fixing retention calibration was not expected to move the Q-bias
number by itself, and if it did, that needed double-checking, not
assuming a bonus win. It did move; explained below, not just reported.

**Fix 1 — full per-facility calibration table.** Computed
`(retention_frac, bg_density_ratio)` for all 24 real facilities present
in both `data/plant_results.json` and `data/emission_estimates.json`
(`FACILITY_RETENTION_TABLE`), not just the 5 used in Task 6. Real
retention ranges from 0.044 (RGundem) to 0.534 (Vindhyachal), **median
0.128** — heavily skewed low, nowhere near Task 6's uniform range's
implied mean (~0.33). The 200-facility dataset now draws
`(retention_frac, bg_density_ratio)` as a **paired bootstrap sample**
from one real facility's exact values (preserving real per-facility
correlation), not two independent uniform draws.

**Fix 2 — a real bug found during this task.** Testing Fix 1 against the
5 validation facilities with their own exact retention (expecting
near-perfect count matches by construction) instead showed a consistent
~2.64x over-count for every facility. Diagnosed directly:
`simulate_sam_day_soundings()`'s `n_near`/`n_bg` counted every footprint
in the 80×80km SAM box, not just those within `IME_NEAR_KM` (27.75km) —
the box (6400 km²) is exactly `box_area/disk_area` = ~2.64x larger than
the near-zone disk (2419 km²). Verified this was **reporting-only**:
`physics_ime.estimate_emission_rate_from_arrays()` computes its own
`dist`/`near_mask` internally from the full lat/lon arrays — direct
check showed its internal near count for Sasan was 7289 (matching the
expected disk-restricted ~7380), while the buggy `n_near` metric
reported 19482 for the same run. Task 6's Q-recovery numbers were never
wrong from this bug; only the separately-reported count metric was.
Fixed by restricting `n_near`/`n_bg` to the same masks `physics_ime.py`
applies. This bug had been *partially masked* in Task 6 by the
shared-range retention scatter (both errors inflated counts together) —
Task 7's precision is what made it visible.

**Result 1 — sounding-count match (what this task targeted): clean win.**

| Facility | Task 6 near-ratio (sim/real) | Task 7 near-ratio (sim/real) |
|---|---|---|
| Sasan | 1.71x | 0.99x |
| Vindhyachal | 1.66x | 0.99x |
| Talcher | 6.37x | 0.99x |
| Rihand | 2.14x | 0.99x |
| Tamnar | 5.22x | 0.99x |

Every facility now lands within 1% of its real sounding count.

**Result 2 — Q-recovery bias, 200 facilities (not targeted by this task):**

| | Task 6 | Task 7 |
|---|---|---|
| Median bias | 2.112x | 1.540x |
| Mean bias | 3.321 | 2.396 |
| sd(log ratio) | 0.901 | 0.918 |
| Within 2x | 43.5% | 53.0% (106/200) |

The median DID move. Checked, not assumed:

- `corr(log_ratio, true_Q)`: -0.691 → -0.643 (**essentially unchanged**
  — the dominant weak-signal noise-floor mechanism was NOT fixed).
- `corr(log_ratio, retention)` = **+0.310** in Task 7 (not computed in
  Task 6) — Q-recovery bias scales *up* with sample count, consistent
  with the IME formula: more near-zone samples at a roughly fixed
  per-sample excess rate scales `IME_kg` and `n_used` ~linearly, so
  `L_eff = sqrt(n_used * area)` scales as `sqrt(retention)`, giving
  `Q ~ retention / sqrt(retention) = sqrt(retention)` — higher retention
  mechanically produces a *larger instance of the same bias*, not a
  different one.
- Task 7's real-facility-sourced retention distribution has a much lower
  typical value (median 0.124) than Task 6's uniform range's implied
  average (~0.33), because most real facilities genuinely have low
  retention.

So the median-bias improvement is a **side effect of correcting the
retention distribution's shape** (most synthetic facilities now
correctly have fewer samples, like most real facilities do), which
mechanically produces a smaller instance of the same still-present bias
— **not evidence the bias's root cause was addressed**. Per-facility
validation accuracy also genuinely improved for specific replayed
facilities (Sasan's `recovered_Q_ratio_mean`: 0.763 → 0.970; Vindhyachal
0.720 → 0.955) — from removing scatter (random shared-range draws
previously sometimes gave these high-retention facilities a
too-low retention, pulling their average down). Talcher and Tamnar (the
smallest true-Q facilities) remain badly biased even with their own
**exact** real retention (2.155x, 2.003x mean) — direct confirmation the
weak-signal mechanism is untouched, exactly where §4.4 said it would be.

**Verdict**: Task 7 is a clean win on its stated target (sounding-count
match, now within 1% for all 5 validated facilities, plus an
independently-found and fixed reporting bug) and a diagnosed, explained,
not-a-bonus-win partial movement on the Q-bias number. Both reported
plainly, not conflated.

### 4.6 Direct mechanistic correction of the noise-floor bias (Task 8) — FAILED two distinct ways; sub-investigation CONCLUDED

**Checked before writing code**: physics_ime.py's own comments and this
project's citations for a documented basis for noise scaling with local
signal strength (§4.6's option (b)). None found — only sampling-noise
and background-definition-sensitivity discussion (the latter, from
`diagnose_talcher.py`, already flagged weak-signal facilities as a
harder case for a *different*, independently-discovered reason). This
ruled out option (b); proceeded with option (a): a closed-form
statistical correction on `physics_ime.py`'s **output**, not a retuned
`SOUNDING_NOISE_STD_PPM` constant.

**The mechanism**: `clip(near_i - bg_mean, 0, None)` is a rectified/
censored Gaussian. For an off-plume sounding under noise
`N ~ Normal(0, sigma)`: `E[max(N,0)] = sigma/sqrt(2*pi)` (not zero), and
`Prob(N>0) = 0.5` (not zero). `bg_std` (real-world-observable, verified
~0.78-0.81ppm against a true injected 0.8ppm) estimates sigma.

**Attempt 1 — correct `IME_kg` only**: failed systematically. Made
*every* validated facility worse, including the two already near 1.0:

| Facility | Uncorrected | IME-kg-only corrected |
|---|---|---|
| Sasan | 0.970 | 0.501 |
| Vindhyachal | 0.955 | 0.485 |
| Talcher | 2.155 | 0.183 |
| Rihand | 0.788 | 0.440 |
| Tamnar | 2.003 | 0.203 |

Diagnosed why: for Sasan, `n_soundings_used=3852` of `n_near_total=7343`
— almost exactly the ~50% pure-noise baseline (3671.5). `L_eff` was
**also** overwhelmingly noise-driven, not just `IME_kg`. The "good"
uncorrected ratio was an *accidental cancellation* between a
noise-inflated numerator and a noise-inflated denominator — correcting
only the numerator broke that cancellation. This is a materially deeper
finding than §4.4/§4.5 reached: near-zone data is noise-dominated even
for large, easy-to-detect facilities, not only weak ones.

**Attempt 2 — correct both `IME_kg` and `n_used`/`L_eff` consistently**
(subtract the same `0.5 * n_near_total` baseline from `n_soundings_used`
before computing `L_eff`): mechanistically the necessary next step, but
failed through **numerical instability**, not a systematic direction
error. Full 200-facility result:

| | Before (Task 7) | After (Task 8) |
|---|---|---|
| Median bias | 1.540x | 1.037x |
| Mean bias | 2.396 | 1.827 |
| sd(log ratio) | 0.918 | **5.096** |
| Within 2x | 53.0% | 48.5% |
| `corr(log_ratio, true_Q)` | -0.643 | +0.255 |

The median moving to 1.037 looks, in isolation, like a near-perfect fix
— exactly what this task's own instructions warned against trusting
without the full picture. In full: 30/200 facilities produced absurd
ratios (near-zero or up to 44x). Root cause: `n_used_corrected = n_used
- 0.5*n_near_total` subtracts two large, comparable-magnitude quantities
to get a small residual (Sasan: 3852 - 3671.5 = 180.5) — a classic
"difference of nearly-equal large numbers" instability. Small relative
errors in the 50%-baseline assumption (exact only in the infinite-sample
limit) get massively amplified in the residual, which then sits in
`L_eff`'s denominator.

**Required check — Talcher/Tamnar, the sharpest test**: sampled several
Task-8-corrected facilities sourced from Talcher's/Tamnar's own real
retention directly from the 200-facility run — no consistent
improvement; several collapse to exactly 0 (complete overcorrection),
one already-accurate large Talcher-sourced facility (0.921) gets pushed
to 4.715. Decisive: the fix does not reliably help even its own target
case.

**Verdict**: two mechanistically distinct, principled correction
attempts both failed — the first systematically, the second through
instability worse in aggregate variance than doing nothing. The
weak-signal bias is real and mechanistically understood down to the
individual-sounding level, but not fixable by a direct closed-form
correction at this level of effort. Fixing it properly would need
materially more sophisticated machinery (a Bayesian/shrinkage estimator,
or substantially larger near-zone samples to reduce subtraction variance
before amplification) — a different, larger undertaking, not a next
incremental patch.

## 5. Final synthesis — eight tasks, eight mechanistically distinct attempts

1. Near-field singularity fix, wind-speed floor, resolution match
   (Task 1) — correct, kept.
2. Area-averaging (Task 2) — correct, kept, insufficient alone.
3. Q-source falsification (Task 3) — ruled out, made things worse.
4. IME-consistent readout geometry (Task 4) — confirmed a forward/
   inverse mismatch, flipped the error direction.
5. Multi-day pooling (Task 5) — confirmed again, near-zero effect,
   specific analytical cause.
6. Real OCO-3 SAM orbital sampling geometry with native IME Q-recovery
   (Task 6) — first genuinely mixed result, scale substantially
   improved.
7. Per-facility retention calibration (Task 7) — clean win on sounding
   counts (within 1% for all 5 validated facilities, plus a real bug
   found and fixed), isolated the remaining bias to one mechanism.
8. Direct mechanistic correction of that mechanism (Task 8) — failed two
   distinct, well-diagnosed ways.

This is a complete, honest negative result for the specific goal of
*"generate physics-simulated (XCO2 tile, mask, Q) training triples whose
Q label matches physics_ime.py's own real-world accuracy profile."* Not
because any one component is wrong — the plume physics (§3), the orbital
sampling geometry (§4.4), and the sounding-count calibration (§4.5) are
all independently verified correct — but because the fundamental
signal-to-noise regime at individual-sounding scale makes accurate Q
recovery from sparse, noisy, single-facility samples inherently
unstable. That is a property of the measurement/estimation problem
itself, not of this simulator's implementation.

## 6. CONCLUDED — not paused, resolved

Per this project's own diminishing-returns discipline (WEEK13_LOG.txt's
Rihand investigation), this sub-investigation is concluded here, not
attempted a ninth way. Recommended path forward:

- **The U-Net's SEGMENTATION stage** (mask quality) has never been shown
  problematic across any of the 8 tasks — the mask is built directly
  from the true, noise-free physics field (§1-3), entirely independent
  of every Q-calibration attempt in §4.1-4.6. It remains a defensible,
  separately-buildable capability.
- **Q REGRESSION as originally scoped in the blueprint** (Paper 2's
  U-Net → CNN regression architecture) should be treated as a
  **documented deviation**: this investigation could not produce a
  training label this project's own diminishing-returns discipline would
  trust, despite eight mechanistically distinct, honestly-reported
  attempts. Attempting Q regression on these labels would train a model
  whose accuracy claims this project could not stand behind.
- **Recommendation**: build the U-Net scoped to segmentation only (plume
  mask quality), with Q regression deferred to a separate mechanism not
  dependent on this simulator's Q calibration — this is new work, not
  started in this session, deferred to a future session with its own
  dedicated scope rather than tacked onto the close of an 8-task
  investigation.

## 7. What is NOT open

- The Gaussian plume physics itself (`plume_model.py`) is validated and
  unchanged throughout all of this — see `tests/test_plume_model.py`.
- The near-field guard, wind floor, resolution match, and area-averaging
  fixes (§3) are all independently verified and not implicated in any
  Task 3-8 finding.
- The multi-day pooling MECHANISM itself (§4.3) is verified correct — it
  reduces variance as expected; it simply doesn't address a bias that
  turned out not to be a sampling-count problem.
- Sounding-count/geometry accuracy (§4.5) is CLOSED as of Task 7 — every
  validated facility matches real counts within 1%. It was never the
  remaining problem.
- The SAM raw footprint GEOMETRY itself (§4.4) is validated correct in
  scale against 4 of 5 real facilities before retention calibration is
  even applied.
- The Q-recovery bias's MECHANISM is understood, not mysterious — a
  rectified-Gaussian noise floor at individual-sounding scale, verified
  down to matching physics_ime.py's own internal near/n_used counts
  (§4.6). It is CLOSED as an open question (the cause is known); it is
  NOT closed as a fixable problem (§4.6's two attempts both failed).
- **The U-Net's segmentation stage is NOT blocked by any of this** — see
  §6. Its mask ground truth never depended on Q calibration.
- **Q regression as originally scoped in the blueprint IS blocked**, and
  per §6's verdict, concluded as a documented deviation rather than left
  open for a ninth attempt.

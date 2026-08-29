"""
Track B DL architecture (4ypblueprint.pdf, Paper 2 / Dumont Le Brazidec 2024):
stage-1 data generator. Produces synthetic (XCO2 tile, plume segmentation
mask, true Q) training pairs for the U-Net -> CNN regression pipeline that
this project's Track B never built (Track B ended up pure physics --
physics_ime.py / physics_gaussian_crosssection.py -- and Track A's CNN is
detection-only, not quantification).

Why synthetic: no real labeled plume-image dataset exists at this project's
scale -- the same reason SMARTCARB used a transport model instead of real
labels. This script uses plume_model.py (this project's own validated
Briggs/Pasquill-Gifford Gaussian plume physics) as the simulation engine,
NOT an external transport model.

Ground-level concentration (kg/m^3) -> column XCO2 enhancement (ppm):
this requires a boundary-layer-height assumption (H_PBL_M below) to turn a
ground concentration into a column mass, then physics_ime.py's own
column_mass_enhancement() (inverted) to turn that column mass into ppm.
This conversion is a SIMULATION-REALISM CHOICE, not a physical measurement
-- there is no way to do it rigorously without a real transport model.

Three known issues found and fixed during prior debugging (see WEEK20_LOG.txt):

BUG 1 (near-field singularity): plume_model.ground_level_concentration()'s
existing guard (x_safe = np.where(x_m > 1.0, x_m, 1.0)) only prevents a
literal div-by-zero, not the physically-unrealistic concentration spike
within the first few tens of meters of the source (sigma_y/sigma_z are
still ~0 at x=1m). This script does NOT call plume_model.plume_grid()
directly -- it recomputes the downwind/crosswind rotation locally (same
formula as plume_model.py's own module docstring / _rotate_to_plume_frame),
clips downwind_x to a minimum of max(3*H_m, 300) before evaluating
concentration (standard few-stack-heights near-field exclusion rule), and
sets true-upwind pixels (downwind_x <= 0) to zero concentration. NOTE: this
same latent gap likely exists in build_plume_maps.py -- flagged as a
possible follow-up, not fixed here (that file produces paper-facing
figures and is out of scope for this change).

BUG 2 (wind-speed floor): sampling wind speed down to 0.5 m/s produced
near-calm outliers with extreme concentrations (1/U singularity). The real
per-overpass wind estimates in data/emission_estimates.json never go below
1.226 m/s across all 24 facilities -- the real matched-day pipeline
apparently never retains a near-calm case. WIND_SPEED_RANGE floors at
1.2 m/s, justified by that real data, not an arbitrary "avoid blowing up"
clip.

BUG 3 (resolution mismatch): tile pixel size matches Track A's actual
convention (SIZE_KM=60, PX=64, ~937.5 m/px, per export_facility_tiles.py),
not a finer arbitrary grid -- a finer grid would under-sample the spatial
averaging real satellite pixels naturally provide.

TASK 2 fix (area-averaging, this file's main addition over the bug fixes
above): point-sampling the analytic concentration function at each pixel's
center coordinate still produces an unrealistic high tail, because a real
satellite sounding spatially integrates over its footprint rather than
sampling a single point -- that averaging is what smooths out near-field
spikes. Each pixel here is evaluated on an AREA_AVG_N x AREA_AVG_N subgrid
spanning its own footprint and averaged, not sampled once at its center.

TASK 3 fixes (Q-source falsification -- see WEEK20_LOG.txt Task 3): tested
whether the residual peak-tail gap was caused by using IME's own Q instead
of an independent ground-truth Q. It was not -- capping Q to CEA's real
range made the gap WORSE, not better, ruling out Q source as the fixable
variable.

TASK 4 fix (this addition -- IME-consistent Q readout, see
SIMULATOR_METHODOLOGY_NOTE.md for the full design rationale): Task 3 left
open the possibility that the forward simulator (point-source Gaussian
dispersion, read out as a single peak pixel) and physics_ime.py's inverse
method (a spatial mass-balance integral: near-plant-zone mean minus
background-annulus mean, over a MUCH larger area than one pixel) are
simply not comparable quantities -- comparing a peak pixel to an IME-
DERIVED Q was never an apples-to-apples realism check. This fix does NOT
change the underlying 2D concentration field, which remains genuine
Gaussian plume physics with the BUG 1 near-field guard and Task 2 area-
averaging already applied. It only changes how Q is READ BACK OUT of that
field for calibration: instead of the single hottest pixel, the readout
now replicates physics_ime.py's own near-plant/background-annulus
geometry (NEAR=0.25 deg, BG_IN/BG_OUT=0.4/0.9 deg, converted to km at the
same ~111 km/deg this project's own comments already use) and computes
mean(near-zone) - mean(background-annulus), exactly matching
process_plant.py's co2_enhancement_ppm definition. Because
physics_ime.py's background annulus (out to ~100 km) is far larger than
Track A's 60km training tile, this readout is computed on a SEPARATE,
larger calibration-only grid (never saved as a training tile) sized to
contain the full annulus.

TASK 5 fix (multi-day aggregation for CALIBRATION ONLY -- see
SIMULATOR_METHODOLOGY_NOTE.md Sec 6.1): Task 4 diagnosed that a single
synthetic tile's near-zone disk mean is diluted almost to zero because
>97% of that disk sits off-plume for any ONE fixed wind direction, while
a real co2_enhancement_ppm value pools soundings across MANY overpass
days, each potentially a different real wind direction. This fix tests
that diagnosis directly: for a synthetic FACILITY (fixed Q, wind speed,
stack height, stability -- the physical characteristics of one plant),
simulate N independent single-direction days (wind direction resampled
fresh each day) and pool their near-zone/background-annulus samples the
same way physics_ime.py pools real per-overpass soundings (concatenate
raw samples across days, THEN take the mean of each pooled zone), rather
than averaging N single-day scalar readouts. N is drawn per facility
from the REAL hit_days distribution (data/plant_results.json, N=30
facilities: min 1, max 25, median 8, mean 9.93) -- not an arbitrary
constant. SCOPING CONSTRAINT, strictly enforced: this changes ONLY the
calibration/verification readout. Each individual simulated day is still
generated by the exact same single-snapshot make_tile() code path as
Tasks 1-4 -- one wind direction, one day, its own exact mask -- and is
saved to the training set completely unaggregated, exactly as it would
be if generated standalone. The only structural change to training-tile
generation is that groups of tiles now share a common (Q, wind speed,
stack height, stability) drawn once per synthetic facility rather than
independently per tile, which is disclosed explicitly here: it does not
change what any individual tile represents or how it is generated, only
how several tiles are grouped together downstream for the pooled
calibration check.

TASK 6 fix (real OCO-3 orbital sampling geometry, see
SIMULATOR_METHODOLOGY_NOTE.md Sec 4.4/6 for the full rationale): Task 5
diagnosed why multi-day pooling failed -- the readout evaluated an
EXHAUSTIVE spatial grid every day, not the SPARSE, orbital-track-shaped
sample a real satellite actually produces, so pooling more full-disk
days couldn't shift the systematic dilution bias. This fix replaces the
exhaustive readout grid with a simulated OCO-3 Snapshot Area Mapping
(SAM) scan: individual footprints 1.6km (cross-track) x 2.2km
(along-track), approximated as a small rectangle rather than a true
rhombus -- the corner-area difference between a rhombus and a rectangle
of the same width/height is a small correction relative to the
footprint-averaging step itself (already a simplification), so a
rectangle was used for implementation simplicity, stated explicitly
rather than silently. Footprints are arranged in frames of 8 across a
~13km swath (8 x 1.6km = 12.8km), spaced contiguously along-track at the
2.2km footprint dimension (a standard pushbroom-instrument design
assumption -- contiguous coverage along track -- used here instead of an
independently-sourced ISS ground-track speed, to avoid citing an
uncertain external constant). SAM mode covers an ~80x80km box around the
target via repeated parallel swaths: 7 swaths (12.8km each, covering
89.6km) x 37 along-track frames (2.2km each, covering 81.4km) x 8
footprints/frame = 2072 raw theoretical footprints per scan, BEFORE any
cloud/data-quality loss.

Background-annulus sampling (44.4-99.9km out) is NOT modeled with the
same SAM-raster geometry -- SAM mode specifically targets a box around
the facility (per the task's own framing) and a single 80km box cannot
physically reach the real background annulus's outer edge (~100km), the
same geometric mismatch already documented in Task 4. Background
soundings are modeled as a much sparser, Poisson-scattered sample within
the annulus, its density set as a fraction of the near-zone's simulated
density -- calibrated (not independently derived from orbital mechanics,
disclosed as the less rigorous half of this fix) against the same real
facility data used to calibrate near-zone retention (see below).

Both the near-zone SAM retention fraction and the background/near
density ratio are drawn from REAL per-facility density ratios, not
invented: computed from 5 real facilities (Sasan, Vindhyachal, Talcher,
Rihand, Tamnar; data/plant_results.json "soundings"/"hit_days" for raw
near-zone density, data/emission_estimates.json
"n_bg_before_month_filter"/hit_days for background density) against this
fix's own theoretical raw geometric footprint count -- retention 0.132
to 0.534, background/near density ratio 0.0352 to 0.0600.

Q is read back out using physics_ime.py's own
estimate_emission_rate_from_arrays() applied DIRECTLY to this sparse,
simulated sounding set (synthetic lat/lon built from the same
km/111 deg conversion used throughout this file) -- not a hand-rolled
near-mean-minus-bg-mean scalar as in Tasks 4-5. This is a materially
different mathematical operation, not just a different sampling
geometry: physics_ime.py's IME_kg is a SUM of positive excess over near
soundings (soundings below background contribute exactly zero, not a
diluting pull toward zero the way an unconditional MEAN does), and its
effective length L_eff = sqrt(n_used * FOOTPRINT_AREA_M2) scales with
how many near soundings actually sit on-plume, not with the fixed
near-zone disk's full radius -- a genuinely different, and specifically
motivated, test of whether this project's own inverse method recovers Q
correctly given a realistically sparse forward-simulated sounding set.
"""
import json
import os

import numpy as np

import physics_ime
import plume_model
from physics_ime import BG_IN as IME_BG_IN_DEG
from physics_ime import BG_OUT as IME_BG_OUT_DEG
from physics_ime import NEAR as IME_NEAR_DEG
from physics_ime import column_mass_enhancement

SEED = 42

# --- Tile geometry: MUST match Track A's export_facility_tiles.py exactly ---
SIZE_KM, PX = 60, 64
PX_SIZE_M = SIZE_KM * 1000.0 / PX  # ~937.5 m/px

# --- Area-averaging subgrid (Task 2 fix) ---
AREA_AVG_N = 5  # 5x5 subsamples per pixel footprint

# --- Column conversion assumption (flagged, not hidden) ---
H_PBL_M = 800.0  # boundary-layer height used to turn ground kg/m^3 into a
                 # column kg/m^2; a reasonable daytime default, NOT a
                 # physical measurement -- see module docstring.
PPM_PER_KG_M2 = 1.0 / column_mass_enhancement(1.0)  # inverse of physics_ime's ppm->kg/m^2

# --- Background / noise, matched to data/plant_results.json's real scale ---
BG_XCO2_PPM = 415.0
SOUNDING_NOISE_STD_PPM = 0.8

# --- Cloud-gap masking ---
CLOUD_GAP_FRAC_RANGE = (0.15, 0.55)

# --- Sampling ranges, drawn from this project's own real data (see module
#     docstring for BUG 2 on the wind floor) ---
# Q range: CEA ground-truth abs_emissions_t_co2 min/max (data/cea_ground_truth_2020_21.json,
# N=30), NOT the raw IME q_t_per_year range used in the first Week 20 attempt.
# Week 20 diagnosed IME's Q distribution as ~2.2x more right-skewed than CEA's
# (P95/median: IME 5.55 vs CEA 2.58 on the matched N=24) -- a real shape
# difference, not just a scale difference -- so this is a genuine, motivated
# retest, not a repeat of the same range under another name.
Q_T_PER_YEAR_RANGE = (2.061468660722863e6, 3.3212767714887384e7)  # CEA min (RayalSeema) / max (Vindhyachal)
WIND_SPEED_RANGE = (1.2, 4.0)         # m/s; floor justified, not arbitrary
STACK_HEIGHT_RANGE = (150.0, 275.0)   # m; matches plume_model.py's own ablation range
STABILITY_CLASSES = ["A", "B", "C"]

N_NEGATIVE = 300
MASK_THRESHOLD_PPM = 0.1  # true (noise-free) enhancement above this -> mask=1

OUT_DIR = "data/simulated_train"
OUT_NPZ = os.path.join(OUT_DIR, "simulated_tiles.npz")
OUT_META = os.path.join(OUT_DIR, "simulated_tiles_meta.json")

# --- TASK 4: IME-consistent readout geometry ---
# physics_ime.py / co2_enhancement.py define NEAR/BG_IN/BG_OUT in degrees
# and compute `dist` as a raw Euclidean degree distance (no cos(lat)
# longitude correction) -- the same ~111 km/deg conversion their own
# comments use ("<0.25 ~25 km", "0.4-0.9 ~45km to ~100km") is reused here,
# not "fixed", to stay consistent with how Q is actually estimated in this
# project.
KM_PER_DEG = 111.0
IME_NEAR_KM = IME_NEAR_DEG * KM_PER_DEG      # ~27.75 km
IME_BG_IN_KM = IME_BG_IN_DEG * KM_PER_DEG    # ~44.4 km
IME_BG_OUT_KM = IME_BG_OUT_DEG * KM_PER_DEG  # ~99.9 km

# physics_ime.py's background annulus (out to ~100 km) is far larger than
# Track A's 60km training tile, so the IME-style readout is evaluated on a
# separate, bigger grid -- calibration-only, never saved as a training
# tile. Resolution matches physics_ime.py's own FOOTPRINT_AREA_M2
# (~1.5km, the real OCO-3 sounding footprint), not Track A's tile pixel
# size, since this grid isn't standing in for a satellite image.
READOUT_HALF_EXTENT_KM = IME_BG_OUT_KM + 10.0  # buffer past the outer annulus edge
READOUT_PX_SIZE_KM = 1.5
READOUT_PX = int(round(2 * READOUT_HALF_EXTENT_KM / READOUT_PX_SIZE_KM))

# --- TASK 5: real per-facility overpass-day counts (data/plant_results.json
# "hit_days", N=30: min 1, max 25, median 8, mean 9.93) -- bootstrap-sampled
# directly from these real values (not a fitted/invented range) to set how
# many single-direction days are pooled per synthetic facility.
HIT_DAYS_POOL = [1, 1, 1, 1, 4, 4, 5, 5, 5, 5, 5, 5, 6, 7, 7, 9, 9, 10, 12,
                 14, 14, 15, 16, 16, 17, 19, 19, 20, 21, 25]
NUM_POSITIVE_FACILITIES = 200  # synthetic facilities; total day-tiles = sum(n_days), reported at runtime

# --- TASK 6: OCO-3 Snapshot Area Mapping (SAM) instrument/orbital geometry
# -- see module docstring for the full rationale. ---
FOOTPRINT_CROSS_TRACK_KM = 1.6
FOOTPRINT_ALONG_TRACK_KM = 2.2
FOOTPRINT_SUBGRID_N = 3  # small-footprint area-average subsamples (Task 2's principle, footprint scale)
FOOTPRINTS_PER_FRAME = 8
FRAME_SPACING_KM = FOOTPRINT_ALONG_TRACK_KM  # contiguous along-track coverage assumption
SWATH_WIDTH_KM = FOOTPRINTS_PER_FRAME * FOOTPRINT_CROSS_TRACK_KM  # 12.8 km, ~"13km swath"
SAM_BOX_HALF_KM = 40.0  # 80km x 80km SAM target box, centered on the facility
N_SWATHS = int(np.ceil(2 * SAM_BOX_HALF_KM / SWATH_WIDTH_KM))          # 7 (covers 89.6 km)
N_FRAMES_PER_SWATH = int(np.ceil(2 * SAM_BOX_HALF_KM / FRAME_SPACING_KM))  # 37 (covers 81.4 km)
RAW_FOOTPRINTS_PER_SAM_SCAN = N_SWATHS * N_FRAMES_PER_SWATH * FOOTPRINTS_PER_FRAME  # 2072
SAM_BOX_AREA_KM2 = (2 * SAM_BOX_HALF_KM) ** 2  # 6400

# TASK 7 (see WEEK20_LOG.txt Task 7 / SIMULATOR_METHODOLOGY_NOTE.md Sec
# 4.5): Task 6's original RETENTION_FRAC_RANGE/BG_DENSITY_RATIO_RANGE
# (below, kept for reference/documentation only, no longer used for
# sampling) were calibrated from only 5 real facilities and drawn as a
# single shared CONTINUOUS UNIFORM range across all synthetic facilities.
# Task 6's own validation showed this produces >10x sounding-count
# mismatches for facilities whose real retention sits at the range's
# extremes (Talcher, Tamnar) -- because a uniform range implicitly
# assumes retention is uniformly distributed between its min and max,
# which the real data does NOT support (see below: the real distribution
# is heavily skewed toward LOW retention, median 0.128, not centered
# near the old range's midpoint ~0.33).
#
# Fix: retention_frac and bg_density_ratio are now drawn as a PAIRED
# bootstrap sample from FACILITY_RETENTION_TABLE -- the ACTUAL per-
# facility (retention, bg_ratio) values computed from all 24 real
# facilities present in BOTH data/plant_results.json ("soundings",
# "hit_days") and data/emission_estimates.json
# ("n_bg_before_month_filter"), not a continuous range fitted to 5. Each
# synthetic facility "borrows" one real facility's actual calibrated
# density characteristics (preserving the real per-facility correlation
# between retention and bg_ratio), rather than sampling the two
# independently from an idealized uniform range.
RETENTION_FRAC_RANGE = (0.132, 0.534)      # Task 6 range; superseded, kept for reference
BG_DENSITY_RATIO_RANGE = (0.0352, 0.0600)  # Task 6 range; superseded, kept for reference

# name: (retention_frac, bg_density_ratio, hit_days, n_soundings_used) for
# all 24 real facilities with both real data sources. retention_frac =
# (soundings/hit_days) / 783.22 (this fix's own theoretical near-zone-only
# footprint count); bg_density_ratio = (n_bg_before_month_filter/hit_days
# density over the 25160 km^2 annulus) / (near-zone density). Computed
# directly from real data, not fitted or invented.
FACILITY_RETENTION_TABLE = {
    "RGundem":          (0.0439, 0.0447, 5, 5),
    "ShriSingajiMalwa": (0.0582, 0.0506, 5, 36),
    "Sagardighi":       (0.0784, 0.0716, 7, 3),
    "Raichur":          (0.0838, 0.0460, 5, 20),
    "RayalSeema":       (0.0838, 0.0521, 9, 44),
    "Kahalgaon":        (0.0947, 0.0606, 14, 61),
    "Korba":            (0.0981, 0.0646, 5, 78),
    "ChandrapurCoal":   (0.1036, 0.0676, 14, 25),
    "Kudgi":            (0.1079, 0.0398, 4, 49),
    "KGudemNew":        (0.1221, 0.0436, 10, 226),
    "TalwandiSabo":     (0.1235, 0.0412, 4, 23),
    "Chhabra":          (0.1240, 0.0637, 9, 29),
    "Talcher":          (0.1316, 0.0600, 21, 105),
    "Tamnar":           (0.1536, 0.0359, 7, 86),
    "Farakka":          (0.1600, 0.0444, 12, 153),
    "Tirora":           (0.1713, 0.0751, 5, 6),
    "Koradi":           (0.1875, 0.0453, 25, 279),
    "Mouda":            (0.1937, 0.0449, 19, 422),
    "Pryagraj(Bara)":   (0.2144, 0.0382, 15, 427),
    "Dadri(Nctpp)":     (0.3185, 0.0409, 20, 757),
    "Anpara":           (0.4603, 0.0388, 16, 858),
    "Rihand":           (0.4845, 0.0390, 16, 1182),
    "Sasan":            (0.4959, 0.0372, 19, 997),
    "Vindhyachal":      (0.5343, 0.0352, 17, 1083),
}
FACILITY_RETENTION_NAMES = list(FACILITY_RETENTION_TABLE.keys())

# The 5 real facilities used for the sounding-count VALIDATION step (run
# each facility's own real Q, wind speed, hit_days -- AND, as of Task 7,
# its own real retention_frac/bg_density_ratio from
# FACILITY_RETENTION_TABLE, not a redraw -- back through this simulator
# and compare simulated sounding counts to that facility's real ones).
SAM_VALIDATION_FACILITIES = {
    # name: (q_t_per_year, wind_speed_ms, hit_days, real_n_soundings_used, real_soundings_total)
    "Sasan":       (3.9279e7, 1.3165, 19, 997, 7379),
    "Vindhyachal": (3.6738e7, 1.2522, 17, 1083, 7114),
    "Talcher":     (6.9146e6, 1.7695, 21, 105, 2165),
    "Rihand":      (4.8347e7, 1.2935, 16, 1182, 6072),
    "Tamnar":      (3.2231e6, 1.2260, 7, 86, 842),
}


def _pixel_centers_km(n_px=PX, size_km=SIZE_KM):
    axis_km = (np.arange(n_px) + 0.5) / n_px * size_km - size_km / 2.0
    east_km, north_km = np.meshgrid(axis_km, axis_km)
    return east_km, north_km


def _pixel_subgrid_offsets_km(px_size_km=None):
    """AREA_AVG_N x AREA_AVG_N offsets spanning one pixel's own footprint,
    centered on 0."""
    if px_size_km is None:
        px_size_km = PX_SIZE_M / 1000.0
    off = (np.arange(AREA_AVG_N) + 0.5) / AREA_AVG_N * px_size_km - px_size_km / 2.0
    return off


def _concentration_field_kg_m3(east_km, north_km, sub_off, Q_t_per_year,
                                wind_speed_ms, wind_from_deg, stack_height_m,
                                stability_class):
    """
    Ground-level concentration (kg/m^3) on an arbitrary (east_km, north_km)
    grid, area-averaged over each cell's own footprint via an
    AREA_AVG_N x AREA_AVG_N subgrid (Task 2 fix), with the near-field
    singularity guard from BUG 1 applied at every subsample point. Shared
    core for both the training-tile grid and the larger Task 4 readout
    grid -- same physics either way, only the grid geometry differs.
    """
    theta = np.radians(wind_from_deg + 180.0)
    near_field_floor_m = max(3.0 * stack_height_m, 300.0)

    acc = np.zeros(east_km.shape, dtype=np.float64)
    for dy in sub_off:
        for dx in sub_off:
            e_m = (east_km + dx) * 1000.0
            n_m = (north_km + dy) * 1000.0
            downwind_x = e_m * np.sin(theta) + n_m * np.cos(theta)
            crosswind_y = e_m * np.cos(theta) - n_m * np.sin(theta)
            dx_clipped = np.where(downwind_x > near_field_floor_m,
                                   downwind_x, near_field_floor_m)
            conc = plume_model.ground_level_concentration(
                Q_t_per_year, wind_speed_ms, stack_height_m,
                dx_clipped, crosswind_y, stability_class)
            conc = np.where(downwind_x <= 0, 0.0, conc)
            acc += conc
    return acc / (len(sub_off) * len(sub_off))


def area_averaged_concentration_kg_m3(Q_t_per_year, wind_speed_ms, wind_from_deg,
                                       stack_height_m, stability_class):
    """Training-tile-resolution (60km/64px) area-averaged concentration."""
    east_km, north_km = _pixel_centers_km()
    sub_off = _pixel_subgrid_offsets_km()
    return _concentration_field_kg_m3(east_km, north_km, sub_off, Q_t_per_year,
                                       wind_speed_ms, wind_from_deg,
                                       stack_height_m, stability_class)


def _readout_pixel_centers_km():
    return _pixel_centers_km(n_px=READOUT_PX, size_km=2 * READOUT_HALF_EXTENT_KM)


def _readout_subgrid_offsets_km():
    return _pixel_subgrid_offsets_km(px_size_km=READOUT_PX_SIZE_KM)


def _cloud_gap_mask(rng, shape=(PX, PX)):
    """Random rectangular cloud patches covering CLOUD_GAP_FRAC_RANGE of
    the grid. A simplification (real cloud fields are not rectangles) --
    flagged, not hidden."""
    target_frac = rng.uniform(*CLOUD_GAP_FRAC_RANGE)
    mask = np.zeros(shape, dtype=bool)
    covered = 0.0
    guard = 0
    min_dim = min(shape)
    while covered < target_frac and guard < 50:
        h = rng.integers(4, max(5, min_dim // 2))
        w = rng.integers(4, max(5, min_dim // 2))
        r0 = rng.integers(0, shape[0] - h)
        c0 = rng.integers(0, shape[1] - w)
        mask[r0:r0 + h, c0:c0 + w] = True
        covered = mask.mean()
        guard += 1
    return mask


def _readout_zone_samples(rng, Q_t_per_year, wind_speed_ms, wind_from_deg,
                           stack_height_m, stability_class):
    """
    One single-direction day's raw near-zone and background-annulus ppm
    samples on the READOUT grid (same genuine physics as the training
    tile -- near-field guard + area-averaging both applied). Shared core
    for both the single-day readout (Task 4) and the multi-day pooled
    readout (Task 5) -- pooling operates on these raw arrays, not on
    already-averaged scalars, matching physics_ime.py's own pooling
    (concatenate raw soundings across days, THEN take the mean).
    """
    east_km, north_km = _readout_pixel_centers_km()
    sub_off = _readout_subgrid_offsets_km()

    if Q_t_per_year > 0:
        conc_kg_m3 = _concentration_field_kg_m3(
            east_km, north_km, sub_off, Q_t_per_year, wind_speed_ms,
            wind_from_deg, stack_height_m, stability_class)
        enhancement_ppm = conc_kg_m3 * H_PBL_M * PPM_PER_KG_M2
    else:
        enhancement_ppm = np.zeros_like(east_km)

    # No fresh cloud-gap mask here, unlike make_tile()'s training tile: a
    # real co2_enhancement_ppm value (process_plant.py) is built from a
    # whole year of overpasses already, i.e. an aggregate over many
    # cloud-free-at-that-pixel observations -- not one cloudy snapshot.
    # Per-sounding measurement noise still applies (real soundings are
    # noisy even on clear days).
    noise = rng.normal(0.0, SOUNDING_NOISE_STD_PPM, size=east_km.shape)
    field = BG_XCO2_PPM + enhancement_ppm + noise

    dist_km = np.sqrt(east_km ** 2 + north_km ** 2)
    near_mask = dist_km < IME_NEAR_KM
    bg_mask = (dist_km > IME_BG_IN_KM) & (dist_km < IME_BG_OUT_KM)
    return field[near_mask].copy(), field[bg_mask].copy()


def ime_style_readout_ppm(rng, Q_t_per_year, wind_speed_ms, wind_from_deg,
                           stack_height_m, stability_class):
    """
    TASK 4: reads Q back out of the SAME genuine Gaussian plume field for
    a SINGLE day, using physics_ime.py's own near-plant/background
    geometry, replicating process_plant.py's co2_enhancement_ppm exactly:
    mean(near-zone ppm) - mean(background-annulus ppm).
    """
    near_vals, bg_vals = _readout_zone_samples(
        rng, Q_t_per_year, wind_speed_ms, wind_from_deg, stack_height_m, stability_class)
    return float(near_vals.mean() - bg_vals.mean())


def multi_day_ime_readout_ppm(rng, Q_t_per_year, wind_speed_ms, stack_height_m,
                               stability_class, n_days):
    """
    TASK 5, calibration/verification ONLY -- see module docstring. Pools
    n_days independent single-direction days' near-zone and background-
    annulus samples (raw arrays concatenated across days, not per-day
    scalars averaged), matching how physics_ime.py pools real
    per-overpass soundings across matched days before computing bg_mean
    and the near/bg difference. Returns (pooled_readout_ppm,
    per_day_wind_from_deg).
    """
    near_all, bg_all, wind_dirs = [], [], []
    for _ in range(n_days):
        wind_from_deg = float(rng.uniform(0.0, 360.0))
        near_vals, bg_vals = _readout_zone_samples(
            rng, Q_t_per_year, wind_speed_ms, wind_from_deg, stack_height_m, stability_class)
        near_all.append(near_vals)
        bg_all.append(bg_vals)
        wind_dirs.append(wind_from_deg)
    near_pool = np.concatenate(near_all)
    bg_pool = np.concatenate(bg_all)
    pooled = float(near_pool.mean() - bg_pool.mean())
    return pooled, wind_dirs


def _sam_scan_footprint_offsets_km(rng, retention_frac):
    """
    TASK 6: the sparse (east_km, north_km) footprint centers one simulated
    SAM scan of the 80x80km target box actually produces -- a raster of
    N_SWATHS parallel swaths x N_FRAMES_PER_SWATH along-track frames x
    FOOTPRINTS_PER_FRAME footprints, then a random cloud/data-quality
    RETENTION_FRAC subset kept. Fixed local (east, north) frame -- real
    satellite ground-track heading is not modeled, a stated simplification
    (see module docstring).
    """
    swath_centers = ((np.arange(N_SWATHS) + 0.5) / N_SWATHS * (2 * SAM_BOX_HALF_KM)
                      - SAM_BOX_HALF_KM)
    frame_positions = ((np.arange(N_FRAMES_PER_SWATH) + 0.5) / N_FRAMES_PER_SWATH
                        * (2 * SAM_BOX_HALF_KM) - SAM_BOX_HALF_KM)
    footprint_cross_offsets = (np.arange(FOOTPRINTS_PER_FRAME) - (FOOTPRINTS_PER_FRAME - 1) / 2.0) \
        * FOOTPRINT_CROSS_TRACK_KM

    east_grid = (swath_centers[:, None, None] + footprint_cross_offsets[None, None, :])
    east_grid = np.broadcast_to(east_grid, (N_SWATHS, N_FRAMES_PER_SWATH, FOOTPRINTS_PER_FRAME))
    north_grid = np.broadcast_to(frame_positions[None, :, None],
                                  (N_SWATHS, N_FRAMES_PER_SWATH, FOOTPRINTS_PER_FRAME))
    east_all = east_grid.ravel()
    north_all = north_grid.ravel()

    keep = rng.random(east_all.size) < retention_frac
    return east_all[keep].copy(), north_all[keep].copy()


def _background_footprint_offsets_km(rng, bg_density_km2):
    """
    TASK 6: sparse Poisson-scattered background-annulus footprint centers
    (see module docstring for why this uses a different, calibrated-only
    model rather than SAM-raster geometry).
    """
    annulus_area_km2 = np.pi * (IME_BG_OUT_KM ** 2 - IME_BG_IN_KM ** 2)
    n_expected = max(bg_density_km2 * annulus_area_km2, 0.0)
    n = int(rng.poisson(n_expected)) if n_expected > 0 else 0
    if n == 0:
        return np.zeros(0), np.zeros(0)
    r = np.sqrt(rng.uniform(IME_BG_IN_KM ** 2, IME_BG_OUT_KM ** 2, size=n))
    theta = rng.uniform(0.0, 2 * np.pi, size=n)
    return r * np.cos(theta), r * np.sin(theta)


def _evaluate_footprints_ppm(rng, east_km, north_km, Q_t_per_year, wind_speed_ms,
                              wind_from_deg, stack_height_m, stability_class):
    """
    TASK 6: small-footprint area-average (FOOTPRINT_SUBGRID_N x
    FOOTPRINT_SUBGRID_N subsamples spanning each footprint's real
    1.6km x 2.2km rectangular extent, consistent with Task 2's per-pixel
    averaging principle applied at footprint scale) ppm value at each
    scattered footprint center, with BUG 1's near-field guard applied at
    every subsample. Same underlying physics as the training tile and the
    Task 4/5 readout grid -- only the sample locations differ.
    """
    if east_km.size == 0:
        return np.zeros(0)

    cross_offsets = ((np.arange(FOOTPRINT_SUBGRID_N) + 0.5) / FOOTPRINT_SUBGRID_N
                      * FOOTPRINT_CROSS_TRACK_KM - FOOTPRINT_CROSS_TRACK_KM / 2.0)
    along_offsets = ((np.arange(FOOTPRINT_SUBGRID_N) + 0.5) / FOOTPRINT_SUBGRID_N
                      * FOOTPRINT_ALONG_TRACK_KM - FOOTPRINT_ALONG_TRACK_KM / 2.0)

    if Q_t_per_year > 0:
        theta = np.radians(wind_from_deg + 180.0)
        near_field_floor_m = max(3.0 * stack_height_m, 300.0)
        acc = np.zeros_like(east_km, dtype=np.float64)
        for dy in along_offsets:
            for dx in cross_offsets:
                e_m = (east_km + dx) * 1000.0
                n_m = (north_km + dy) * 1000.0
                downwind_x = e_m * np.sin(theta) + n_m * np.cos(theta)
                crosswind_y = e_m * np.cos(theta) - n_m * np.sin(theta)
                dx_clipped = np.where(downwind_x > near_field_floor_m,
                                       downwind_x, near_field_floor_m)
                conc = plume_model.ground_level_concentration(
                    Q_t_per_year, wind_speed_ms, stack_height_m,
                    dx_clipped, crosswind_y, stability_class)
                conc = np.where(downwind_x <= 0, 0.0, conc)
                acc += conc
        conc_kg_m3 = acc / (FOOTPRINT_SUBGRID_N * FOOTPRINT_SUBGRID_N)
        enhancement_ppm = conc_kg_m3 * H_PBL_M * PPM_PER_KG_M2
    else:
        enhancement_ppm = np.zeros_like(east_km, dtype=np.float64)

    noise = rng.normal(0.0, SOUNDING_NOISE_STD_PPM, size=east_km.shape)
    return BG_XCO2_PPM + enhancement_ppm + noise


def simulate_sam_day_soundings(rng, day_int, Q_t_per_year, wind_speed_ms, wind_from_deg,
                                stack_height_m, stability_class, retention_frac, bg_density_ratio):
    """
    TASK 6: one simulated facility-day's full sparse sounding set (near
    SAM-box scan + background-annulus scatter), in the same lat/lon/xco2/
    day array shape physics_ime.estimate_emission_rate_from_arrays()
    expects. Synthetic lat/lon are built from the same km/111-deg
    conversion used throughout this file (KM_PER_DEG), with the facility
    placed at (lat, lon) = (0, 0) -- an arbitrary but harmless reference
    since only relative distance matters and physics_ime.py's own `dist`
    computation applies no cos(lat) correction either.
    """
    near_east, near_north = _sam_scan_footprint_offsets_km(rng, retention_frac)
    near_zone_density_km2 = retention_frac * RAW_FOOTPRINTS_PER_SAM_SCAN / SAM_BOX_AREA_KM2
    bg_density_km2 = bg_density_ratio * near_zone_density_km2
    bg_east, bg_north = _background_footprint_offsets_km(rng, bg_density_km2)

    east_all = np.concatenate([near_east, bg_east])
    north_all = np.concatenate([near_north, bg_north])
    xco2_all = _evaluate_footprints_ppm(rng, east_all, north_all, Q_t_per_year, wind_speed_ms,
                                         wind_from_deg, stack_height_m, stability_class)
    lat_all = north_all / KM_PER_DEG
    lon_all = east_all / KM_PER_DEG
    day_all = np.full(east_all.shape, day_int, dtype=np.int64)

    # TASK 7 bug fix: n_near/n_bg must report the counts physics_ime.py's
    # own near_mask/bg_mask would actually select (dist < IME_NEAR_KM /
    # IME_BG_IN_KM < dist < IME_BG_OUT_KM), NOT len(near_east)/len(bg_east)
    # -- _sam_scan_footprint_offsets_km() returns footprints across the
    # WHOLE 80x80km SAM box (needed so physics_ime.py's own masking can
    # select the disk-restricted subset from real, full-box-shaped data),
    # so len(near_east) over-counts by (box_area/near_disk_area) ~2.6x.
    # physics_ime.estimate_emission_rate_from_arrays() computes its own
    # dist/near_mask internally from lat_all/lon_all and was NEVER
    # affected by this -- this fixes a reporting-only bug in the
    # validation/count metrics, not the Q-recovery computation itself.
    near_dist_km = np.sqrt(near_east ** 2 + near_north ** 2)
    n_near_disk_restricted = int((near_dist_km < IME_NEAR_KM).sum())
    bg_dist_km = np.sqrt(bg_east ** 2 + bg_north ** 2) if bg_east.size else np.zeros(0)
    n_bg_annulus_restricted = int(((bg_dist_km > IME_BG_IN_KM) & (bg_dist_km < IME_BG_OUT_KM)).sum())
    return lat_all, lon_all, xco2_all, day_all, n_near_disk_restricted, n_bg_annulus_restricted


def recover_q_from_sam_scans(rng, facility_name, Q_t_per_year, wind_speed_ms, stack_height_m,
                              stability_class, n_days, retention_frac, bg_density_ratio,
                              day_start=20200101):
    """
    TASK 6: simulates n_days independent SAM-scan days (fresh wind
    direction each day) for one facility, pools all their sparse soundings
    the way physics_ime.py itself pools real per-overpass soundings for
    one plant, and reads Q back out via
    physics_ime.estimate_emission_rate_from_arrays() -- the project's own,
    unmodified IME implementation, not a hand-rolled readout. Returns
    (result_dict_or_None, per_day_wind_from_deg, n_near_total, n_bg_total,
    simple_ppm_readout_or_None).
    """
    lat_all, lon_all, xco2_all, day_all = [], [], [], []
    wind_dirs = []
    n_near_total = n_bg_total = 0
    for i in range(n_days):
        wind_from_deg = float(rng.uniform(0.0, 360.0))
        day_int = day_start + i
        lat, lon, xco2, day, n_near, n_bg = simulate_sam_day_soundings(
            rng, day_int, Q_t_per_year, wind_speed_ms, wind_from_deg,
            stack_height_m, stability_class, retention_frac, bg_density_ratio)
        lat_all.append(lat)
        lon_all.append(lon)
        xco2_all.append(xco2)
        day_all.append(day)
        wind_dirs.append(wind_from_deg)
        n_near_total += n_near
        n_bg_total += n_bg

    lat_all = np.concatenate(lat_all)
    lon_all = np.concatenate(lon_all)
    xco2_all = np.concatenate(xco2_all)
    day_all = np.concatenate(day_all)

    dist_km = np.sqrt(lat_all ** 2 + lon_all ** 2) * KM_PER_DEG
    near_mask = dist_km < IME_NEAR_KM
    bg_mask = (dist_km > IME_BG_IN_KM) & (dist_km < IME_BG_OUT_KM)
    simple_ppm_readout = (float(xco2_all[near_mask].mean() - xco2_all[bg_mask].mean())
                           if near_mask.sum() > 0 and bg_mask.sum() > 0 else None)

    plant_row = {"lat": 0.0, "lon": 0.0}
    wind_series = {int(day_start + i): wind_speed_ms for i in range(n_days)}
    result = physics_ime.estimate_emission_rate_from_arrays(
        facility_name, lat_all, lon_all, xco2_all, day_all, plant_row, wind_series)
    return result, wind_dirs, n_near_total, n_bg_total, simple_ppm_readout


def validate_sam_sounding_counts(seed=SEED, n_repeats=5):
    """
    TASK 6/7 REQUIRED VALIDATION -- run BEFORE trusting anything
    downstream. For each of SAM_VALIDATION_FACILITIES, replays that REAL
    facility's own (Q, wind speed, hit_days) through this simulator
    n_repeats times, AS OF TASK 7 using that SAME facility's own real
    retention_frac/bg_density_ratio from FACILITY_RETENTION_TABLE (not a
    redraw from a shared range) -- isolating remaining repeat-to-repeat
    variation to wind direction, background Poisson count, and
    measurement noise, not "which facility's retention got drawn this
    time". Compares simulated sounding counts to that facility's actual
    real counts. If simulated density were off by an order of magnitude
    from real, the geometry parameters would be wrong and nothing
    downstream should be trusted.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for name, (q, wind_speed, hit_days, real_n_used, real_soundings) in SAM_VALIDATION_FACILITIES.items():
        retention, bg_ratio, _, _ = FACILITY_RETENTION_TABLE[name]
        sim_near, sim_bg, sim_n_used, sim_q_ratio = [], [], [], []
        for _ in range(n_repeats):
            result, _, n_near, n_bg, _ = recover_q_from_sam_scans(
                rng, f"{name}_sim", q, wind_speed, 220.0, "B", hit_days, retention, bg_ratio)
            sim_near.append(n_near)
            sim_bg.append(n_bg)
            if result is not None:
                sim_n_used.append(result["n_soundings_used"])
                sim_q_ratio.append(result["q_t_per_year"] / q)
        rows.append(dict(
            facility=name,
            real_soundings_total=real_soundings,
            sim_near_total_mean=float(np.mean(sim_near)),
            sim_near_total_range=(int(np.min(sim_near)), int(np.max(sim_near))),
            near_ratio_sim_over_real=float(np.mean(sim_near)) / real_soundings,
            real_n_soundings_used=real_n_used,
            sim_n_soundings_used_mean=float(np.mean(sim_n_used)) if sim_n_used else None,
            n_used_ratio_sim_over_real=(float(np.mean(sim_n_used)) / real_n_used) if sim_n_used else None,
            true_q_t_per_year=q,
            recovered_q_ratio_mean=float(np.mean(sim_q_ratio)) if sim_q_ratio else None,
            recovered_q_ratio_range=((min(sim_q_ratio), max(sim_q_ratio)) if sim_q_ratio else None),
            n_repeats_with_valid_result=len(sim_q_ratio),
        ))
    return rows


def make_tile(rng, positive, q=None, wind_speed=None, stack_height=None, stability=None,
              wind_from_deg=None):
    """
    Generates ONE single-snapshot training tile (one wind direction, one
    simulated day, its own exact mask) -- unchanged mechanism from Tasks
    1-4. The optional q/wind_speed/stack_height/stability let TASK 5's
    facility grouping (see module docstring) share those facility-level
    physical characteristics across several calls. wind_from_deg is
    resampled fresh inside this function UNLESS explicitly passed in --
    TASK 6 passes the SAME day's wind_from_deg used to generate that
    day's SAM scan, so one simulated SAM-style scan = one training tile
    (same physical realization), per the explicit requirement that the
    sparse-sampling geometry change must not decouple a tile's image from
    its calibration readout. Calling this with no optional args (the
    Tasks 1-4 behavior) samples everything independently, exactly as
    before.
    """
    if positive:
        if q is None:
            q = float(np.exp(rng.uniform(np.log(Q_T_PER_YEAR_RANGE[0]),
                                          np.log(Q_T_PER_YEAR_RANGE[1]))))
        if wind_speed is None:
            wind_speed = float(rng.uniform(*WIND_SPEED_RANGE))
        if wind_from_deg is None:
            wind_from_deg = float(rng.uniform(0.0, 360.0))
        if stack_height is None:
            stack_height = float(rng.uniform(*STACK_HEIGHT_RANGE))
        if stability is None:
            stability = str(rng.choice(STABILITY_CLASSES))

        conc_kg_m3 = area_averaged_concentration_kg_m3(
            q, wind_speed, wind_from_deg, stack_height, stability)
        column_mass_kg_m2 = conc_kg_m3 * H_PBL_M
        enhancement_ppm = column_mass_kg_m2 * PPM_PER_KG_M2
    else:
        q = 0.0
        wind_speed = float(rng.uniform(*WIND_SPEED_RANGE))
        wind_from_deg = float(rng.uniform(0.0, 360.0))
        stack_height = float(rng.uniform(*STACK_HEIGHT_RANGE))
        stability = str(rng.choice(STABILITY_CLASSES))
        enhancement_ppm = np.zeros((PX, PX), dtype=np.float64)

    mask = (enhancement_ppm > MASK_THRESHOLD_PPM).astype(np.uint8)

    noise = rng.normal(0.0, SOUNDING_NOISE_STD_PPM, size=(PX, PX))
    tile = BG_XCO2_PPM + enhancement_ppm + noise

    cloud_mask = _cloud_gap_mask(rng)
    tile = tile.copy()
    tile[cloud_mask] = np.nan

    peak_ppm = float(enhancement_ppm.max())
    ime_readout_ppm = ime_style_readout_ppm(
        rng, q, wind_speed, wind_from_deg, stack_height, stability)
    params = dict(
        positive=positive,
        q_t_per_year=q,
        wind_speed_ms=wind_speed,
        wind_from_deg=wind_from_deg,
        stack_height_m=stack_height,
        stability_class=stability,
        cloud_gap_frac=float(cloud_mask.mean()),
        peak_enhancement_ppm=peak_ppm,
        ime_readout_ppm=ime_readout_ppm,
    )
    return tile.astype(np.float32), mask, params


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    tiles, masks, q_values, all_params = [], [], [], []
    facility_records = []

    # TASK 6: positive tiles are still generated in synthetic-facility
    # groups (Task 5's structure), but the calibration readout now uses
    # simulated OCO-3 SAM-mode sparse sounding geometry + physics_ime.py's
    # own IME logic, replacing Task 4/5's exhaustive-grid mean-difference.
    # Each facility draws ONE (Q, wind speed, stack height, stability,
    # retention_frac, bg_density_ratio); each of its n_days (HIT_DAYS_POOL)
    # draws ONE fresh wind direction, shared between that day's training
    # tile (make_tile(), unchanged single-snapshot mechanism) and that
    # day's simulated SAM scan -- one simulated SAM-style scan = one
    # training tile, per the explicit requirement. Pooling across a
    # facility's days happens via physics_ime.estimate_emission_rate_from_arrays()
    # itself, exactly matching how the real per-plant pipeline pools
    # soundings across hit_days -- never touching the saved tiles/masks.
    n_facilities_insufficient = 0
    for facility_id in range(NUM_POSITIVE_FACILITIES):
        q = float(np.exp(rng.uniform(np.log(Q_T_PER_YEAR_RANGE[0]),
                                      np.log(Q_T_PER_YEAR_RANGE[1]))))
        wind_speed = float(rng.uniform(*WIND_SPEED_RANGE))
        stack_height = float(rng.uniform(*STACK_HEIGHT_RANGE))
        stability = str(rng.choice(STABILITY_CLASSES))
        n_days = int(rng.choice(HIT_DAYS_POOL))
        # TASK 7: paired bootstrap from a REAL facility's own (retention,
        # bg_ratio), not independent draws from a shared continuous range
        # -- see FACILITY_RETENTION_TABLE's docstring comment.
        source_facility = str(rng.choice(FACILITY_RETENTION_NAMES))
        retention_frac, bg_density_ratio, _, _ = FACILITY_RETENTION_TABLE[source_facility]
        day_start = 20200101 + facility_id * 40  # keep each facility's days in a distinct, non-overlapping dummy date block

        lat_all, lon_all, xco2_all, day_all = [], [], [], []
        day_wind_dirs, n_near_total, n_bg_total = [], 0, 0
        for day_idx in range(n_days):
            wind_from_deg = float(rng.uniform(0.0, 360.0))
            day_int = day_start + day_idx

            tile, mask, params = make_tile(rng, positive=True, q=q, wind_speed=wind_speed,
                                            stack_height=stack_height, stability=stability,
                                            wind_from_deg=wind_from_deg)
            params["facility_id"] = facility_id
            tiles.append(tile)
            masks.append(mask)
            q_values.append(params["q_t_per_year"])
            all_params.append(params)

            lat, lon, xco2, day, n_near, n_bg = simulate_sam_day_soundings(
                rng, day_int, q, wind_speed, wind_from_deg, stack_height, stability,
                retention_frac, bg_density_ratio)
            lat_all.append(lat)
            lon_all.append(lon)
            xco2_all.append(xco2)
            day_all.append(day)
            day_wind_dirs.append(wind_from_deg)
            n_near_total += n_near
            n_bg_total += n_bg

        lat_all = np.concatenate(lat_all)
        lon_all = np.concatenate(lon_all)
        xco2_all = np.concatenate(xco2_all)
        day_all = np.concatenate(day_all)

        dist_km = np.sqrt(lat_all ** 2 + lon_all ** 2) * KM_PER_DEG
        near_mask = dist_km < IME_NEAR_KM
        bg_mask = (dist_km > IME_BG_IN_KM) & (dist_km < IME_BG_OUT_KM)
        simple_ppm_readout = (float(xco2_all[near_mask].mean() - xco2_all[bg_mask].mean())
                               if near_mask.sum() > 0 and bg_mask.sum() > 0 else None)

        plant_row = {"lat": 0.0, "lon": 0.0}
        wind_series = {int(day_start + i): wind_speed for i in range(n_days)}
        result = physics_ime.estimate_emission_rate_from_arrays(
            f"synthetic_facility_{facility_id}", lat_all, lon_all, xco2_all, day_all,
            plant_row, wind_series)
        if result is None:
            n_facilities_insufficient += 1

        facility_records.append(dict(
            facility_id=facility_id,
            q_t_per_year=q,
            wind_speed_ms=wind_speed,
            stack_height_m=stack_height,
            stability_class=stability,
            n_days=n_days,
            retention_frac=retention_frac,
            bg_density_ratio=bg_density_ratio,
            retention_source_facility=source_facility,
            per_day_wind_from_deg=day_wind_dirs,
            n_near_soundings_total=n_near_total,
            n_bg_soundings_total=n_bg_total,
            simple_ppm_readout=simple_ppm_readout,
            ime_result_available=result is not None,
            ime_recovered_q_t_per_year=(result["q_t_per_year"] if result is not None else None),
            ime_recovered_q_ratio=((result["q_t_per_year"] / q) if result is not None else None),
            ime_n_soundings_used=(result["n_soundings_used"] if result is not None else None),
        ))

    for _ in range(N_NEGATIVE):
        tile, mask, params = make_tile(rng, positive=False)
        tiles.append(tile)
        masks.append(mask)
        q_values.append(params["q_t_per_year"])
        all_params.append(params)

    tiles = np.stack(tiles, axis=0)
    masks = np.stack(masks, axis=0)
    q_values = np.array(q_values, dtype=np.float64)

    np.savez_compressed(OUT_NPZ, tiles=tiles, masks=masks, q_t_per_year=q_values)

    pos_peaks = np.array([p["peak_enhancement_ppm"] for p in all_params if p["positive"]])
    pos_ime_readout = np.array([p["ime_readout_ppm"] for p in all_params if p["positive"]])
    neg_ime_readout = np.array([p["ime_readout_ppm"] for p in all_params if not p["positive"]])
    sam_simple_readouts = np.array([r["simple_ppm_readout"] for r in facility_records
                                     if r["simple_ppm_readout"] is not None])
    valid_facilities = [r for r in facility_records if r["ime_result_available"]]
    sam_q_ratios = np.array([r["ime_recovered_q_ratio"] for r in valid_facilities])
    sam_n_soundings_used = np.array([r["ime_n_soundings_used"] for r in valid_facilities])
    n_positive_tiles = int(pos_peaks.size)
    real_min, real_max = -1.277, 3.698  # data/plant_results.json co2_enhancement_ppm, N=24
    # Full real distribution (not just min/max), data/plant_results.json
    # co2_enhancement_ppm, N=24:
    real_p10, real_median, real_mean, real_p90, real_p99 = -0.866, 0.619, 0.620, 1.839, 3.409

    def _dist_stats(vals):
        return dict(
            min=float(vals.min()),
            p10=float(np.percentile(vals, 10)),
            median=float(np.median(vals)),
            mean=float(vals.mean()),
            p90=float(np.percentile(vals, 90)),
            p99=float(np.percentile(vals, 99)),
            max=float(vals.max()),
        )

    print("Running TASK 6 required sounding-count validation against 5 real facilities "
          "(their own real Q/wind speed/hit_days replayed through this simulator) "
          "before trusting anything else...")
    sam_validation = validate_sam_sounding_counts(seed=SEED, n_repeats=5)

    calibration_report = dict(
        n_positive_facilities=NUM_POSITIVE_FACILITIES,
        n_positive_tiles=n_positive_tiles,
        n_negative=N_NEGATIVE,
        n_facilities_with_insufficient_soundings=n_facilities_insufficient,
        hit_days_pool_source="data/plant_results.json hit_days, N=30 (min 1, max 25, median 8, mean 9.93)",
        retention_calibration_method="TASK 7: paired bootstrap from FACILITY_RETENTION_TABLE (24 real facilities), not a shared range",
        retention_frac_range_legacy_task6=RETENTION_FRAC_RANGE,
        bg_density_ratio_range_legacy_task6=BG_DENSITY_RATIO_RANGE,
        sam_sounding_count_validation=sam_validation,
        peak_enhancement_ppm_stats=_dist_stats(pos_peaks),
        ime_readout_ppm_stats=_dist_stats(pos_ime_readout),
        ime_readout_ppm_stats_negative_tiles=_dist_stats(neg_ime_readout),
        sam_facility_simple_ppm_readout_stats=_dist_stats(sam_simple_readouts),
        sam_recovered_q_ratio_stats=_dist_stats(sam_q_ratios),
        sam_n_soundings_used_stats=_dist_stats(sam_n_soundings_used),
        real_reference_distribution_ppm=dict(
            min=real_min, p10=real_p10, median=real_median, mean=real_mean,
            p90=real_p90, p99=real_p99, max=real_max, n_facilities=24,
            source="data/plant_results.json co2_enhancement_ppm"),
        frac_positive_tiles_within_real_range_peak=float(
            np.mean((pos_peaks >= real_min) & (pos_peaks <= real_max))),
        frac_positive_tiles_within_real_range_ime_readout=float(
            np.mean((pos_ime_readout >= real_min) & (pos_ime_readout <= real_max))),
        frac_facilities_within_real_range_sam_readout=float(
            np.mean((sam_simple_readouts >= real_min) & (sam_simple_readouts <= real_max))),
        h_pbl_m_assumption=H_PBL_M,
        area_avg_n=AREA_AVG_N,
        tile_px=PX,
        tile_size_km=SIZE_KM,
        px_size_m=PX_SIZE_M,
        ime_near_km=IME_NEAR_KM,
        ime_bg_in_km=IME_BG_IN_KM,
        ime_bg_out_km=IME_BG_OUT_KM,
        readout_half_extent_km=READOUT_HALF_EXTENT_KM,
        readout_px=READOUT_PX,
        readout_px_size_km=READOUT_PX_SIZE_KM,
        raw_footprints_per_sam_scan=RAW_FOOTPRINTS_PER_SAM_SCAN,
    )

    meta = dict(
        seed=SEED,
        params=all_params,
        facility_records=facility_records,
        calibration_report=calibration_report,
    )
    with open(OUT_META, "w") as f:
        json.dump(meta, f, indent=2)

    print("=== simulate_training_pairs.py summary ===")
    print(f"Wrote {len(tiles)} tiles ({n_positive_tiles} positive across "
          f"{NUM_POSITIVE_FACILITIES} facilities, {N_NEGATIVE} negative) to {OUT_NPZ}")
    print(f"Tile shape: {tiles.shape[1:]} ({SIZE_KM}km / {PX}px, ~{PX_SIZE_M:.1f}m/px)")
    print(f"n_days per facility drawn from real hit_days "
          f"(data/plant_results.json, N=30, min 1 max 25 median 8 mean 9.93)")
    print(f"n_facilities with insufficient soundings for physics_ime (excluded from Q-recovery stats): "
          f"{n_facilities_insufficient}/{NUM_POSITIVE_FACILITIES}")
    print()
    print("[TASK 6 VALIDATION] simulated vs real sounding counts, 5 real facilities' own "
          "(Q, wind, hit_days) replayed, 5 repeats each:")
    for row in sam_validation:
        print(f"  {row['facility']}: real_soundings={row['real_soundings_total']} "
              f"sim_near_mean={row['sim_near_total_mean']:.0f} "
              f"(ratio sim/real={row['near_ratio_sim_over_real']:.2f}x)  "
              f"real_n_used={row['real_n_soundings_used']} "
              f"sim_n_used_mean={row['sim_n_soundings_used_mean']:.0f} "
              f"(ratio={row['n_used_ratio_sim_over_real']:.2f}x)  "
              f"recovered_Q_ratio_mean={row['recovered_q_ratio_mean']:.3f}")
    print()
    print(f"[legacy] Peak enhancement (ppm), {n_positive_tiles} positive tiles:")
    for k, v in calibration_report["peak_enhancement_ppm_stats"].items():
        print(f"  {k}: {v:.3f}")
    print(f"[TASK 4] Single-day exhaustive-grid readout (ppm), {n_positive_tiles} positive tiles "
          f"(near<{IME_NEAR_KM:.1f}km minus bg {IME_BG_IN_KM:.1f}-{IME_BG_OUT_KM:.1f}km):")
    for k, v in calibration_report["ime_readout_ppm_stats"].items():
        print(f"  {k}: {v:.3f}")
    print(f"[TASK 6] SAM-sparse simple ppm readout (near.mean()-bg.mean() on the sparse "
          f"sounding set), {NUM_POSITIVE_FACILITIES} synthetic facilities:")
    for k, v in calibration_report["sam_facility_simple_ppm_readout_stats"].items():
        print(f"  {k}: {v:.3f}")
    print(f"[TASK 6] physics_ime.py RECOVERED Q / TRUE Q ratio, "
          f"{len(valid_facilities)}/{NUM_POSITIVE_FACILITIES} synthetic facilities with a valid IME result:")
    for k, v in calibration_report["sam_recovered_q_ratio_stats"].items():
        print(f"  {k}: {v:.3f}")
    print(f"Real full distribution (ppm), N=24 (data/plant_results.json co2_enhancement_ppm):")
    for k, v in calibration_report["real_reference_distribution_ppm"].items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"Fraction within real min/max range -- peak pixel: "
          f"{calibration_report['frac_positive_tiles_within_real_range_peak']:.3f}, "
          f"single-day exhaustive-grid readout: {calibration_report['frac_positive_tiles_within_real_range_ime_readout']:.3f}, "
          f"SAM-sparse simple readout: {calibration_report['frac_facilities_within_real_range_sam_readout']:.3f}")
    print(f"Meta + calibration report written to {OUT_META}")


if __name__ == "__main__":
    main()

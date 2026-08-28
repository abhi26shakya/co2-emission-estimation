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
"""
import json
import os

import numpy as np

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

N_POSITIVE = 600
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


def ime_style_readout_ppm(rng, Q_t_per_year, wind_speed_ms, wind_from_deg,
                           stack_height_m, stability_class):
    """
    TASK 4: reads Q back out of the SAME genuine Gaussian plume field
    (near-field guard + area-averaging both applied, identical physics to
    the training tile) using physics_ime.py's own near-plant/background
    geometry, replicating process_plant.py's co2_enhancement_ppm exactly:
    mean(near-zone ppm) - mean(background-annulus ppm). Computed on the
    separate, larger READOUT grid (see module docstring) since the real
    background annulus doesn't fit inside a 60km training tile.
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

    near_mean = float(field[near_mask].mean())
    bg_mean = float(field[bg_mask].mean())
    return near_mean - bg_mean


def make_tile(rng, positive):
    if positive:
        q = float(np.exp(rng.uniform(np.log(Q_T_PER_YEAR_RANGE[0]),
                                      np.log(Q_T_PER_YEAR_RANGE[1]))))
        wind_speed = float(rng.uniform(*WIND_SPEED_RANGE))
        wind_from_deg = float(rng.uniform(0.0, 360.0))
        stack_height = float(rng.uniform(*STACK_HEIGHT_RANGE))
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

    for _ in range(N_POSITIVE):
        tile, mask, params = make_tile(rng, positive=True)
        tiles.append(tile)
        masks.append(mask)
        q_values.append(params["q_t_per_year"])
        all_params.append(params)

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
    real_min, real_max = -1.277, 3.698  # data/plant_results.json co2_enhancement_ppm, N=24

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

    calibration_report = dict(
        n_positive=N_POSITIVE,
        n_negative=N_NEGATIVE,
        peak_enhancement_ppm_stats=_dist_stats(pos_peaks),
        ime_readout_ppm_stats=_dist_stats(pos_ime_readout),
        ime_readout_ppm_stats_negative_tiles=_dist_stats(neg_ime_readout),
        real_reference_range_ppm=dict(min=real_min, max=real_max, n_facilities=24,
                                       source="data/plant_results.json co2_enhancement_ppm"),
        frac_positive_tiles_within_real_range_peak=float(
            np.mean((pos_peaks >= real_min) & (pos_peaks <= real_max))),
        frac_positive_tiles_within_real_range_ime_readout=float(
            np.mean((pos_ime_readout >= real_min) & (pos_ime_readout <= real_max))),
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
    )

    meta = dict(
        seed=SEED,
        params=all_params,
        calibration_report=calibration_report,
    )
    with open(OUT_META, "w") as f:
        json.dump(meta, f, indent=2)

    print("=== simulate_training_pairs.py summary ===")
    print(f"Wrote {len(tiles)} tiles ({N_POSITIVE} positive, {N_NEGATIVE} negative) "
          f"to {OUT_NPZ}")
    print(f"Tile shape: {tiles.shape[1:]} ({SIZE_KM}km / {PX}px, ~{PX_SIZE_M:.1f}m/px)")
    print(f"[legacy] Peak enhancement (ppm), {N_POSITIVE} positive tiles:")
    for k, v in calibration_report["peak_enhancement_ppm_stats"].items():
        print(f"  {k}: {v:.3f}")
    print(f"[TASK 4] IME-consistent readout (ppm), {N_POSITIVE} positive tiles "
          f"(near<{IME_NEAR_KM:.1f}km minus bg {IME_BG_IN_KM:.1f}-{IME_BG_OUT_KM:.1f}km):")
    for k, v in calibration_report["ime_readout_ppm_stats"].items():
        print(f"  {k}: {v:.3f}")
    print(f"[TASK 4] IME-consistent readout (ppm), {N_NEGATIVE} negative (no-plume) tiles "
          f"(sanity check, should be ~0):")
    for k, v in calibration_report["ime_readout_ppm_stats_negative_tiles"].items():
        print(f"  {k}: {v:.3f}")
    print(f"Real reference range: [{real_min}, {real_max}] ppm (N=24 facilities)")
    print(f"Fraction of positive tiles within real range (peak pixel): "
          f"{calibration_report['frac_positive_tiles_within_real_range_peak']:.3f}")
    print(f"Fraction of positive tiles within real range (IME-consistent readout): "
          f"{calibration_report['frac_positive_tiles_within_real_range_ime_readout']:.3f}")
    print(f"Meta + calibration report written to {OUT_META}")


if __name__ == "__main__":
    main()

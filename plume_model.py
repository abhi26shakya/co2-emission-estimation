"""
Phase 1 of NOVEL_METHODOLOGY_PROPOSAL.md: a wind-conditioned Gaussian
plume model that spatially distributes each plant's Track B mass-balance
emission rate (Q) into a ground-level concentration field -- the spatial
"hotspot map" capability this project did not previously have (see
NOVEL_METHODOLOGY_PROPOSAL.md Sec 1 for the gap analysis: despite the
filename, physics_gaussian.py contains no actual 2D dispersion equation,
only a scalar near/background mass-balance split).

Standard, citable physics, newly applied in this project:

  C(x, y, 0) = Q / (pi * U * sigma_y(x) * sigma_z(x))
               * exp(-y^2 / (2*sigma_y(x)^2))
               * exp(-H^2 / (2*sigma_z(x)^2))

  Pasquill-Gifford dispersion coefficients sigma_y(x), sigma_z(x) via the
  Briggs (1973) rural formulas, as tabulated in Turner, D.B., "Workbook of
  Atmospheric Dispersion Estimates" (2nd ed., 1994) -- the standard
  reference implementation of the original Pasquill-Gifford curves.

Explicit, documented assumptions (not invented, not hidden):
  - Stability class: fixed at Pasquill class B (moderately unstable) by
    default, justified by this project's own per-overpass wind data (mean
    ~1.2-1.8 m/s across the prototype facilities, consistent with light-
    wind daytime conditions) and OCO-3's typical daytime overpass timing.
    This is NOT derived from a full insolation/cloud-cover scheme --
    flagged as a simplification. Classes A and C are supported as an
    explicit sensitivity ablation (see build_plume_maps.py).
  - Effective stack height H: 220m by default, citing India's CPCB/MoEFCC
    stack-height norms for coal-fired thermal power plants (minimum 220m
    for units/aggregate capacity >=500MW; all facilities in this project
    exceed that threshold by a wide margin). Included as an explicit
    sensitivity ablation at 150/220/275m.
  - Q (emission rate): the RAW Track B physics estimate (q_t_per_year from
    data/emission_estimates.json), not the ground-truth-corrected value.
    q_correction_model.py's correction is a statistical accuracy
    adjustment for benchmarking against CEA data, not a re-derivation of
    the physical mass flux -- conflating the two would misrepresent what
    physically flows through this equation.

Honesty constraint (NOVEL_METHODOLOGY_PROPOSAL.md Sec 4.2): this produces
a physically-derived VISUALIZATION calibrated to a validated total mass
flux, not a directly-observed spatial measurement -- OCO-3's sparse
per-sounding footprint cannot support a pixel-accuracy claim. Validation
against real sounding locations is a spatial CONSISTENCY check, not a
pixel-accuracy benchmark (see validate_plume_spatial_consistency.py,
Phase 2).
"""
import numpy as np

SEC_PER_YEAR = 3600 * 24 * 365
DEFAULT_STABILITY_CLASS = "B"
DEFAULT_STACK_HEIGHT_M = 220.0

# Briggs (1973) rural sigma_y(x) = a * x * (1 + b*x)^p, x in meters.
_SIGMA_Y_COEFFS = {
    "A": (0.22, 0.0001, -0.5),
    "B": (0.16, 0.0001, -0.5),
    "C": (0.11, 0.0001, -0.5),
    "D": (0.08, 0.0001, -0.5),
    "E": (0.06, 0.0001, -0.5),
    "F": (0.04, 0.0001, -0.5),
}

# Briggs (1973) rural sigma_z(x) = a * x * (1 + b*x)^p (b=0 => sigma_z = a*x).
_SIGMA_Z_COEFFS = {
    "A": (0.20, 0.0, 0.0),
    "B": (0.12, 0.0, 0.0),
    "C": (0.08, 0.0002, -0.5),
    "D": (0.06, 0.0015, -0.5),
    "E": (0.03, 0.0003, -1.0),
    "F": (0.016, 0.0003, -1.0),
}


def _sigma_y(x_m, stability_class):
    a, b, p = _SIGMA_Y_COEFFS[stability_class]
    return a * x_m * (1 + b * x_m) ** p


def _sigma_z(x_m, stability_class):
    a, b, p = _SIGMA_Z_COEFFS[stability_class]
    if b == 0.0:
        return a * x_m
    return a * x_m * (1 + b * x_m) ** p


def ground_level_concentration(Q_t_per_year, U_ms, H_m, x_m, y_m,
                                stability_class=DEFAULT_STABILITY_CLASS):
    """
    Q_t_per_year: emission rate, tons CO2/year (converted internally to kg/s)
    U_ms: wind speed, m/s
    H_m: effective stack height, m
    x_m, y_m: downwind / crosswind distance from the source, meters
              (scalars or same-shape arrays; x<=0 is upwind of the source)
    Returns ground-level CO2 concentration enhancement, kg/m^3.
    """
    if U_ms <= 0:
        raise ValueError("wind speed must be > 0 -- the Gaussian plume "
                          "equation is undefined for calm conditions")
    Q_kg_s = Q_t_per_year * 1000.0 / SEC_PER_YEAR
    x_m = np.asarray(x_m, dtype=np.float64)
    y_m = np.asarray(y_m, dtype=np.float64)
    x_safe = np.where(x_m > 1.0, x_m, 1.0)  # sigma formulas are undefined at x=0
    sy = _sigma_y(x_safe, stability_class)
    sz = _sigma_z(x_safe, stability_class)
    conc = (Q_kg_s / (np.pi * U_ms * sy * sz)
            * np.exp(-y_m ** 2 / (2 * sy ** 2))
            * np.exp(-H_m ** 2 / (2 * sz ** 2)))
    return np.where(x_m > 1.0, conc, 0.0)  # physically zero upwind of the source


def plume_grid(Q_t_per_year, wind_speed_ms, wind_from_deg,
                H_m=DEFAULT_STACK_HEIGHT_M, stability_class=DEFAULT_STABILITY_CLASS,
                extent_km=30, resolution_m=500):
    """
    Builds a 2D ground-level concentration grid centered on the plant, in a
    north-up local (east_km, north_km) frame, with the plume rotated to
    point in the actual downwind direction.

    wind_from_deg: meteorological convention -- the compass direction the
    wind is blowing FROM (matches plant_results.json's existing wind_deg
    field, computed the same way in process_plant.py / physics_gaussian.py).
    The plume travels in the direction wind_from_deg + 180.

    Returns (grid_kg_m3, east_km_axis, north_km_axis).
    """
    n = int(2 * extent_km * 1000 / resolution_m) + 1
    axis_m = np.linspace(-extent_km * 1000, extent_km * 1000, n)
    east_m, north_m = np.meshgrid(axis_m, axis_m)

    theta = np.radians(wind_from_deg + 180.0)
    downwind_x = east_m * np.sin(theta) + north_m * np.cos(theta)
    crosswind_y = east_m * np.cos(theta) - north_m * np.sin(theta)

    conc = ground_level_concentration(Q_t_per_year, wind_speed_ms, H_m,
                                       downwind_x, crosswind_y, stability_class)
    return conc, axis_m / 1000.0, axis_m / 1000.0

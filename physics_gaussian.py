"""
physics_gaussian.py
-------------------
Nassar et al. (2017) Gaussian-plume inversion -- your PHYSICS BASELINE.

This is NOT machine learning. It is the analytical method every ML/DL model
in this field is compared against. Runs on CPU in milliseconds.

Core idea (the "one breath" version):
    A power plant emits CO2. Wind blows it into a plume. A satellite measures a
    cross-section of the column-CO2 enhancement (XCO2 above background). The
    enhancement across the plume is ~Gaussian. Fit that Gaussian, combine with
    wind speed, and back-calculate the emission rate Q.

Use it three ways in your project:
    1. Reproduce Nassar's ~1-17% error on real OCO-2 overpasses.
    2. Generate / sanity-check labels (Q) for training the CNN.
    3. Validation anchor: report "Gaussian-Q" next to "CNN-Q" in every results table.

Author scaffold for: Abhishek Kumar Shakya
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import curve_fit


# ----------------------------------------------------------------------
# 1. The Gaussian cross-section model
# ----------------------------------------------------------------------
def gaussian_cross_section(y, amplitude, center, sigma, background):
    """
    XCO2 enhancement as a function of cross-wind distance y.

    Parameters
    ----------
    y          : array, cross-wind distance from an arbitrary origin [m]
    amplitude  : peak enhancement above background [ppm]  (the "bump" height)
    center     : plume center position along y [m]
    sigma      : plume width (std dev of the bell) [m]
    background : residual baseline XCO2 [ppm]

    Returns
    -------
    XCO2(y) [ppm]
    """
    return amplitude * np.exp(-((y - center) ** 2) / (2.0 * sigma ** 2)) + background


# ----------------------------------------------------------------------
# 2. Fit the model to a measured slice
# ----------------------------------------------------------------------
def fit_plume(y, xco2, p0=None):
    """
    Least-squares fit of the Gaussian cross-section to a measured XCO2 slice.

    Parameters
    ----------
    y    : array [m]      cross-wind distance of each sounding
    xco2 : array [ppm]    measured column-averaged CO2 for each sounding
    p0   : optional initial guess (amplitude, center, sigma, background)

    Returns
    -------
    params : dict with amplitude, center, sigma, background (+ 1-sigma errors)
    """
    y = np.asarray(y, dtype=float)
    xco2 = np.asarray(xco2, dtype=float)

    # Smooth a copy to locate the peak robustly under noise (low-SNR defence).
    if len(xco2) >= 5:
        kernel = np.ones(5) / 5.0
        xco2_smooth = np.convolve(xco2, kernel, mode="same")
    else:
        xco2_smooth = xco2

    if p0 is None:
        # Sensible automatic initial guess -- this matters for convergence.
        bg0 = np.percentile(xco2, 20)          # background ~ the low end
        amp0 = max(xco2_smooth.max() - bg0, 0.1)
        center0 = y[np.argmax(xco2_smooth)]    # peak from the SMOOTHED slice
        sigma0 = (y.max() - y.min()) / 8.0     # rough width
        p0 = [amp0, center0, sigma0, bg0]

    span = y.max() - y.min()
    bounds = (
        [0.0,    y.min(), span * 0.02, xco2.min() - 5],   # lower: sigma floor
        [np.inf, y.max(), span * 0.30, xco2.max() + 5]    # upper: sigma cap
    )

    popt, pcov = curve_fit(
        gaussian_cross_section, y, xco2, p0=p0, bounds=bounds, maxfev=10000
    )
    perr = np.sqrt(np.diag(pcov))

    return {
        "amplitude":   popt[0], "amplitude_err":  perr[0],
        "center":      popt[1], "center_err":     perr[1],
        "sigma":       popt[2], "sigma_err":      perr[2],
        "background":  popt[3], "background_err": perr[3],
    }


# ----------------------------------------------------------------------
# 3. Convert the fitted plume + wind into an emission rate Q
# ----------------------------------------------------------------------
# Physical constants
M_CO2 = 44.01e-3          # kg/mol, molar mass of CO2
M_AIR = 28.97e-3          # kg/mol, molar mass of dry air
G     = 9.80665           # m/s^2
P0    = 101325.0          # Pa, standard surface pressure

def xco2_ppm_to_column_density(amplitude_ppm, sigma_m, surface_pressure_pa=P0):
    """
    Integrate the Gaussian enhancement across the plume to get the total
    excess CO2 *line density* (kg of CO2 per metre of along-wind length).

    XCO2 is a column-averaged DRY-AIR MOLE FRACTION. To turn a ppm enhancement
    into a mass column we use the total dry-air column:
        air column [mol/m^2] = surface_pressure / (g * M_AIR)
    Excess CO2 column [mol/m^2] = (amplitude_ppm * 1e-6) * air_column
    Integral of the Gaussian across y = amplitude * sigma * sqrt(2*pi).
    """
    air_col_mol_m2 = surface_pressure_pa / (G * M_AIR)          # mol(air)/m^2
    peak_excess_co2_mol_m2 = (amplitude_ppm * 1e-6) * air_col_mol_m2
    # Cross-wind integral of the bell curve:
    integrated_mol_m = peak_excess_co2_mol_m2 * sigma_m * np.sqrt(2 * np.pi)
    line_density_kg_m = integrated_mol_m * M_CO2               # kg(CO2)/m
    return line_density_kg_m


def estimate_emission_rate(fit_params, wind_speed_ms, surface_pressure_pa=P0):
    """
    Mass-balance: Q = (cross-wind integrated excess CO2 column) x wind speed.

    Returns Q in kg/s, kt/day, and Mt/yr.

    Intuition (the sink analogy): the plume is water in a sink. The bump is the
    water level (line density); the wind is how fast the drain pulls it away.
    Level x drain-speed = tap inflow = emission rate.
    """
    line_density_kg_m = xco2_ppm_to_column_density(
        fit_params["amplitude"], fit_params["sigma"], surface_pressure_pa
    )
    Q_kg_s = line_density_kg_m * wind_speed_ms       # (kg/m) * (m/s) = kg/s

    return {
        "Q_kg_s":   Q_kg_s,
        "Q_kt_day": Q_kg_s * 86400.0 / 1e6,          # kg/s -> kilotonnes/day
        "Q_Mt_yr":  Q_kg_s * 86400.0 * 365.0 / 1e9,  # kg/s -> megatonnes/year
    }


def invert_overpass(y, xco2, wind_speed_ms, surface_pressure_pa=P0, verbose=True):
    """
    End-to-end convenience wrapper: slice + wind  ->  emission rate Q.
    This is the function you call on each real OCO-2 overpass.
    """
    fit = fit_plume(y, xco2)
    Q = estimate_emission_rate(fit, wind_speed_ms, surface_pressure_pa)
    if verbose:
        print(f"  fitted amplitude  : {fit['amplitude']:.3f} +/- {fit['amplitude_err']:.3f} ppm")
        print(f"  fitted sigma      : {fit['sigma']/1000:.2f} km")
        print(f"  fitted background : {fit['background']:.2f} ppm")
        print(f"  wind speed        : {wind_speed_ms:.2f} m/s")
        print(f"  --> emission Q    : {Q['Q_kt_day']:.2f} kt/day   ({Q['Q_Mt_yr']:.2f} Mt/yr)")
    return {"fit": fit, "emission": Q}


# ----------------------------------------------------------------------
# 4. Self-test: make a synthetic plume with a KNOWN Q, then recover it
# ----------------------------------------------------------------------
def _self_test():
    """
    Forward-simulate a plume from a known emission rate, add noise, then run
    the inversion and check we recover Q. This validates the math end-to-end.
    """
    rng = np.random.default_rng(0)

    # --- Ground-truth scenario (a large, well-observed plant: Nassar's usable case) ---
    true_Q_Mt_yr = 30.0                      # a very large coal plant (e.g. ~30 Mt/yr)
    true_Q_kg_s  = true_Q_Mt_yr * 1e9 / (86400 * 365)
    wind = 3.0                               # m/s (lower wind -> taller, clearer bump)
    sigma_true = 4000.0                      # 4 km plume width
    background = 410.0                       # ppm ambient

    # Invert the physics to find what amplitude this Q implies:
    air_col = P0 / (G * M_AIR)
    line_density = true_Q_kg_s / wind                       # kg/m
    integrated_mol_m = line_density / M_CO2
    peak_mol_m2 = integrated_mol_m / (sigma_true * np.sqrt(2*np.pi))
    amplitude_true = peak_mol_m2 / air_col / 1e-6           # back to ppm

    # --- Build a noisy measured slice (what the satellite would see) ---
    y = np.linspace(-20000, 20000, 120)      # 120 fine soundings across +/-20 km
    clean = gaussian_cross_section(y, amplitude_true, 0.0, sigma_true, background)
    noise = rng.normal(0, 0.3, size=y.shape) # ~0.3 ppm retrieval noise
    measured = clean + noise

    print("=" * 60)
    print("SELF-TEST: recover a known emission rate from a noisy plume")
    print("=" * 60)
    print(f"  TRUE emission     : {true_Q_Mt_yr:.2f} Mt/yr")
    print(f"  (implied peak amp : {amplitude_true:.3f} ppm)")
    print("-" * 60)
    result = invert_overpass(y, measured, wind)
    recovered = result["emission"]["Q_Mt_yr"]
    err = abs(recovered - true_Q_Mt_yr) / true_Q_Mt_yr * 100
    print("-" * 60)
    print(f"  RECOVERED         : {recovered:.2f} Mt/yr")
    print(f"  RELATIVE ERROR    : {err:.1f} %   (Nassar reports 1-17%)")
    print("=" * 60)
    return err


if __name__ == "__main__":
    _self_test()

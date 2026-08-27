"""Streamlit demo: 30-plant Q-vs-CEA map + Week 13 Rihand-style diagnostics.

Run: streamlit run app.py  (project's conda env: /opt/miniconda3/envs/co2)
All numbers are read from data/*.json at runtime -- nothing here is hardcoded.
"""
import json

import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

from physics_gaussian import NEAR, BG_IN, BG_OUT

DATA = "data"
DEFAULT_HIGHLIGHTS = ["Vindhyachal", "Rihand"]

st.set_page_config(page_title="CO2 Q vs CEA", layout="wide")


def loo_capacity_only_estimate(cap, cea):
    """LOO log-space linear fit of CEA ~ capacity_mw, same method as
    baseline_capacity.py's predictor A (loo_linear_log_predictions)."""
    log_cap, log_cea = np.log(cap.to_numpy()), np.log(cea.to_numpy())
    n = len(log_cap)
    design = np.column_stack([np.ones(n), log_cap])
    preds = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        coef, *_ = np.linalg.lstsq(design[mask], log_cea[mask], rcond=None)
        preds[i] = design[i] @ coef
    return np.exp(preds)


@st.cache_data
def load_data():
    plants = pd.read_csv(f"{DATA}/candidate_plants.csv")

    q = json.load(open(f"{DATA}/q_correction_model_results.json"))
    ft = pd.DataFrame(q["feature_table"])[
        ["plant", "our_q", "cea_truth", "log_ratio", "hit_days"]
    ]

    dens = json.load(open(f"{DATA}/overpass_density_results.json"))
    coverage_threshold = dens["threshold_days"]

    snr = json.load(open(f"{DATA}/snr_all_plants.json"))
    snr_df = pd.DataFrame(snr["plants"])[["plant", "signal_to_noise"]]
    snr_median = snr["q2_rihand"]["median_snr_all_plants"]

    bg = json.load(open(f"{DATA}/bg_sensitivity_all_plants.json"))
    bg_df = pd.DataFrame(bg["plants"])[["plant", "ime_proxy_range_pct_of_default"]]
    bg_threshold = bg_df["ime_proxy_range_pct_of_default"].median()

    wind = json.load(open(f"{DATA}/wind_match_quality_all_plants.json"))
    wind_df = pd.DataFrame(wind["plants"])[["plant", "wind_match_rate"]]
    wind_threshold = wind_df["wind_match_rate"].quantile(0.25)

    baseline = json.load(open(f"{DATA}/baseline_capacity_results.json"))
    r2_by_label = {p["label"]: p["loo_r2"] for p in baseline["predictors"]}
    r2_capacity = next(v for k, v in r2_by_label.items() if k.startswith("A:"))
    r2_satellite = next(v for k, v in r2_by_label.items() if k.startswith("B:"))

    df = plants.merge(ft, left_on="name", right_on="plant", how="left")
    for other in (snr_df, bg_df, wind_df):
        df = df.merge(other, on="plant", how="left")

    has_estimate = df["log_ratio"].notna()
    df["capacity_only_est"] = np.nan
    df.loc[has_estimate, "capacity_only_est"] = loo_capacity_only_estimate(
        df.loc[has_estimate, "capacity_mw"], df.loc[has_estimate, "cea_truth"]
    )

    df["coverage_pass"] = df["hit_days"] >= coverage_threshold
    df["snr_pass"] = df["signal_to_noise"] >= snr_median
    df["bg_pass"] = df["ime_proxy_range_pct_of_default"] <= bg_threshold
    df["wind_pass"] = df["wind_match_rate"] > wind_threshold
    df["ratio"] = df["our_q"] / df["cea_truth"]
    df["pct_error"] = (df["ratio"] - 1) * 100
    df["abs_log_ratio"] = df["log_ratio"].abs()
    df["status"] = df["log_ratio"].notna().map({True: "has estimate", False: "no estimate"})

    thresholds = dict(
        coverage=coverage_threshold, snr=snr_median, bg=bg_threshold, wind=wind_threshold
    )
    baseline_r2 = dict(capacity=r2_capacity, satellite=r2_satellite)
    return df, thresholds, baseline_r2


@st.cache_data
def load_soundings(plant_name):
    """Per-plant OCO-3 lat/lon/xco2 point cloud, classified into the same
    near-plant / background zones physics_gaussian.py's IME calculation
    uses (NEAR, BG_IN, BG_OUT), so the map matches what the estimate does."""
    try:
        d = np.load(f"{DATA}/{plant_name}_soundings.npz")
    except FileNotFoundError:
        return None
    lat, lon = d["lat"], d["lon"]
    prow = df[df["name"] == plant_name].iloc[0]
    dist = np.sqrt((lat - prow["latitude"]) ** 2 + (lon - prow["longitude"]) ** 2)
    zone = np.where(dist < NEAR, "near-plant",
                     np.where((dist > BG_IN) & (dist < BG_OUT), "background", "other"))
    return pd.DataFrame({"lat": lat, "lon": lon, "xco2": d["xco2"], "zone": zone})


df, thresholds, baseline_r2 = load_data()

st.title("Satellite CO2 Q Estimate vs CEA Ground Truth")
st.caption(
    "30 candidate coal plants. Color = |log ratio| of our IME Q estimate vs CEA's "
    "FY2020-21 baseline. Diagnostics are the four Week 13 checks run against Rihand's "
    "unexplained +134% error (WEEK13_LOG.txt)."
)
st.markdown(
    f"For context: plant capacity alone predicts CEA emissions better than this "
    f"satellite estimate (LOO R^2 {baseline_r2['capacity']:.2f} vs {baseline_r2['satellite']:.2f}, "
    f"see baseline_capacity_results.json). This demo visualizes where and why the satellite "
    f"method succeeds or fails, not a competing prediction tool."
)

fig = px.scatter_geo(
    df,
    lat="latitude",
    lon="longitude",
    color="abs_log_ratio",
    color_continuous_scale="Reds",
    hover_name="name",
    custom_data=["name"],
    symbol=df["name"].isin(DEFAULT_HIGHLIGHTS).map({True: "star", False: "circle"}),
    size=df["name"].isin(DEFAULT_HIGHLIGHTS).map({True: 18, False: 9}),
    scope="asia",
    title="Click a plant marker to load its detail panel below",
)
fig.update_geos(lataxis_range=[8, 32], lonaxis_range=[68, 92], showcountries=True)
fig.update_layout(height=520, margin=dict(l=0, r=0, t=40, b=0))

event = st.plotly_chart(fig, width="stretch", on_select="rerun", key="map")

if "selected_plant" not in st.session_state:
    st.session_state.selected_plant = "Rihand"
points = event.selection.get("points", []) if event else []
if points:
    st.session_state.selected_plant = points[0]["customdata"][0]

st.subheader("Highlighted by default")
cols = st.columns(2)
for col, name in zip(cols, DEFAULT_HIGHLIGHTS):
    row = df[df["name"] == name].iloc[0]
    with col:
        st.metric(
            f"{name} ({'best result' if name == 'Vindhyachal' else 'unexplained outlier'})",
            f"{row['pct_error']:+.0f}% vs CEA",
        )

st.divider()
plant_options = sorted(df["name"])
selected = st.selectbox(
    "Or pick a plant directly",
    plant_options,
    index=plant_options.index(st.session_state.selected_plant),
)
st.session_state.selected_plant = selected
row = df[df["name"] == selected].iloc[0]

st.subheader(selected)
if pd.isna(row["log_ratio"]):
    st.warning("No Q estimate for this plant (insufficient OCO-3 coverage).")
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("Our Q estimate", f"{row['our_q']/1e6:,.1f} Mt/yr")
    c2.metric("CEA ground truth", f"{row['cea_truth']/1e6:,.1f} Mt/yr")
    c3.metric("Ratio (ours / CEA)", f"{row['ratio']:.2f}x", f"{row['pct_error']:+.0f}%")

    c4, c5, c6 = st.columns(3)
    c4.metric("Plant capacity", f"{row['capacity_mw']:,.0f} MW")
    cap_ratio = row["capacity_only_est"] / row["cea_truth"]
    c5.metric("Capacity-only estimate", f"{row['capacity_only_est']/1e6:,.1f} Mt/yr")
    c6.metric("Capacity-only ratio", f"{cap_ratio:.2f}x", f"{(cap_ratio - 1) * 100:+.0f}%")

st.markdown("**Week 13 diagnostics** (pass = scores at or better than the 24-plant median/threshold)")
diag = pd.DataFrame(
    [
        ("Overpass coverage", row["hit_days"], f">= {thresholds['coverage']} hit-days", row["coverage_pass"]),
        ("Signal-to-noise", row["signal_to_noise"], f">= median ({thresholds['snr']:.2f})", row["snr_pass"]),
        ("Background sensitivity", row["ime_proxy_range_pct_of_default"], f"<= median ({thresholds['bg']:.1f}%)", row["bg_pass"]),
        ("Wind-match quality", row["wind_match_rate"], f"> bottom quartile ({thresholds['wind']:.2f})", row["wind_pass"]),
    ],
    columns=["Diagnostic", "Value", "Pass threshold", "Pass"],
)
diag["Pass"] = diag["Pass"].map({True: "PASS", False: "FAIL"}).fillna("no data")
st.table(diag)

st.subheader("Sounding density")
soundings = load_soundings(selected)
if soundings is None:
    st.caption("No saved OCO-3 soundings file for this plant.")
else:
    st.caption(
        f"{len(soundings)} OCO-3 soundings. Near-plant zone (< {NEAR} deg) and background "
        f"annulus ({BG_IN}-{BG_OUT} deg) are the same zones physics_gaussian.py's IME "
        "calculation uses -- this is what the Q estimate is actually computed from."
    )
    layer_choice = st.radio("Layer", ["Heatmap", "Grid"], horizontal=True)
    if layer_choice == "Heatmap":
        density_layer = pdk.Layer(
            "HeatmapLayer", data=soundings, get_position=["lon", "lat"],
            get_weight="xco2", radius_pixels=40,
        )
    else:
        density_layer = pdk.Layer(
            "GridLayer", data=soundings, get_position=["lon", "lat"],
            cell_size=2000, extruded=False, get_color_weight="xco2",
            color_aggregation="MEAN", pickable=True,
        )
    plant_layer = pdk.Layer(
        "ScatterplotLayer",
        data=pd.DataFrame([{"lat": row["latitude"], "lon": row["longitude"]}]),
        get_position=["lon", "lat"], get_color=[0, 200, 255], get_radius=800,
    )
    view_state = pdk.ViewState(latitude=row["latitude"], longitude=row["longitude"], zoom=8.5, pitch=0)
    st.pydeck_chart(pdk.Deck(
        layers=[density_layer, plant_layer], initial_view_state=view_state,
        map_style=None, tooltip={"text": "XCO2: {xco2} ppm\nzone: {zone}"},
    ))

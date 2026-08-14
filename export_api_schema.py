"""
Phase 7 of NOVEL_METHODOLOGY_PROPOSAL.md: exports one record per facility,
matching data/schema/emission_record_schema.json, for downstream
integration with an external visualization platform. Assembles from
already-computed result files only -- nothing is recomputed here, and
nothing is invented for facilities lacking a given piece (fields are
null/available:false rather than guessed).

Includes a lightweight hand-rolled schema check (required top-level keys
present, const fields correct) run before saving, since jsonschema isn't
an existing dependency of this project and adding it for one validation
pass isn't worth a new requirements.txt entry -- catches the most likely
mistakes (a forgotten field, a validated:true slip) without a new
dependency.
"""
import datetime
import json
import os

SOURCE_FILES = [
    "data/plant_results.json", "data/emission_estimates.json",
    "data/cea_ground_truth_2020_21.json", "data/q_correction_model_results.json",
    "data/q_correction_model_strengthened_results.json", "data/activity_signals.json",
    "data/lofo_track_a_results.json", "data/climate_trace_comparison.json",
    "data/temporal_q_model_results.json", "data/plume_maps/prototype_summary.json",
    "data/gradcam_3channel_summary.json", "data/shap_correction_model_results.json",
]


def load_or_empty(path, default):
    return json.load(open(path)) if os.path.exists(path) else default


def build_records():
    plant_results = {r["plant"]: r for r in json.load(open("data/plant_results.json"))}
    emission_estimates = {r["plant"]: r for r in json.load(open("data/emission_estimates.json"))}
    cea = load_or_empty("data/cea_ground_truth_2020_21.json", {"facilities": {}})["facilities"]
    q_corr = load_or_empty("data/q_correction_model_results.json", {"feature_table": []})
    q_corr_ft = {r["plant"]: r for r in q_corr.get("feature_table", [])}
    q_corr_loo = q_corr.get("correction", {}).get("loo_predictions_log_ratio", {})
    q_corr_sig = load_or_empty("data/q_correction_model_strengthened_results.json",
                                {"bootstrap": {"distinguishable_from_noise": False}})
    activity = load_or_empty("data/activity_signals.json", [])
    activity = {r["plant"]: r for r in activity}
    lofo = load_or_empty("data/lofo_track_a_results.json", {"per_facility": []})
    lofo_by_plant = {r["plant"]: r for r in lofo.get("per_facility", [])}
    ct = load_or_empty("data/climate_trace_comparison.json", {"facilities": []})
    ct_by_plant = {f["plant"]: f for f in ct.get("facilities", []) if "our_q_t_per_year" in f}
    temporal = load_or_empty("data/temporal_q_model_results.json", {"results": {}})["results"]
    plume_summary = load_or_empty("data/plume_maps/prototype_summary.json", {"facilities": {}})["facilities"]
    gradcam = load_or_empty("data/gradcam_3channel_summary.json", {"facilities": {}})["facilities"]
    shap_result = load_or_empty("data/shap_correction_model_results.json", {"per_facility": {}})
    shap_per_facility = shap_result.get("per_facility", {})
    candidates_names = set(plant_results)

    import csv
    capacity = {}
    with open("data/candidate_plants.csv") as f:
        for row in csv.DictReader(f):
            capacity[row["name"]] = float(row["capacity_mw"])

    records = []
    for name in sorted(candidates_names):
        pr = plant_results[name]
        est = emission_estimates.get(name)
        cea_row = cea.get(name)
        act = activity.get(name)
        lofo_row = lofo_by_plant.get(name)
        ct_row = ct_by_plant.get(name)
        temp_row = temporal.get(name)
        plume_row = plume_summary.get(name)
        gradcam_row = gradcam.get(name)
        shap_row = shap_per_facility.get(name)

        act_prob_mean = act["activity_prob_mean"] if act else None
        track_a = {
            "activity_prob_mean": act_prob_mean,
            "activity_prob_std": act["activity_prob_std"] if act else None,
            "exhaustive_lofo_recall": lofo_row["recall"] if lofo_row else None,
            "validation_status": "exhaustive_lofo_tested" if lofo_row else "not_tested",
        }

        raw_estimate = None
        if est:
            raw_estimate = {
                "q_t_per_year": est["q_t_per_year"], "q_t_per_year_std": est["q_t_per_year_std"],
                "wind_mode": est["wind_mode"], "month_stratified": est.get("month_stratified", False),
                "hit_days": pr["hit_days"], "n_soundings": pr["soundings"],
            }

        corrected_q = None
        if name in q_corr_loo and est:
            import math
            corrected_q = est["q_t_per_year"] * math.exp(-q_corr_loo[name])
        # "matched" must mean a real, usable comparison exists (both a CEA
        # row AND a Track B raw estimate) -- not merely that the facility's
        # name happens to appear in the raw CEA download. Mundra is a real
        # example this caught: it's in the CEA database, but has no Track B
        # estimate at all (0 near-plant soundings), so there is nothing to
        # correct or compare, and reporting it as "matched" would be wrong.
        has_real_match = cea_row is not None and raw_estimate is not None
        ground_truth_correction = {
            "available": corrected_q is not None,
            "corrected_q_t_per_year": corrected_q,
            "correction_significant": q_corr_sig.get("bootstrap", {}).get("distinguishable_from_noise", False),
            "validation_status": "cea_ground_truth_matched" if has_real_match else "not_matched",
        }

        climate_trace_comparison = None
        if ct_row:
            climate_trace_comparison = {
                "climate_trace_co2_t": ct_row["climate_trace_co2_t"],
                "ratio_ours_over_ct": ct_row["ratio_ours_over_ct"],
                "bracketed_by_our_interval": ct_row["bracketed_by_our_interval"],
            }

        plume = {
            "available": plume_row is not None,
            "validated": False,
            "validation_summary": ("Physics-consistent visualization calibrated to a validated total "
                                    "mass flux. Two robustness checks (random-sector null baseline, "
                                    "per-overpass day-matched wind) found the predicted spatial "
                                    "orientation does not reliably align with real OCO-3 data for the "
                                    "large majority of facilities tested -- NOT a validated spatial "
                                    "prediction. See NOVEL_METHODOLOGY_PROPOSAL.md Sec 11-12."),
            "grid_reference": f"plume_maps/{name}_default.npz" if plume_row else None,
            "default_assumptions": {"stability_class": "B", "stack_height_m": 220.0} if plume_row else None,
        }

        temporal_out = None
        if temp_row and not temp_row.get("skipped"):
            temporal_out = {
                "available": True,
                "n_months_usable": temp_row["n_months_usable"],
                "months_with_data": sorted(int(m) for m in temp_row["monthly_series"]),
                "absolute_values_comparable_to_annual": False,
                "monsoon_sampling_gap_caveat": ("Months 7/8/10 have no usable data for ANY facility in "
                                                 "this dataset, likely an OCO-3 monsoon-cloud-cover "
                                                 "retrieval gap, not a genuine emissions pattern."),
            }

        explainability = {
            "grad_cam": {
                "available": gradcam_row is not None,
                "prob_plant": gradcam_row["prob_plant"] if gradcam_row else None,
                "dominant_channel": gradcam_row["dominant_channel"] if gradcam_row else None,
                "channel_shares": gradcam_row["channel_shares"] if gradcam_row else None,
                "image_reference": "gradcam_3channel.png" if gradcam_row else None,
            },
            "shap": {
                "available": shap_row is not None,
                "top_features": shap_row["top_3_features"] if shap_row else None,
            },
        }

        records.append({
            "facility_id": name, "name": name, "lat": pr["lat"], "lon": pr["lon"],
            "capacity_mw": capacity.get(name),
            "track_a": track_a,
            "track_b": {"raw_estimate": raw_estimate, "ground_truth_correction": ground_truth_correction,
                        "climate_trace_comparison": climate_trace_comparison},
            "plume": plume,
            "temporal": temporal_out,
            "explainability": explainability,
            "provenance": {"exported_at": datetime.datetime.utcnow().isoformat() + "Z",
                           "source_files": SOURCE_FILES, "methodology_reference": "NOVEL_METHODOLOGY_PROPOSAL.md"},
        })
    return records


def validate_record(r, errors, path=""):
    required_top = ["facility_id", "name", "lat", "lon", "track_a", "track_b", "plume", "provenance"]
    for k in required_top:
        if k not in r:
            errors.append(f"{path}: missing required field '{k}'")
    if r.get("plume", {}).get("validated") is not False:
        errors.append(f"{path}: plume.validated must be exactly false")
    if r.get("provenance", {}).get("methodology_reference") != "NOVEL_METHODOLOGY_PROPOSAL.md":
        errors.append(f"{path}: provenance.methodology_reference mismatch")


def main():
    records = build_records()
    errors = []
    for r in records:
        validate_record(r, errors, path=r["name"])
    if errors:
        print(f"[!] {len(errors)} schema validation errors:")
        for e in errors:
            print(f"  {e}")
        raise SystemExit(1)

    print(f"All {len(records)} records pass validation.")
    out = {"schema_reference": "data/schema/emission_record_schema.json",
           "n_facilities": len(records), "facilities": records}
    json.dump(out, open("data/api_export/facilities.json", "w"), indent=2)
    print("[SAVED] data/api_export/facilities.json")


if __name__ == "__main__":
    main()

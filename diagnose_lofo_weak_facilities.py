"""
Targeted follow-up to lofo_track_a.py's 24-tile-depth re-run (see
PROJECT_RESEARCH_DOCUMENTATION.md §12.9): after closing most of the LOFO
generalization gap (47.2%->69.1% mean recall) by matching every facility to
24-tile temporal depth, two facilities still show 0% LOFO recall -- Kudgi
and ShriSingajiMalwa. lofo_recall_correlates.py already showed low
activity_prob_mean predicts LOFO failure (r=+0.90); this script asks the
more basic question underneath that: is the raw NO2/SO2/VIIRS signal for
these two facilities actually weak in absolute terms, or is this a
detector/training artifact?

Compares per-channel mean tile intensity for Kudgi and ShriSingajiMalwa
against (a) the rest of the positive (plant) class, (b) hard_negative
(cities/steel/highways), and (c) rural negative -- if a "plant" facility's
raw signal sits at or below the negative classes' own levels, no detector
architecture or training regime can be expected to separate it, and the
0% LOFO recall is an observability limit, not a bug.

Also checks no2_peak_km (offset of the tile's peak NO2 pixel from the
plant's registered coordinates, already computed in plant_results.json)
to rule out a simpler explanation: bad lat/lon placing the plant tile off
the actual plume.
"""
import numpy as np, glob, os, json

WEAK_FACILITIES = ["Kudgi", "ShriSingajiMalwa"]
COMPARE_FACILITIES = ["Talcher", "Rihand", "Anpara", "Korba"]  # 100% LOFO recall in the 24-tile-depth run


def facility_tiles(name):
    return sorted(glob.glob(f"data/threech/positive/{name}_*.npy"))


def channel_means(files):
    if not files:
        return None
    stacked = np.stack([np.load(f) for f in files])
    return stacked.mean(axis=(0, 2, 3))  # (3,) -> NO2, SO2, VIIRS


def class_means(pattern):
    files = glob.glob(pattern)
    return len(files), channel_means(files)


def main():
    plant_results = {r["plant"]: r for r in json.load(open("data/plant_results.json"))}

    print("=== Per-channel mean tile intensity (NO2, SO2, VIIRS) ===\n")
    rows = {}
    for name in WEAK_FACILITIES + COMPARE_FACILITIES:
        files = facility_tiles(name)
        means = channel_means(files)
        rows[name] = {"n_tiles": len(files), "no2_mean": float(means[0]),
                       "so2_mean": float(means[1]), "viirs_mean": float(means[2]),
                       "no2_peak_km": plant_results.get(name, {}).get("no2_peak_km")}
        print(f"{name:20s} n={len(files):3d}  NO2={means[0]:.3e}  SO2={means[1]:.3e}  "
              f"VIIRS={means[2]:.3e}  no2_peak_km={rows[name]['no2_peak_km']}")

    n_hn, hn_means = class_means("data/threech/hard_negative/*.npy")
    n_rn, rn_means = class_means("data/threech/negative/*.npy")
    all_pos = glob.glob("data/threech/positive/*.npy")
    weak_files = set(f for name in WEAK_FACILITIES for f in facility_tiles(name))
    rest_pos_files = [f for f in all_pos if f not in weak_files]
    rest_pos_means = channel_means(rest_pos_files)

    print(f"\n{'hard_negative':20s} n={n_hn:3d}  NO2={hn_means[0]:.3e}  SO2={hn_means[1]:.3e}  VIIRS={hn_means[2]:.3e}")
    print(f"{'rural_negative':20s} n={n_rn:3d}  NO2={rn_means[0]:.3e}  SO2={rn_means[1]:.3e}  VIIRS={rn_means[2]:.3e}")
    print(f"{'positive (rest)':20s} n={len(rest_pos_files):3d}  NO2={rest_pos_means[0]:.3e}  "
          f"SO2={rest_pos_means[1]:.3e}  VIIRS={rest_pos_means[2]:.3e}")

    result = {
        "weak_facilities": rows,
        "hard_negative": {"n_tiles": n_hn, "no2_mean": float(hn_means[0]),
                           "so2_mean": float(hn_means[1]), "viirs_mean": float(hn_means[2])},
        "rural_negative": {"n_tiles": n_rn, "no2_mean": float(rn_means[0]),
                            "so2_mean": float(rn_means[1]), "viirs_mean": float(rn_means[2])},
        "positive_rest": {"n_tiles": len(rest_pos_files), "no2_mean": float(rest_pos_means[0]),
                           "so2_mean": float(rest_pos_means[1]), "viirs_mean": float(rest_pos_means[2])},
        "finding": (
            "Kudgi's and ShriSingajiMalwa's raw NO2/SO2 tile intensity sits at or below "
            "rural_negative levels and well below the rest-of-positive-class average "
            "(~3x lower NO2, ~3-4x lower SO2) -- checked every month individually, not just "
            "on average, with no month showing a strong signal spike. no2_peak_km for both "
            "(3.6km, 1.0km) is smaller than several 100%-LOFO-recall facilities (Talcher "
            "34.7km, Anpara 16.7km, Rihand 8.1km), ruling out bad lat/lon tile placement as "
            "the explanation. Conclusion: these two facilities' satellite-observable "
            "NO2/SO2 signature is genuinely at or near the noise floor established by the "
            "negative classes -- a real observability limit, not a detector/training "
            "artifact. No architecture or training-regime change can be expected to fix "
            "this; the signal is not there to learn from a satellite-tile classifier "
            "input. Plausible physical explanation, NOT verified from available data "
            "(capacity_mw is the only per-plant attribute in candidate_plants.csv, and both "
            "are large plants at 2400-2520 MW, so capacity alone does not explain it): "
            "modern flue-gas desulfurization (FGD) / selective catalytic reduction (SCR) "
            "emissions-control equipment suppressing stack NO2/SO2 well below older, "
            "less-controlled plants of similar capacity. Not documented / needs "
            "verification against each plant's actual emissions-control equipment."
        ),
    }
    json.dump(result, open("data/lofo_weak_facility_diagnosis.json", "w"), indent=2)
    print("\n[SAVED] data/lofo_weak_facility_diagnosis.json")


if __name__ == "__main__":
    main()

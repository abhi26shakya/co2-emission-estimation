"""
Week 13, experiment 2: is the CO2 signal above background noise, for all
30 plants -- not just Talcher?

diagnose_talcher.py computed signal-to-noise (near-plant mean minus
background mean, divided by background std) for exactly two plants:
Talcher (SNR=0.18, a known-bad case) and Rihand (SNR=1.27, used there as
the "robust" comparison plant). This script reuses that same
near_bg_stats() function -- unchanged, not reimplemented -- across all 30
plants, then asks:

1. Does SNR predict |log_ratio| (our Q vs CEA truth, log scale) better
   than hit_days did? Week 12's feature table gives r(hit_days,
   |log_ratio|) = -0.31.
2. Rihand had good overpass coverage (16 days) but a bad Q estimate
   (Week 13 experiment 1: +134% vs CEA even at full coverage). Does
   Rihand actually have a low SNR despite its day count -- something
   experiment 1's day-count-only lens couldn't see?
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import physics_ime as pg
from diagnose_talcher import near_bg_stats

# vindhyachal_soundings.npz is lowercase, unlike every other plant's file
pg.NPZ_PATHS["Vindhyachal"] = "data/vindhyachal_soundings.npz"


def plant_snr_row(name, plant_row):
    npz_path = pg.NPZ_PATHS.get(name, f"data/{name}_soundings.npz")
    d = np.load(npz_path)
    n_soundings = int(len(d["xco2"]))
    hit_days = int(len(np.unique(d["day"]))) if "day" in d.files else None

    stats = near_bg_stats(name, plant_row)
    if stats is None:
        print(f"[{name}] too few near/bg soundings, skipping")
        return None

    row = {
        "plant": name,
        "n_soundings": n_soundings,
        "hit_days": hit_days,
        "near_n": stats["near_n"],
        "bg_n": stats["bg_n"],
        "signal_to_noise": stats["signal_to_noise"],
        "frac_near_above_bg": stats["frac_near_above_bg"],
    }
    print(f"[{name}] SNR={row['signal_to_noise']:.3f}  "
          f"frac_above_bg={row['frac_near_above_bg']:.1%}  "
          f"hit_days={hit_days}  n_soundings={n_soundings}")
    return row


def main():
    plant_rows = {r["plant"]: r for r in json.load(open("data/plant_results.json"))}
    q_correction = json.load(open("data/q_correction_model_results.json"))
    feature_table = {r["plant"]: r for r in q_correction["feature_table"]}

    rows = []
    for name in sorted(plant_rows):
        row = plant_snr_row(name, plant_rows[name])
        if row is not None:
            rows.append(row)

    # --- Q1: SNR vs hit_days as a predictor of |log_ratio|, on the N=24
    # plants that actually have a log_ratio (both Q estimate and CEA truth) ---
    matched = [(r, feature_table[r["plant"]]) for r in rows if r["plant"] in feature_table]
    snr_vals = np.array([r["signal_to_noise"] for r, _ in matched])
    hit_days_vals = np.array([r["hit_days"] for r, _ in matched], dtype=float)
    abs_log_ratio = np.array([abs(f["log_ratio"]) for _, f in matched])

    r_snr = float(np.corrcoef(snr_vals, abs_log_ratio)[0, 1])
    r_hit_days = float(np.corrcoef(hit_days_vals, abs_log_ratio)[0, 1])

    print(f"\n=== Q1: SNR vs hit_days as predictors of |log_ratio| (N={len(matched)}) ===")
    print(f"  r(SNR, |log_ratio|)       = {r_snr:+.3f}")
    print(f"  r(hit_days, |log_ratio|)  = {r_hit_days:+.3f}  (Week 12 feature table)")
    better = "SNR" if abs(r_snr) > abs(r_hit_days) else "hit_days"
    print(f"  -> {better} is the stronger (|r| larger) predictor")

    # --- Q2: Rihand's SNR despite good coverage ---
    rihand = next(r for r in rows if r["plant"] == "Rihand")
    talcher = next((r for r in rows if r["plant"] == "Talcher"), None)
    median_snr = float(np.median([r["signal_to_noise"] for r in rows]))
    print(f"\n=== Q2: Rihand SNR despite good coverage (hit_days={rihand['hit_days']}) ===")
    print(f"  Rihand SNR = {rihand['signal_to_noise']:.3f}  "
          f"(median across all plants = {median_snr:.3f}"
          + (f", Talcher = {talcher['signal_to_noise']:.3f})" if talcher else ")"))
    rihand_low_snr = rihand["signal_to_noise"] < median_snr
    print(f"  Rihand SNR {'IS' if rihand_low_snr else 'is NOT'} below the plant-set median")

    out = {
        "n_plants": len(rows),
        "plants": sorted(rows, key=lambda r: r["signal_to_noise"]),
        "q1_snr_vs_hit_days": {
            "n_matched": len(matched),
            "r_snr_abs_log_ratio": r_snr,
            "r_hit_days_abs_log_ratio": r_hit_days,
            "stronger_predictor": better,
        },
        "q2_rihand": {
            "rihand_snr": rihand["signal_to_noise"],
            "rihand_hit_days": rihand["hit_days"],
            "median_snr_all_plants": median_snr,
            "talcher_snr": talcher["signal_to_noise"] if talcher else None,
            "rihand_below_median": rihand_low_snr,
        },
    }
    out_path = "data/snr_all_plants.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\n[SAVED] {out_path}")

    # --- plot ---
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(snr_vals, abs_log_ratio, color="tab:blue")
    for r, f in matched:
        label = r["plant"] if r["plant"] in ("Rihand", "Talcher") else None
        if label:
            ax.annotate(label, (r["signal_to_noise"], abs(f["log_ratio"])),
                        textcoords="offset points", xytext=(5, 5), fontsize=9, color="tab:red")
            ax.scatter([r["signal_to_noise"]], [abs(f["log_ratio"])], color="tab:red", zorder=5)
    ax.set_xlabel("signal-to-noise ratio")
    ax.set_ylabel("|log_ratio| (our Q vs CEA, log scale)")
    ax.set_title(f"SNR vs |log_ratio|  (r={r_snr:+.2f}, N={len(matched)})")
    fig.tight_layout()
    plot_path = "data/snr_vs_log_ratio_plot.png"
    fig.savefig(plot_path, dpi=150)
    print(f"[SAVED] {plot_path}")


if __name__ == "__main__":
    main()

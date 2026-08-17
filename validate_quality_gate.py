"""
Week 12: is the "hit_days >= 10 AND wind_co2_diff_deg <= 60" quality gate a real
finding, or selection bias from fitting the threshold to the same 24 points we
then judged it on?

Three tests, all against data/q_correction_model_results.json's N=24
feature_table:

1. Leave-one-facility-out (LOO): for each facility, hold it out, pick the best
   (hit_days_min, wind_diff_max) threshold pair using ONLY the other 23
   (minimising sd(log_ratio) among passing plants, subject to a minimum gate
   size), then apply that threshold to the held-out plant. This tells us
   whether a gate selected without seeing a plant still predicts that plant
   will be easier.

2. Permutation test (the decisive one): shuffle log_ratio labels across
   facilities 1000 times, keeping hit_days/wind_diff fixed, and re-run the
   SAME best-gate search on the shuffled data. If shuffled (label-scrambled)
   data produces gates just as good as the real one, the search procedure is
   finding structure in the threshold grid, not in the data -- i.e. selection
   bias / multiple-comparisons artifact.

3. Threshold stability across the 24 LOO folds: if the selected threshold
   jumps around, the gate is not a stable feature of the data.

Writes data/quality_gate_validation.json. Seed 42.
"""
import json
import math
from collections import Counter

import numpy as np

SEED = 42
DATA_PATH = "data/q_correction_model_results.json"
OUTPUT_PATH = "data/quality_gate_validation.json"

MIN_GATE_N = 5  # minimum plants a candidate gate must keep to be considered.
# Chosen to roughly match the scale of the published N=7 gate (and because
# sd() needs n>=2 to be defined at all); this is the one judgment call in
# this script.
PUBLISHED_GATE = (10, 60)  # (hit_days_min, wind_co2_diff_deg_max)
N_PERMUTATIONS = 1000


def load_feature_table():
    with open(DATA_PATH) as f:
        d = json.load(f)
    return d["feature_table"]


def grid_search_best_gate(rows, min_n=MIN_GATE_N):
    """Search (hit_days_min, wind_diff_max) pairs drawn from the values present
    in `rows`, keep only pairs whose gate passes >= min_n plants, and return
    the one with lowest sd(log_ratio) among passing plants.

    Tie-break, in order: lowest sd, largest passing count, smallest hit_days
    threshold, smallest wind_diff threshold. This makes the search
    deterministic when multiple thresholds tie exactly.

    Returns None if no threshold pair meets min_n (only possible if
    len(rows) < min_n).
    """
    hit_days_candidates = sorted({r["hit_days"] for r in rows})
    wind_candidates = sorted({r["wind_co2_diff_deg"] for r in rows})

    best = None
    for h in hit_days_candidates:
        for w in wind_candidates:
            passing = [r for r in rows if r["hit_days"] >= h and r["wind_co2_diff_deg"] <= w]
            n = len(passing)
            if n < min_n:
                continue
            log_ratios = np.array([r["log_ratio"] for r in passing])
            sd = float(np.std(log_ratios, ddof=1))
            candidate = (sd, -n, h, w, passing)
            if best is None or candidate[:4] < best[:4]:
                best = candidate

    if best is None:
        return None
    sd, neg_n, h, w, passing = best
    return {
        "hit_days_min": h,
        "wind_diff_max": w,
        "sd_log_ratio": sd,
        "n": -neg_n,
        "plants": [r["plant"] for r in passing],
    }


def group_stats(log_ratios):
    """bias / sd / within-2x for a list of log_ratio values."""
    n = len(log_ratios)
    if n == 0:
        return {"n": 0, "bias": None, "sd_log_ratio": None, "within_2x": 0, "within_2x_frac": None}
    arr = np.array(log_ratios)
    bias = float(math.exp(np.mean(arr)))
    sd = float(np.std(arr, ddof=1)) if n >= 2 else None
    within_2x = int(np.sum(np.abs(arr) <= math.log(2)))
    return {"n": n, "bias": bias, "sd_log_ratio": sd, "within_2x": within_2x,
            "within_2x_frac": within_2x / n}


def loo_validation(rows, min_n=MIN_GATE_N):
    per_fold = []
    for i in range(len(rows)):
        held_out = rows[i]
        train = rows[:i] + rows[i + 1:]
        gate = grid_search_best_gate(train, min_n)
        held_pass = (held_out["hit_days"] >= gate["hit_days_min"]
                     and held_out["wind_co2_diff_deg"] <= gate["wind_diff_max"])
        per_fold.append({
            "held_out_plant": held_out["plant"],
            "selected_hit_days_min": gate["hit_days_min"],
            "selected_wind_diff_max": gate["wind_diff_max"],
            "training_gate_sd": gate["sd_log_ratio"],
            "training_gate_n": gate["n"],
            "held_out_passed_gate": held_pass,
            "held_out_log_ratio": held_out["log_ratio"],
            "held_out_abs_log_ratio": abs(held_out["log_ratio"]),
        })

    gated_log_ratios = [f["held_out_log_ratio"] for f in per_fold if f["held_out_passed_gate"]]
    ungated_log_ratios = [f["held_out_log_ratio"] for f in per_fold if not f["held_out_passed_gate"]]

    gated_stats = group_stats(gated_log_ratios)
    ungated_stats = group_stats(ungated_log_ratios)

    ungated_avg_abs_log_ratio = (
        float(np.mean(np.abs(ungated_log_ratios))) if ungated_log_ratios else None
    )
    for f in per_fold:
        f["lower_error_than_ungated_avg"] = (
            f["held_out_abs_log_ratio"] < ungated_avg_abs_log_ratio
            if ungated_avg_abs_log_ratio is not None else None
        )

    return {
        "per_fold": per_fold,
        "gated_stats": gated_stats,
        "ungated_stats": ungated_stats,
        "ungated_avg_abs_log_ratio": ungated_avg_abs_log_ratio,
    }


def threshold_stability(per_fold, published_gate=PUBLISHED_GATE):
    pairs = [(f["selected_hit_days_min"], f["selected_wind_diff_max"]) for f in per_fold]
    exact_matches = sum(1 for p in pairs if p == published_gate)
    distribution = Counter(f"{h},{w}" for h, w in pairs)
    return {
        "published_gate": {"hit_days_min": published_gate[0], "wind_diff_max": published_gate[1]},
        "n_folds": len(pairs),
        "exact_matches_to_published_gate": exact_matches,
        "exact_match_frac": exact_matches / len(pairs),
        "distribution": dict(distribution),
    }


def permutation_test(rows, min_n=MIN_GATE_N, n_perm=N_PERMUTATIONS, seed=SEED):
    real_best = grid_search_best_gate(rows, min_n)
    real_best_sd = real_best["sd_log_ratio"]

    rng = np.random.default_rng(seed)
    log_ratios = np.array([r["log_ratio"] for r in rows])

    shuffled_sds = []
    for _ in range(n_perm):
        shuffled = rng.permutation(log_ratios)
        shuffled_rows = [dict(r, log_ratio=float(shuffled[j])) for j, r in enumerate(rows)]
        best = grid_search_best_gate(shuffled_rows, min_n)
        if best is not None:
            shuffled_sds.append(best["sd_log_ratio"])

    shuffled_sds = np.array(shuffled_sds)
    as_good_or_better = int(np.sum(shuffled_sds <= real_best_sd))
    p_value = (as_good_or_better + 1) / (n_perm + 1)

    return {
        "n_permutations": n_perm,
        "seed": seed,
        "real_grid_search_optimal_gate": real_best,
        "shuffled_best_sd_mean": float(np.mean(shuffled_sds)),
        "shuffled_best_sd_min": float(np.min(shuffled_sds)),
        "shuffled_best_sd_max": float(np.max(shuffled_sds)),
        "shuffles_as_good_or_better_than_real": as_good_or_better,
        "p_value": p_value,
    }


def published_gate_stats(rows, gate=PUBLISHED_GATE):
    h, w = gate
    passing = [r for r in rows if r["hit_days"] >= h and r["wind_co2_diff_deg"] <= w]
    stats = group_stats([r["log_ratio"] for r in passing])
    stats["hit_days_min"] = h
    stats["wind_diff_max"] = w
    stats["plants"] = [r["plant"] for r in passing]
    return stats


def make_verdict(loo_result, perm_result, stability_result):
    p_value = perm_result["p_value"]
    gated = loo_result["gated_stats"]
    ungated = loo_result["ungated_stats"]

    permutation_supports_gate = p_value < 0.05
    loo_shows_real_improvement = (
        gated["n"] > 0 and ungated["n"] > 0
        and gated["sd_log_ratio"] is not None and ungated["sd_log_ratio"] is not None
        and gated["sd_log_ratio"] < ungated["sd_log_ratio"]
        and (gated["within_2x_frac"] or 0) >= (ungated["within_2x_frac"] or 0)
    )

    holds = permutation_supports_gate and loo_shows_real_improvement

    gated_sd_str = f"{gated['sd_log_ratio']:.3f}" if gated["sd_log_ratio"] is not None else "n/a"
    ungated_sd_str = f"{ungated['sd_log_ratio']:.3f}" if ungated["sd_log_ratio"] is not None else "n/a"
    gated_2x_str = f"{gated['within_2x']}/{gated['n']}" if gated["n"] else "n/a"
    ungated_2x_str = f"{ungated['within_2x']}/{ungated['n']}" if ungated["n"] else "n/a"

    reasoning = (
        f"Permutation test: p = {p_value:.3f} "
        f"({perm_result['shuffles_as_good_or_better_than_real']}/{perm_result['n_permutations']} "
        "shuffles matched or beat the real grid-search-optimal gate's sd of "
        f"{perm_result['real_grid_search_optimal_gate']['sd_log_ratio']:.3f}). "
        + ("This is below 0.05, so the gate is unlikely to be a pure multiple-comparisons artifact. "
           if permutation_supports_gate else
           "This is NOT below 0.05 -- random label-scrambled data finds an equally good gate "
           "about as often as the real data does, which is the signature of selection bias. ")
        + f"LOO out-of-fold: gated held-out plants (n={gated['n']}) sd(log_ratio)={gated_sd_str}, "
        f"within-2x {gated_2x_str}; ungated held-out plants (n={ungated['n']}) "
        f"sd(log_ratio)={ungated_sd_str}, within-2x {ungated_2x_str}. "
        + ("Gated plants show real out-of-fold improvement over ungated. "
           if loo_shows_real_improvement else
           "Gated plants do NOT show consistent out-of-fold improvement over ungated. ")
        + f"Threshold stability: the LOO-selected gate matched the published (10, 60) gate exactly in "
        f"{stability_result['exact_matches_to_published_gate']}/{stability_result['n_folds']} folds "
        f"({stability_result['exact_match_frac']:.0%})."
    )

    label = "GATE HOLDS" if holds else "GATE IS SELECTION BIAS"
    return {"label": label, "reasoning": reasoning}


def main():
    rows = load_feature_table()

    loo_result = loo_validation(rows)
    stability_result = threshold_stability(loo_result["per_fold"])
    perm_result = permutation_test(rows)
    published = published_gate_stats(rows)
    verdict = make_verdict(loo_result, perm_result, stability_result)

    output = {
        "seed": SEED,
        "n_facilities": len(rows),
        "min_gate_n": MIN_GATE_N,
        "published_gate_stats": published,
        "loo": loo_result,
        "threshold_stability": stability_result,
        "permutation_test": perm_result,
        "verdict": verdict["label"],
        "verdict_reasoning": verdict["reasoning"],
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print("=" * 70)
    print("Week 12: quality gate validation (hit_days>=10, wind_diff<=60)")
    print("=" * 70)
    print(f"Published gate on full N=24: n={published['n']}, bias={published['bias']:.3f}, "
          f"sd(log_ratio)={published['sd_log_ratio']:.3f}, "
          f"within_2x={published['within_2x']}/{published['n']}")
    print()
    print("--- LOO per-fold detail (24 folds) ---")
    ungated_avg = loo_result["ungated_avg_abs_log_ratio"]
    if ungated_avg is not None:
        print(f"(ungated avg |log_ratio| = {ungated_avg:.3f})")
    print(f"{'Plant':<20}{'Gate(h,w)':<14}{'Passed':<9}{'|log_ratio|':<14}{'< ungated avg?'}")
    for f in loo_result["per_fold"]:
        gate_str = f"({f['selected_hit_days_min']},{f['selected_wind_diff_max']})"
        print(f"{f['held_out_plant']:<20}{gate_str:<14}{str(f['held_out_passed_gate']):<9}"
              f"{f['held_out_abs_log_ratio']:<14.3f}{f['lower_error_than_ungated_avg']}")
    print()
    print("--- LOO aggregate ---")
    print(f"Gated (n={loo_result['gated_stats']['n']}): "
          f"bias={loo_result['gated_stats']['bias']}, "
          f"sd(log_ratio)={loo_result['gated_stats']['sd_log_ratio']}, "
          f"within_2x={loo_result['gated_stats']['within_2x']}/{loo_result['gated_stats']['n']}")
    print(f"Ungated (n={loo_result['ungated_stats']['n']}): "
          f"bias={loo_result['ungated_stats']['bias']}, "
          f"sd(log_ratio)={loo_result['ungated_stats']['sd_log_ratio']}, "
          f"within_2x={loo_result['ungated_stats']['within_2x']}/{loo_result['ungated_stats']['n']}")
    print()
    print("--- Threshold stability ---")
    print(f"Exact match to published (10,60): "
          f"{stability_result['exact_matches_to_published_gate']}/{stability_result['n_folds']}")
    print(f"Distribution: {stability_result['distribution']}")
    print()
    print("--- Permutation test ---")
    print(f"Real grid-search-optimal gate: {perm_result['real_grid_search_optimal_gate']}")
    print(f"Shuffled best-sd: mean={perm_result['shuffled_best_sd_mean']:.3f}, "
          f"min={perm_result['shuffled_best_sd_min']:.3f}, max={perm_result['shuffled_best_sd_max']:.3f}")
    print(f"p-value: {perm_result['p_value']:.4f}")
    print()
    print(f"VERDICT: {verdict['label']}")
    print(verdict["reasoning"])
    print()
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

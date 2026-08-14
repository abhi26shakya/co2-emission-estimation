"""
First genuine Track A/Track B fusion step, scoped to respect the same
constraint reliability_model.py already established: Climate TRACE is an
independent benchmark, never a training label (RESEARCH_PLAN.md Sec 7), and
no other per-facility ground-truth emissions source exists for Indian coal
plants. A Q-correcting regression (RESEARCH_PLAN.md Sec 9's A3: "does the
activity signal correct physics' residual?") is therefore not buildable in
the literal sense -- there is no legitimate residual to fit against.

This script instead builds what IS buildable without a banned label: a
combined Track A + Track B *reliability/trust* score per facility, then
checks (evaluates, does not fit) whether that combined score is a better
predictor of trustworthiness than either track's signal alone, using
Climate TRACE's bracketed_by_our_interval flag purely as an outcome to
correlate against -- never as something coefficients are fit to minimize
error against.

Two questions:
1. Does Track A's activity_prob_mean add anything to Track B's own best
   single-feature uncertainty predictor (hit_days, r=-0.617, LOO R^2=0.212
   per reliability_model.py) in a 2-feature LOO-CV fit? Still deliberately
   linear/interpretable at N=17, per the same underpowered-multi-feature-
   model caution reliability_model.py and RESEARCH_PLAN.md Sec 8 state.
2. Does a composite fusion trust score (low predicted q_rel_std + high
   activity confidence = high trust) actually track which facilities get
   bracketed by their own uncertainty interval when checked against
   Climate TRACE? This is evaluation-only: the fusion score is built from
   q_rel_std/activity_prob_mean, never touches Climate TRACE data, and is
   then simply compared against the (separately computed) bracketing
   outcome.
"""
import json
import numpy as np


def loo_multifeature_fit(X, y):
    """Leave-one-out CV for an ordinary least-squares fit with an
    intercept, generalizing reliability_model.py's single-feature
    np.polyfit LOO loop to arbitrary feature count via np.linalg.lstsq."""
    n = len(y)
    Xb = np.column_stack([X, np.ones(n)])
    preds = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        coef, *_ = np.linalg.lstsq(Xb[mask], y[mask], rcond=None)
        preds[i] = Xb[i] @ coef
    mae = float(np.mean(np.abs(preds - y)))
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return preds, mae, r2


def main():
    rel = json.load(open("data/reliability_model_results.json"))
    rows = rel["feature_table"]
    n = len(rows)
    print(f"Loaded reliability_model.py's N={n} feature table "
          f"(best single feature: {rel['best_single_feature']}, "
          f"LOO R^2={rel['loo_cv']['r2']:.3f})\n")

    plants = [r["plant"] for r in rows]
    y = np.array([r["q_rel_std"] for r in rows])
    hit_days = np.array([r["hit_days"] for r in rows], dtype=np.float64)
    activity = np.array([r["activity_prob_mean"] for r in rows], dtype=np.float64)

    print("=== Question 1: does activity_prob_mean add to hit_days-only LOO fit? ===\n")
    _, mae1, r2_1 = loo_multifeature_fit(hit_days.reshape(-1, 1), y)
    print(f"  hit_days only            LOO MAE={mae1:.3f}  LOO R^2={r2_1:.3f}  "
          f"(matches reliability_model.py: R^2={rel['loo_cv']['r2']:.3f})")

    X2 = np.column_stack([hit_days, activity])
    preds2, mae2, r2_2 = loo_multifeature_fit(X2, y)
    print(f"  hit_days + activity_mean LOO MAE={mae2:.3f}  LOO R^2={r2_2:.3f}")
    improved = r2_2 > r2_1
    print(f"\n  {'Improvement' if improved else 'No improvement'}: "
          f"R^2 {'rose' if improved else 'did not rise'} from {r2_1:.3f} to {r2_2:.3f} "
          f"by adding Track A's activity signal.")

    print("\n=== Question 2: does a fused trust score track Climate TRACE bracketing? ===\n")
    # Normalize each signal to [0,1] across the N facilities, min-max.
    def norm(x, invert=False):
        x = np.asarray(x, dtype=np.float64)
        lo, hi = x.min(), x.max()
        n = (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)
        return 1 - n if invert else n

    trust_from_q = norm(y, invert=True)          # low q_rel_std -> high trust
    trust_from_activity = norm(activity)          # high activity_prob_mean -> high trust
    fusion_trust = (trust_from_q + trust_from_activity) / 2

    ct = json.load(open("data/climate_trace_comparison.json"))
    bracketed = {f["plant"]: f["bracketed_by_our_interval"]
                 for f in ct["facilities"] if "bracketed_by_our_interval" in f}

    matched = [(p, t, bracketed[p]) for p, t in zip(plants, fusion_trust) if p in bracketed]
    print(f"  {len(matched)}/{n} facilities have both a fusion trust score and a "
          f"Climate TRACE bracketing outcome (evaluation-only join, CT never seen by the "
          f"fit above)\n")
    for p, t, b in sorted(matched, key=lambda r: -r[1]):
        print(f"  {p:18s} fusion_trust={t:.3f}  bracketed_by_own_interval={b}")

    bracketed_scores = [t for _, t, b in matched if b]
    missed_scores = [t for _, t, b in matched if not b]
    mean_bracketed = float(np.mean(bracketed_scores)) if bracketed_scores else float("nan")
    mean_missed = float(np.mean(missed_scores)) if missed_scores else float("nan")
    print(f"\n  mean fusion_trust | bracketed=True  (n={len(bracketed_scores)}): {mean_bracketed:.3f}")
    print(f"  mean fusion_trust | bracketed=False (n={len(missed_scores)}): {mean_missed:.3f}")
    direction = ("higher trust facilities were MORE often bracketed (expected direction)"
                 if mean_bracketed > mean_missed else
                 "higher trust facilities were NOT more often bracketed (unexpected / no signal)")
    print(f"  -> {direction}")

    result = {
        "n_facilities": n,
        "question_1_two_feature_fit": {
            "single_feature_hit_days_loo_r2": r2_1,
            "single_feature_hit_days_loo_mae": mae1,
            "two_feature_hit_days_plus_activity_loo_r2": r2_2,
            "two_feature_hit_days_plus_activity_loo_mae": mae2,
            "activity_signal_improves_fit": bool(improved),
            "predictions": {p: float(v) for p, v in zip(plants, preds2)},
        },
        "question_2_fusion_trust_vs_climate_trace": {
            "fusion_trust_score": {p: float(t) for p, t in zip(plants, fusion_trust)},
            "n_matched_to_climate_trace": len(matched),
            "mean_trust_when_bracketed": mean_bracketed,
            "mean_trust_when_missed": mean_missed,
            "expected_direction_observed": bool(mean_bracketed > mean_missed) if matched else None,
        },
        "caveat": (
            "N=17 facilities (N<=17 for the CT-matched subset in question 2). The fusion "
            "trust score is built ONLY from q_rel_std and activity_prob_mean (both already "
            "computed independently of Climate TRACE) and is never fit to Climate TRACE -- "
            "the comparison in question 2 is a post-hoc evaluation of an already-fixed "
            "score, not model training or tuning. Per RESEARCH_PLAN.md Sec 7, Climate "
            "TRACE is not, and must not become, a training label; per Sec 8, no "
            "multi-feature nonlinear model was attempted at this facility count. This is "
            "still an indicative feasibility result, not a validated correction model -- "
            "the genuine Q-correcting A1-A5 ablation ladder remains unbuilt and requires an "
            "independent ground-truth emissions source this project does not have."
        ),
    }
    json.dump(result, open("data/track_fusion_model_results.json", "w"), indent=2)
    print("\n[SAVED] data/track_fusion_model_results.json")


if __name__ == "__main__":
    main()

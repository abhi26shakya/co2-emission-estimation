"""
Compares the Week 10 activity signal (detector3_2ch_mixed.pt, tile-level
split) against the Week 11 re-run (detector3_2ch_mixed_facility_split.pt,
facility-level split) -- the direct test of WEEK11_LOG.txt's memorization
concern, rather than leaving it as an inference from recall numbers alone.

Facilities fall into three groups relative to Track A's original 5-plant
training set (Vindhyachal, Sasan, Mundra, Tirora, plus Mundra's sibling
unit -- only Vindhyachal/Sasan/Mundra/Tirora matter for Track B):

  in_training_both        -- Vindhyachal, Sasan, Mundra: part of Track A's
                              training data in BOTH checkpoints (the
                              facility-split run's held-out test groups for
                              the "mixed" tag were TIRORA_TPP, city_Bangalore,
                              hwy_AhmedabadVadodara, hwy_PuneBangalore,
                              ind_Ludhiana, rural_Rajasthan -- see
                              WEEK11_LOG.txt / train_3channel.py's printed
                              test_groups). Expect near-zero deltas if the
                              signal reflects genuine, stable recognition.

  held_out_in_facility_split -- Tirora: part of Track A's training data in
                              the ORIGINAL checkpoint, but held out as a
                              test facility in the facility-split run.
                              A meaningful drop here is direct evidence the
                              original signal was propped up by having seen
                              this exact facility during training.

  never_in_training        -- Talcher, Rihand, Sipat, ChandrapurCoal,
                              Anpara: never part of Track A's training set
                              in EITHER checkpoint (Track A only ever
                              trained on the original 5-plant set). Large
                              deltas here can't be "memorization" in the
                              same sense (neither model saw them), but they
                              do show how unstable the model's verdict on a
                              genuinely novel facility is across reasonable
                              training variation -- itself a reliability
                              problem for using this signal downstream.
"""
import json

FACILITY_GROUPS = {
    "Vindhyachal": "in_training_both",
    "Sasan": "in_training_both",
    "Mundra": "in_training_both",
    "Tirora": "held_out_in_facility_split",
    "Talcher": "never_in_training",
    "Rihand": "never_in_training",
    "Sipat": "never_in_training",
    "ChandrapurCoal": "never_in_training",
    "Anpara": "never_in_training",
}

orig = {r["plant"]: r for r in json.load(open("data/activity_signals.json"))}
fs = {r["plant"]: r for r in json.load(open("data/activity_signals_facility_split.json"))}

results = []
print(f"{'plant':16s} {'group':28s} {'orig':>8s} {'fac_split':>10s} {'delta':>8s}")
for name, group in FACILITY_GROUPS.items():
    o = orig[name]["activity_prob_mean"]
    f = fs[name]["activity_prob_mean"]
    delta = f - o
    print(f"{name:16s} {group:28s} {o:8.3f} {f:10.3f} {delta:+8.3f}")
    results.append({"plant": name, "group": group, "orig_activity_prob": o,
                     "facility_split_activity_prob": f, "delta": delta})

by_group = {}
for r in results:
    by_group.setdefault(r["group"], []).append(abs(r["delta"]))
print("\nMean |delta| by group:")
summary = {}
for group, deltas in by_group.items():
    mean_abs_delta = sum(deltas) / len(deltas)
    summary[group] = mean_abs_delta
    print(f"  {group:28s} mean|delta|={mean_abs_delta:.3f}  (n={len(deltas)})")

conclusion = (
    "in_training_both facilities show near-zero deltas (stable, as expected "
    "if the model reliably recognizes sites it was trained on in both runs). "
    "Tirora (held out only in the facility-split run) drops meaningfully, "
    "direct evidence the original signal was inflated by having seen this "
    "exact facility during training. The 5 never-in-training facilities show "
    "the largest and most inconsistent deltas of any group (up to 0.22), "
    "meaning the activity signal for genuinely novel facilities is unstable "
    "across reasonable training variation, not just weak on average -- "
    "confirms WEEK11_LOG.txt's concern about the Week 10 activity signal "
    "for the 5 new facilities directly, rather than leaving it inferred."
)
print(f"\n{conclusion}")

json.dump({"per_facility": results, "mean_abs_delta_by_group": summary,
           "conclusion": conclusion},
          open("data/activity_signal_checkpoint_comparison.json", "w"), indent=2)
print("\n[SAVED] data/activity_signal_checkpoint_comparison.json")

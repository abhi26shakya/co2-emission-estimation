"""
Compares the tile-level-split "mixed" checkpoint (detector3_2ch_mixed.pt)
against the facility-level-split one (detector3_2ch_mixed_facility_split.pt)
-- the direct test of the memorization concern WEEK11_LOG.txt raised,
rather than leaving it as an inference from recall numbers alone.

Both checkpoints were retrained this session on the full 20-facility
positive set (up from the original 4 physical sites) after
export_new_positive_tiles.py filled in Track A positive-class tiles for the
16 newly processed Track B facilities. Facilities are grouped by whether
they were part of the "mixed" facility-split run's held-out test set,
taken directly from that run's printed test_groups (Anpara, ChandrapurCoal,
SASAN_UMPP, Talcher, plus non-plant groups not relevant here):

  held_out_in_facility_split -- Anpara, ChandrapurCoal, Sasan, Talcher:
                              excluded from training in the facility-split
                              checkpoint, but included (via their tiles
                              landing in the training pool) in the
                              tile-level checkpoint. A meaningful delta
                              here is evidence the tile-level signal was
                              propped up by the model having seen this
                              exact facility during training.

  in_training_both          -- every other facility: assumed present in
                              the training pool for both checkpoints (the
                              facility-level split only holds out the
                              groups above; the tile-level split holds out
                              individual tiles, not whole facilities, so
                              with 20 facilities contributing tiles it is
                              very likely -- though not logged explicitly
                              by train_3channel.py -- that each of these
                              facilities kept at least some tiles in train
                              for the tile-level run too).
"""
import json

# ALIAS_TO_PLANT mirrors extract_activity_signal.py's TRACK_A_ALIAS mapping
# (inverted): the 4 original sites' filename prefixes -> plant_results.json
# short names used as test_groups entries by train_3channel.py.
ALIAS_TO_PLANT = {"VINDH_CHAL_STPS": "Vindhyachal", "SASAN_UMPP": "Sasan",
                   "MUNDRA_TPP": "Mundra", "TIRORA_TPP": "Tirora"}

# test_groups printed by this session's train_3channel.py run for the
# "2ch_mixed" facility-level split; non-plant entries (city/hwy/rural) are
# irrelevant here and dropped when building HELD_OUT.
MIXED_FACILITY_SPLIT_TEST_GROUPS = [
    "Anpara", "ChandrapurCoal", "SASAN_UMPP", "Talcher",
    "city_Bangalore", "hwy_AhmedabadVadodara", "hwy_PuneBangalore",
    "ind_Ludhiana", "rural_Rajasthan",
]
HELD_OUT = {ALIAS_TO_PLANT.get(g, g) for g in MIXED_FACILITY_SPLIT_TEST_GROUPS}

orig = {r["plant"]: r for r in json.load(open("data/activity_signals.json"))}
fs = {r["plant"]: r for r in json.load(open("data/activity_signals_facility_split.json"))}

FACILITY_GROUPS = {
    name: ("held_out_in_facility_split" if name in HELD_OUT else "in_training_both")
    for name in sorted(set(orig) & set(fs))
}

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

held_out_mean = summary.get("held_out_in_facility_split", 0.0)
in_train_mean = summary.get("in_training_both", 0.0)
conclusion = (
    f"held_out_in_facility_split facilities (excluded from training only in the "
    f"facility-split checkpoint) show mean|delta|={held_out_mean:.3f}; "
    f"in_training_both facilities (assumed present in both training pools) show "
    f"mean|delta|={in_train_mean:.3f}. "
    + ("The held-out group's larger delta is consistent with the tile-level "
       "signal being inflated by having seen these exact facilities during "
       "training -- the same memorization pattern WEEK11_LOG.txt flagged, now "
       "checked directly against the retrained 20-facility checkpoints."
       if held_out_mean > in_train_mean else
       "Unlike the original 4-facility comparison, the held-out group's delta "
       "is no longer larger than the in-training group's -- with 20 facilities "
       "in the positive class, holding out 4 of them for the facility-split "
       "run no longer produces the clear memorization signature seen in "
       "WEEK11_LOG.txt's 4-facility comparison.")
)
print(f"\n{conclusion}")

json.dump({"per_facility": results, "mean_abs_delta_by_group": summary,
           "conclusion": conclusion},
          open("data/activity_signal_checkpoint_comparison.json", "w"), indent=2)
print("\n[SAVED] data/activity_signal_checkpoint_comparison.json")

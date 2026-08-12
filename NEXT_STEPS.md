# Project Status & Next Steps (as of Week 11)

This document consolidates the project's history and remaining work across both tracks. It complements `README.md` (setup/usage) and `RESEARCH_PLAN.md` (the original research roadmap, written after Week 7) with a running "where we are now" view, since that synthesis previously required reading all of `WEEK2_LOG.txt`–`WEEK11_LOG.txt` together.

## Context

The project has two parallel tracks estimating CO2 emissions from Indian coal power plants using satellite data:

- **Track A (plant detector):** a CNN classifying satellite tiles as "power plant" vs. not, using NO2/SO2/VIIRS channels.
- **Track B (CO2 estimation):** a physics-based mass-balance (IME) method estimating tons CO2/yr directly from OCO-3 XCO2 soundings.

`RESEARCH_PLAN.md` §14 laid out a 7-step roadmap to turn this into a research contribution. This document tracks progress against that roadmap and records unplanned but important findings discovered along the way.

## Step-by-step history

### Weeks 2–5 — Track A detector, incremental channel fusion
- Week 2: NO2-only baseline, 91.2% accuracy (easy negatives) — later shown to really be a "concentrated combustion detector," not plant-specific.
- Week 3: added hard negatives (cities, steel plants, highways) → accuracy dropped to 77.1%, exposing the confound.
- Week 4: added SO2 channel → 79.2%, fixed city false alarms but not steel plants (which also emit SO2).
- Week 5: added VIIRS thermal → tied at 79.2%, reduced steel-plant false alarms specifically but highways became relatively more confusable.

### Weeks 6–8 — Track B physics pipeline + uncertainty
- Week 6: built `physics_gaussian.py`, an IME mass-balance estimator. Vindhyachal (44.6 Mt/yr) and Sasan (37.2 Mt/yr) landed in plausible range; Tirora too low (thin coverage); Mundra skipped (57 soundings only).
- Week 7: expanded highway hard-negatives (5→10), first Track A accuracy improvement since Week 4 (81.2%). Flagged the tile-level train/test split as a leakage risk (fixed later, Week 11).
- Week 8: per-overpass wind matching + uncertainty propagation into `physics_gaussian.py` (roadmap step 1) — replaced single annual-mean wind with per-overpass ERA5 matching, added checkpointing for long-running OCO-3 scans.

### Week 9 — facility-set expansion (roadmap step 2)
- Extended candidate list from 5 to 20 plants (`pick_plants.py`), generalized `process_plant.py` to read from a CSV registry instead of hardcoded plants.
- Processed 5 new plants (Talcher, Rihand, Sipat, ChandrapurCoal, Anpara): went from 4→9 total plants processed, 2/4→7/9 producing a plausible estimate.
- Measured real cost: ~50–65 min/plant (network-bound OCO-3 scan) — a documented constraint on how far facility expansion can go in one session. 10 candidates remain unprocessed.

### Week 10 — activity signal, reliability check, Climate TRACE benchmark, Talcher diagnosis
- Roadmap step 3: extracted an activity-probability signal from Track A's existing 3-channel CNN for all 9 Track B facilities. Broadly tracked the physics-based Q estimate.
- Roadmap step 4 (scoped down): tested whether activity signal / wind alignment / metadata predicts `physics_gaussian.py`'s own uncertainty. Negative result — best single-feature correlation didn't survive leave-one-out CV at N=7, reinforcing that more facilities are needed.
- Roadmap step 6: pulled Climate TRACE India power-sector CO2 data, matched to all 9 facilities. 5/7 facilities with a Track B estimate fell within stated uncertainty. Two misses: Sasan overestimated (2.13x), Talcher underestimated (0.50x) despite having the tightest interval of the session.
- Unplanned: diagnosed why Talcher was both most-confident and most-wrong. Root cause: thin CO2 signal-to-noise ratio (0.18 vs. 1.27 for a well-bracketed plant) and high sensitivity to the background-annulus definition (IME swings 20% across reasonable background choices) — a structural uncertainty source the model wasn't accounting for.

### Same-day follow-ups (commits `a0d70d0`, `fc3655b`, `5baa198`) + Week 11
- Folded the background-definition sensitivity found during Talcher diagnosis into `physics_gaussian.py` as a third uncertainty term (alongside wind and IME-sampling noise), computed for all 9 facilities. Talcher's interval widened modestly (±23%→±25%) but still doesn't bracket its Climate TRACE benchmark — an honest widening, not a forced fix.
- **Week 11:** fixed the Week-7-flagged Track A leakage by adding a facility-level train/test split (`facility_level_split()` in `train_3channel.py`), run side-by-side with the old tile-level split for comparison. Result: leakage was real and substantial — "hard_only" accuracy dropped 81.2%→67.3% (facility-level), and "mixed" model's plant recall collapsed 53%→8%, revealing the model was largely memorizing the 4 original training facilities.
- Re-ran the activity-signal extraction against the new facility-split checkpoint and built `compare_activity_signal_checkpoints.py` to diff results per facility by training exposure. Found the Week 10 activity signal for the 5 newer facilities is low-confidence for two compounding reasons: weak recall on unseen facilities (8%), and high variance in what that weak signal outputs for a given novel facility (e.g. Sipat/ChandrapurCoal swung ~0.22 between checkpoints; Rihand was stable). Conclusion: this doesn't invalidate the Track B physics estimates, only the CNN-derived activity-signal cross-check.

## Roadmap status vs. RESEARCH_PLAN.md §14

| Step | Status |
|---|---|
| 1. Per-overpass wind + uncertainty in `physics_gaussian.py` | Done (Week 8), later strengthened with a 3rd uncertainty term (background-sensitivity) |
| 2. Facility-set expansion (10–20 candidates) | Partial — 9/20 processed; 10 candidates remain |
| 3. Activity signal extraction from Track A CNN | Done (Week 10), but flagged low-confidence for 5/9 facilities after Week 11's leakage fix |
| 4. Correction model (A1→A5 ablation ladder, §9) | Scoped down to a negative-result feasibility check (Week 10); full correction model not built — blocked on facility count |
| 5. Leave-one-facility-out CV harness | Not started as a general harness (though Week 11 effectively did one LOFO-style split for Track A) |
| 6. Climate TRACE benchmark comparison | Done (Week 10) |
| 7. Full evaluation figure set (§12) | Not started |

## What's needed next

1. **Facility-set expansion (highest-leverage, explicitly named as blocking in Weeks 9–11 logs).** 10 candidates remain (Korba, ShriSingajiMalwa, Koradi, Tamnar, Kudgi, Kahalgaon, Mouda, Chhabra, Farakka, Simhadri) at the measured ~50–65 min/plant cost. This is the shared bottleneck for: Track B's reliability-model retry (needs N>7), Track A's classifier evaluation (only 5 positive-class facilities today), and the correction model in step 4.
2. **More Track A positive-class facilities**, contingent on step 1 — would require exporting NO2/SO2/VIIRS tiles for any newly processed plants, not just OCO-3 soundings.
3. **Build the actual correction model (roadmap step 4)** once facility count supports it — currently only a negative-result feasibility check exists.
4. **General leave-one-facility-out CV harness (roadmap step 5)** — currently only exists ad hoc (Week 11's facility split for Track A).
5. **Formalize wind/CO2 offset-diff as an explicit per-plant quality flag** (noted as a TODO in Week 9's log) rather than just printing it.
6. **Full evaluation figure set (roadmap step 7 / RESEARCH_PLAN.md §12)** — not started: predicted-vs-actual, residuals, error distribution, facility-level comparison, uncertainty calibration plot, ablation table, feature importance.
7. Optional smaller follow-up already flagged in Week 10: ChandrapurCoal's per-overpass wind fell back to annual-mean mode (0 days met the match threshold) despite 28 near-plant soundings — worth a closer look given its bad wind/CO2 alignment (177°, the worst seen).

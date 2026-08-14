# Tests

Stdlib `unittest`, no extra dependency required. Run from the repo root:

```
python -m unittest discover -s tests -v
```

## Scope

These are unit tests for the deterministic, pure-function building blocks
of Track A/Track B, not end-to-end integration tests — nothing here touches
Earth Engine, OCO-3 downloads, or the real `data/` directory.

- `test_physics_gaussian.py` — the IME mass-balance math and the
  month-stratification logic added to fix the ShriSingajiMalwa
  seasonal-sampling artifact.
- `test_build_3channel.py` — NO2/SO2/VIIRS tile pairing, gap-filling, and
  the SO2/VIIRS negative-value clamp.
- `test_lofo_track_a.py` — `facility_fold_indices()`, guarding against a
  regression of the Week 11 tile-level-leakage bug (the same physical
  facility's tiles landing in both train and test).

## What's not covered

Anything requiring network access (Earth Engine, `earthaccess`), GPU/CPU
model training end-to-end, or real satellite data — those are exercised by
actually running the pipeline scripts, not by this test suite. There's no
CI configured to run these automatically; run them manually before trusting
a change to `physics_gaussian.py`, `build_3channel.py`, or the LOFO
harnesses.

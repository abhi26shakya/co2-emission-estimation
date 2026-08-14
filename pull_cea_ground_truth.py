"""
Pulls the independent, non-Climate-TRACE ground-truth emissions source
RESEARCH_PLAN.md's A1-A5 Q-correcting ablation ladder has been blocked on
all session: the Central Electricity Authority (CEA, Government of India,
Ministry of Power)'s CO2 Baseline Database. This is fundamentally
different in kind from Climate TRACE: it is NOT satellite/thermal-proxy
inferred, but a bottom-up calculation from each plant's own reported fuel
consumption x gross calorific value x IPCC/national emission factor --
the same accounting basis used for real CDM/carbon-credit determinations
under the Kyoto Protocol / UNFCCC. Plant-level activity data is "provided
by the respective power utilities" per the database's own disclaimer, not
inferred from remote sensing. This makes it usable where Climate TRACE
explicitly is not (RESEARCH_PLAN.md Sec 7): as a genuine training/
validation label, not just an independent benchmark.

Pulls Version 17.0 (published 2022-02, covers FY 2020-21 -- the closest
single-fiscal-year match to Track B's OCO-3-derived 2020-calendar-year
estimates; India's fiscal year is April-March, so FY2020-21 overlaps 9 of
Track B's 12 sounding-collection months). Matches station names against
data/plant_results.json's facility list via a hand-verified name mapping
(CEA's naming convention turned out to closely mirror this project's own
internal aliases -- e.g. "VINDH_CHAL STPS" vs. this project's
"VINDH_CHAL_STPS" -- suggesting the original facility set may have been
sourced from this same database). All 30 candidate facilities (20
committed + 10 from the in-progress facility-set-expansion batch) matched
cleanly; no fuzzy/ambiguous matches were accepted into MAPPING.

Source: https://cea.nic.in/cdm-co2-baseline-database/?lang=en (public,
no login required). Downloads the FY2020-21 archive directly rather than
committing the ~2.3MB source .xlsm to the repo.
"""
import io
import json
import os
import urllib.request
import zipfile

import openpyxl

URL = "https://cea.nic.in/wp-content/uploads/baseline/2022/02/database_17_.zip"
FY_COLUMN_LABEL = "2020-21"

# our facility name -> CEA "Data" worksheet station name (unit_no == 0 row).
# Hand-verified against a full fuzzy-match pass over every CEA station name
# (see PROJECT_RESEARCH_DOCUMENTATION.md for the verification method) --
# every one of these is an exact or unambiguous match, not a guess.
MAPPING = {
    "Vindhyachal": "VINDH_CHAL STPS",
    "Mundra": "MUNDRA TPP",
    "Sasan": "SASAN UMPP",
    "Tirora": "TIRORA TPP",
    "Talcher": "TALCHER STPS",
    "Rihand": "RIHAND",
    "Sipat": "SIPAT STPS",
    "ChandrapurCoal": "CHANDRAPUR_COAL",
    "Anpara": "ANPARA",
    "RGundem": "R_GUNDEM STPS",
    "Korba": "KORBA STPS",
    "ShriSingajiMalwa": "SHRI SINGAJI MALWA TPP",
    "Koradi": "KORADI",
    "Tamnar": "TAMNAR TPP",
    "Kudgi": "KUDGI",
    "Kahalgaon": "KAHALGAON",
    "Mouda": "MOUDA STPS",
    "Chhabra": "CHHABRA TPS",
    "Farakka": "FARAKKA STPS",
    "Simhadri": "SIMHADRI",
    "TalwandiSabo": "TALWANDI SABO",
    "Lalitpur": "LALITPUR TPP",
    "Pryagraj(Bara)": "PRYAGRAJ (BARA) TPP",
    "Dadri(Nctpp)": "DADRI (NCTPP)",
    "Akaltara": "AKALTARA TPP",
    "KGudemNew": "K_GUDEM NEW",
    "Raichur": "RAICHUR",
    "Bellary": "BELLARY TPS",
    "RayalSeema": "RAYAL SEEMA",
    "Sagardighi": "SAGARDIGHI TPP",
}


def fetch_workbook():
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        zip_bytes = resp.read()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        xlsm_name = next(n for n in zf.namelist() if n.lower().endswith((".xlsm", ".xlsx")))
        with zf.open(xlsm_name) as f:
            data = f.read()
    return openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True, keep_vba=False)


def main():
    wb = fetch_workbook()
    ws = wb["Data"]
    rows = list(ws.iter_rows(values_only=True))
    header = rows[0]

    col_gen = col_abs = col_spec = None
    for i, h in enumerate(header):
        if not h or FY_COLUMN_LABEL not in str(h):
            continue
        if "Net" in h and "Generation" in h:
            col_gen = i
        elif "Absolute" in h:
            col_abs = i
        elif "Specific" in h:
            col_spec = i
    assert None not in (col_gen, col_abs, col_spec), "FY column headers not found -- CEA workbook layout changed"

    stations = {}
    for row in rows[1:]:
        name, unit_no = row[1], row[2]
        if name is None or unit_no != 0:
            continue  # unit_no==0 is the station-level aggregate row
        stations[name.strip().upper()] = {
            "net_gen_gwh": row[col_gen],
            "abs_emissions_t_co2": row[col_abs],
            "spec_emissions_t_per_mwh": row[col_spec],
        }

    out = {}
    missing = []
    for our_name, cea_name in MAPPING.items():
        s = stations.get(cea_name)
        if s is None or s["abs_emissions_t_co2"] is None:
            missing.append((our_name, cea_name))
            continue
        out[our_name] = {"cea_station_name": cea_name, **s}
        print(f"[{our_name:20s}] {cea_name:28s} gen={s['net_gen_gwh']:>12,.1f} GWh  "
              f"abs={s['abs_emissions_t_co2']:>16,.0f} t CO2")

    if missing:
        print(f"\n[!] {len(missing)} facilities not matched: {missing}")

    result = {
        "source": "Central Electricity Authority (CEA), Government of India -- CO2 Baseline Database v17.0",
        "source_url": "https://cea.nic.in/cdm-co2-baseline-database/?lang=en",
        "fiscal_year": FY_COLUMN_LABEL,
        "fiscal_year_caveat": (
            "India's fiscal year is April-March. FY2020-21 = 2020-04-01 to 2021-03-31, "
            "overlapping 9 of the 12 months Track B's 2020-calendar-year OCO-3 soundings "
            "cover (Apr-Dec 2020) -- closest available single-FY match, not exact."
        ),
        "methodology_note": (
            "Bottom-up: fuel consumption x gross calorific value x IPCC/national emission "
            "factor, using activity data reported by each plant's own operator to CEA. NOT "
            "satellite/thermal-proxy inferred -- fundamentally different in kind from "
            "Climate TRACE, and therefore usable as a training/validation label per "
            "RESEARCH_PLAN.md Sec 7's Climate-TRACE-specific restriction."
        ),
        "n_matched": len(out),
        "n_missing": len(missing),
        "facilities": out,
    }
    json.dump(result, open("data/cea_ground_truth_2020_21.json", "w"), indent=2)
    print(f"\n[SAVED] {len(out)}/{len(MAPPING)} facilities -> data/cea_ground_truth_2020_21.json")


if __name__ == "__main__":
    main()

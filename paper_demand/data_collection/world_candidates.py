"""
world_candidates.py — enumerate the bike-share networks worldwide
that are eligible for the IMD-augmented demand-prediction study.

Inputs the global audit catalogue produced by the GBFS Audit
Catalogue paper (1,505 systems audited across 48+ countries) and
applies the eligibility filter:

  - status == "ok"           : feed reachable, station_information parseable
  - station_type docked       : true dock-based deployment (not free-floating)
  - n_stations >= N_MIN       : large enough for a station-level panel
  - A1 not flagged            : not a car-sharing system mislabelled as BSS
  - A2 not flagged            : capacity is real, not a placeholder constant
  - A3 not flagged            : capacity not blown up by conditional averaging
  - A7 share <= A7_MAX_SHARE  : the capacity field is mostly populated

The output splits systems into two tiers:

  T1 — "gold-standard trip log" : networks with a publicly downloadable
       historical trip-record archive (NA Lyft + a small curated list)
  T2 — "GBFS pseudo-flow"      : every other eligible system ; trip-level
       demand has to be reconstructed from station_status polling
       (the converter at tier2_pseudo_flow.py)

Usage:
  python world_candidates.py [--n-min 20] [--max-a7 0.10]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

GLOBAL_AUDIT_CSV = Path(
    "/home/rohanfosse/Bureau/Recherche/gbfs-audit-catalogue/experiments/"
    "e2_threshold_sensitivity/global_audit_results_typed.csv"
)
OUT_DIR = Path(
    "/home/rohanfosse/Bureau/Recherche/imd-national-catalogue/paper_demand/"
    "data_collection/world_candidates_output"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Known open-trip-history networks (Tier 1) ────────────────────────────────
# Lyft-operated North American systems: same monthly CSV schema on s3.
# Plus a small curated list of other publicly-archived trip logs.
TIER1_KNOWN = {
    # --- USA Lyft / Motivate-stack networks (same s3 schema since 2020) ----
    "Citi Bike": ("US", "https://s3.amazonaws.com/tripdata/"),
    "Capital Bike Share": ("US", "https://s3.amazonaws.com/capitalbikeshare-data/"),
    "Bay Wheels": ("US", "https://s3.amazonaws.com/baywheels-data/"),
    "Blue Bikes": ("US", "https://s3.amazonaws.com/hubway-data/"),  # Boston
    "Bluebikes": ("US", "https://s3.amazonaws.com/hubway-data/"),
    "Divvy": ("US", "https://divvy-tripdata.s3.amazonaws.com/"),
    "Indego": ("US", "https://www.rideindego.com/about/data/"),  # Philly
    "Metro Bike Share": ("US", "https://bikeshare.metro.net/about/data/"),  # LA
    "BIKETOWN": ("US", "https://www.biketownpdx.com/system-data"),
    "Nike Biketown": ("US", "https://www.biketownpdx.com/system-data"),
    "Bublr Bikes": ("US", "https://bublrbikes.org/system-data/"),  # Milwaukee
    "CapMetro": ("US", "https://data.austintexas.gov/Transportation-and-Mobility/Austin-MetroBike-Trips/tyfh-5r8s"),
    # --- Canada ---------------------------------------------------------------
    "BIXI Montréal": ("CA", "https://bixi.com/en/open-data"),
    "BIXI Montreal": ("CA", "https://bixi.com/en/open-data"),
    "Bike Share Toronto": ("CA", "https://open.toronto.ca/dataset/bike-share-toronto-ridership-data/"),
    "Mobi Bike Share": ("CA", "https://www.mobibikes.ca/en/system-data"),  # Vancouver
    # --- UK ------------------------------------------------------------------
    "Santander Cycles": ("GB", "https://cycling.data.tfl.gov.uk/"),
    # --- Latin America -------------------------------------------------------
    "Ecobici": ("MX", "https://ecobici.cdmx.gob.mx/datos-abiertos/"),  # also AR
    "Mibici Guadalajara": ("MX", "https://www.mibici.net/es/datos-abiertos/"),
    "Bike Itaú - Rio": ("BR", "https://tembici.com.br/dados-abertos/"),
    "Bike Itaú - Sampa": ("BR", "https://tembici.com.br/dados-abertos/"),
    "Bike Itaú - Pernambuco": ("BR", "https://tembici.com.br/dados-abertos/"),
    "Bike Itaú - Salvador": ("BR", "https://tembici.com.br/dados-abertos/"),
    "Bike Itaú - Brasilia": ("BR", "https://tembici.com.br/dados-abertos/"),
    "Bike Itaú - Porto Alegre": ("BR", "https://tembici.com.br/dados-abertos/"),
    # --- Continental Europe -------------------------------------------------
    "Bicing": ("ES", "https://opendata-ajuntament.barcelona.cat/data/en/dataset/estacions-bicing"),
    "Velomagg": ("FR", "open release via TAM Montpellier"),
    "Vélomagg": ("FR", "open release via TAM Montpellier"),
}


def is_tier1(name: str) -> tuple[bool, str | None]:
    """Match a system name against the curated Tier 1 list."""
    n = name.strip()
    for key, (country, url) in TIER1_KNOWN.items():
        if key.lower() in n.lower() or n.lower() in key.lower():
            return True, url
    return False, None


# ── Eligibility filter ───────────────────────────────────────────────────────
def filter_eligible(df: pd.DataFrame, n_min: int = 20,
                    max_a7_share: float = 0.10) -> pd.DataFrame:
    e = df.copy()
    e["n_stations"] = pd.to_numeric(e["n_stations"], errors="coerce")
    e["a7_share"] = (pd.to_numeric(e["n_capacity_nan"], errors="coerce")
                     / e["n_stations"].replace(0, pd.NA)).fillna(0)

    # Status reachable
    status_ok = e["status"] == "ok"
    # Dock-based
    dock_ok = e["station_type"].astype(str).str.contains("docked", case=False)
    # Size
    size_ok = e["n_stations"].fillna(0) >= n_min
    # No structural anomaly
    a1_ok = ~(e["A1_n_stations"].fillna(0) > 0)
    a2_ok = ~e["A2_flagged"].fillna(False)
    a3_ok = ~(e["A3_n_stations"].fillna(0) > 0)
    # Capacity field mostly populated
    a7_ok = e["a7_share"] <= max_a7_share

    eligible = status_ok & dock_ok & size_ok & a1_ok & a2_ok & a3_ok & a7_ok
    e["eligible"] = eligible
    e["filter_reason"] = ""
    for col, cond, label in [
        (status_ok, status_ok, "status_not_ok"),
        (dock_ok, dock_ok, "not_docked"),
        (size_ok, size_ok, f"under_{n_min}_stations"),
        (a1_ok, a1_ok, "A1_carsharing"),
        (a2_ok, a2_ok, "A2_placeholder_capacity"),
        (a3_ok, a3_ok, "A3_over_capacity"),
        (a7_ok, a7_ok, f"A7_over_{int(max_a7_share*100)}pct_nan"),
    ]:
        e.loc[~cond, "filter_reason"] = (
            e.loc[~cond, "filter_reason"]
            .where(e.loc[~cond, "filter_reason"] != "", "")
            .astype(str) + (label + ";")
        )
    return e


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-min", type=int, default=20)
    p.add_argument("--max-a7", type=float, default=0.10)
    args = p.parse_args()

    print(f"Loading {GLOBAL_AUDIT_CSV.name}")
    df = pd.read_csv(GLOBAL_AUDIT_CSV)
    print(f"  raw: {len(df)} systems × {len(df.columns)} cols")

    audited = filter_eligible(df, n_min=args.n_min, max_a7_share=args.max_a7)
    eligible = audited[audited["eligible"]].copy()
    print(f"  eligible after filters (N_min={args.n_min}, max_A7={args.max_a7}): "
          f"{len(eligible)} / {len(df)} ({100*len(eligible)/len(df):.1f}%)")

    # Tier classification
    tier_info = eligible["name"].apply(is_tier1)
    eligible["tier"] = ["T1" if t[0] else "T2" for t in tier_info]
    eligible["tier1_archive_url"] = [t[1] if t[0] else "" for t in tier_info]

    tier1 = eligible[eligible["tier"] == "T1"]
    tier2 = eligible[eligible["tier"] == "T2"]

    print(f"\n--- TIER BREAKDOWN ---")
    print(f"  T1 (open trip-history):  {len(tier1):4d}  systems   "
          f"{int(tier1['n_stations'].sum()):7d} stations")
    print(f"  T2 (GBFS pseudo-flow):   {len(tier2):4d}  systems   "
          f"{int(tier2['n_stations'].sum()):7d} stations")

    # By country
    print(f"\n--- BY COUNTRY (top 25 eligible systems) ---")
    by_country = eligible.groupby("country").agg(
        n_systems=("name", "count"),
        n_stations=("n_stations", "sum"),
        n_t1=("tier", lambda x: (x == "T1").sum()),
        n_t2=("tier", lambda x: (x == "T2").sum()),
    ).sort_values("n_systems", ascending=False).head(25)
    print(by_country.to_string())

    # Filter reasons for excluded
    excluded = audited[~audited["eligible"]]
    print(f"\n--- EXCLUSION REASONS ({len(excluded)} systems excluded) ---")
    reason_counts = excluded["filter_reason"].str.split(";").explode().str.strip()
    reason_counts = reason_counts[reason_counts != ""].value_counts()
    print(reason_counts.head(15).to_string())

    # Tier 1 list
    print(f"\n--- TIER 1 (known open trip-history archives) ---")
    print(tier1[["name", "country", "n_stations", "tier1_archive_url"]]
          .sort_values("n_stations", ascending=False).to_string(index=False))

    # Tier 2 by country (top countries)
    print(f"\n--- TIER 2 BY COUNTRY (top 15) ---")
    t2_by_country = (tier2.groupby("country")
                     .agg(n_systems=("name", "count"),
                          n_stations=("n_stations", "sum"))
                     .sort_values("n_stations", ascending=False).head(15))
    print(t2_by_country.to_string())

    # Save
    eligible.to_csv(OUT_DIR / "eligible_systems.csv", index=False)
    tier1.to_csv(OUT_DIR / "tier1_systems.csv", index=False)
    tier2.to_csv(OUT_DIR / "tier2_systems.csv", index=False)
    by_country.to_csv(OUT_DIR / "_eligible_by_country.csv")

    summary = {
        "n_total_audited": int(len(df)),
        "n_eligible": int(len(eligible)),
        "n_tier1": int(len(tier1)),
        "n_tier2": int(len(tier2)),
        "n_stations_eligible": int(eligible["n_stations"].sum()),
        "n_stations_tier1": int(tier1["n_stations"].sum()),
        "n_stations_tier2": int(tier2["n_stations"].sum()),
        "n_countries_eligible": int(eligible["country"].nunique()),
        "thresholds": {"n_min": args.n_min, "max_a7_share": args.max_a7},
    }
    (OUT_DIR / "_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nOutput files written to {OUT_DIR}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

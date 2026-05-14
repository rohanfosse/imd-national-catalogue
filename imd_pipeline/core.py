"""Join the B20 / B21 commune-level outputs into a single catalogue.

The pipeline composes two artefacts produced upstream by Paper 03's
`b20_national_imd4.py` and `b21_national_ies.py`:

  source/imd4_national_communes.csv  — 34,858 communes × 12 columns
                                       (IMD-4 + M/I/D components,
                                       insee_part_velo validator)
  source/ies_national_communes.csv   — 31,266 communes × 7 columns
                                       (income_median, IES, poverty_rate)

The unified catalogue is one row per INSEE 2024 commune (PK
``code_commune``), with the IES columns carrying NaN for the 3,592
communes that lack a Filosofi income value and an explicit
``ies_available`` flag to make filtering unambiguous. A
``double_penalty`` boolean materialises the cycling-poverty desert
definition from Paper 04 (bottom-33% IMD AND top-33% poverty).

Public API
----------
- ``build_catalogue() -> DataFrame``  the 34,858-row reference table
- ``score_communes()  -> DataFrame``  alias of ``build_catalogue``
- ``audit_summary(df) -> DataFrame``  per-strata aggregates
- ``load_sources()    -> tuple``      raw IMD + IES DataFrames
- ``CATALOGUE_COLUMNS -> list[str]``  the locked column order
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
SOURCE_DIR = _ROOT / "data" / "source"
CATALOGUE_DIR = _ROOT / "catalogue"

CATALOGUE_COLUMNS: list[str] = [
    "code_commune",
    "nom",
    "population",
    "surface_km2",
    "M_per_km2",
    "I_per_km2",
    "D_log",
    "M_z",
    "I_z",
    "D_z",
    "IMD4_national",
    "insee_part_velo",
    "income_median",
    "poverty_rate",
    "IES",
    "ies_available",
    "double_penalty",
]


def load_sources(source_dir: Path | str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the two upstream CSVs with ``code_commune`` kept as string.

    INSEE codes are zero-padded (``"01001"``) and must not be parsed as
    integers — pandas would otherwise drop the leading zero on Corsican
    and overseas communes.
    """
    d = Path(source_dir) if source_dir is not None else SOURCE_DIR
    imd = pd.read_csv(d / "imd4_national_communes.csv", dtype={"code_commune": str})
    ies = pd.read_csv(d / "ies_national_communes.csv", dtype={"code_commune": str})
    return imd, ies


def build_catalogue(source_dir: Path | str | None = None) -> pd.DataFrame:
    """Materialise the 34,858-row commune-level catalogue.

    The IES columns are kept NaN-able rather than split into a second
    table so that downstream consumers can filter on ``ies_available``
    without rejoining.
    """
    imd, ies = load_sources(source_dir)

    if imd["code_commune"].duplicated().any():
        raise ValueError("IMD source has duplicate code_commune keys")
    if ies["code_commune"].duplicated().any():
        raise ValueError("IES source has duplicate code_commune keys")
    if not ies["code_commune"].isin(imd["code_commune"]).all():
        raise ValueError("IES file contains code_commune values absent from IMD")

    catalogue = imd.merge(
        ies[["code_commune", "income_median", "IES", "poverty_rate"]],
        on="code_commune",
        how="left",
        validate="one_to_one",
    )

    catalogue["ies_available"] = catalogue["IES"].notna()

    # Double-penalty = cycling-poverty desert (Paper 04, Section IES):
    # commune in the bottom 33% of IMD-4 AND in the top 33% of poverty.
    # Computed on the panel that has *both* IMD and poverty (≈4,408
    # communes); communes without a poverty value get False.
    with_pov = catalogue.dropna(subset=["poverty_rate"])
    imd_q33 = with_pov["IMD4_national"].quantile(0.33)
    pov_q66 = with_pov["poverty_rate"].quantile(0.66)
    catalogue["double_penalty"] = (
        catalogue["poverty_rate"].notna()
        & (catalogue["IMD4_national"] < imd_q33)
        & (catalogue["poverty_rate"] > pov_q66)
    )

    catalogue["population"] = catalogue["population"].astype("Int64")

    return catalogue[CATALOGUE_COLUMNS].copy()


def score_communes(source_dir: Path | str | None = None) -> pd.DataFrame:
    """Public alias kept stable since the v0.1 scaffold."""
    return build_catalogue(source_dir)


def audit_summary(catalogue: pd.DataFrame) -> pd.DataFrame:
    """Per-population-strata aggregate: count, mean IMD, IES coverage, deserts.

    Strata mirror the ones used in the B20 stratified-validation figure
    (panel-of-7 thresholds), so the audit summary lines up one-to-one
    with the paper's national-validation table.
    """
    thresholds = [0, 1000, 5000, 10000, 20000, 50000, 100000]
    rows = []
    for thr in thresholds:
        sub = catalogue[catalogue["population"].fillna(0) >= thr]
        if len(sub) == 0:
            continue
        rows.append(
            {
                "stratum": f"pop_ge_{thr}",
                "n_communes": int(len(sub)),
                "n_with_ies": int(sub["ies_available"].sum()),
                "n_double_penalty": int(sub["double_penalty"].sum()),
                "mean_imd4": float(sub["IMD4_national"].mean()),
                "median_imd4": float(sub["IMD4_national"].median()),
                "mean_insee_part_velo": float(sub["insee_part_velo"].mean()),
            }
        )
    return pd.DataFrame(rows)

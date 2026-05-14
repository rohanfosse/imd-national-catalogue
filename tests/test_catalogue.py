"""Schema and value-bound tests for the materialised catalogue.

These tests pin the contract of the v0.2 release. They run against the
output of ``build_catalogue()`` (i.e. the in-memory frame), not the
on-disk Parquet — so they pass even before a release is shipped.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import imd_pipeline
from imd_pipeline import (
    CATALOGUE_COLUMNS,
    audit_summary,
    build_catalogue,
    load_sources,
)


@pytest.fixture(scope="module")
def catalogue() -> pd.DataFrame:
    return build_catalogue()


def test_version_is_v0_2():
    assert imd_pipeline.__version__ == "0.2.0"


def test_row_count_matches_paper(catalogue):
    assert len(catalogue) == 34858, "Paper 04 headline: 34,858 communes"


def test_unique_primary_key(catalogue):
    assert not catalogue["code_commune"].duplicated().any()


def test_column_order_locked(catalogue):
    assert list(catalogue.columns) == CATALOGUE_COLUMNS


def test_code_commune_is_string(catalogue):
    # INSEE codes must keep their leading zeros (e.g. "01001"). The
    # dtype is whatever pandas decides for string-typed columns
    # (object on pandas <=2.x, StringDtype on pandas >=3.x); the
    # invariant we care about is value preservation.
    assert pd.api.types.is_string_dtype(catalogue["code_commune"])
    assert catalogue["code_commune"].iloc[0] == "01001"


def test_population_is_nullable_int(catalogue):
    assert str(catalogue["population"].dtype) == "Int64"


def test_ies_coverage(catalogue):
    # 31,266 of 34,858 communes have a Filosofi income median.
    assert int(catalogue["ies_available"].sum()) == 31266
    assert catalogue["IES"].notna().sum() == 31266


def test_ies_available_flag_matches_ies_nan(catalogue):
    assert (catalogue["ies_available"] == catalogue["IES"].notna()).all()


def test_double_penalty_count_matches_paper(catalogue):
    # Paper 04 headline: 362 cycling-poverty deserts.
    assert int(catalogue["double_penalty"].sum()) == 362


def test_z_score_components_are_centred(catalogue):
    # B20 upstream z-scores on the pre-filter panel (~34,875 rows)
    # then drops ~17 rows where IMD or insee_part_velo are non-finite.
    # So mean/std differ from the ideal (0, 1) by O(1e-3); tolerances
    # reflect that, not numerical precision.
    for col in ["M_z", "I_z", "D_z"]:
        assert abs(catalogue[col].mean()) < 1e-2, f"{col} not centred"
        assert math.isclose(catalogue[col].std(ddof=0), 1.0, abs_tol=1e-2)


def test_imd4_is_finite(catalogue):
    assert catalogue["IMD4_national"].notna().all()
    assert np.isfinite(catalogue["IMD4_national"]).all()


def test_audit_summary_strata(catalogue):
    summary = audit_summary(catalogue)
    # Seven strata, the most permissive (pop ≥ 0) covers everything.
    assert len(summary) == 7
    all_row = summary.iloc[0]
    assert all_row["stratum"] == "pop_ge_0"
    assert all_row["n_communes"] == 34858
    assert all_row["n_with_ies"] == 31266
    assert all_row["n_double_penalty"] == 362


def test_load_sources_keys_disjoint_in_expected_way():
    imd, ies = load_sources()
    # All IES communes are in IMD; reverse is not true (3,592 communes
    # lack Filosofi income).
    assert ies["code_commune"].isin(imd["code_commune"]).all()
    assert (~imd["code_commune"].isin(ies["code_commune"])).sum() == 3592

"""Public API placeholder for v0.1.

The scoring functions land here once the b20/b21 scripts from
papers/03_imd_bayesian/experiments/ are ported. Keeping the public
names stable from the start so the dataset card, the Streamlit app
and the tests can be wired up against the same import path.
"""

from __future__ import annotations

import pandas as pd


def score_communes() -> pd.DataFrame:
    """Materialise the 34,858-row commune-level IMD/IES table.

    Returns
    -------
    pd.DataFrame
        One row per INSEE commune, columns to be finalised in v0.2.
    """
    raise NotImplementedError(
        "v0.1 scaffold — implementation lands with the Phase 2 release"
    )


def audit_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-region/strata aggregate (mean IMD, deserts count, etc.)."""
    raise NotImplementedError(
        "v0.1 scaffold — implementation lands with the Phase 2 release"
    )

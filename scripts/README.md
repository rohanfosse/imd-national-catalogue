# `scripts/`

One-shot CLI entry points (mirrors of `gbfs-audit-catalogue/scripts/`).
Populated in Phase 2; the planned entry points are:

* `regenerate_catalogue.py` — rebuild `catalogue/communes_imd_national.parquet`
  from the INSEE meta + Gold Standard GBFS + Bayesian weights.
* `make_figures.py` — produce the paper's B14–B22 figures from the
  pinned catalogue.
* `push_to_huggingface.py` — upload the catalogue and the dataset card
  to `rohanfosse/imd-national-catalogue`.

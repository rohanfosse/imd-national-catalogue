# `catalogue/`

This directory will hold the released data products at `v1.0.0`:

* `communes_imd_national.parquet` — 34,858 rows × (IMD-4, IES, M/I/S/T
  components, INSEE meta, double-penalty flag).
* `communes_imd_audit_summary.csv` — per-region / per-strata
  aggregates and the list of cycling-poverty deserts.

In the v0.1 scaffold these files are intentionally absent; consume the
data from Hugging Face (`rohanfosse/imd-national-catalogue`) or from
the Zenodo DOI once `v1.0.0` is tagged.

Both files are released under the
[ODbL v1.0](../LICENSE-DATA); the Python code that produced them sits
under [`../imd_pipeline/`](../imd_pipeline/) (MIT).

# `data/source/`

Frozen inputs that the catalogue is built from. Bundled in the repo
so a clone is enough to rebuild `catalogue/communes_imd_national.parquet`
without any external dependency.

## Files

| File | Rows | Origin |
|------|------|--------|
| `imd4_national_communes.csv` | 34,858 | produced by `b20_national_imd4.py` (Paper 03) |
| `ies_national_communes.csv`  | 31,266 | produced by `b21_national_ies.py` (Paper 03) |

## Upstream chain

The B20 / B21 scripts in turn consume the open-data sources listed in
[`../../LICENSE-DATA`](../../LICENSE-DATA):

* `Tableau de bord des mobilités durables` (data.gouv.fr) — linéaire
  cyclable, nombre de stations TC lourdes, part-vélo-travail INSEE
  Mobilités Professionnelles 2022.
* INSEE Filosofi — niveau de vie médian, taux de pauvreté.
* INSEE — surface, population, code commune COG 2024.
* The Bayesian posterior weights (M, I, T, D) come from the B7
  calibration on the 59-city VLS panel against the Baromètre FUB
  and the EMP 2019.

## Reproduction

The full upstream pipeline (raw inputs → these two CSVs) is
intentionally **not** vendored here — it lives in the methodology
working tree, at
`papers/03_imd_bayesian/experiments/b20_national_imd4.py` and
`b21_national_ies.py` of the
[`bikeshare-data-explorer`](https://github.com/rohanfosse/bikeshare-data-explorer)
repository. This separation matches the gbfs-audit-catalogue
convention: this repo distributes the **catalogue**, not the raw
ingestion stack.

To rebuild only the unified Parquet from these CSVs:

```bash
pip install -e .            # editable install of imd_pipeline
python scripts/regenerate.py
```

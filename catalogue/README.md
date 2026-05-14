# `catalogue/`

Data products of the IMD National Catalogue.

| File | Shape | Description |
|------|-------|-------------|
| `communes_imd_national.parquet` | 34,858 × 17 | One row per INSEE 2024 commune; full schema in the [project README](../README.md#schema). |
| `communes_imd_audit_summary.csv` | 7 × 7      | Per-population-stratum aggregate: count, IES coverage, deserts count, mean/median IMD. |

Both files are released under the [ODbL v1.0](../LICENSE-DATA); the
Python code that produced them sits under
[`../imd_pipeline/`](../imd_pipeline/) (MIT).

To regenerate from the bundled CSV sources:

```bash
python scripts/regenerate.py
```

The script reads `data/source/imd4_national_communes.csv` and
`data/source/ies_national_communes.csv`, applies the join + the
double-penalty definition (bottom-33% IMD-4 ∩ top-33% poverty rate)
and writes the two artefacts above.

## Headline counts (matches Paper 04)

| Quantity | Value |
|---|---|
| Communes total | 34,858 |
| With Filosofi income (IES computable) | 31,266 |
| Cycling-poverty deserts (`double_penalty`) | 362 |
| Largest desert (population) | shown in the audit summary |

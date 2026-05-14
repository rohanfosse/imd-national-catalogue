# IMD National Catalogue

> A commune-level reference dataset for the **Indice de Mobilité Douce
> (IMD)** and the **Indice d'Équité Sociale (IES)** across **34,858
> French communes**, with the open-source pipeline that produced it.

[![Code: MIT](https://img.shields.io/badge/code-MIT-green)](LICENSE)
[![Data: ODbL](https://img.shields.io/badge/data-ODbL%20v1.0-blue)](LICENSE-DATA)
[![Status](https://img.shields.io/badge/status-v0.1%20scaffold-orange)](#roadmap)

> **Status — v0.1 scaffold.** This repository ships the licensing,
> package layout and citation metadata. The Parquet catalogue, the
> Hugging Face dataset, the Streamlit dashboard and the Zenodo DOI are
> being staged in [the roadmap below](#roadmap). Pin to a tagged
> release once `v1.0.0` is cut.

## What this is

Volumetric metrics (stations per capita, kilometres of cycle lane)
predict cycling *supply*, not cycling *quality*. The IMD is a four-
component composite — **M**ultimodality, **I**nfrastructure,
**S**afety, **T**opography — whose weights are calibrated against the
Baromètre FUB and the EMP 2019 via a Bayesian simplex. Once calibrated
at urban-area level (Paper 03), the same weights are pushed to **all
34,858 French communes** (Paper 04) and paired with the **Indice
d'Équité Sociale (IES)**, which flags communes that under-perform
relative to their socio-economic profile — the *cycling-poverty
deserts*.

Headline results (Paper 04):

* IMD-4 wins **4 / 5** of the tournament references against the
  Cerema infrastructure baseline.
* **+18 pts R²** over the standard Cerema indicator
  (Cerema-residual regression).
* National panel ρ = **+0.41** overall, ρ = **+0.55** on the 42 cities
  with population ≥ 100 k.
* **362 cycling-poverty deserts** identified, 6 of the top-15 in
  La Réunion.
* ρ(IMD, median income) = +0.001 in the urban subset — the IMD is
  **not** a wealth proxy.

## Quick start (once v1.0.0 is published)

```python
from datasets import load_dataset

imd = load_dataset("rohanfosse/imd-national-catalogue", split="train").to_pandas()

# The 362 priority deserts: high social fragility, low IMD
deserts = imd.query("ies_quintile == 1 and population >= 5000") \
             .nsmallest(362, "imd4_national")
```

## Companion artefacts

| Resource | Where |
| --- | --- |
| Sister catalogue (stations, Paper 01) | [github.com/rohanfosse/gbfs-audit-catalogue](https://github.com/rohanfosse/gbfs-audit-catalogue) |
| Manuscript (LaTeX) | [`paper/`](paper/) — Paper 04 of the BikeShare-ICT series |
| Bayesian methodology (Paper 03) | upstream calibration; weights ingested by `imd_pipeline/` |
| Dataset card + full schema | _coming with v1.0.0 on Hugging Face_ |
| Live dashboard | _coming with v1.0.0 on Streamlit Cloud_ |
| Zenodo DOI | _minted at v1.0.0 release_ |

## Roadmap

* **Phase 1 — Scaffold** (this commit). Licensing, README, package
  layout, citation metadata.
* **Phase 2 — Pipeline & catalogue.** Port the `b20_national_imd4.py`
  and `b21_national_ies.py` scoring scripts into the `imd_pipeline/`
  package and materialise `catalogue/communes_imd_national.parquet`
  (34,858 rows). Adds tests on schema, z-score bounds, and
  Cerema-residual consistency.
* **Phase 3 — Hugging Face.** Dataset card with explicit recipes
  (top-15 deserts, jointure INSEE, Cerema-vs-IMD comparison) and
  Croissant metadata.
* **Phase 4 — Streamlit.** Commune-level map with strata filters,
  IES double-penalty layer, M / I / S / T decomposition tab.
* **Phase 5 — Zenodo.** `v1.0.0` tag, archived DOI, cite-as updated
  across the paper, the dataset card and `CITATION.cff`.

## Relation to the BikeShare-ICT series

This repository is the data-release companion of **Paper 04** of the
BikeShare-ICT series (CESI LINEACT, 2025–2026). The four-paper series:

1. **Paper 01** — GBFS audit & infrastructure
   → [`gbfs-audit-catalogue`](https://github.com/rohanfosse/gbfs-audit-catalogue).
2. **Paper 02** — original IMD/IES exploratory draft (superseded
   by Paper 04).
3. **Paper 03** — station-level Bayesian methodology (weights
   producer).
4. **Paper 04** — national application on 34,858 communes (this
   repository's release).

## Citation

A [`CITATION.cff`](CITATION.cff) is provided. Once the manuscript is
deposited, the BibTeX block below will be filled in.

```bibtex
@article{Fosse2026imdnational,
  author  = {Foss\'e, Rohan and Pallares, Ga\"el},
  title   = {A National Cycling-Environment Composite Indicator for
             French Communes: The IMD and the Cycling-Poverty Deserts},
  year    = {2026},
  note    = {Manuscript in preparation; submission target:
             Transportation Research Part D / Transport Policy /
             Journal of Transport Geography}
}
```

## Licences

Code under [MIT](LICENSE); data (once released) under
[ODbL v1.0](LICENSE-DATA). Upstream attributions are listed in
`LICENSE-DATA`.

## Contact

**Rohan Fossé** ([rfosse@cesi.fr](mailto:rfosse@cesi.fr)) — CESI École d'Ingénieurs, Montpellier.
**Gaël Pallares** — CESI LINEACT (EA 7527), Montpellier.

Issues and contributions are welcome on the
[issue tracker](https://github.com/rohanfosse/imd-national-catalogue/issues).

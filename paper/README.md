# `paper/`

Source of **Paper 04** of the BikeShare-ICT series — *A National
Cycling-Environment Composite Indicator for French Communes: the IMD
and the Cycling-Poverty Deserts*.

Submission target: Transportation Research Part D / Transport Policy /
Journal of Transport Geography.

## Layout

```
paper/
├── imd_national.tex      # main manuscript
├── references.bib        # local bibliography (self-contained)
└── figures/              # 7 publication PDFs (B14–B22 subset)
    ├── b14_tournament_with_insee.pdf
    ├── b15_decomposition.pdf
    ├── b18_holdout_cv.pdf
    ├── b19_national_scatter.pdf
    ├── b20_national_imd4_scatter.pdf
    ├── b21_ies_panel.pdf
    └── b22_regional_political.pdf
```

## Build

```bash
cd paper
pdflatex imd_national && bibtex imd_national && \
    pdflatex imd_national && pdflatex imd_national
```

The bibliography style is `elsarticle-harv` (Elsevier Harvard).

## Source of truth

This vendored copy is synchronised from
[`papers/04_imd_national/overleaf/`](https://github.com/rohanfosse/bikeshare-data-explorer/tree/main/papers/04_imd_national/overleaf)
in the upstream `bikeshare-data-explorer` working repository. The two
deviations from the upstream copy are intentional and minimal:

1. `\bibliography{../references}` → `\bibliography{references}` so
   the repo is self-contained (the local `references.bib` is the
   merged bibliography).
2. `\graphicspath{{figures/}}` is unchanged — the seven figures are
   bundled flat under `figures/`.

When iterating on the paper inside this repository, edit
`imd_national.tex` directly. Any structural change (new figures,
new section reorganisation) should also be ported back upstream to
keep both copies in lockstep until the manuscript is frozen.

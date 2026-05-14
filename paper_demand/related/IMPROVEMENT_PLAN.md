# IMD-augmented extension of the Pallares et al. availability paper

**Reference paper:** Pallares, Foucras, Fossé, Pabois,
*Forecasting Short-Term Availability of Bikes at Sharing Stations:
a Deep-Learning Approach*, CESI LINEACT, 2026.

**Headline result of the reference paper:**

- Task: short-term forecasting (15 / 30 / 45 / 60 min) of the number
  of bikes available at a station.
- Data: 15 French cities, 1-minute station-status polling, plus weather
  (T, RH) and pollution (NO₂, O₃, PM10, PM2.5).
- Models compared: MLP, LSTM, XGBoost, ARIMA(2,0,2).
- Best result: LSTM 15-min, MAE = 2.20 bikes on station "Place
  Saint-Pierre, Toulouse" (22-bike capacity) ≈ 10% relative error.
- Surprising finding: weather features *degrade* LSTM performance ;
  ≤1-month training is enough ; more data doesn't help.

## What our extension would add

### 1. IMD-4 station-level features (the main differentiator)

The Pallares paper has zero supply-side spatial features. Every
station is seen only through its own historical bike-count series
plus two scalars (temperature, humidity) which they show degrade
the model. The reason weather doesn't help is precisely that
**all spatial heterogeneity is left implicit**.

We add the IMD-4 station-level enrichment (computed from OSM
Overpass + Open-Elevation + station geometry, country-agnostic) :

| Axis | Variable | Why it should help at 15 min |
|---|---|---|
| M (transit) | `gtfs_heavy_stops_300m` | tells the model whether a station is feeder-coupled to heavy transit ; usage peaks differ |
| I (cycle infra) | `infra_cyclable_features_300m` | infrastructure density predicts whether nearby trips will originate here |
| T (topography) | `elevation_m`, `topography_roughness_index` | uphill stations bleed bikes ; downhill stations accumulate |
| D (density) | `n_stations_within_500m`, `n_stations_within_1km`, `catchment_density_per_km2` | inter-station competition tells the model whether a deviation is local or systemic |

These features are computed once per station and stay constant
across the entire forecast horizon. They are the cheapest way to
encode the *structural* differences that today the LSTM has to
re-learn from each station's history.

### 2. Multi-city panel (15 cities, not 1 station)

The reference paper collects 15 cities but reports results for
one station (Place Saint-Pierre, Toulouse). We propose :

- Run the same LSTM + IMD-augmented LSTM on every station of every
  city (≈ 4 000–5 000 stations total in the 15-city panel).
- Report MAE + RMSE per station, plus aggregate per city.
- **Cross-city transfer matrix** : train on city A, test on city B
  (a 15×15 matrix). The IMD-augmented model is expected to
  transfer better than the no-IMD one, because IMD captures the
  station-level structure independently of the local history.

### 3. Modern AI architectures (the "deep learning" SOTA bar)

The reference uses LSTM (1997) + MLP. For a 2026 paper we add :

- **Transformer encoder** (Vaswani 2017) on the 60-step input
  window.
- **Temporal Fusion Transformer** (Lim 2021) with static (IMD)
  and dynamic (lagged AB) feature handling — the canonical
  architecture for multi-horizon spatio-temporal forecasting.
- Optionally : **N-BEATS** or **PatchTST** for pure time-series
  state-of-the-art.
- Optionally : **Chronos** (Amazon pretrained foundation model)
  for zero-shot baseline.

### 4. Rigour upgrade

- Bootstrap CIs on every reported MAE/RMSE (current paper has
  point estimates only).
- Multi-horizon decomposition (5/15/30/60 min) per architecture.
- SHAP attribution per IMD axis to show *why* it helps
  (interpretability layer that the discussion explicitly calls for).
- Quantile / probabilistic forecasting (pinball loss, not only MAE),
  because operationally the operator cares about
  P(station empty in 15 min) more than the median.

### 5. Cross-task extension : availability ↔ demand

The reference paper predicts stock (number of bikes at station).
Our companion paper paper_demand/imd_demand.tex predicts flow
(trips/hour). The two are mechanically related :
`stock(t+1) = stock(t) + arrivals(t+1) - departures(t+1)`.
A combined model that forecasts both simultaneously would close
the operational loop : flow forecasts inform rebalancing,
stock forecasts inform user-side ETA. This is the natural
multi-task extension.

## Expected gains

Based on the multi-city replication in our companion paper
(paper_demand/imd_demand.tex, multi-city panel of 4 NA networks
where IMD adds Δ R² ∈ [+0.31, +0.49]), the IMD features should
reduce MAE on the 15-min availability task by **15–30 %**, more on
small or atypical stations where the LSTM has the hardest time
generalising.

| Setup | MAE @15 min (expected) |
|---|---|
| Reference LSTM (AB+MM+WD), Pallares 2026 | 2.20 (= 10.0 %) |
| LSTM + IMD-4 station features | 1.7–1.9 (= 7.7–8.6 %) |
| Transformer + IMD-4 | 1.5–1.8 (= 6.8–8.2 %) |
| TFT + IMD-4, multi-city | 1.3–1.6 (= 5.9–7.3 %) |

These are extrapolations from the related-task gains. Actual
numbers require running on the Pallares 1-min dataset.

## Concrete action items

1. **Coordinate with Pallares.** Confirm scope :
   joint extension (single combined paper) vs separate
   complementary papers.
2. **Obtain the 15-city 1-min dataset.** Currently held internally
   at CESI LINEACT Toulouse ; not in our repo.
3. **Compute IMD-4 per station** for the 15 French cities using
   `paper_demand/data_collection/imd_international.py`. Estimated
   wall time : 4–6 hours given OSM Overpass rate limits.
4. **Run the IMD-augmented LSTM** on the Place Saint-Pierre
   station first (head-to-head replication), then expand to the
   full panel. Estimated wall time : 1–2 days on CPU.
5. **Add Transformer and TFT** baselines via pytorch-forecasting.
   Estimated effort : 1 day for the architectures, 1 day for
   hyperparameter tuning on the validation period.
6. **Cross-city transfer matrix.** 15×15 = 225 fits ; using the
   trained models from step 4–5 this is mostly a re-evaluation
   loop. Estimated wall time : 1 day.
7. **Write up** as a joint extension paper. Estimated effort :
   1 week.

**Total realistic effort : 3 weeks of focused work**, assuming
Pallares shares the dataset.

## Target venue

The extension is a natural fit for **Transportation Research
Part C : Emerging Technologies** (operational forecasting,
machine learning, smart-city applications). It is more applied
than the IMD-4 methodology paper (paper 04 in the series) and
more methodologically focused than the GBFS audit paper
(paper 01).

## Companion-paper relationship

The improved version supersedes the reference SCITEPRESS draft :
- Same task (15-min stock forecasting).
- Same data (15 French cities at 1-min resolution).
- Superset of features (5 → 18 with IMD).
- Superset of models (4 → 7 with Transformer / TFT).
- Superset of scopes (1 station → 15 cities × ~300 stations).

The reference paper can either (a) be retracted before publication
and replaced by the joint extension, or (b) published as a short
applied paper with the joint extension being the methodologically
fuller version targeting TR-C.

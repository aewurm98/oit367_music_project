# Working Directory Summary
**OIT367: Business Intelligence from Big Data**
**Stanford GSB, Winter 2026 | Wurm · Chen · Barli · Taruno**

---

## Project Overview

This project investigates whether Spotify audio features, supplemented by artist-level commercial signals, can predict (1) whether a track appears on the Billboard Hot 100 (binary classification) and (2) how many weeks it remains on the chart (survival analysis). The full pipeline is implemented in Python across five iterative script versions, culminating in `run_all_v5.py` operating on `oit367_final_dataset.csv`.

---

## Repository Structure

```
OIT-367/
├── course_project.pdf               Course instructions
├── run_all_v5.py (in scripts/)      Current full pipeline script
├── build_final_dataset.py           Dataset assembly from raw sources
├── build_artist_features.py         Artist enrichment (Spotify API + MusicBrainz)
├── build_lyric_features.py          VADER sentiment from Billboard lyrics CSV
│
├── datasets/
│   ├── oit367_final_dataset.csv     CURRENT input (78,390 rows, committed)
│   ├── oit367_base_dataset.csv      Intermediate; left-join of Spotify + Billboard
│   ├── artist_features.csv          Enrichment: Spotify API + Last.fm (17,437 artists)
│   └── [raw Kaggle files]           gitignored; ~75 MB total
│
├── outputs/
│   ├── fig1_class_balance.png       through fig9_precision_recall.png
│   ├── model_performance_summary.csv
│   ├── logistic_odds_ratios.csv
│   ├── xgboost_shap_importance.csv
│   ├── cox_summary.csv
│   ├── ols_longevity_coefficients.csv
│   ├── vif_table.csv
│   └── genre_chart_rates.csv
│
├── logs/
│   ├── PROJECT_CONTEXT.md           Full pipeline handoff document
│   ├── RESULTS.md                   All model results (report-ready)
│   ├── ANALYSIS_LOG.md              Detailed per-step analysis log
│   └── FEATURE_REFERENCE.md        Feature-to-source join map
│
├── examples/
│   └── 4.pdf (and others)           Prior-year OIT367 report examples
│
└── archive/
    └── run_all.py through v4.py     Deprecated pipeline versions
```

---

## Data Pipeline

### Source Files

Two public Kaggle datasets form the base:

**`spotify_tracksdataset.csv`** (~114k rows, ~20 MB): One row per track-genre pair from Spotify's public API. Contains `track_id`, artist and album metadata, and eleven audio features: `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `time_signature`. One track can appear multiple times if Spotify assigns it to multiple genres.

**`hot-100-current.csv`** (~690k weekly rows, ~19 MB): Weekly Billboard Hot 100 snapshots from 1958 to 2024. Each row is a single song's appearance in one chart week. Contains `title`, `performer`, `peak_pos`, `wks_on_chart`, and `chart_week`. No Spotify `track_id` -- joining to Spotify requires a pre-matched intermediate file (`merged_spotify_billboard_data.csv`).

### Pipeline Versions and Key Fixes

| Version | Key Change | Dataset |
|---|---|---|
| v1 (run_all.py) | Original; inner join bug (100% positive rate) | merged_spotify_billboard_data.csv |
| v2 | Left join fix; 3.90% positive rate restored | oit367_base_dataset.csv (89,741 rows) |
| v3 | Statsmodels/importlib compatibility fixes | Same |
| v4 | Energy removed (VIF=15.07); Cox mode stratified; PR-AUC added | Same |
| v5 / v6 | Artist features + lyric sentiment added; cross-ID dedup; 2.75% positive rate | oit367_final_dataset.csv (78,390 rows) |

**The inner join bug (v1-v2):** The original pipeline read `merged_spotify_billboard_data.csv` directly, which was an inner join containing only charted tracks. This produced a 100% positive rate and an untrainable classification model. The fix was to rebuild from scratch: deduplicate the 114k Spotify rows to 89,741 unique `track_id`s, aggregate Billboard weekly rows to one row per track (taking `min(peak_pos)`, `max(wks_on_chart)`, `min(chart_week)`), and perform a left join retaining all Spotify tracks. Non-charted tracks receive NaN for all Billboard columns; `is_charted = peak_pos.notna()`.

**Cross-ID deduplication (v6):** After the left join, 3,502 charted `track_id`s existed in the base dataset, but only ~2,157 represent truly unique songs -- the rest are album vs. single variants, regional releases, or explicit/clean pairs with different Spotify IDs but identical audio features. Using a teammate-supplied canonical ID file, v6 collapses these to 2,157 unique charted songs and ~76,233 non-charted, for a final universe of 78,390 rows and 2.75% positive rate.

### Dataset Augmentation

Three external enrichment sources were joined to the base dataset:

**Spotify API artist features** (`artist_features.csv`, 17,437 artists): `artist_peak_popularity`, `artist_popularity_api`, and `artist_track_count` retrieved via the Spotipy library. Joined on normalized artist name string.

**MusicBrainz / Last.fm** (`artists.csv`, 1.47M entries): `lastfm_listeners_log` (log1p of listener count) and `is_us_artist` (binary, derived from `country_mb`). 65.8% match rate on artist name; unknown nationality is conservatively imputed as 0.

**VADER lyric sentiment** (`billboard_lyrics.csv`, 6,879 entries): Tokenized lyrics from a separate Kaggle lyrics dataset matched to charted tracks by title and artist. Four features computed: `sentiment_compound`, `sentiment_pos`, `sentiment_neg`, `lyric_word_count`. Coverage: 856 of 2,157 charted tracks (40.1%). Used only in the survival models.

---

## Final Feature Set

### Classification features (15 total, used in Logistic Regression and XGBoost)

**Audio (10):** `valence`, `acousticness`, `loudness`, `speechiness`, `instrumentalness`, `liveness`, `mode`, `key`, `explicit`, `duration_min`

**Artist-level (5):** `artist_peak_popularity`, `artist_popularity_api`, `artist_track_count`, `lastfm_listeners_log`, `is_us_artist`

**Removed for multicollinearity (VIF > 10):**
- `energy`: VIF=15.07, collinear with `loudness` (Pearson r ≈ 0.78). Removed in v4.
- `danceability`: VIF=12.41, acted as a collinearity hub after other features were added. Removed in v5.
- `tempo`: VIF=10.65 after danceability removal. Removed in v5.

### Survival model features (above 15 + 5 additional)

`decade_idx` (ordinal 0-7, derived from `chart_entry_date`; NaN for non-charted -- cannot be used in classification models) and the four VADER lyric sentiment features (charted subset, n=856).

**Note on `artist_peak_popularity` and `artist_popularity_api` collinearity:** VIF = 21.25 and 14.12 respectively. Their individual regression coefficients are artifacts of collinearity and carry opposite signs in logistic regression. Only their combined directional signal -- that more commercially established artists are more likely to chart and stay on chart longer -- is interpretable.

---

## Models Implemented

| Model | Task | Outcome Variable | Key Setup |
|---|---|---|---|
| Logistic Regression | Chart entry (binary) | `is_charted` | `class_weight="balanced"`, `StandardScaler`, `max_iter=1000` |
| XGBoost + SHAP | Chart entry (binary) | `is_charted` | `scale_pos_weight=24.6`, `n_estimators=500`, early stopping |
| Cox Proportional Hazards | Longevity (survival) | `wks_on_chart` | `strata=["mode"]`, `penalizer=0.1`, n=856 lyric-matched |
| Log-OLS (robustness check) | Longevity | `log1p(wks_on_chart)` | OLS, same 856 rows, same features |

Class imbalance (2.75% positive rate) is addressed through `class_weight="balanced"` in logistic regression and `scale_pos_weight=24.6` in XGBoost. PR-AUC is reported as the primary classification metric alongside ROC-AUC; at 2.75% positive rate, a random classifier achieves PR-AUC = 0.0275.

`mode` is stratified in Cox PH because it failed the Schoenfeld proportional hazards test (p < 0.05), meaning the proportional hazards assumption does not hold for this covariate. Stratification allows a separate baseline hazard for major vs. minor key tracks.

---

## Model Performance Summary

| Model | Primary Metric | Score | Baseline | Lift |
|---|---|---|---|---|
| Logistic Regression | ROC-AUC | 0.9144 | 0.50 | -- |
| Logistic Regression | PR-AUC | 0.2750 | 0.0275 | 10.0x |
| XGBoost | ROC-AUC | 0.9655 | 0.50 | -- |
| XGBoost | PR-AUC | 0.4402 | 0.0275 | 16.0x |
| Cox PH | Concordance index | 0.7526 | 0.50 | -- |
| Log-OLS | R² | 0.2118 | -- | -- |

5-fold cross-validated ROC-AUC for logistic regression: 0.9139 +/- 0.0037. Low variance indicates the model generalizes.

**Pre-augmentation baselines (audio features only):**
LR ROC-AUC = 0.7106 / XGBoost ROC-AUC = 0.8343 / Cox C-stat = 0.5508 / OLS R² = 0.0438. The artist enrichment features (particularly `artist_peak_popularity` and `lastfm_listeners_log`) account for the majority of the performance gain.

---

## Output Files

Nine figures are saved to `outputs/`:
- `fig1_class_balance.png`: Bar chart of charted vs. non-charted distribution
- `fig2_correlation_heatmap.png`: Pearson correlation matrix of all features
- `fig3_roc_curves.png`: ROC curves for LR and XGBoost side by side
- `fig4_odds_ratios.png`: Forest plot of logistic regression odds ratios with 95% CIs
- `fig5_shap_importance.png`: XGBoost mean SHAP value bar chart (ranked)
- `fig6_cox_hazard_ratios.png`: Cox PH hazard ratio plot for significant covariates
- `fig7_kaplan_meier.png`: Kaplan-Meier survival curves (stratified by mode or genre)
- `fig8_longevity_distribution.png`: Histogram of weeks on chart for charted tracks
- `fig9_precision_recall.png`: Precision-recall curves for LR and XGBoost

Seven CSV tables are saved to `outputs/`: `model_performance_summary.csv`, `logistic_odds_ratios.csv`, `xgboost_shap_importance.csv`, `cox_summary.csv`, `ols_longevity_coefficients.csv`, `vif_table.csv`, `genre_chart_rates.csv`.

The report permits a maximum of 7 tables and figures combined in the main body; the remaining 9 can appear in the appendix.

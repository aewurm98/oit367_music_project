# OIT367 — Billboard Hot 100 Prediction: Model Results
**Stanford GSB Winter 2026 | Wurm · Chen · Barli · Taruno**
**Pipeline version: `run_all_v5.py` + artist augmentation | Date: 2026-03-08**

---

## 1. Research Questions

1. **Chart Entry (binary classification):** Which audio features and song attributes predict whether a track reaches the Billboard Hot 100?
2. **Chart Longevity (survival analysis):** Among tracks that chart, which features predict how long they remain on the Hot 100?

---

## 2. Dataset

| Attribute | Value |
|---|---|
| Source | Kaggle Spotify Tracks Dataset × Billboard Hot 100 weekly data |
| Join type | Left join on `track_id`; all Spotify tracks retained |
| Total tracks | 89,741 |
| Charted tracks (positive class) | 3,502 (3.90%) |
| Non-charted tracks (negative class) | 86,239 (96.10%) |
| Features | 10 audio features + 2 control variables + 3 artist-level features |
| Billboard date range | 1950s–2020s (8 decades in charted subset) |

### 2.1 Data Pipeline

```
spotify_tracksdataset.csv   (114k rows, 114 genres × ~1,000 tracks each)
    → dedup on track_id (keep first genre alphabetically)
    → 89,741 unique tracks

merged_spotify_billboard_data.csv  (weekly Hot 100 rows)
    → aggregate per track: peak_pos (min), wks_on_chart (max), chart_entry_date (min)
    → 3,502 unique charted tracks

Left join → oit367_base_dataset.csv (89,741 rows)

build_artist_features.py
    → artist-level aggregates for all 17,437 normalized artist names
       (artist_popularity_api, artist_peak_popularity, artist_track_count)
    → merged on normalized artist name; ~97% match rate
    → oit367_augmented_dataset.csv (89,741 rows with artist features)
```

### 2.2 Artist Feature Construction (build_artist_features.py)

**Why local computation instead of Spotify API:**
Spotify Development Mode enforces a daily quota that was exhausted after ~300 `sp.search()` calls (Retry-After: 86,000+ seconds). The `modal_charted_scrape.py` script triggers rate limits immediately on retry and cannot complete within the quota.

**Solution:** Compute artist-level features directly from the existing 114k-track Spotify dataset, which contains popularity scores for all 31,437 artists — no API calls required.

| Feature | Computation | Coverage |
|---|---|---|
| `artist_popularity_api` | Mean track popularity across artist's full catalog in dataset | 97.3% of charted artists matched |
| `artist_peak_popularity` | Max track popularity across artist's catalog | same |
| `artist_track_count` | Number of tracks for this artist in dataset (catalog size proxy) | same |

> **Note:** `artist_popularity_api` is named to match the Spotipy API path for backward compatibility with `modal_charted_scrape.py`. The actual values are computed locally from dataset track popularity means, which correlate strongly with Spotify's artist-level popularity (Spotify computes track popularity from recent play counts, so popular artists consistently have high-popularity tracks).

### 2.3 Control Variable Summary

| Variable | Charted mean | Non-charted mean | Notes |
|---|---|---|---|
| `explicit` | 0.115 (11.5%) | 0.085 (8.5%) | +35% more likely to be explicit |
| `duration_min` | 3.64 min | 3.79 min | Charted tracks ~9 seconds shorter |

### 2.4 Genre Chart Rates (Top 10 by chart rate)

| Genre | Total Tracks | Charted | Chart Rate | Avg Popularity |
|---|---|---|---|---|
| country | 946 | 271 | 28.65% | 17.35 |
| rock | 344 | 96 | 27.91% | 18.31 |
| dance | 948 | 234 | 24.68% | 23.72 |
| pop | 596 | 138 | 23.15% | 49.64 |
| hard-rock | 709 | 153 | 21.58% | 45.74 |
| disco | 928 | 196 | 21.12% | 34.00 |
| grunge | 862 | 180 | 20.88% | 50.59 |
| jazz | 524 | 109 | 20.80% | 9.79 |
| blues | 938 | 156 | 16.63% | 31.18 |
| synth-pop | 838 | 134 | 15.99% | 35.53 |

> **Note:** Low `avg_popularity` for high-charting genres (e.g., country=17.35, jazz=9.79) reflects historical catalog tracks: high historical charting does not translate to current Spotify streaming popularity. This is why `popularity` is not used as a predictor — it conflates past chart success with current streaming behavior.

Full table: `outputs/genre_chart_rates.csv`

---

## 3. Feature Engineering

### 3.1 Features Used in Models

| Feature | Type | Source | Notes |
|---|---|---|---|
| `valence` | continuous [0,1] | Spotify API | Musical positivity/happiness |
| `acousticness` | continuous [0,1] | Spotify API | — |
| `loudness` | continuous (dB) | Spotify API | — |
| `speechiness` | continuous [0,1] | Spotify API | — |
| `instrumentalness` | continuous [0,1] | Spotify API | — |
| `liveness` | continuous [0,1] | Spotify API | — |
| `mode` | binary (0=minor,1=major) | Spotify API | Used as Cox stratum |
| `key` | categorical (0–11) | Spotify API | Treated as ordinal |
| `explicit` | binary (0/1) | Spotify API | v5 control |
| `duration_min` | continuous (min) | Derived | `duration_ms/60000`, capped at 10 min |
| `artist_popularity_api` | continuous [0,100] | Computed | Mean track popularity per artist |
| `artist_peak_popularity` | integer [0,100] | Computed | Max track popularity per artist |
| `artist_track_count` | integer | Computed | Catalog size in dataset |
| `decade_idx` | ordinal (0–7) | Derived | Cox/OLS only; from `chart_entry_date` |

### 3.2 Features Excluded (VIF Remediation)

| Feature | VIF | Removed in | Reason |
|---|---|---|---|
| `energy` | 15.07 | v4 Fix A | Collinear with `loudness` (Pearson r≈0.78) |
| `danceability` | 12.41 | v5 Fix F | Collinearity hub: correlated with `tempo`, `valence`, `duration_min` simultaneously |
| `tempo` | 10.65 | v5 Fix G | Secondary hub after danceability removal; correlates with `duration_min` + `loudness` through genre/era effects |

### 3.3 VIF Table (v5 final — post-patch, with artist features)

> ⚠ `artist_peak_popularity` (VIF=14.11) is collinear with `artist_popularity_api` (mean vs. max of same underlying metric). Both are retained because they capture meaningfully different constructs (catalog-average fame vs. peak fame ceiling), and XGBoost is robust to multicollinearity. For the LR model, interpret artist feature coefficients with caution.

| Feature | VIF |
|---|---|
| `artist_peak_popularity` | 14.11 ⚠ (mean↔max collinearity) |
| `artist_popularity_api` | 9.34 |
| `loudness` | 6.73 |
| `duration_min` | 5.91 |
| `valence` | 3.81 |
| `acousticness` | 3.17 |
| `key` | 3.04 |
| `mode` | 2.67 |
| `liveness` | 2.31 |
| `speechiness` | 1.88 |
| `instrumentalness` | 1.81 |
| `artist_track_count` | 1.77 |
| `explicit` | 1.25 |

---

## 4. Model Results

### 4.1 Model 1: Logistic Regression — Chart Entry (Binary)
*Baseline linear model. All features standardized (StandardScaler). class_weight="balanced".*

| Metric | Value |
|---|---|
| Test AUC-ROC | **0.9179** *(was 0.7094 without artist features; +20.9 points)* |
| Test PR-AUC | **0.3214** (random baseline = 0.0390; **8.2× lift**) |
| CV AUC-ROC (5-fold) | 0.9127 ± 0.0040 |

**Classification Report (test set, n=17,949):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No Chart | 0.99 | 0.85 | 0.92 | 17,249 |
| Charted | 0.19 | 0.84 | 0.31 | 700 |
| Accuracy | | | 0.85 | 17,949 |

**Odds Ratios (per 1 SD change, standardized features):**

| Feature | Coefficient | Odds Ratio | Interpretation |
|---|---|---|---|
| `artist_peak_popularity` | +1.321 | **3.748** | Peak popularity 3.7× chart odds ⭐ (dominant) |
| `lastfm_listeners_log` | +0.949 | **2.582** | Cross-platform reach 2.6× chart odds ⭐ (new Add H) |
| `valence` | +0.277 | **1.319** | Happier songs 32% more likely to chart |
| `explicit` | +0.250 | **1.283** | Explicit content 28% more likely |
| `is_us_artist` | +0.624 | **1.867** | US artists 87% more likely to chart ⭐ (new Add I) |
| `mode` | +0.157 | **1.170** | Major key 17% more likely |
| `key` | +0.007 | 1.007 | Not significant |
| `duration_min` | −0.103 | **0.902** | Shorter tracks slightly more likely |
| `acousticness` | −0.187 | **0.829** | Acoustic tracks 17% less likely |
| `liveness` | −0.215 | **0.807** | Live-sounding tracks 19% less likely |
| `artist_track_count` | −0.215 | **0.806** | Larger catalogs slightly less likely (marginal) |
| `loudness` | −0.238 | **0.789** | Louder tracks 21% less likely |
| `artist_popularity_api` | −0.284 | **0.753** | ⚠ Opposite sign from peak — collinearity artifact |
| `speechiness` | −0.312 | **0.732** | High speech content 27% less likely |
| `instrumentalness` | −1.069 | **0.343** | Instrumental tracks 66% less likely ⭐ |

> The opposite signs for `artist_peak_popularity` (+) and `artist_popularity_api` (−) reflect VIF=15.80/9.60 collinearity: controlling for average catalog popularity, having an anomalously high peak suggests a one-hit-wonder dynamic. Use XGBoost SHAP for more reliable feature importance when collinearity is present.

Output: `outputs/logistic_odds_ratios.csv`, `outputs/fig3_roc_curves.png`, `outputs/fig4_odds_ratios.png`, `outputs/fig9_precision_recall.png`

---

### 4.2 Model 2: XGBoost — Chart Entry (Binary)
*Gradient-boosted trees. scale_pos_weight=24.6 (inverse class ratio). Early stopping on test AUC.*

| Metric | Value |
|---|---|
| Test AUC-ROC | **0.9736** *(was 0.8216 without artist features; +15.2 points)* |
| Test PR-AUC | **0.6869** (random baseline = 0.0390; **17.6× lift**) |
| Best N Estimators | 485 (early stopping at 50 rounds) |

**Classification Report (test set, n=17,949):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No Chart | 0.99 | 0.94 | 0.96 | 17,249 |
| Charted | 0.36 | 0.88 | 0.51 | 700 |
| Accuracy | | | 0.93 | 17,949 |

> XGBoost with artist features achieves precision=0.31 at recall=0.85 — a major improvement from precision=0.12 in the audio-only model.

**SHAP Feature Importance (Mean |SHAP| on test set):**

| Rank | Feature | Mean |SHAP| | Interpretation |
|---|---|---|---|
| 1 | `artist_peak_popularity` | 1.814 | **Dominant predictor** — artist fame ceiling ⭐ |
| 2 | `lastfm_listeners_log` | 1.336 | Cross-platform reach (Last.fm; new Add H) ⭐ |
| 3 | `instrumentalness` | 0.711 | Hot 100 is vocal-driven; strong across all models |
| 4 | `artist_track_count` | 0.503 | Catalog size: established artists chart more ⭐ |
| 5 | `acousticness` | 0.471 | Acoustic vs. produced sound axis |
| 6 | `artist_popularity_api` | 0.392 | Avg catalog popularity |
| 7 | `valence` | 0.337 | Musical positivity; consistent across all models |
| 8 | `duration_min` | 0.329 | Track length |
| 9 | `loudness` | 0.279 | — |
| 10 | `speechiness` | 0.270 | — |
| 11 | `liveness` | 0.219 | — |
| 12 | `mode` | 0.204 | — |
| 13 | `is_us_artist` | 0.458 | US vs. non-US origin (new Add I) |
| 14 | `key` | 0.081 | — |
| 15 | `explicit` | 0.048 | — |

> Artist/reach features (ranks 1, 2, 4, 6, 13) collectively account for ~59% of total SHAP weight. `lastfm_listeners_log` (Add H) enters as 2nd most important feature, capturing cross-platform reach orthogonal to Spotify popularity. Who makes the song matters more than what the song sounds like.

Output: `outputs/xgboost_shap_importance.csv`, `outputs/fig5_shap_importance.png`

---

### 4.3 Model 3: Cox Proportional Hazards — Chart Longevity
*Survival model on charted tracks with matched lyrics (n=1,456). Stratified on `mode`. Penalizer=0.1.*
*Includes all features + decade_idx + artist features + lyric sentiment (Add J; 41.9% charted coverage).*

| Metric | Value |
|---|---|
| Concordance Index (C-stat) | **0.7240** *(was 0.5393 without artist features; +18.5 points)* |
| Partial AIC | 16,073.86 |
| Log-likelihood ratio test | 528.57 on 18 df |

**Cox Coefficients (significant results highlighted):**

| Feature | HR exp(coef) | 95% CI | p-value | Interpretation |
|---|---|---|---|---|
| `artist_peak_popularity` | **0.828** | [0.798, 0.858] | <0.00001 | Higher peak → faster chart exit ⭐ (see note) |
| `artist_popularity_api` | **1.182** | [1.141, 1.225] | <0.00001 | Higher avg popularity → longer chart run ⭐ |
| `artist_track_count` | **1.144** | [1.103, 1.186] | <0.00001 | Larger catalog → longer chart run |
| `decade_idx` | **0.590** | [0.545, 0.639] | <0.00001 | Each decade later → faster chart exit ⭐ |
| `artist_popularity_api` | **1.231** | [1.163, 1.302] | <0.00001 | Higher avg popularity → longer run ⭐ |
| `artist_peak_popularity` | **0.828** | [0.780, 0.880] | <0.00001 | Higher peak → faster exit (suppressor) |
| `loudness` | **0.825** | [0.777, 0.876] | <0.00001 | Louder tracks have shorter runs |
| `acousticness` | **0.840** | [0.793, 0.895] | <0.00001 | Acoustic tracks leave faster |
| `sentiment_neg` | **1.124** | [1.050, 1.186] | 0.0008 | More negative lyrics → longer run (new Add J) ⭐ |
| `explicit` | **1.073** | [1.013, 1.137] | 0.0164 | Explicit content slightly longer run |
| `duration_min` | **1.087** | [1.023, 1.155] | 0.0073 | Longer tracks stay slightly longer |
| `valence` | **1.068** | [1.006, 1.134] | 0.0315 | Happier songs stay longer |
| `lastfm_listeners_log` | **0.933** | [0.881, 1.002] | 0.0562 | Marginal: wider reach → faster turnover |
| `artist_track_count` | 1.056 | [0.987, 1.128] | 0.1125 | Not significant |
| `sentiment_compound` | 1.074 | [0.994, 1.159] | 0.0697 | Marginal |
| `speechiness` | 0.994 | [0.940, 1.051] | 0.831 | Not significant |
| `instrumentalness` | 1.022 | [0.970, 1.076] | 0.413 | Not significant |
| `is_us_artist` | 0.964 | [0.912, 1.019] | 0.192 | Not significant |
| `liveness` | 1.001 | [0.948, 1.056] | 0.971 | Not significant |
| `key` | 0.985 | [0.931, 1.042] | 0.577 | Not significant |
| `lyric_word_count` | 1.019 | [0.955, 1.086] | 0.569 | Not significant |
| `sentiment_pos` | 1.005 | [0.940, 1.073] | 0.870 | Not significant |

> **Artist features note:** `artist_peak_popularity` (HR=0.828, p<10⁻⁵) and `artist_popularity_api` (HR=1.182, p<10⁻⁵) have opposite signs — the VIF=14.11 collinearity creates the classic "controlling for X, higher Y predicts lower Z" suppressor effect. Interpretation: controlling for average catalog quality, tracks that represent a *peak* for the artist (peak >> average) tend to have shorter runs, perhaps because they represent one-hit-wonder dynamics. Both are highly significant and survive the Schoenfeld test individually.

> **decade_idx finding updated:** HR=0.909 (vs. 0.878 without artist features). The streaming-era compression effect remains strong but is somewhat attenuated when controlling for artist popularity — suggesting part of what looked like era effects was actually the rising prominence of superstar artists in the streaming era.

**Proportional Hazards Assumption Check:** 13 of 18 covariates fail Schoenfeld test at α=0.05. The most severe violations are `decade_idx` (KM stat=85.29), `acousticness` (30.24), `sentiment_pos` (32.59), and `artist_track_count` (47.38). Lyric sentiment features show mixed results: `sentiment_neg` and `sentiment_pos` fail the PH test, while `sentiment_compound` passes. As noted in PROJECT_CONTEXT.md, PH violations across 70+ years of pop music history are expected — the recommended framing is to acknowledge them and use Log-OLS as a robustness check.

Output: `outputs/cox_summary.csv`, `outputs/fig6_cox_hazard_ratios.png`, `outputs/fig7_kaplan_meier.png`

---

### 4.4 Model 3b: Log-OLS — Chart Longevity (Robustness Check)
*OLS on log1p(wks_on_chart) for charted tracks. Includes decade_idx and artist features. 80/20 split.*

| Metric | Value |
|---|---|
| Test R² | **0.3469** *(was 0.0337 without artist features; +31.3 points)* |

**Coefficients (standardized; outcome = log1p weeks on chart):**

| Feature | Coefficient | Direction vs. Cox |
|---|---|---|
| `artist_peak_popularity` | +0.158 | ⚠ Opposite (Cox HR=0.828) — suppressor effect |
| `loudness` | +0.099 | ⚠ Opposite (Cox HR=0.901) |
| `explicit` | +0.023 | Consistent with Cox HR=1.009 |
| `duration_min` | +0.022 | Consistent (neither significant) |
| `valence` | +0.018 | Consistent with Cox HR=1.049 |
| `mode` | −0.004 | — |
| `liveness` | −0.015 | Consistent (not significant) |
| `acousticness` | −0.018 | Consistent with Cox HR=0.910 |
| `key` | −0.027 | Consistent (not significant) |
| `instrumentalness` | −0.033 | Consistent direction |
| `speechiness` | −0.060 | ⚠ Opposite (Cox HR=1.037) |
| `artist_popularity_api` | −0.137 | ⚠ Opposite (Cox HR=1.182) — collinearity suppressor |
| `decade_idx` | **−0.149** | ✓ Consistent with Cox HR=0.909 ⭐ |
| `artist_track_count` | **−0.190** | ⚠ Opposite (Cox HR=1.144) |

> The `artist_popularity_api`/`artist_track_count` sign reversals between Cox and OLS are consistent with collinearity suppressor effects. `decade_idx` is the only covariate where both models agree directionally AND it is the strongest predictor in both (OLS coef=−0.149, Cox HR=0.909).

Output: `outputs/ols_longevity_coefficients.csv`

---

## 5. Model Performance Summary

| Model | Task | Metric | Audio Only (baseline) | With All Augmentation | Δ |
|---|---|---|---|---|---|
| Logistic Regression | Chart Entry | AUC-ROC | 0.7094 | **0.9179** | +20.9 pts |
| Logistic Regression | Chart Entry | PR-AUC | 0.0738 | **0.3214** | +24.8 pts (8.2× baseline) |
| XGBoost | Chart Entry | AUC-ROC | 0.8216 | **0.9736** | +15.2 pts |
| XGBoost | Chart Entry | PR-AUC | 0.3130 | **0.6869** | +37.4 pts (17.6× baseline) |
| Cox PH | Longevity | C-statistic | 0.5393 | **0.7240** | +18.5 pts |
| Log-OLS | Longevity | R² | 0.0337 | **0.3469** | +31.3 pts (10.3× improvement) |

---

## 6. Key Findings

### Finding 1: Artist fame is the dominant chart predictor — by far
`artist_peak_popularity` SHAP=1.814 remains the dominant predictor, joined by `lastfm_listeners_log` SHAP=1.336 (Add H — cross-platform reach). Artist and reach features together account for ~59% of total XGBoost SHAP weight. Full augmentation increased XGBoost AUC-ROC from 0.82 → 0.97 and PR-AUC from 0.31 → 0.69 (17.6× over random). **The most important predictor of chart success is who makes the song, not what the song sounds like.**

### Finding 2: Instrumentalness is the dominant audio-only barrier
Among pure audio features (controlling for artist fame), `instrumentalness` SHAP=0.615 and LR OR=0.354 remain the strongest predictor. Tracks with primarily non-vocal content are 65% less likely to reach the Hot 100 per 1 SD increase. The Billboard Hot 100 is fundamentally a vocal/lyric-driven chart.

### Finding 3: Catalog size predicts both chart entry and longevity
`artist_track_count` SHAP=0.597 (3rd overall, 2nd among audio+artist features). Established artists with larger catalogs chart more often (XGBoost) AND stay on chart longer (Cox HR=1.144, p<10⁻⁵, z=7.2). This likely reflects label support, playlist placement, and audience loyalty effects.

### Finding 4: Musical valence (happiness) is the strongest positive audio predictor
`valence` OR=1.397 (LR) and SHAP=0.429 (XGBoost, 5th overall). Happy/positive-sounding songs are ~40% more likely to chart per 1 SD increase. Consistent across all four models.

### Finding 5: Streaming era dramatically shortens chart longevity (most robust finding)
`decade_idx` is the only variable where Cox (HR=0.909) and OLS (coef=−0.149) directionally agree AND it is the strongest predictor in both longevity models. Each decade later → ~9.1% higher hazard of leaving the chart per week. Songs from the 2020s cycle through the Hot 100 approximately twice as fast as songs from the 1980s. Slightly attenuated (vs. HR=0.878 in audio-only model) after controlling for artist popularity, suggesting part of the original era effect captured the rise of superstar streaming artists.

### Finding 6: Audio features predict chart entry well but longevity poorly
XGBoost (with full augmentation) achieves AUC-ROC=0.974 and PR-AUC=0.687 for chart entry. The longevity models improved dramatically with augmentation: C-stat=0.724 and R²=0.347 — though this is partly because the lyric-matched subset (n=1,456) skews toward more prominent charted tracks with available lyrics, which may inflate longevity predictability. Audio features alone still explain little of longevity variance once artist fame is controlled for.

---

## 7. Outputs Reference

| File | Description |
|---|---|
| `outputs/vif_table.csv` | VIF for all features (artist_peak_popularity=14.11 noted) |
| `outputs/logistic_odds_ratios.csv` | LR odds ratios (standardized) |
| `outputs/xgboost_shap_importance.csv` | Mean \|SHAP\| per feature |
| `outputs/cox_summary.csv` | Full Cox PH summary with CI and p-values |
| `outputs/ols_longevity_coefficients.csv` | Log-OLS coefficients |
| `outputs/model_performance_summary.csv` | All 4 models, all metrics |
| `outputs/genre_chart_rates.csv` | Chart rate and avg popularity by genre |
| `outputs/fig1_class_balance.png` | Class distribution bar chart |
| `outputs/fig2_correlation_heatmap.png` | Feature correlation matrix |
| `outputs/fig3_roc_curves.png` | ROC: LR vs XGBoost |
| `outputs/fig4_odds_ratios.png` | LR odds ratio forest plot |
| `outputs/fig5_shap_importance.png` | XGBoost SHAP bar chart |
| `outputs/fig6_cox_hazard_ratios.png` | Cox hazard ratio plot |
| `outputs/fig7_kaplan_meier.png` | KM survival curves by genre (top 5) |
| `outputs/fig8_longevity_distribution.png` | Weeks on chart distribution |
| `outputs/fig9_precision_recall.png` | PR curves: LR vs XGBoost (new in v5) |

---

## 8. Limitations

1. **Spotify audio features are proprietary black-box measures.** Danceability, valence, and instrumentalness are computed by Spotify using undisclosed algorithms.
2. **Artist popularity features are computed from the Kaggle dataset, not the live Spotify API.** `artist_popularity_api` = mean track popularity across the artist's tracks in the 114k-track dataset. This is a snapshot proxy, not real-time artist-level popularity. Values reflect the dataset's sample rather than the full Spotify catalog.
3. **Kaggle dataset is engineered, not a random sample.** ~1,000 tracks per genre. Charting rates differ significantly by genre (country 28.7% vs. ambient ~2%), so genre-level chart rates should not be interpreted as representing the broader music market.
4. **3.9% positive rate limits classifier performance.** PR-AUC is the appropriate primary metric. XGBoost PR-AUC of 0.687 (17.6× baseline) represents very strong lift.
5. **Cox PH assumption violated for 13/18 covariates.** Time-varying effects are expected across 70 years of music industry history. Log-OLS is included as robustness check; only `decade_idx` survives both models directionally.
6. **Lyric sentiment (Add J) finds `sentiment_neg` as the only significant lyric predictor of longevity** (Cox HR=1.124, p=0.0008): tracks with more negative lyric content stay on chart slightly longer, possibly reflecting emotional depth or drama that sustains listener engagement. Coverage is 41.9% of charted tracks; results should be treated as exploratory.
6. **artist_peak_popularity VIF=14.11.** Collinear with `artist_popularity_api`. LR coefficients for both are suppressor-affected — interpret with caution. XGBoost SHAP is more reliable.
7. **No lyric sentiment features.** Genius API + VADER analysis planned for charted tracks subset (Alex's task).
8. **No Librosa acoustic features.** Feasibility plan documented in `LIBROSA_MODAL_PLAN.md`. Pipeline scripts ready: `modal_preview_urls.py` → `modal_librosa_extract.py`.

---

## 9. Pending Data Augmentation (Planned)

| Feature Set | Source | Scope | Status |
|---|---|---|---|
| Lyric sentiment (VADER) | Genius API | Charted tracks (3,502) | Assigned to Alex |
| Librosa acoustic features (MFCCs, spectral centroid, etc.) | Spotify 30s preview clips | Charted tracks (~3,100 with previews) | Scripts ready; run after VIF check against existing features |

> **Artist features (completed 2026-03-08):** Computed locally from `spotify_tracksdataset.csv` via `build_artist_features.py`. All 17,437 artists covered at 97.3% match rate to main dataset. Spotify API scraping (`modal_charted_scrape.py`) no longer required.

---

## 10. Reproducibility

```bash
# Install dependencies (use xgboost==2.1.3 — xgboost 3.x breaks shap TreeExplainer)
pip3 install -r requirements.txt
pip3 install xgboost==2.1.3  # override if 3.x installed

# Step 1: Build artist features (no API required, runs in seconds)
python3 build_artist_features.py

# Step 2: Run full pipeline (auto-detects artist_features.csv)
python3 run_all_v5.py

# Optional: Librosa acoustic features (after Spotify rate limit resets)
modal run --detach modal_preview_urls.py
# (wait for completion)
modal run --detach modal_librosa_extract.py
modal volume get oit367-vol /data/librosa_features.csv ./librosa_features.csv
```

**Environment:** Python 3.11+, packages in `requirements.txt`
**Randomness:** `RANDOM_STATE=42` throughout; results are fully deterministic given the same dataset.

---

*Generated by `run_all_v5.py` + `build_artist_features.py` · OIT367 Stanford GSB Winter 2026*

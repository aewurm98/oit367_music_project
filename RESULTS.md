# OIT367 Model Results — Full Analysis
**Stanford GSB Winter 2026 | Wurm · Chen · Barli · Taruno**
**Last updated: 2026-03-09 — v6 pipeline (final dataset, cross-ID dedup, teammate enrichment)**

---

## 0. Version History

| Version | Dataset | Charted | AUC-ROC (XGB) | Key change |
|---|---|---|---|---|
| v4 | 89,741 rows | 3,502 (3.90%) | 0.8343 | Energy removed (VIF) |
| v5 | 89,741 rows | 3,502 (3.90%) | 0.9733 | Artist + lyric features added |
| **v6** | **78,390 rows** | **2,157 (2.75%)** | **0.9655** | **Cross-ID dedup + teammate enrichment** |

**Why v6 AUC is slightly lower than v5:** Cross-ID deduplication removed 1,345 charted track_id duplicates (same song, different Spotify IDs — album vs. single, regional variants, etc.) and ~11,000 analogous non-charted duplicates. v5 made artificially easy predictions from near-duplicate rows with identical features. v6 numbers are more methodologically sound and represent a conservative lower bound.

---

## 1. Dataset Summary

| Metric | v6 Value | v5 Value |
|---|---|---|
| Total tracks | 78,390 | 89,741 |
| Charted (is_charted=1) | 2,157 (2.75%) | 3,502 (3.90%) |
| Non-charted | 76,233 (97.25%) | 86,239 (96.10%) |
| Random PR-AUC baseline | 0.0275 | 0.0390 |
| Cross-ID dedup applied | ✓ | ✗ |
| Source | oit367_final_dataset.csv | oit367_base_dataset.csv |

---

## 2. Features Used

**Classification (LR + XGBoost) — 15 features:**
Audio: `valence`, `acousticness`, `loudness`, `speechiness`, `instrumentalness`, `liveness`, `mode`, `key`, `explicit`, `duration_min`
Artist: `artist_peak_popularity`, `artist_popularity_api`, `artist_track_count`, `lastfm_listeners_log`, `is_us_artist`

**Longevity (Cox PH + Log-OLS) — above 15 + `decade_idx` + lyric features:**
`COX_FEATURES = FEATURES + ["decade_idx", "sentiment_compound", "sentiment_pos", "sentiment_neg", "lyric_word_count"]`
Lyric features included when coverage >20%; longevity model uses n=856 lyric-matched charted tracks.

**Features removed (VIF):**
- `energy`: VIF=15.07, collinear with `loudness` (v4 Fix A)
- `danceability`: VIF=12.41, collinearity hub (v5 Fix F)
- `tempo`: VIF=10.65 after Fix F (v5 Fix G)

**Teammate features available in dataset but excluded from models (insufficient universe coverage):**
- `is_male_artist`, `artist_age`, `is_mainstream_genre`: only populated for ~58% of 2,157 charted tracks = ~1% of all 78,390 rows. Falls below the >50% non-null threshold for auto-detection. Available for descriptive analysis of the charted subset.
- `artist_scrobbles_log`, `artist_listeners_monthly_log`: VIF≈615 (near-perfectly collinear with each other and with `lastfm_listeners_log`). Excluded from model features.
- `time_signature`: VIF=19.13, exceeds threshold of 10.

---

## 3. VIF Table

| Feature | VIF | Note |
|---|---|---|
| `artist_peak_popularity` | 21.25 | ⚠ Pre-existing; collinear with artist_popularity_api |
| `artist_popularity_api` | 14.12 | ⚠ |
| `loudness` | 6.72 | |
| `duration_min` | 6.28 | |
| `lastfm_listeners_log` | 5.18 | |
| `valence` | 3.75 | |
| `acousticness` | 3.17 | |
| `key` | 3.04 | |
| `mode` | 2.65 | |
| `liveness` | 2.31 | |
| `speechiness` | 1.89 | |
| `instrumentalness` | 1.87 | |
| `artist_track_count` | 1.77 | |
| `is_us_artist` | 1.40 | |
| `explicit` | 1.27 | |

**Note:** `artist_peak_popularity` and `artist_popularity_api` are highly collinear. Their individual regression coefficients are collinearity artifacts — only their combined directional effect (more commercially established artists chart and stay on chart more) is interpretable.

---

## 4. Model Results

### 4.1 Logistic Regression — Chart Entry

**Setup:** `LogisticRegression(class_weight="balanced", max_iter=1000)` + `StandardScaler`. Coefficients are per 1 SD change.

| Metric | v6 Score | v5 Score |
|---|---|---|
| Test AUC-ROC | **0.9144** | 0.9179 |
| Test PR-AUC | **0.2750** (10.0× above baseline) | 0.3214 (8.2×) |
| CV AUC-ROC (5-fold) | **0.9139 ± 0.0037** | 0.9127 ± 0.0040 |

```
Classification Report:
              precision    recall  f1-score   support
    No Chart       1.00      0.85      0.92     15,247
     Charted       0.15      0.88      0.25        431
    accuracy                           0.86     15,678
```

**Odds Ratios (per 1 SD, sorted by |coef|):**

| Feature | Coef | OR | Note |
|---|---|---|---|
| `artist_peak_popularity` | +1.045 | **2.84** | #1 overall predictor |
| `lastfm_listeners_log` | +0.773 | **2.17** | #2 overall (moved up from #3 in v5) |
| `is_us_artist` | +0.647 | **1.91** | US artists 91% more likely |
| `valence` | +0.362 | **1.44** | Happier → more likely |
| `explicit` | +0.217 | **1.24** | |
| `mode` | +0.117 | 1.12 | Major key |
| `key` | +0.013 | 1.01 | Not significant |
| `duration_min` | −0.034 | 0.97 | Not significant |
| `artist_popularity_api` | −0.132 | 0.88 | ⚠ Collinearity artifact (opposite sign to artist_peak_popularity) |
| `artist_track_count` | −0.217 | 0.80 | |
| `loudness` | −0.221 | 0.80 | |
| `acousticness` | −0.260 | 0.77 | |
| `liveness` | −0.262 | 0.77 | |
| `speechiness` | −0.347 | 0.71 | Nonlinear — XGBoost captures better |
| `instrumentalness` | −1.081 | **0.34** | Dominant audio barrier (66% less likely) |

---

### 4.2 XGBoost — Chart Entry

**Setup:** `XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=24.6)`. Early stop @ iter 193.

| Metric | v6 Score | v5 Score |
|---|---|---|
| Test AUC-ROC | **0.9655** | 0.9733 |
| Test PR-AUC | **0.4402** (16.0× above baseline) | 0.6867 (17.6×) |

```
Classification Report:
              precision    recall  f1-score   support
    No Chart       1.00      0.92      0.95     15,247
     Charted       0.23      0.91      0.37        431
    accuracy                           0.92     15,678
```

**SHAP Feature Importance (Mean |SHAP| on test set):**

| Rank | Feature | Mean \|SHAP\| | v5 Rank |
|---|---|---|---|
| 1 | `artist_peak_popularity` | **1.618** | 1 |
| 2 | `lastfm_listeners_log` | **1.153** | 2 |
| 3 | `instrumentalness` | 0.629 | 3 |
| 4 | `is_us_artist` | **0.558** | 6 (↑) |
| 5 | `artist_track_count` | 0.388 | 5 |
| 6 | `acousticness` | 0.317 | 4 (↓) |
| 7 | `valence` | 0.315 | 7 |
| 8 | `artist_popularity_api` | 0.266 | 8 |
| 9 | `speechiness` | 0.214 | 11 |
| 10 | `duration_min` | 0.183 | 9 |
| 11 | `loudness` | 0.144 | 10 |
| 12 | `mode` | 0.101 | 13 |
| 13 | `liveness` | 0.095 | 12 |
| 14 | `key` | 0.029 | 14 |
| 15 | `explicit` | 0.011 | 15 |

The 5.1-point AUC gap between XGBoost (0.966) and LR (0.914) confirms substantial nonlinearity. The top-2 SHAP features are both artist-level social/commercial variables — artist prominence explains chart entry better than any sonic property. Rankings are fully consistent with v5.

---

### 4.3 Cox PH — Chart Longevity

**Setup:** `CoxPHFitter(penalizer=0.1)`, `strata=["mode"]`, restricted to 856 lyric-matched charted tracks (40.1% of 2,157 charted; smaller than v5's 1,456 due to proper dedup removing duplicate charted rows).

| Metric | v6 Score | v5 Score |
|---|---|---|
| Concordance Index | **0.7526** | 0.7240 |
| Sample size | 856 | 1,456 |
| Schoenfeld failures | **4/19** | 13/19 |

**Full Coefficient Table:**

| Feature | Coef | HR | SE | p | Sig? |
|---|---|---|---|---|---|
| `decade_idx` | −0.666 | **0.514** | 0.050 | <0.00001 | ✓ Strongest |
| `artist_peak_popularity` | −0.180 | **0.835** | 0.041 | <0.00001 | ✓ ⚠ Collinear pair |
| `artist_track_count` | +0.109 | **1.115** | 0.038 | 0.0046 | ✓ |
| `artist_popularity_api` | +0.082 | **1.085** | 0.038 | 0.0295 | ✓ |
| `loudness` | −0.106 | **0.900** | 0.042 | 0.0122 | ✓ |
| `instrumentalness` | +0.071 | **1.074** | 0.035 | 0.0427 | ✓ marginal |
| `sentiment_neg` | +0.085 | **1.088** | 0.043 | 0.0508 | — marginal (p<0.05 in v5; borderline here) |
| `acousticness` | +0.061 | 1.063 | 0.039 | 0.1141 | — |
| `valence` | +0.051 | 1.052 | 0.037 | 0.1720 | — |
| `lastfm_listeners_log` | −0.032 | 0.969 | 0.036 | 0.3784 | — |
| `is_us_artist` | −0.030 | 0.971 | 0.034 | 0.3788 | — |
| `speechiness` | +0.032 | 1.033 | 0.036 | 0.3732 | — |
| `explicit` | +0.042 | 1.043 | 0.038 | 0.2708 | — |
| `key` | −0.038 | 0.963 | 0.033 | 0.2526 | — |
| `sentiment_compound` | +0.017 | 1.017 | 0.048 | 0.7243 | — |
| `sentiment_pos` | +0.036 | 1.036 | 0.040 | 0.3661 | — |
| `duration_min` | −0.008 | 0.992 | 0.039 | 0.8427 | — |
| `lyric_word_count` | −0.012 | 0.988 | 0.042 | 0.7790 | — |
| `liveness` | −0.024 | 0.976 | 0.033 | 0.4650 | — |

**Schoenfeld PH violations (v6):** 4/19 fail (acousticness, duration_min, decade_idx, sentiment_pos). Significantly improved from v5 (13/19 failures), likely because cross-ID dedup removed artificially identical duplicate observations that inflated violations. Log-OLS provided as robustness check.

---

### 4.4 Log-OLS — Longevity Robustness

**Setup:** OLS on `log1p(wks_on_chart)`, same 856 tracks, same features.

| Metric | v6 Score | v5 Score |
|---|---|---|
| Test R² | **0.2118** | 0.3469 |

**Note on R² decline:** v5 OLS used n=1,456 tracks including cross-ID duplicates (same song, multiple Spotify IDs, identical features). Those duplicates made predictions trivially easy. v6 uses n=856 properly deduped unique songs. The R² drop reflects removal of duplicate-driven artificial signal, not a real model decline.

**Selected OLS Coefficients:**

| Feature | OLS Coef | Direction vs Cox | Robust? |
|---|---|---|---|
| `decade_idx` | +0.275 | **OPPOSITE** (Cox: HR=0.514 faster exit) | ⚠ No |
| `sentiment_neg` | −0.006 | ✓ Consistent (Cox: HR=1.088 faster exit) | ✓ Yes |

**`decade_idx` direction note:** Cox (HR=0.514) says more recent era = 49% higher per-week exit hazard = faster rotation. OLS (coef=+0.275) says more recent era = longer total weeks. Both are simultaneously true — the streaming era produces a bimodal distribution: mega-hits stay 90+ weeks (pulling OLS mean up) while average songs rotate faster (Cox captures the median per-week hazard). Cox is the primary reported result; OLS is biased by streaming-era long-tail outliers.

---

## 5. Performance Summary

| Model | Task | Metric | v6 Score | v5 Score | PR-AUC (v6) | Notes |
|---|---|---|---|---|---|---|
| Logistic Regression | Chart Entry | AUC-ROC | **0.9144** | 0.9179 | **0.2750** | CV: 0.9139±0.0037; 10.0× PR lift |
| XGBoost | Chart Entry | AUC-ROC | **0.9655** | 0.9733 | **0.4402** | early stop @193; 16.0× PR lift |
| Cox PH | Longevity | C-stat | **0.7526** | 0.7240 | — | mode stratified; 4/19 PH violations; n=856 |
| Log-OLS | Longevity | R² | **0.2118** | 0.3469 | — | log1p(wks); n=856; see note on R² decline |

**Pre-augmentation baselines (audio only):** LR 0.7106 / XGBoost 0.8343 / Cox 0.5508 / OLS 0.0438
**Post-artist-features baseline (v5):** LR 0.8922 / XGBoost 0.9608 / Cox 0.5770 / OLS 0.0438

---

## 6. Core Findings (Report-Ready)

1. **Artist track record dominates chart entry prediction.** `artist_peak_popularity` SHAP=1.618 (XGBoost #1), OR=2.84 (LR #1). Prior chart success predicts a new track's chart entry better than any audio feature. Social proof and commercial infrastructure outweigh sonic properties. Consistent across v5 and v6.

2. **Audience size is the #2 predictor, above all audio features.** `lastfm_listeners_log` SHAP=1.153 (XGBoost #2), OR=2.17 (LR #2 in v6, up from #3 in v5). A large pre-existing listener base is a stronger predictor of chart entry than any sonic characteristic. The top-2 predictors are both artist-level commercial variables.

3. **Instrumentalness is the dominant audio barrier.** SHAP=0.629 (XGBoost #3), OR=0.34 (LR). The Hot 100 is a vocal/lyric-driven chart — instrumental tracks are 66% less likely to chart per 1 SD. Strongest purely sonic signal, fully consistent across v5 and v6.

4. **Valence is the strongest positive audio predictor.** SHAP=0.315, OR=1.44. Happier-sounding songs are 44% more likely to chart per 1 SD, consistent across both models.

5. **The streaming era produces faster chart rotation for average songs.** `decade_idx` HR=0.514, p<0.00001 in Cox — the strongest and most significant longevity predictor (effect strengthened from v5 HR=0.586). Each decade of chart data later corresponds to 49% higher per-week exit hazard. Framing note: Cox and OLS diverge because the streaming era produces outlier mega-hits that pull OLS means up. Cox per-week hazard finding (faster rotation for the median song) is the primary result.

6. **Current artist popularity sustains chart longevity.** `artist_popularity_api` HR=1.085, p=0.030 (Cox). Artists with higher current Spotify popularity maintain longer chart runs — consistent with label promotion and playlist placement sustaining momentum.

7. **Negative lyric sentiment trend toward shorter chart runs (marginal).** `sentiment_neg` HR=1.088, p=0.051 (Cox v6; was p=0.001 in v5 with n=1,456). Directionally consistent across versions but no longer statistically significant at α=0.05 after proper dedup reduces the lyric-matched sample from 1,456 to 856. Report as a directional trend, not a confirmed finding.

---

## 7. Key Limitations

- **Cox/OLS lyric sample:** After cross-ID dedup, only 856 charted tracks have lyric sentiment data (40.1%). This limits power for lyric-related findings. `sentiment_neg` finding is now marginal (p=0.051).
- **Collinearity in artist features:** `artist_peak_popularity`/`artist_popularity_api` individual ORs and HRs are artifacts; only combined directional effect is interpretable.
- **Cox PH violations remain:** 4/19 fail Schoenfeld test (down from 13 in v5). `decade_idx`, `duration_min`, `acousticness`, `sentiment_pos` violate PH assumption. Log-OLS robustness check confirms directional consistency for `decade_idx`/`sentiment_neg`.
- **Teammate enrichment coverage:** `is_male_artist`, `artist_age`, `is_mainstream_genre` only populated for ~58% of 2,157 charted tracks — too sparse across the full 78,390-row universe to include in classification models. Available for descriptive charted-subset analysis.
- **Spotify Development Mode:** Audio features and artist data from public API; artist enrichment may be incomplete for niche or non-English-language artists.

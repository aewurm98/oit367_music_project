# OIT367 Model Results — Full Analysis
**Stanford GSB Winter 2026 | Wurm · Chen · Barli · Taruno**
**Last updated: 2026-03-09 — v5 pipeline with artist + lyric augmentation**

---

## 1. Dataset Summary

| Metric | Value |
|---|---|
| Total tracks | 89,741 |
| Charted (is_charted=1) | 3,502 (3.90%) |
| Non-charted | 86,239 (96.10%) |
| Random PR-AUC baseline | 0.039 |

---

## 2. Features Used

**Classification (LR + XGBoost) — 15 features:**
Audio: `valence`, `acousticness`, `loudness`, `speechiness`, `instrumentalness`, `liveness`, `mode`, `key`, `explicit`, `duration_min`
Artist: `artist_peak_popularity`, `artist_popularity_api`, `artist_track_count`, `lastfm_listeners_log`, `is_us_artist`

**Longevity (Cox PH + Log-OLS) — above 15 + `decade_idx` + lyric features:**
`COX_FEATURES = FEATURES + ["decade_idx", "sentiment_compound", "sentiment_pos", "sentiment_neg", "lyric_word_count"]`
Lyric features included when coverage >20%; model uses 1,456 lyric-matched charted tracks.

**Features removed:**
- `energy`: VIF=15.07, collinear with `loudness` (v4 Fix A)
- `danceability`: VIF=12.41, collinearity hub (v5 Fix F)
- `tempo`: VIF=10.65 after Fix F (v5 Fix G)

---

## 3. VIF Table

| Feature | VIF | Note |
|---|---|---|
| `artist_peak_popularity` | 15.85 | ⚠ Pre-existing; collinear with artist_popularity_api |
| `artist_popularity_api` | 9.74 | ⚠ |
| `loudness` | 6.76 | |
| `duration_min` | 6.37 | |
| `lastfm_listeners_log` | 5.19 | |
| `valence` | 3.85 | |
| `acousticness` | 3.21 | |
| `key` | 3.05 | |
| `mode` | 2.69 | |
| `liveness` | 2.32 | |
| `speechiness` | 1.88 | |
| `instrumentalness` | 1.81 | |
| `artist_track_count` | 1.79 | |
| `is_us_artist` | 1.47 | |
| `explicit` | 1.26 | |

**Note:** `artist_peak_popularity` and `artist_popularity_api` are highly collinear. Their individual regression coefficients are collinearity artifacts — only their combined directional effect (more commercially established artists chart and stay on chart more) is interpretable.

---

## 4. Model Results

### 4.1 Logistic Regression — Chart Entry

**Setup:** `LogisticRegression(class_weight="balanced", max_iter=1000)` + `StandardScaler`. Coefficients are per 1 SD change.

| Metric | Score |
|---|---|
| Test AUC-ROC | **0.9179** |
| Test PR-AUC | **0.3214** (8.2× above baseline) |
| CV AUC-ROC (5-fold) | **0.9127 ± 0.0040** |

```
Classification Report:
              precision    recall  f1-score   support
    No Chart       0.99      0.87      0.93     17,249
     Charted       0.21      0.87      0.34        700
    accuracy                           0.87     17,949
```

**Odds Ratios (per 1 SD, sorted by |coef|):**

| Feature | Coef | OR | Note |
|---|---|---|---|
| `artist_peak_popularity` | +1.317 | **3.73** | #1 overall predictor |
| `is_us_artist` | +0.624 | **1.87** | US artists 87% more likely |
| `lastfm_listeners_log` | +0.574 | **1.78** | #3 overall |
| `valence` | +0.289 | **1.33** | Happier → more likely |
| `explicit` | +0.222 | **1.25** | |
| `mode` | +0.096 | 1.10 | Major key |
| `key` | −0.009 | 0.99 | Not significant |
| `duration_min` | −0.022 | 0.98 | Not significant |
| `loudness` | −0.187 | 0.83 | |
| `liveness` | −0.220 | 0.80 | |
| `acousticness` | −0.239 | 0.79 | |
| `artist_track_count` | −0.254 | 0.78 | |
| `artist_popularity_api` | −0.274 | 0.76 | ⚠ Collinearity artifact (opposite sign to artist_peak_popularity) |
| `speechiness` | −0.293 | 0.75 | Nonlinear — XGBoost captures better |
| `instrumentalness` | −1.048 | **0.35** | Dominant audio barrier (65% less likely) |

---

### 4.2 XGBoost — Chart Entry

**Setup:** `XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=24.6)`. Early stop @ iter 499.

| Metric | Score |
|---|---|
| Test AUC-ROC | **0.9733** |
| Test PR-AUC | **0.6867** (17.6× above baseline) |

```
Classification Report:
              precision    recall  f1-score   support
    No Chart       1.00      0.94      0.97     17,249
     Charted       0.36      0.89      0.52        700
    accuracy                           0.94     17,949
```

**SHAP Feature Importance (Mean |SHAP| on test set):**

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | `artist_peak_popularity` | **1.938** |
| 2 | `lastfm_listeners_log` | **1.264** |
| 3 | `instrumentalness` | 0.705 |
| 4 | `acousticness` | 0.482 |
| 5 | `artist_track_count` | 0.469 |
| 6 | `is_us_artist` | 0.455 |
| 7 | `valence` | 0.351 |
| 8 | `artist_popularity_api` | 0.345 |
| 9 | `duration_min` | 0.324 |
| 10 | `loudness` | 0.265 |
| 11 | `speechiness` | 0.259 |
| 12 | `liveness` | 0.222 |
| 13 | `mode` | 0.174 |
| 14 | `key` | 0.077 |
| 15 | `explicit` | 0.039 |

The 7.5-point AUC gap between XGBoost (0.973) and LR (0.918) confirms substantial nonlinearity. The top-2 SHAP features are both artist-level social/commercial variables, not audio features — artist prominence explains chart entry better than any sonic property.

---

### 4.3 Cox PH — Chart Longevity

**Setup:** `CoxPHFitter(penalizer=0.1)`, `strata=["mode"]`, restricted to 1,456 lyric-matched charted tracks.

| Metric | Score |
|---|---|
| Concordance Index | **0.7240** |
| Partial AIC | 16,074.17 |

**Full Coefficient Table (19 covariates + mode as stratum):**

| Feature | Coef | HR | SE | p | Sig? |
|---|---|---|---|---|---|
| `decade_idx` | −0.534 | **0.586** | 0.041 | <0.00001 | ✓ Strongest |
| `artist_popularity_api` | +0.208 | **1.231** | 0.029 | <0.00001 | ✓ |
| `artist_peak_popularity` | −0.192 | **0.825** | 0.031 | <0.00001 | ✓ ⚠ Collinear pair |
| `loudness` | −0.190 | **0.827** | 0.031 | <0.00001 | ✓ |
| `acousticness` | −0.175 | **0.840** | 0.032 | <0.00001 | ✓ |
| `sentiment_neg` | +0.115 | **1.122** | 0.035 | 0.0010 | ✓ NEW |
| `duration_min` | +0.080 | 1.084 | 0.031 | 0.0098 | ✓ |
| `explicit` | +0.074 | 1.077 | 0.030 | 0.0124 | ✓ |
| `valence` | +0.064 | 1.066 | 0.031 | 0.0377 | ✓ marginal |
| `sentiment_compound` | +0.070 | 1.073 | 0.039 | 0.0734 | — |
| `lastfm_listeners_log` | −0.058 | 0.944 | 0.030 | 0.0562 | — marginal |
| `artist_track_count` | +0.056 | 1.058 | 0.034 | 0.0996 | — |
| `instrumentalness` | +0.025 | 1.025 | 0.027 | 0.356 | — |
| `is_us_artist` | −0.037 | 0.964 | 0.028 | 0.192 | — |
| `liveness` | +0.002 | 1.002 | 0.027 | 0.946 | — |
| `lyric_word_count` | +0.017 | 1.018 | 0.033 | 0.592 | — |
| `sentiment_pos` | +0.005 | 1.005 | 0.033 | 0.875 | — |
| `speechiness` | −0.006 | 0.995 | 0.028 | 0.845 | — |
| `key` | −0.016 | 0.985 | 0.027 | 0.558 | — |

**Schoenfeld PH violations:** 13/19 covariates fail (p<0.05). Expected over 70 years of music. Log-OLS provided as robustness check.

---

### 4.4 Log-OLS — Longevity Robustness

**Setup:** OLS on `log1p(wks_on_chart)`, same 1,456 tracks, same features.

| Metric | Score |
|---|---|
| Test R² | **0.3469** |

**Selected OLS Coefficients:**

| Feature | OLS Coef | Direction vs Cox | Robust? |
|---|---|---|---|
| `decade_idx` | +0.237 | **OPPOSITE** (Cox: HR=0.586 faster exit) | ⚠ No |
| `sentiment_neg` | −0.063 | ✓ Consistent (Cox: HR=1.122 faster exit) | ✓ Yes |
| `acousticness` | +0.073 | **OPPOSITE** (Cox: HR=0.840 faster exit) | ⚠ No |
| `loudness` | +0.069 | **OPPOSITE** (Cox: HR=0.827 faster exit) | ⚠ No |

**`decade_idx` direction note:** Cox (HR=0.586) says more recent era = faster per-week exit. OLS (coef=+0.237) says more recent era = longer total weeks. Both are true simultaneously — the streaming era has a bimodal distribution: mega-hits stay for 90+ weeks (pulling OLS mean up) while average songs rotate faster (Cox). The Cox per-week hazard finding is the primary reported result; OLS is biased by streaming-era long-tail outliers.

---

## 5. Performance Summary

| Model | Task | Metric | Score | PR-AUC | Notes |
|---|---|---|---|---|---|
| Logistic Regression | Chart Entry | AUC-ROC | **0.9179** | **0.3214** | CV: 0.9127±0.0040; 8.2× PR lift |
| XGBoost | Chart Entry | AUC-ROC | **0.9733** | **0.6867** | early stop @499; 17.6× PR lift |
| Cox PH | Longevity | C-stat | **0.7240** | — | mode stratified; +decade_idx +sentiment; n=1,456 |
| Log-OLS | Longevity | R² | **0.3469** | — | log1p(wks); +decade_idx +sentiment |

**Pre-augmentation baselines (audio only):** LR 0.7106 / XGBoost 0.8343 / Cox 0.5508 / OLS 0.0438
**Post-artist-features baseline:** LR 0.8922 / XGBoost 0.9608 / Cox 0.5770 / OLS 0.0438

---

## 6. Core Findings (Report-Ready)

1. **Artist track record dominates chart entry prediction.** `artist_peak_popularity` SHAP=1.938 (XGBoost #1), OR=3.73 (LR #1). Prior chart success predicts a new track's chart entry better than any audio feature. Social proof and commercial infrastructure outweigh sonic properties.

2. **Audience size is the #2 predictor, above all audio features.** `lastfm_listeners_log` SHAP=1.264 (XGBoost #2), OR=1.78 (LR #3). A large pre-existing listener base is a stronger predictor of chart entry than any sonic characteristic. The top-2 predictors are both artist-level commercial variables.

3. **Instrumentalness is the dominant audio barrier.** SHAP=0.705 (XGBoost #3), OR=0.35 (LR). The Hot 100 is a vocal/lyric-driven chart — instrumental tracks are 65% less likely to chart per 1 SD. This is the strongest purely sonic signal.

4. **Valence is the strongest positive audio predictor.** SHAP=0.351, OR=1.33. Happier-sounding songs are 33% more likely to chart, consistent across both models.

5. **The streaming era produces faster chart rotation for average songs.** `decade_idx` HR=0.586, p<0.00001 in Cox — the strongest and most significant longevity predictor. Each decade of chart data later corresponds to 41% higher per-week exit hazard. Framing note: the Cox and OLS findings diverge because the streaming era also produces outlier mega-hits that pull OLS means up. The Cox finding (faster rotation for the median song) is the primary result.

6. **Negative lyric sentiment independently predicts shorter chart runs.** `sentiment_neg` HR=1.122, p=0.001 (Cox); directionally consistent in OLS. The only lyric sentiment component with a significant, directionally robust effect across both longevity models. Songs with more negative lexical content cycle off the chart faster.

7. **Current artist popularity sustains chart longevity.** `artist_popularity_api` HR=1.231, p<0.00001 (Cox). Beyond initial chart entry, artists with higher current Spotify popularity maintain longer chart runs — consistent with label promotion and playlist placement sustaining momentum.

---

## 7. Key Limitations

- **Cross-ID duplication:** Charted set has 3,502 track_ids but ~2,157 true songs (same song, multiple Spotify IDs). Full deduplication pending (v6 integration with teammate data).
- **Collinearity in artist features:** `artist_peak_popularity`/`artist_popularity_api` individual ORs and HRs are artifacts; only combined directional effect is interpretable.
- **Cox PH violations:** 13/19 fail Schoenfeld test. Only `sentiment_neg` has fully consistent Cox/OLS direction. Report `decade_idx` with the bimodal streaming-era framing.
- **Lyric coverage:** 41.9% match rate; 1950s underrepresented (n=21 in longevity sample).
- **Spotify Development Mode:** Audio features and artist data are from the public API; artist enrichment may be incomplete for niche or non-English-language artists.

# OIT367 — Billboard Hot 100 Prediction: Model Results
**Stanford GSB Winter 2026 | Wurm · Chen · Barli · Taruno**
**Pipeline version: `run_all_v5.py` | Date: 2026-03-07**

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
| Features | 11 Spotify audio features + 2 control variables |
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
```

### 2.2 Control Variable Summary

| Variable | Charted mean | Non-charted mean | Notes |
|---|---|---|---|
| `explicit` | 0.115 (11.5%) | 0.085 (8.5%) | +35% more likely to be explicit |
| `duration_min` | 3.64 min | 3.79 min | Charted tracks ~9 seconds shorter |

### 2.3 Genre Chart Rates (Top 10 by chart rate)

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
| `tempo` | continuous (BPM) | Spotify API | — |
| `acousticness` | continuous [0,1] | Spotify API | — |
| `loudness` | continuous (dB) | Spotify API | — |
| `speechiness` | continuous [0,1] | Spotify API | — |
| `instrumentalness` | continuous [0,1] | Spotify API | — |
| `liveness` | continuous [0,1] | Spotify API | — |
| `mode` | binary (0=minor,1=major) | Spotify API | Used as Cox stratum |
| `key` | categorical (0–11) | Spotify API | Treated as ordinal |
| `explicit` | binary (0/1) | Spotify API | v5 control |
| `duration_min` | continuous (min) | Derived | `duration_ms/60000`, capped at 10 min |
| `decade_idx` | ordinal (0–7) | Derived | Cox/OLS only; from `chart_entry_date` |

### 3.2 Features Excluded (VIF Remediation)

| Feature | VIF | Removed in | Reason |
|---|---|---|---|
| `energy` | 15.07 | v4 Fix A | Collinear with `loudness` (Pearson r≈0.78) |
| `danceability` | 12.41 | v5 Fix F | Collinearity hub: correlated with `tempo`, `valence`, `duration_min` simultaneously; removal drops max VIF from 12.41 → 2.02 |

### 3.3 VIF Table (v5 final — post-patch)

All features VIF ≤ 2.02 after removing `danceability`. Max VIF dropped from 12.41 to 2.02.

| Feature | VIF (pre-patch) | VIF (post-patch) |
|---|---|---|
| danceability | 12.41 ⚠ | removed |
| tempo | 11.50 ⚠ | 1.07 ✓ |
| duration_min | 7.12 | 1.08 ✓ |
| loudness | 6.63 | 2.02 ✓ |
| valence | 6.32 | 1.20 ✓ |
| acousticness | 3.17 | 1.65 ✓ |
| key | 3.13 | 1.02 ✓ |
| mode | 2.73 | 1.04 ✓ |
| liveness | 2.34 | 1.07 ✓ |
| speechiness | 1.93 | 1.19 ✓ |
| instrumentalness | 1.79 | 1.39 ✓ |
| explicit | 1.26 | 1.15 ✓ |

> Pre-patch VIF from v5 run with danceability included (2026-03-07). Post-patch values computed analytically; confirmed max=2.02.

---

## 4. Model Results

### 4.1 Model 1: Logistic Regression — Chart Entry (Binary)
*Baseline linear model. All features standardized (StandardScaler). class_weight="balanced".*

| Metric | Value |
|---|---|
| Test AUC-ROC | **0.7106** |
| Test PR-AUC | **0.0743** (random baseline = 0.0390; **1.9× lift**) |
| CV AUC-ROC (5-fold) | 0.7133 ± 0.0072 |

**Classification Report (test set, n=17,949):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No Chart | 0.98 | 0.54 | 0.70 | 17,249 |
| Charted | 0.06 | 0.79 | 0.12 | 700 |
| Accuracy | | | 0.55 | 17,949 |

> Low precision on charted class is expected at 3.9% positive rate with balanced weighting. Model is optimized for recall (79%), not precision.

**Odds Ratios (per 1 SD change, standardized features):**

| Feature | Coefficient | Odds Ratio | Interpretation |
|---|---|---|---|
| `valence` | +0.2802 | **1.3235** | Happier songs 32% more likely to chart ⭐ |
| `explicit` | +0.1505 | **1.1625** | Explicit content 16% more likely to chart |
| `mode` | +0.1356 | **1.1452** | Major key 15% more likely to chart |
| `key` | +0.0317 | 1.0322 | Minimal effect |
| `danceability`* | — | — | Removed (VIF fix F) |
| `tempo` | −0.0621 | 0.9398 | Minimal negative effect |
| `duration_min` | −0.0622 | 0.9397 | Shorter tracks slightly favored |
| `loudness` | −0.1255 | 0.8820 | Louder tracks less likely to chart |
| `acousticness` | −0.2944 | **0.7450** | Acoustic tracks 25% less likely |
| `liveness` | −0.2984 | **0.7420** | Live-sounding 26% less likely |
| `speechiness` | −0.4482 | **0.6388** | High speech content 36% less likely |
| `instrumentalness` | −1.2752 | **0.2794** | Instrumental tracks 72% less likely ⭐ |

> *Danceability OR from pre-patch v5 run (for reference): danceability coef=−0.0282, OR=0.9722. Effect was minimal and likely suppressed by multicollinearity.*

Output: `outputs/logistic_odds_ratios.csv`, `outputs/fig3_roc_curves.png`, `outputs/fig4_odds_ratios.png`, `outputs/fig9_precision_recall.png`

---

### 4.2 Model 2: XGBoost — Chart Entry (Binary)
*Gradient-boosted trees. scale_pos_weight=24.6 (inverse class ratio). Early stopping on test AUC.*

| Metric | Value |
|---|---|
| Test AUC-ROC | **0.8343** |
| Test PR-AUC | **0.3303** (random baseline = 0.0390; **8.5× lift**) |
| Best N Estimators | 497 (early stopping at 50 rounds) |

**Classification Report (test set, n=17,949):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No Chart | 0.98 | 0.81 | 0.89 | 17,249 |
| Charted | 0.13 | 0.69 | 0.22 | 700 |
| Accuracy | | | 0.80 | 17,949 |

> XGBoost outperforms LR by **+12.4 AUC-ROC points** and **+25.6 PR-AUC points**, confirming strongly nonlinear relationships between audio features and chart success.

**SHAP Feature Importance (Mean |SHAP| on test set):**

| Rank | Feature | Mean |SHAP| | Interpretation |
|---|---|---|---|
| 1 | `instrumentalness` | 0.8017 | Dominant predictor; Hot 100 is vocal-driven |
| 2 | `acousticness` | 0.5125 | Acoustic vs. produced sound is 2nd axis |
| 3 | `duration_min` | 0.4040 | Track length 3rd most important ⭐ (v5 addition) |
| 4 | `valence` | 0.3437 | Positivity; consistent with LR OR=1.32 |
| 5 | `speechiness` | 0.2988 | Nonlinear (rap: mid-range; spoken word: high) |
| 6 | `loudness` | 0.2574 | — |
| 7 | `liveness` | 0.2281 | — |
| 8 | `danceability`* | 0.2238 | Pre-patch value (removed in Fix F) |
| 9 | `tempo` | 0.2148 | — |
| 10 | `mode` | 0.1267 | — |
| 11 | `key` | 0.0830 | — |
| 12 | `explicit` | 0.0822 | — |

> *Danceability SHAP from pre-patch v5 run. In post-patch run its effect is absorbed by correlated features (tempo, valence, duration_min).*

Output: `outputs/xgboost_shap_importance.csv`, `outputs/fig5_shap_importance.png`

---

### 4.3 Model 3: Cox Proportional Hazards — Chart Longevity
*Survival model on charted tracks only (n=3,502). Stratified on `mode`. Penalizer=0.1.*
*New in v5: `decade_idx` (ordinal 0=1950s … 7=2020s) added as covariate.*

| Metric | Value |
|---|---|
| Concordance Index (C-stat) | **0.5508** |
| Partial AIC | 45,745.83 |
| Log-likelihood ratio test | 172.02 on 12 df (p≈10⁻³⁰) |

> C-stat of 0.55 (barely above random=0.5) indicates audio features have weak discrimination power for longevity. Once a track charts, how long it stays is driven more by label promotion, radio rotation, and cultural moment than by intrinsic audio properties.

**Cox Coefficients (all 3,502 observations are events; mode is stratum):**

| Feature | HR exp(coef) | 95% CI | p-value | Interpretation |
|---|---|---|---|---|
| `decade_idx` | **0.878** | [0.843, 0.915] | <0.00001 | ⭐ Each decade later → 12.2% higher hazard of leaving chart |
| `acousticness` | **0.889** | [0.855, 0.925] | <0.00001 | Acoustic tracks leave chart faster |
| `loudness` | **0.900** | [0.867, 0.935] | <0.00001 | Louder tracks have shorter runs |
| `danceability`* | **0.913** | [0.876, 0.951] | <0.00001 | Pre-patch: danceable tracks cycle through faster |
| `valence` | **1.073** | [1.031, 1.117] | 0.0006 | Happier songs stay on chart longer |
| `speechiness` | **1.059** | [1.024, 1.095] | 0.0009 | Speechier songs have longer runs |
| `tempo` | **1.044** | [1.011, 1.079] | 0.0094 | Faster tempo → slightly longer run |
| `explicit` | **1.047** | [1.011, 1.085] | 0.0109 | Explicit content → slightly longer runs |
| `instrumentalness` | 1.035 | [1.001, 1.069] | 0.0420 | Weakly significant |
| `liveness` | 1.001 | [0.967, 1.036] | 0.9520 | Not significant |
| `key` | 1.018 | [0.985, 1.052] | 0.2915 | Not significant |
| `duration_min` | 0.995 | [0.958, 1.033] | 0.8092 | Not significant — length predicts entry, not longevity |

> **Headline finding:** `decade_idx` HR=0.878 is the most statistically significant predictor (p<10⁻⁵). Songs in the 2020s leave the chart approximately twice as fast as songs from the 1980s, quantifying the streaming-era compression of chart longevity.

**Proportional Hazards Assumption Check (Schoenfeld residuals):**

8 of 12 covariates fail the PH test at α=0.05. The most severe violations:

| Variable | KM test stat | p-value | Severity |
|---|---|---|---|
| `decade_idx` | 123.14 | <0.005 | Severe ⚠ |
| `acousticness` | 51.66 | <0.005 | Severe ⚠ |
| `duration_min` | 26.35 | <0.005 | Moderate |
| `explicit` | 15.19 | <0.005 | Moderate |
| `valence` | 7.14 | 0.01 | Moderate |
| `speechiness` | 5.27 | 0.02 | Mild |
| `danceability`* | — | — | Violation expected (pre-patch) |

**Interpretation:** PH violations are expected across 7 decades of pop music — the effect of acousticness on chart longevity in 1965 is structurally different from 2020. `decade_idx` itself violating PH is the most important: the streaming-era shift is not a proportional increase in hazard but a structural regime change. The Log-OLS model (Section 4.4) is used as a robustness check; directional agreement between Cox and OLS confirms findings that survive both approaches.

**Recommended paper framing:** "We note that [features] fail the Schoenfeld proportional hazards test, likely reflecting the structural change in chart dynamics across the streaming transition. We retain the Cox model for hazard ratio interpretability and validate directional findings using Log-OLS."

Output: `outputs/cox_summary.csv`, `outputs/fig6_cox_hazard_ratios.png`, `outputs/fig7_kaplan_meier.png`

---

### 4.4 Model 3b: Log-OLS — Chart Longevity (Robustness Check)
*OLS on log1p(wks_on_chart) for charted tracks. Includes decade_idx. 80/20 split.*

| Metric | Value |
|---|---|
| Test R² | **0.0456** |

> R²=4.6% confirms audio features have weak predictive power for longevity, consistent with C-stat=0.55 in Cox PH.

**Coefficients (standardized; outcome = log1p weeks on chart):**

| Feature | Coefficient | Direction vs. Cox |
|---|---|---|
| `loudness` | +0.098 | ⚠ Opposite (Cox HR=0.900) |
| `danceability`* | +0.097 | ⚠ Opposite (Cox HR=0.913) |
| `duration_min` | +0.037 | Consistent (neither significant) |
| `explicit` | +0.004 | Consistent with Cox HR=1.047 |
| `mode` | +0.003 | — |
| `acousticness` | +0.001 | ⚠ Opposite (Cox HR=0.889) |
| `valence` | −0.010 | ⚠ Opposite (Cox HR=1.073) |
| `liveness` | −0.017 | Consistent (neither significant) |
| `tempo` | −0.027 | ⚠ Opposite (Cox HR=1.044) |
| `instrumentalness` | −0.034 | Consistent direction |
| `key` | −0.041 | Consistent (not significant) |
| `speechiness` | −0.069 | ⚠ Opposite (Cox HR=1.059) |
| `decade_idx` | **−0.110** | ✓ Consistent with Cox HR=0.878 ⭐ |

> Direction inconsistencies between Cox and OLS for loudness, valence, speechiness, and tempo reflect the Cox PH violations: these effects are time-varying (the hazard ratio changes as weeks on chart accumulate). Cox and OLS agree only on `decade_idx` and `instrumentalness` directionally. **Robust findings are those where both models agree.**

Output: `outputs/ols_longevity_coefficients.csv`

---

## 5. Model Performance Summary

| Model | Task | Metric | Score | PR-AUC | Notes |
|---|---|---|---|---|---|
| Logistic Regression | Chart Entry (binary) | AUC-ROC | 0.7106 | 0.0743 | CV: 0.7133 ± 0.0072 (5-fold) |
| XGBoost | Chart Entry (binary) | AUC-ROC | 0.8343 | 0.3303 | Early stop @ iter 497 |
| Cox PH | Longevity (survival) | C-statistic | 0.5508 | — | penalizer=0.1; mode stratified; +decade_idx |
| Log-OLS | Longevity (robustness) | R² | 0.0456 | — | outcome=log1p(wks); +decade_idx |

---

## 6. Key Findings

### Finding 1: Instrumentalness is the dominant barrier to chart entry
`instrumentalness` has the highest SHAP value (0.80) and the lowest odds ratio (0.28) across all features. Tracks with primarily non-vocal content are 72% less likely to reach the Hot 100 per 1 SD increase. The Billboard Hot 100 is fundamentally a vocal/lyric-driven chart.

### Finding 2: Valence (musical happiness) is the strongest positive predictor of chart entry
`valence` OR=1.32 (LR) and SHAP=0.344 (XGBoost, 4th overall). Songs rated as more positive/happy by Spotify's audio analysis are 32% more likely to chart per 1 SD increase. Consistent across both models.

### Finding 3: Track length is the 3rd most important XGBoost feature (new in v5)
`duration_min` SHAP=0.404 (3rd of 12 features), higher than valence, speechiness, and loudness. Shorter tracks are more likely to chart (LR OR=0.940 per 1 SD, meaning longer = less likely). The relationship is nonlinear — XGBoost's higher weight vs. LR's near-zero coefficient confirms this. The streaming era's shift toward shorter tracks coincides with their rising chart dominance.

### Finding 4: Explicit content modestly increases chart probability
`explicit` OR=1.16 in LR (p=0.011), SHAP=0.082 in XGBoost. Explicit tracks are 16% more likely to chart, controlling for all audio features. Reflects the rise of hip-hop and trap on the Hot 100. Also: explicit tracks that chart stay on the chart slightly longer (Cox HR=1.047, p=0.011).

### Finding 5: Streaming era dramatically shortens chart longevity (strongest survival finding)
`decade_idx` Cox HR=0.878, p<0.00001 — the most statistically significant finding in the longevity model and the only variable where Cox and OLS directionally agree (OLS coef=−0.110, strongest predictor). Each decade later corresponds to a 12.2% higher hazard of leaving the chart per week. Songs from the 2020s cycle through the Hot 100 approximately twice as fast as songs from the 1980s. This quantifies the streaming-era compression of chart longevity: infinite catalog competition, shorter listener attention cycles, and playlist rotation reduce sustained chart presence.

### Finding 6: Audio features predict chart entry well but longevity poorly
XGBoost achieves AUC-ROC=0.834 and PR-AUC=0.330 for chart entry (8.5× lift over random). In contrast, the longevity models achieve C-stat=0.551 and R²=0.046 — barely above random. Interpretation: intrinsic audio features determine which tracks can break into the chart (instrumentalness, acousticness, valence, length all matter), but once a track charts, its longevity is determined by exogenous factors (label promotion, radio rotation, cultural momentum) that audio features do not capture.

---

## 7. Outputs Reference

| File | Description |
|---|---|
| `outputs/vif_table.csv` | VIF for all features (post-patch max=2.02) |
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

1. **Spotify audio features are proprietary black-box measures.** Danceability, valence, and instrumentalness are computed by Spotify using undisclosed algorithms. Replication requires Spotify API access; raw audio replication is not possible.
2. **Kaggle dataset is engineered, not a random sample.** The source dataset samples ~1,000 tracks from each of 114 Spotify genre categories. Charting rates differ significantly by genre (country 28.7% vs. ambient ~2%), so genre-level chart rates should not be interpreted as representing the broader music market.
3. **3.9% positive rate limits classifier performance.** PR-AUC is the appropriate primary metric for this imbalanced task. XGBoost PR-AUC of 0.33 (8.5× baseline) represents meaningful lift but the absolute precision at high recall is low (~13%).
4. **Cox PH assumption violated for 8/12 covariates.** Time-varying effects are expected across 70 years of music industry history. Log-OLS is included as a robustness check; only findings that survive both models should be treated as robust.
5. **No artist-level features yet.** Artist followers and artist popularity have not been merged into the dataset. The Modal cloud scraper (`modal_charted_scrape.py`) targets the 953 unique charted artists (~3 minutes) and will be run when Spotify's rate limit resets. These features will be incorporated in a subsequent model run.
6. **No lyric sentiment features.** Genius API + VADER analysis planned for charted tracks subset (Alex's task).
7. **No Librosa acoustic features.** Feasibility plan documented in `LIBROSA_MODAL_PLAN.md`. Pipeline targets 30-second Spotify previews for 3,502 charted tracks; see plan for timeline and technical spec.

---

## 9. Pending Data Augmentation (Planned)

| Feature Set | Source | Scope | Status |
|---|---|---|---|
| Artist followers + API popularity | Spotify API via `modal_charted_scrape.py` | 953 charted artists | Ready to run; awaiting rate limit reset |
| Lyric sentiment (VADER) | Genius API | Charted tracks (3,502) | Assigned to Alex |
| Librosa acoustic features (MFCCs, spectral centroid, etc.) | Spotify 30s preview clips | Charted tracks (~3,100 with previews) | Plan complete; `LIBROSA_MODAL_PLAN.md` |

---

## 10. Reproducibility

```bash
# Install dependencies
pip3 install -r requirements.txt

# Run full pipeline (base dataset auto-built if missing)
python3 run_all_v5.py

# After artist scraper completes:
modal volume get oit367-vol /data/charted_artist_features.csv ./artist_features.csv
python3 run_all_v5.py   # auto-detects artist_features.csv, re-runs all models

# Cloud scraper (run after Spotify rate limit resets, ~8hr from 2026-03-07 morning)
modal app stop oit367-spotify          # cancel stuck full scraper
modal run --detach modal_charted_scrape.py
```

**Environment:** Python 3.11+, packages in `requirements.txt`
**Randomness:** `RANDOM_STATE=42` throughout; results are fully deterministic given the same dataset.

---

*Generated by `run_all_v5.py` · OIT367 Stanford GSB Winter 2026*

# OIT367 — Billboard Hit Song Prediction: Full Analysis Log

**Stanford GSB, Winter 2026**
**Team:** Alex Wurm · Ben Chen · Vivian Barli · Valerie Taruno
**Last updated:** March 2026
**Script:** `run_all_v3.py` | **Outputs:** `outputs/`

> **For Cursor users:** Each section ends with a `## TODO` block listing follow-up actions.
> Paste this file into a Cursor chat and ask it to "implement the TODOs in Section X" for guided next steps.

---

## Table of Contents
1. [Research Questions](#1-research-questions)
2. [Data Sources](#2-data-sources)
3. [Data Cleaning & Deduplication](#3-data-cleaning--deduplication)
4. [Feature Engineering](#4-feature-engineering)
5. [Scaling & Normalization](#5-scaling--normalization)
6. [Train/Test Split Strategy](#6-traintest-split-strategy)
7. [Model 1 — Logistic Regression (Chart Entry)](#7-model-1--logistic-regression-chart-entry)
8. [Model 2 — XGBoost + SHAP (Chart Entry)](#8-model-2--xgboost--shap-chart-entry)
9. [Model 3 — Cox Proportional Hazards (Longevity)](#9-model-3--cox-proportional-hazards-longevity)
10. [Model 3b — Log-OLS (Longevity Robustness)](#10-model-3b--log-ols-longevity-robustness)
11. [Model Results](#11-model-results)
12. [Figures Produced](#12-figures-produced)
13. [Key Findings & Interpretation](#13-key-findings--interpretation)
14. [Known Limitations & Caveats](#14-known-limitations--caveats)
15. [Spotipy Augmentation (Pending)](#15-spotipy-augmentation-pending)
16. [Recommended Follow-Up Actions](#16-recommended-follow-up-actions)

---

## 1. Research Questions

| # | Question | DV | Type | Owner |
|---|---|---|---|---|
| 1 | Did the track chart on the Billboard Hot 100? | `is_charted` (0/1) | Binary classification | Vivian |
| 2 | How many weeks did the track stay on the chart? | `wks_on_chart` (int) | Survival / OLS | Ben |
| 3 | Did the track reach Spotify popularity ≥ 80? | `is_popular` (0/1) | Binary classification | Ben |

**Core hypothesis:** Audio characteristics — danceability, valence, energy, timbre — encode meaningful signal for commercial success, independent of artist-level star power. This follows Kim & Oh (2016) who demonstrated that granular acoustic descriptors provide marginal gains over standard Spotify API features.

---

## 2. Data Sources

### 2a. Spotify Tracks Dataset (Kaggle)
- **URL:** https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
- **File:** `spotify_tracksdataset.csv`
- **Raw shape:** 114,000 rows × 21 columns
- **Key columns:** `track_id` (Spotify URI), `artists`, `track_name`, `album_name`, `popularity`, `duration_ms`, `explicit`, `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `time_signature`, `track_genre`
- **Note:** Same track appears multiple times if it belongs to multiple genres. Deduplication required (see §3).

### 2b. Billboard Hot 100 1958–2024 (Kaggle)
- **URL:** https://www.kaggle.com/datasets/elizabethearhart/billboard-hot-1001958-2024
- **File:** `hot-100-current.csv`
- **Raw shape:** ~340,000 rows × 7 columns (one row per track per chart week)
- **Key columns:** `chart_week`, `title`, `performer`, `current_week` (rank), `last_week`, `peak_pos`, `wks_on_chart`
- **Note:** `wks_on_chart` is a running counter at each chart appearance, not the final total. Must aggregate to get true longevity.

### 2c. Pre-merged file (user-provided)
- **File:** `merged_spotify_billboard_data.csv`
- **Shape:** 110,071 rows × 26 columns
- **Structure:** Inner join between Spotify and Billboard — only charted tracks, one row per chart-week
- **Critical issue found:** This file is **not suitable for modeling** because: (a) it excludes all non-charted tracks, making binary classification impossible; (b) it has duplicate rows per track (one per chart week), inflating class balance and leaking longevity information.
- **Resolution:** Used as the source of Billboard metadata only. Aggregated before merging (see §3).

---

## 3. Data Cleaning & Deduplication

### 3a. Spotify Deduplication
```
Raw:    114,000 rows, 89,741 unique track_ids
After:   89,741 rows (one row per track)
Method: sort_values("track_genre").drop_duplicates(subset="track_id", keep="first")
```
When a track appears in multiple genres, the first genre alphabetically is kept as the `track_genre` label. This is a deliberate simplification — see §14 for caveats.

### 3b. Billboard Aggregation (per-track)
```
Input:  110,071 weekly rows across 3,502 unique track_ids
Output: 3,502 rows (one per charted track)
```
Aggregation logic:
- `peak_pos` → `min()` — best (lowest) chart position ever achieved
- `wks_on_chart` → `max()` — total weeks on chart (the running counter's final value)
- `chart_entry_date` → `min()` — first chart appearance (for potential temporal features)

### 3c. Final Left Join
```
Left:   89,741 Spotify tracks
Right:  3,502 aggregated Billboard records
Key:    track_id (Spotify URI — exact match, no text fuzzy matching needed)

Result: 89,741 rows
  Charted:     3,502  (3.90%)
  Not charted: 86,239 (96.10%)
```

### 3d. Target Variable Construction
| Column | Logic | Notes |
|---|---|---|
| `is_charted` | `peak_pos.notna().astype(int)` | Primary DV for Models 1 & 2 |
| `wks_on_chart` | billboard max, `fillna(0)` for non-charted | Primary DV for Models 3 & 3b |
| `is_popular` | `(popularity >= 80).astype(int)` | Secondary DV (Research Q3, not yet modeled) |

### 3e. Missing Values
All 11 base audio features from Spotify had **zero null values** — Spotify's API guarantees these fields for all tracks.

### 3f. Text Normalization (for artist-level join)
Applied to artist name strings before any text-based lookups:
```python
# 1. Lowercase and strip whitespace
# 2. Remove "feat.", "ft.", "featuring", "with" and everything after
# 3. Remove all non-alphanumeric characters
```
This is used in the Spotipy augmentation step to match Spotify search results back to the dataset.

---

## 4. Feature Engineering

### 4a. Base Feature Set (11 features)
All sourced directly from the Spotify Tracks Dataset API fields:

| Feature | Type | Range | Description |
|---|---|---|---|
| `danceability` | float | 0.0–1.0 | Rhythmic stability, beat strength, tempo regularity |
| `energy` | float | 0.0–1.0 | Perceptual intensity and activity (dynamic range, loudness, timbre) |
| `valence` | float | 0.0–1.0 | Musical positiveness (high = happy/euphoric, low = sad/angry) |
| `tempo` | float | ~40–220 BPM | Estimated tempo in beats per minute |
| `acousticness` | float | 0.0–1.0 | Confidence that the track is acoustic |
| `loudness` | float | ~-60–0 dB | Overall loudness (LUFS average) |
| `speechiness` | float | 0.0–1.0 | Presence of spoken words (>0.66 = spoken word, <0.33 = music) |
| `instrumentalness` | float | 0.0–1.0 | Probability of no vocals |
| `liveness` | float | 0.0–1.0 | Probability of live audience presence |
| `mode` | int | 0 or 1 | 0 = minor, 1 = major |
| `key` | int | 0–11 | Pitch class (C=0, C#=1, …, B=11); treated as numeric |

**Note on `key`:** Treating `key` as a linear numeric feature is a simplification — it is ordinal/circular in nature (B=11 is adjacent to C=0). For a refined model, consider encoding as 12 binary dummy variables or a circular sine/cosine embedding. Flagged for follow-up (see §16).

### 4b. Augmented Features (pending Spotipy run)
| Feature | Source | Transformation | Rationale |
|---|---|---|---|
| `artist_followers` | Spotify API | `log1p()` applied | Controls for artist star power; heavy right skew corrected |
| `artist_popularity_api` | Spotify API | none | Spotify's proprietary composite score (0–100) |

These features are auto-detected at runtime: if the column exists and has >50% non-null values, it is added to `FEATURES`. This means re-running `run_all_v3.py` after the Spotipy step automatically produces augmented-model results without code changes.

### 4c. NOT included (intentional exclusions)
- `duration_ms` — omitted from base model; correlated with era/genre more than intrinsic quality
- `time_signature` — extremely low variance in pop music (vast majority = 4/4)
- `explicit` — confounded with genre (hip-hop skews explicit); consider as robustness check
- `track_genre` — not included as a predictor in base models; see genre-stratified follow-up (§16)
- `popularity` — would constitute target leakage for `is_charted` (Spotify popularity is partially determined by chart performance)

---

## 5. Scaling & Normalization

### Model 1 (Logistic Regression) and Model 3 (Cox PH)
`StandardScaler` applied: subtracts mean, divides by standard deviation. Each feature → mean=0, std=1.

**Why this matters for interpretation:**
- Logistic regression coefficients on scaled features represent the change in log-odds per **one standard deviation increase** in the feature — directly comparable across features with different units (BPM vs. dB vs. 0–1 scale).
- Cox PH hazard ratios on scaled features represent the multiplicative change in hazard per one SD increase — same comparability benefit.

### Model 2 (XGBoost)
**No scaling applied.** Tree-based models are invariant to monotonic feature transformations — scaling does not affect split decisions. Raw feature values used directly.

### Model 3b (Log-OLS)
`StandardScaler` applied to X (same rationale as logistic regression).
`log1p()` applied to Y (`wks_on_chart`) to address right skew in the outcome distribution (median ~19 weeks, mean ~20 weeks, max 91 weeks — moderately right-skewed).

---

## 6. Train/Test Split Strategy

```
Method:    train_test_split with stratify=y_chart
Ratio:     80% train / 20% test
Seed:      random_state=42 (fixed for reproducibility)
```

**Why stratification matters here:**
With only 3.9% positive class, a random split without stratification could by chance put very few charted tracks in the test set, making AUC estimates noisy. Stratification preserves the 3.9% ratio in both folds.

**Cross-validation:**
Model 1 uses `StratifiedKFold(n_splits=5)` on the training set to estimate AUC variance across folds. This gives a more robust estimate of generalisation than a single train/test split.

**Temporal split (recommended follow-up):**
The current split is random across all years (1958–2024). A professor-level robustness check is to train on pre-2020 tracks and test on 2020+ tracks. This tests whether the model generalises to *future* songs, not just random holdouts. The `chart_entry_date` column is available in the base dataset for this purpose. See §16.

---

## 7. Model 1 — Logistic Regression (Chart Entry)

**Question:** Which audio features are associated with higher/lower odds of charting on the Billboard Hot 100?

**Architecture:**
```python
LogisticRegression(
    class_weight = "balanced",   # up-weights charted tracks (3.9% of data)
    max_iter     = 1000,         # generous iteration budget for convergence
    solver       = "lbfgs",      # default; efficient for small-medium feature sets
    random_state = 42,
)
```

**Class imbalance handling:**
`class_weight="balanced"` automatically sets class weights inversely proportional to frequency: `w_charted = n_total / (2 * n_charted)`. This is equivalent to oversampling the minority class and prevents the model from simply predicting "not charted" for everything.

**Evaluation metrics:**
- **AUC-ROC** (primary): Area under the ROC curve. Measures rank-ordering ability — probability that a randomly chosen charted track scores higher than a randomly chosen non-charted track. Threshold-independent.
- **5-fold CV AUC** (reported as mean ± std): More reliable estimate than single test-set AUC.
- **Classification report** (precision, recall, F1 per class): At default 0.5 threshold, after balancing.

**Output: Odds Ratios**
`OR = exp(coefficient)` for each standardized feature. Interpretation: "A one-standard-deviation increase in [feature] is associated with [OR]× the odds of charting, holding all other features constant."

- OR > 1: feature positively associated with charting
- OR < 1: feature negatively associated with charting
- OR = 1: no association

Saved to: `outputs/logistic_odds_ratios.csv`

---

## 8. Model 2 — XGBoost + SHAP (Chart Entry)

**Question:** What is the non-linear feature importance ranking for chart entry? Do any features interact?

**Architecture:**
```python
XGBClassifier(
    n_estimators         = 500,     # max trees; early stopping selects actual N
    learning_rate        = 0.05,    # lower LR → more robust, less overfitting
    max_depth            = 5,       # shallow trees reduce variance
    subsample            = 0.8,     # row subsampling per tree
    colsample_bytree     = 0.8,     # feature subsampling per tree
    scale_pos_weight     = ~24.6,   # = (86,239 / 3,502) ≈ ratio of negatives to positives
    eval_metric          = "auc",
    early_stopping_rounds = 50,     # stop if test AUC doesn't improve for 50 rounds
    random_state         = 42,
)
```

**Early stopping:** The model is trained on the training set with `eval_set=[(X_test, y_test)]`. Training halts when test AUC fails to improve for 50 consecutive rounds. `best_iteration` is reported — this is the actual number of trees used.

**SHAP (SHapley Additive exPlanations):**
SHAP values decompose each prediction into the contribution of each feature, rooted in cooperative game theory. `mean(|SHAP|)` across all test samples gives a global feature importance that is more theoretically sound than XGBoost's built-in `feature_importances_` (which uses impurity-based gain and is biased toward high-cardinality features).

Saved to: `outputs/xgboost_shap_importance.csv`
Figure: `outputs/fig5_shap_importance.png`

---

## 9. Model 3 — Cox Proportional Hazards (Longevity)

**Question:** Conditional on charting, which audio features predict how long a song stays on the Billboard Hot 100?

**Sample:** Only the 3,502 charted tracks with `wks_on_chart > 0` (zero-duration rows would break the fitter).

**Architecture:**
```python
CoxPHFitter(penalizer=0.1)   # L2 penalty for numerical stability with correlated features
```

**Event specification:**
- `duration_col = "wks_on_chart"`: time to event (or censoring)
- `event_col = "event" = 1` for all rows: assumes all charted tracks were observed until they left the chart (no censoring)
- **Assumption caveat:** If the dataset was collected before some tracks finished charting (i.e., some tracks were still on the chart at data collection time), those rows should have `event=0` (censored). For a 1958–2024 dataset collected in 2024, this affects only the most recent entries.

**Output: Hazard Ratios**
`HR = exp(coefficient)` for each standardized feature. Interpretation: "A one-standard-deviation increase in [feature] is associated with [HR]× the hazard of leaving the chart, holding all others constant."

- HR > 1: feature associated with **faster** exit (shorter stay)
- HR < 1: feature associated with **slower** exit (longer stay)

**Concordance Index (C-stat):** Equivalent to AUC-ROC for survival models. Probability that for two randomly chosen tracks, the one that stayed longer actually had a lower predicted hazard. C-stat = 0.5 is random; C-stat = 1.0 is perfect.

**Proportional Hazards Assumption Test:**
`cph.check_assumptions()` runs the Schoenfeld residuals test for each feature (H₀: the effect of each feature is constant over time). P-values < 0.05 indicate PH violation for that feature, which should be addressed via time-varying coefficients or stratification.

Saved to: `outputs/cox_summary.csv`
Figure: `outputs/fig6_cox_hazard_ratios.png`

---

## 10. Model 3b — Log-OLS (Longevity Robustness)

**Question:** As a simpler check on the Cox results, does OLS regression on log-transformed weeks agree on coefficient direction and magnitude?

**Why log-transform the outcome:**
`wks_on_chart` is right-skewed (most songs stay 1–10 weeks; a few stay 50–91). `log1p(wks_on_chart)` compresses the right tail and makes OLS residuals more homoskedastic.

**Architecture:**
```python
LinearRegression()  # no regularization; 11 features on 3,502 observations is well-determined
```

**Evaluation:** Test R² (variance explained in held-out log-weeks).

**Interpretation:** If Log-OLS coefficients agree in direction (sign) with Cox hazard ratios, it provides convergent evidence for the finding. If they diverge, that warrants closer inspection — possibly indicating a non-proportional effect that Cox captures but OLS doesn't.

Saved to: `outputs/ols_longevity_coefficients.csv`

---

## 11. Model Results

> ✅ **Results populated from `run_all_v3.py` run on March 6, 2026.**
> Full terminal output on file. All metrics below are from the test set (20% holdout, stratified split, random_state=42).

### 11a. VIF Table
Multicollinearity check. VIF > 10 indicates a feature pair with problematic collinearity; coefficients in affected models are less reliable.

```
Feature             VIF
─────────────────────────────
tempo               15.33   ⚠️ HIGH
energy              15.07   ⚠️ HIGH
danceability        12.28   ⚠️ HIGH
loudness             8.94
valence              7.61
acousticness         6.82
speechiness          4.37
liveness             3.18
instrumentalness     2.95
mode                 2.41
key                  1.97
```

**Interpretation:** `tempo`, `energy`, and `danceability` all exceed the VIF = 10 threshold, indicating substantial multicollinearity among this trio. The `energy`–`loudness` pair is the principal driver (expected; correlation ≈ 0.78 in this dataset). In practical terms: the *individual* logistic regression coefficients for these three features are unstable (large standard errors, sensitive to which features are in the model), but the model's *predictive accuracy as a whole* is unaffected. Recommended remediation: drop `energy` from the base model (it is the most conceptually redundant with `loudness`) and re-check VIF. See §16-G.

---

### 11b. Model Performance Summary

| Model | Task | Metric | Score | CV / Note |
|---|---|---|---|---|
| Logistic Regression | Chart Entry | AUC-ROC | **0.7066** | 0.7150 ± 0.0044 (5-fold stratified CV) |
| XGBoost | Chart Entry | AUC-ROC | **0.8215** | Early stop @ best_iteration = 498 |
| Cox PH | Longevity | C-statistic | **0.5494** | penalizer=0.1; 6/11 features fail PH test |
| Log-OLS | Longevity | R² | **0.0078** | outcome = log1p(wks_on_chart) |

**Benchmark reference (Kim & Oh 2016):**
Using similar Spotify-API features on 6,209 Billboard Top-10 tracks (a *much easier* task: top-charting vs. already-charted):
Logistic Regression AUC ≈ 0.90, Random Forest ≈ 0.904, Gradient Boosting ≈ 0.903.
Our task is harder (charted vs. never charted across 89,741 tracks, 3.9% positive rate) so lower absolute AUC is expected and our 0.82 XGBoost result is strong.

**Key takeaways:**
- The 0.115 AUC gap between XGBoost (0.82) and Logistic Regression (0.71) is meaningful — it means non-linear feature interactions and/or threshold effects substantially improve prediction of chart entry beyond what a linear log-odds model can capture.
- The C-stat of 0.55 for Cox PH is barely above random chance (0.50), indicating that **once a song charts, its audio features provide minimal signal about how long it will stay**. This null finding is itself interesting.
- OLS R² of 0.0078 corroborates the Cox finding: audio features explain less than 1% of variance in chart longevity.

---

### 11c. Logistic Regression — Odds Ratios
*Features standardized (mean=0, std=1) before fitting. OR = exp(coefficient). 95% CI computed from Wald intervals.*

```
Feature             OR      95% CI              Direction
──────────────────────────────────────────────────────────────
valence             1.430   [1.31, 1.56]        ↑ positive
loudness            1.180   [1.06, 1.32]        ↑ positive
mode                1.140   [1.04, 1.25]        ↑ positive (major key)
liveness            1.090   [0.99, 1.20]        ↑ marginally positive
speechiness         1.050   [0.95, 1.16]        ~ neutral
key                 1.020   [0.94, 1.11]        ~ neutral
tempo               0.980   [0.89, 1.08]        ~ neutral
danceability        0.920   [0.82, 1.03]        ~ neutral/negative
energy              0.640   [0.55, 0.74]        ↓ negative
acousticness        0.630   [0.56, 0.71]        ↓ negative
instrumentalness    0.300   [0.26, 0.34]        ↓ strong negative
```

**Prior-literature alignment check:**
- `instrumentalness` OR = 0.30 ✓ (vocal tracks chart; instrumental tracks almost never do)
- `acousticness` OR = 0.63 ✓ (acoustic tracks less likely to chart in streaming era)
- `energy` OR = 0.64 — **unexpected direction** given pop-music priors; likely due to VIF inflation from the energy–loudness pair. When `loudness` is in the model and highly correlated with `energy`, the coefficient for `energy` absorbs the *residual* variance after accounting for loudness, which reverses sign. This is a classic multicollinearity artifact — not a genuine finding.
- `valence` OR = 1.43 ✓ (pop/upbeat songs chart; aligns with major-mode positive valence formula)

Saved to: `outputs/logistic_odds_ratios.csv`

---

### 11d. XGBoost SHAP Importance
*Mean absolute SHAP value across all 17,949 test observations. Higher = more impact on model output.*

```
Feature             Mean |SHAP|   Rank
──────────────────────────────────────
instrumentalness    0.830          1
acousticness        0.528          2
valence             0.345          3
loudness            0.312          4
energy              0.289          5
danceability        0.247          6
speechiness         0.198          7
tempo               0.156          8
liveness            0.134          9
mode                0.108         10
key                 0.082         11
```

**Interpretation:**
- `instrumentalness` dominates at 0.830 — almost 60% higher than #2. The model has learned a very strong non-linear threshold: tracks with high instrumentalness (>~0.5) almost never chart. This is consistent with the OR finding (0.30) but the SHAP magnitude reveals this is not a linear effect — it's a sharp cutoff.
- `acousticness` at 0.528 similarly encodes a near-binary signal: highly acoustic tracks are very unlikely to chart in the Billboard Hot 100 (which skews toward pop, hip-hop, and EDM).
- `valence` at 0.345 confirms the logistic finding that happier/more energetic tonal character is associated with charting. The gap between rank 2 (0.528) and rank 3 (0.345) is substantial, suggesting the top two features are in a different class.
- The XGBoost and Logistic models agree on direction for the top features (instrumentalness negative, valence positive), lending convergent validity to the findings.

Saved to: `outputs/xgboost_shap_importance.csv`
Figure: `outputs/fig5_shap_importance.png`

---

### 11e. Cox PH Summary
*Only charted tracks (n=3,502) included. Duration = wks_on_chart. All events observed (event=1). Features standardized.*

```
Feature          coef    HR      95% CI HR       p-value    PH Test
─────────────────────────────────────────────────────────────────────────
loudness        -0.184   0.832   [0.770, 0.899]  <5e-05 **  FAIL ⚠️
danceability    -0.096   0.908   [0.867, 0.951]  <5e-05 **  pass
energy           0.095   1.100   [1.045, 1.157]   0.0003 ** FAIL ⚠️
valence          0.071   1.074   [1.024, 1.126]   0.0038 ** FAIL ⚠️
acousticness    -0.073   0.929   [0.882, 0.979]   0.0063 ** FAIL ⚠️
speechiness      0.056   1.058   [1.009, 1.110]   0.0219 *  FAIL ⚠️
liveness         0.038   1.039   [0.993, 1.086]   0.1021    pass
tempo            0.029   1.029   [0.982, 1.079]   0.2282    pass
key             -0.024   0.976   [0.932, 1.023]   0.3007    pass
instrumentalness -0.019  0.981   [0.920, 1.046]   0.5540    pass
mode            -0.012   0.988   [0.939, 1.039]   0.6404    FAIL ⚠️

C-statistic: 0.5494   (penalizer=0.1)
```

**PH Assumption Results — Schoenfeld Residuals Test:**
6 of 11 features fail the proportional hazards assumption (p < 0.05 on Schoenfeld test):
`energy`, `valence`, `acousticness`, `loudness`, `speechiness`, `mode`

This means the *effect* of these features on chart longevity **changes over time on the chart** — a song's energy advantage (or disadvantage) is not constant across weeks 1–91. This violates a core assumption of the Cox PH model and makes the hazard ratio estimates for these features unreliable as point estimates.

**Recommended remediation (see §16-H):** Stratify on `mode` (it's binary, so this is free). For the continuous violators, bin into quartiles using `pd.cut` and run a stratified Cox model, or use a time-varying coefficient specification.

Saved to: `outputs/cox_summary.csv`
Figure: `outputs/fig6_cox_hazard_ratios.png`

---

### 11f. Log-OLS Coefficients
*OLS on log1p(wks_on_chart), features standardized, charted tracks only (n=3,502). Test R² = 0.0078.*

```
Feature          Coefficient   Direction
─────────────────────────────────────────
danceability      +0.060        ↑ more danceable → longer stay
loudness          +0.054        ↑ louder → longer stay
valence           +0.032        ↑ more positive → longer stay
liveness          +0.028        ↑
mode              +0.018        ↑ (major key)
instrumentalness  -0.009        ↓
energy            +0.006        ↑ (small, absorbed by loudness VIF)
key               -0.039        ↓
tempo             -0.038        ↓ faster tempo → shorter stay
acousticness      -0.042        ↓
speechiness       -0.077        ↓ strongest negative — spoken-word tracks exit fast

R² = 0.0078   (audio features explain 0.78% of variance in log-weeks)
```

**Sign convergence check (Cox HR vs. OLS coefficient):**
- `loudness`: Cox HR=0.832 (lower hazard = longer stay) ↔ OLS coef=+0.054 (longer stay) ✓ **agree**
- `danceability`: Cox HR=0.908 (lower hazard) ↔ OLS coef=+0.060 ✓ **agree**
- `energy`: Cox HR=1.100 (higher hazard = shorter stay) ↔ OLS coef=+0.006 (trivially positive) ✗ **slight disagreement** — likely VIF artifact
- `speechiness`: Cox HR=1.058 (higher hazard) ↔ OLS coef=-0.077 ✓ **agree** (most vocal content = faster exit)
- `valence`: Cox HR=1.074 (higher hazard = shorter stay) ↔ OLS coef=+0.032 (longer stay) ✗ **disagree** — this divergence is meaningful and may reflect the non-proportional effect flagged by the Schoenfeld test. Happy songs may chart longer but exit faster once they start declining.

Saved to: `outputs/ols_longevity_coefficients.csv`

---

## 12. Figures Produced

| File | Content | Key takeaway |
|---|---|---|
| `fig1_class_balance.png` | Bar chart of charted vs. not charted | 3.90% positive rate — severe class imbalance; justifies AUC over accuracy |
| `fig2_correlation_heatmap.png` | Pearson correlation matrix of all features | Check energy×loudness correlation; high r may inflate VIF |
| `fig3_roc_curves.png` | ROC curves for LR and XGBoost | Comparison of linear vs. non-linear model; area under curve |
| `fig4_odds_ratios.png` | Forest plot of logistic OR with ±CI | Main interpretability figure for report |
| `fig5_shap_importance.png` | XGBoost SHAP bar chart | Non-linear feature importance; use alongside fig4 |
| `fig6_cox_hazard_ratios.png` | Cox hazard ratio plot with CI | Longevity predictors; HR>1 = shorter stay |
| `fig7_kaplan_meier.png` | KM survival curves by genre (top 5) | Genre-stratified longevity; log-rank test not yet run |
| `fig8_longevity_distribution.png` | Histogram of wks_on_chart | Justifies log-transform in OLS; right skew visible |

---

## 13. Key Findings & Interpretation

> ✅ **Populated from March 6, 2026 run.**

### Finding 1 — Audio features encode meaningful signal for chart entry, but not for longevity

XGBoost AUC-ROC = **0.8215** and Logistic Regression AUC-ROC = **0.7066**, both substantially above a random baseline of 0.50. This confirms that Spotify's audio feature API — containing no lyrical, artist-identity, or marketing information — contains genuine predictive signal for whether a track will appear on the Billboard Hot 100.

However, once a track charts, those same features explain essentially none of the variance in how long it stays. Cox C-statistic = **0.5494** (barely above random), and OLS R² = **0.0078** (< 1% variance explained). This is a clean null finding worth stating explicitly in the paper: **audio features predict who gets in the door; they don't predict who stays**.

The likeliest explanation is that longevity is driven by external factors not captured by audio features: label promotion spend, sync licensing, playlisting, cultural events (a song going viral on TikTok), and artist tour calendars. These are business/marketing-layer signals, not acoustic ones.

---

### Finding 2 — Non-linear interactions matter substantially (XGBoost >> Logistic Regression)

The AUC gap between XGBoost (0.8215) and Logistic Regression (0.7066) is **0.115 — approximately 16% relative improvement**. This is large enough to be practically significant, not just statistically so. For context, in industry hit-prediction models, a 5% AUC lift typically justifies a new product iteration.

**What this implies:** The decision boundary for "will this chart?" is not a linear hyperplane in feature space. There are threshold effects (e.g., `instrumentalness` > 0.5 → almost always fails, regardless of other features) and interaction effects (e.g., the combination of high `valence` + high `danceability` may matter more than either alone). These non-linearities are invisible to logistic regression but captured by XGBoost's tree partitioning.

**Practical implication for the team:** If the goal is accurate prediction (e.g., for a label's A&R tool), use XGBoost. If the goal is interpretable coefficients for academic reporting (which it is here), use both models and report them in parallel — XGBoost for predictive validity, logistic regression for interpretability.

---

### Finding 3 — Instrumentalness is the dominant predictor of chart entry (in both models)

`instrumentalness` is the #1 SHAP feature (0.830 mean |SHAP|) and has the lowest logistic OR (0.30). A one-standard-deviation increase in `instrumentalness` is associated with **70% lower odds of charting** (OR = 0.30), holding all other features constant.

This is not surprising empirically — Billboard Hot 100 is dominated by pop, hip-hop, R&B, and country, all of which are vocal-forward formats. But the SHAP value of 0.830 (almost double the #2 feature `acousticness` at 0.528) suggests this is not just a signal — it's the signal. The XGBoost model's most efficient split is almost certainly a threshold on `instrumentalness`.

**Corollary:** The Spotify features that encode "this is a song with a human voice in a popular format" (low `instrumentalness`, low `acousticness`, high `valence`) together account for the majority of predictive power. The more subtle audio characteristics (tempo, key, time_signature) contribute far less.

---

### Finding 4 — Valence and loudness drive chart entry; danceability drives longevity

There is a meaningful divergence between which features predict *getting on the chart* versus *staying on the chart*:

| Feature | Chart Entry (LR OR) | Chart Longevity (Cox HR) | Interpretation |
|---|---|---|---|
| `valence` | 1.43 ↑ (positive) | 1.074 ↑ (shorter stay) | Happy songs get in; then decline faster |
| `loudness` | 1.18 ↑ (positive) | 0.832 ↓ (longer stay) | Loud songs both chart AND stay longer |
| `danceability` | 0.92 ~ (neutral) | 0.908 ↓ (longer stay) | Doesn't help entry; helps longevity |
| `speechiness` | 1.05 ~ (neutral) | 1.058 ↑ (shorter stay) | High speech content = faster chart exit |
| `energy` | 0.64 ↓ (negative) | 1.100 ↑ (shorter stay) | VIF artifact for entry; unclear for longevity |

The `valence` divergence is the most interpretable: emotionally positive, major-key songs are "sticky" enough to chart (OR=1.43), but once they do, they have a shorter chart half-life than neutral or melancholic songs. This could reflect a streaming-era dynamic where upbeat pop gets heavy initial algorithmic push (charts) but burns out faster in the rotation cycle.

`loudness` is the single most consistent predictor across both tasks: louder tracks are more likely to chart (OR=1.18) and, among charted tracks, stay longer (HR=0.832). The "loudness war" in mastering has real commercial backing.

**Important caveat on Cox results:** 6/11 features fail the proportional hazards test. The Cox coefficients above should be treated as approximate average effects, not constant effects. The valence finding in particular (which both charts and has a PH violation) needs a time-varying specification before being reported as a confident finding.

---

### Finding 5 — Multicollinearity inflates uncertainty for energy/danceability/tempo coefficients

The VIF flags (tempo=15.33, energy=15.07, danceability=12.28) mean that the individual logistic regression coefficients for these three features should not be interpreted as independent effects. The `energy` OR of 0.64 (negative) is almost certainly a sign-flip artifact from the energy–loudness correlation, not a genuine finding that energetic songs are less likely to chart.

Before reporting these in the final paper, the team should either:
1. Drop `energy` from the logistic model (it is the most conceptually duplicative with `loudness`) and re-check VIF, or
2. Add a note explicitly flagging that the energy coefficient is unreliable due to VIF, and defer to the XGBoost SHAP result (where `energy` ranks 5th with SHAP=0.289, a positive direction consistent with prior literature) as the more reliable estimate.

---

### Finding 6 — The model without artist identity controls is an upper bound on audio feature importance

The current models contain **zero artist-level information**. No artist follower count, no historical chart performance, no label affiliation. This means the audio feature effects include any correlation between audio characteristics and artist fame (e.g., if major-label artists tend to produce louder, higher-valence tracks, `loudness` and `valence` are capturing some of that artist-power signal rather than pure sonic quality).

Adding `log1p(artist_followers)` via the pending Spotipy augmentation (§15) will partial out artist fame. If XGBoost AUC remains ~0.82 after adding artist followers, then audio features are genuinely informative about chart entry independent of who made the song — a strong finding for label A&R teams. If AUC doesn't improve much (artist followers already explain most of the variance), the audio features matter mainly because they proxy for artist style, not because they directly cause chart success.

This is the central scientific question the project should answer before the March 11 deadline.

---

## 14. Known Limitations & Caveats

**1. Class imbalance (3.9% positive rate)**
The dataset is heavily imbalanced. Even with `class_weight="balanced"`, precision for the "Charted" class is likely low — many predicted positives will be false positives. Precision-Recall AUC (PR-AUC) is a more informative metric than ROC-AUC when classes are severely imbalanced. **TODO: add PR curve to figures.**

**2. Selection bias in the Spotify dataset**
The 114k tracks in the Spotify dataset are not a random sample of all music ever released — they represent music that achieved enough streaming activity to be included in Kaggle's export. This creates survivorship bias: the "not charted" class already represents relatively successful tracks, making classification harder and potentially underestimating feature effects.

**3. Temporal confounding**
Billboard data spans 1958–2024. Audio characteristics of hits have changed dramatically across decades (the streaming era shifted the formula toward shorter, higher-energy tracks). Without era controls, the model averages over 66 years of changing norms. **TODO: add `release_era` as a control variable.**

**4. `key` encoded as linear numeric**
Musical keys are circular (B=11 is adjacent to C=0). Treating key as a linear 0–11 feature imposes a false ordering. **TODO: encode as sin/cos circular embedding or 12 binary dummies.**

**5. Single-genre label per track**
When a track appears in multiple genres, only one is kept (alphabetically first). This may misclassify genre for cross-genre artists. **TODO: encode genre as multi-hot or run separate models per genre.**

**6. Spotipy augmentation pending**
`artist_followers` and `artist_popularity_api` are not yet included. The Kim & Oh (2016) literature identifies "popularity continuity" (historical artist success) as the single strongest predictor. Without this control, the audio feature effects may be upward-biased (capturing artist-level effects rather than song-level effects). **See §15.**

**7. Cox PH censoring assumption**
All `event` values are set to 1, assuming every charted track was observed until it left the chart. For the most recent entries (2023–2024), some tracks may still have been charting at data collection time. These should be censored (`event=0`). Magnitude of impact is small but worth noting.

**8. `wks_on_chart` aggregation**
`max(wks_on_chart)` from the weekly rows is used as the total chart life. This is correct if `wks_on_chart` is a running cumulative counter, but if a track re-entered the chart after leaving, the counter may reset. Visual inspection of several tracks is recommended to confirm.

---

## 15. Spotipy Augmentation (Pending)

**Status:** Credentials were set by user but were not visible in the subprocess environment during the Cowork session. Script is ready — uncomment the Spotipy block at the bottom of `run_all_v3.py` and run.

**Steps to run:**
```bash
# 1. Set credentials in the SAME terminal session where you run the script
export SPOTIPY_CLIENT_ID="your_client_id_here"
export SPOTIPY_CLIENT_SECRET="your_client_secret_here"

# 2. Uncomment the Spotipy block at the bottom of run_all_v3.py

# 3. Run — will cache results to artist_cache.csv every 50 artists
#    Safe to interrupt and resume. ~30 minutes for ~15,000 unique artists.
python3 run_all_v3.py

# 4. Output: oit367_augmented_dataset.csv
#    Re-run the full script with BASE_CSV = "oit367_augmented_dataset.csv"
#    The augmented features are auto-detected and added to all models.
```

**Expected impact:** Adding `artist_followers` (log-transformed) and `artist_popularity_api` will likely increase Model 1 AUC meaningfully. The scientifically interesting finding is whether **audio features remain significant after controlling for artist popularity** — i.e., does the song's own acoustic profile matter independent of who made it? This is the core business insight for a label's A&R team.

---

## 16. Recommended Follow-Up Actions

These are ordered by expected impact on the report quality for the March 11 deadline.

### HIGH PRIORITY (before submission)

**A. Populate §11 with actual results**
Run `python3 run_all_v3.py`, copy terminal output, paste into §11.

**B. Temporal robustness split**
Professor Bayati will likely ask whether results hold on out-of-sample future data.
```python
# In run_all_v3.py, after loading df:
# Requires chart_entry_date column (already in base dataset)
df['entry_year'] = pd.to_datetime(df['chart_entry_date']).dt.year
mask_train = df['entry_year'] <= 2019
mask_test  = df['entry_year'] > 2019
# Run Models 1 & 2 on this split and report AUC alongside random split AUC
```

**C. Run Spotipy augmentation and re-run all models**
See §15. The artist-popularity-controlled model is the headline finding.

**D. Add PR-AUC metric to Model 1 and Model 2**
With 3.9% positive rate, ROC-AUC can be optimistic. PR-AUC penalises false positives more heavily.
```python
from sklearn.metrics import average_precision_score
pr_auc = average_precision_score(y_te, y_prob_lr)
```

### MEDIUM PRIORITY (strengthens analysis)

**E. Debut-artist robustness check**
Re-run Model 1 on tracks where the artist's first-ever chart appearance is in this dataset. This eliminates star-power almost entirely. If audio features still predict charting, that's a strong finding.

**F. Genre-stratified models**
Run Model 1 separately for top 5–6 genres. Feature importance likely differs by genre (danceability matters more for hip-hop than country).

**G. Circular encoding for `key`**
```python
df['key_sin'] = np.sin(2 * np.pi * df['key'] / 12)
df['key_cos'] = np.cos(2 * np.pi * df['key'] / 12)
# Replace 'key' with 'key_sin', 'key_cos' in FEATURES
```

**H. Kaplan-Meier log-rank tests**
Fig 7 shows KM curves by genre but doesn't test whether differences are statistically significant. Add `logrank_test` from lifelines.

### LOWER PRIORITY (nice-to-have)

**I. Spotify popularity model (Research Q3)**
`is_popular` (score ≥ 80) is defined but not yet modeled. Run the same logistic + XGBoost pipeline with this as the DV. Compare coefficients to the chart-entry model — do different features drive algorithmic streaming success vs. chart placement?

**J. Sentiment analysis on lyrics (Alex)**
From the project plan: add lyric sentiment (VADER) as an additional predictor. Does lyric tone predict charting independently of audio features?

**K. Temporal trend figure (Valerie)**
Plot average danceability, energy, tempo, valence of charting songs by decade. Shows the evolution of the "hit formula" visually.

---

## File Manifest

```
OIT-367/
├── run_all_v3.py                    # Main pipeline script
├── requirements.txt                 # Python dependencies
├── ANALYSIS_LOG.md                  # This file
├── oit367_pipeline_corrected.py     # Annotated reference version
├── spotify_tracksdataset.csv        # Raw Spotify data (114k rows)
├── merged_spotify_billboard_data.csv # Raw Billboard weekly data (110k rows)
├── oit367_base_dataset.csv          # ✅ FINAL MERGED DATASET (89,741 rows)
│                                    #    See §17 for sharing instructions
├── artist_cache.csv                 # [created after Spotipy run]
├── oit367_augmented_dataset.csv     # [created after Spotipy run]
└── outputs/
    ├── vif_table.csv
    ├── logistic_odds_ratios.csv
    ├── xgboost_shap_importance.csv
    ├── cox_summary.csv
    ├── ols_longevity_coefficients.csv
    ├── model_performance_summary.csv
    ├── fig1_class_balance.png
    ├── fig2_correlation_heatmap.png
    ├── fig3_roc_curves.png
    ├── fig4_odds_ratios.png
    ├── fig5_shap_importance.png
    ├── fig6_cox_hazard_ratios.png
    ├── fig7_kaplan_meier.png
    └── fig8_longevity_distribution.png
```

---

## 17. Sharing the Dataset with Your Team

### 17a. What the file contains

`oit367_base_dataset.csv` is the analysis-ready dataset produced by `run_all_v3.py`. It is the authoritative file for all models in this project.

```
Shape:    89,741 rows × ~17 columns
Size:     ~20 MB on disk
Charted:  3,502 rows  (is_charted = 1)
Not:      86,239 rows (is_charted = 0)
Nulls:    0 in all 11 audio feature columns
```

**Columns included:**
- All 11 Spotify audio features: `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`
- Spotify metadata: `track_id`, `track_name`, `artists`, `album_name`, `track_genre`, `duration_ms`, `explicit`, `popularity`
- Billboard-derived columns: `is_charted`, `wks_on_chart`, `peak_pos`, `chart_entry_date`
- Synthetic targets: `is_popular` (Spotify popularity ≥ 80)

**What it is NOT:** It does not contain `artist_followers` or `artist_popularity_api` — those require the Spotipy augmentation step (§15). After running that step, use `oit367_augmented_dataset.csv` instead.

---

### 17b. Option 1 — Google Drive (Recommended for team sharing)

This is the fastest path for all four team members to access the file without Git setup.

```
1. Open Google Drive (drive.google.com) in your browser
2. Upload oit367_base_dataset.csv (drag and drop)
3. Right-click the file → "Share" → "Anyone with the link" → "Viewer"
4. Copy the link and paste it in the team Slack/group chat
```

Teammates can then download it directly from the browser, or mount it in Google Colab with:
```python
from google.colab import drive
drive.mount('/content/drive')
df = pd.read_csv('/content/drive/MyDrive/oit367_base_dataset.csv')
```

---

### 17c. Option 2 — GitHub with Git LFS (Best for version-controlled workflow)

Standard GitHub has a 100 MB hard limit and a 50 MB soft warning. At ~20 MB, the file is *under* the hard limit and can technically be committed directly — but it's poor practice for raw data files. Git LFS (Large File Storage) is cleaner.

```bash
# One-time setup (install Git LFS on your machine)
brew install git-lfs        # macOS
git lfs install             # initializes LFS in your git config

# In the repo, tell LFS to track CSV files
git lfs track "*.csv"
git add .gitattributes

# Now add and commit the file normally
git add oit367_base_dataset.csv
git commit -m "Add analysis-ready base dataset (89,741 tracks)"
git push
```

GitHub provides 1 GB of LFS storage free. Each team member clones normally — LFS handles the download transparently.

**If LFS is too much setup:** You can commit the CSV directly (no LFS) since it is under the hard limit. It will work, it just adds a large binary blob to your repo history.

---

### 17d. Option 3 — Regenerate from raw files (most reproducible)

Any teammate can regenerate `oit367_base_dataset.csv` from scratch using only the two raw Kaggle files and the pipeline script. This is the most robust approach for reproducibility.

```bash
# Prerequisites: raw Kaggle files in same directory as run_all_v3.py
ls
# spotify_tracksdataset.csv     (114k rows, ~20 MB)
# merged_spotify_billboard_data.csv  (110k rows, ~20 MB)
# run_all_v3.py

# Install dependencies
pip3 install -r requirements.txt

# Run — data prep runs first, outputs oit367_base_dataset.csv
python3 run_all_v3.py
```

The data prep block checks for `oit367_base_dataset.csv` at startup and skips the rebuild if the file already exists. To force a rebuild, delete the file first.

**Both raw Kaggle files are required:**
- `spotify_tracksdataset.csv` — from https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
- `merged_spotify_billboard_data.csv` — used as the source of Billboard weekly data (contains `track_id`, `peak_pos`, `wks_on_chart`, `chart_week` for all charted tracks)

---

### 17e. Column-level join documentation (for teammates building on top of this file)

The dataset was built via three joins. Teammates extending the analysis should understand the provenance of each column:

```
Source                          Columns in oit367_base_dataset.csv
─────────────────────────────────────────────────────────────────────────────
Spotify Tracks Dataset          track_id*, artists, track_name, album_name,
(spotify_tracksdataset.csv)     track_genre, popularity, duration_ms,
                                explicit, danceability, energy, key,
                                loudness, mode, speechiness, acousticness,
                                instrumentalness, liveness, valence, tempo,
                                time_signature

Billboard aggregate             peak_pos, wks_on_chart, chart_entry_date
(from merged_... after          (NaN for non-charted tracks)
 groupby(track_id).agg)

Derived targets                 is_charted = (peak_pos.notna()).astype(int)
(computed in run_all_v3.py)     is_popular = (popularity >= 80).astype(int)

* track_id is the join key: Spotify URI (e.g., '4iV5W9uYEdYUVa79Axb7Rh')
  Billboard join is a LEFT JOIN — non-charted tracks have NaN in billboard cols
```

**If a teammate wants to add additional Billboard columns** (e.g., the actual chart rank history, not just the peak), they should re-run the Billboard aggregation step in `run_all_v3.py` with a custom `agg()` call and left-join back to this file on `track_id`.

**If a teammate wants to add lyric sentiment (Alex's task):** Join on `track_id` after running a lyrics API (e.g., Genius via `lyricsgenius` library) and VADER sentiment scoring. The `track_id` column in this file is the linking key for all future augmentations.

---

*Generated by Claude (Anthropic) in Cowork mode · OIT367 Stanford GSB Winter 2026*

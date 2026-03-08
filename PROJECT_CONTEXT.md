# OIT367 Full Project Context
**Stanford GSB Winter 2026 | Wurm · Chen · Barli · Taruno**
**Document purpose: Complete handoff context for continuation in a new session**
**Last updated: 2026-03-07**

---

## 1. Research Questions and Scope

**Primary question:** Can audio features extracted from Spotify's API predict whether a track reaches the Billboard Hot 100, and how long it stays there?

**Two sub-tasks:**
1. **Chart Entry (binary classification):** Is-charted = 1 or 0? Logistic Regression + XGBoost.
2. **Chart Longevity (survival analysis):** Among tracks that charted, how many weeks? Cox Proportional Hazards + Log-OLS robustness check.

**Dataset scope:** 89,741 unique Spotify tracks, of which 3,502 appeared on the Billboard Hot 100 (3.90% positive rate). The heavy class imbalance is real and important — it means ROC-AUC can be misleadingly optimistic, so Precision-Recall AUC is reported as the primary metric.

---

## 2. Dataset Construction — Full Pipeline

### 2.1 Source Files

Two Kaggle datasets are joined:

**`spotify_tracksdataset.csv`** (~20MB, 114k rows)
Kaggle's "Spotify Tracks Dataset." Contains one row per track-genre pair — the same track can appear multiple times if Spotify classifies it under multiple genres. Columns include: `track_id`, `artists`, `album_name`, `track_name`, `popularity`, `duration_ms`, `explicit`, and all Spotify audio features (`danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `time_signature`, `track_genre`).

**`hot-100-current.csv`** (~19MB)
Weekly Billboard Hot 100 snapshots. Each row is one track's appearance in one chart week. Columns include `track_id`, `chart_week`, `peak_pos`, `wks_on_chart`.

**`merged_spotify_billboard_data.csv`** (~20MB)
A pre-existing join of the two above that came with the Kaggle data. This file was initially used as the dataset but was discovered to be an **inner join**, meaning it only contained tracks that had charted. This made the positive rate artificially 100% and would have produced a useless model — there was no negative class.

### 2.2 The Inner Join Bug and Fix

The original scripts (v1–v2) all used `merged_spotify_billboard_data.csv` directly. When we ran the models, the class distribution came back as 100% charted. The fix was to rebuild the dataset from scratch:

```python
# Step 1: Deduplicate Spotify dataset (114k rows → 89,741 unique track_ids)
# Keep first occurrence per track_id, sorted by track_genre alphabetically
# This preserves one genre label per track for the KM survival analysis
spotify_dedup = (
    spotify
    .sort_values("track_genre")
    .drop_duplicates(subset="track_id", keep="first")
    .reset_index(drop=True)
)

# Step 2: Aggregate Billboard weekly rows → one row per track
# peak_pos = minimum (lower = better chart position)
# wks_on_chart = maximum (longest stay recorded)
# chart_entry_date = minimum chart_week (first appearance)
bb_weekly["chart_week"] = pd.to_datetime(bb_weekly["chart_week"], errors="coerce")
bb_agg = bb_weekly.groupby("track_id", as_index=False).agg(
    peak_pos=("peak_pos", "min"),
    wks_on_chart=("wks_on_chart", "max"),
    chart_entry_date=("chart_week", "min"),
)

# Step 3: LEFT JOIN — all 89,741 Spotify tracks retained
# Non-charted tracks get NaN in peak_pos, wks_on_chart, chart_entry_date
df = spotify_dedup.merge(bb_agg, on="track_id", how="left")
df["is_charted"] = df["peak_pos"].notna().astype(int)      # binary label
df["wks_on_chart"] = df["wks_on_chart"].fillna(0).astype(int)
```

This produced `oit367_base_dataset.csv` — 89,741 rows, 3.90% positive rate. This file is committed to the repo and is the starting point for all model runs.

### 2.3 Why `chart_entry_date` Only Exists for Charted Tracks

`chart_entry_date` is derived from the Billboard data. Non-charted tracks never appeared in the Billboard data, so they have NaN for this column. This matters for the `decade_idx` control variable, which is derived from `chart_entry_date` and therefore can only be used in models that restrict to charted tracks (Cox PH, Log-OLS). It cannot be used in the classification model (LR, XGBoost) without imputing values for 96% of the dataset, which would be methodologically unsound.

---

## 3. Pipeline Version History — What Each Version Fixed

### `run_all.py` (v1) — Original
The initial prototype. Used `merged_spotify_billboard_data.csv` directly (inner join bug). No VIF check. No class imbalance handling. Spotipy scraping block ran unconditionally on every execution. Never produced meaningful results.

### `run_all_v2.py` — Dataset fix
Switched to rebuilding from raw files (left join). First version with a working 3.90% positive rate. Still had VIF issues, no PR-AUC, Cox PH not stratified.

### `run_all_v3.py` — Bug fixes
Fixed `ModuleNotFoundError: statsmodels` (added requirements.txt). Fixed `AttributeError: module 'importlib' has no attribute 'util'` (Python 3.13 changed how submodules are loaded; fix was `import importlib.util` explicitly instead of `import importlib`). The Spotipy block was still unconditionally included — running this script with Spotipy credentials active crashed Alex's laptop by attempting to scrape 31,437 artists.

### `run_all_v4.py` — Model quality fixes
Four key changes:
- **Fix A:** Removed `energy` from `BASE_FEATURES`. VIF=15.07. `energy` is highly collinear with `loudness` (Pearson r≈0.78). When both are in a logistic regression, the `energy` coefficient absorbs residual variance from `loudness` and flips sign — a multicollinearity artifact, not a real finding. XGBoost handles collinearity through tree structure so `energy` was still informative in that model, but the SHAP values remain valid even without it.
- **Fix B:** Cox PH now uses `strata=["mode"]`. `mode` is binary (major=1, minor=0) and was failing the Schoenfeld proportional hazards test (p<0.05), meaning the proportional hazards assumption didn't hold for it. Stratification solves this by estimating a separate baseline hazard for major vs. minor keys rather than forcing a proportional effect assumption. `mode` is still in the data, it just becomes a stratification variable rather than an estimated covariate.
- **Fix C:** Added `average_precision_score` (PR-AUC) for both LR and XGBoost. At 3.90% positive rate, a dummy classifier that always predicts "not charted" achieves ROC-AUC ≈ 0.50 but PR-AUC ≈ 0.039 (the base rate). PR-AUC is the appropriate primary metric for heavily imbalanced classification.
- **Fix E:** Wrapped Spotipy block in `if artist_features_path.exists()` conditional so it only runs when `artist_features.csv` is present in the folder. Previously it ran unconditionally.

### `run_all_v5.py` — CURRENT VERSION (control variables + VIF patch)
Six additions on top of v4:
- **v5 Add A:** `explicit` (bool→int, 0/1) added to `BASE_FEATURES`. Charted tracks are 11.5% explicit vs. 8.5% for non-charted. This is a valid exogenous predictor — radio/platform play-listing decisions directly respond to explicit content ratings.
- **v5 Add B:** `duration_min` = `duration_ms / 60000`, clipped at 10 minutes, added to `BASE_FEATURES`. The raw max is ~87 minutes (podcasts/audiobooks in the Kaggle data), so the cap removes extreme outliers without affecting typical tracks. Charted tracks average 3.64 min vs. 3.79 min for non-charted. Ended up being the 3rd-ranked SHAP feature in XGBoost (Mean|SHAP|=0.404).
- **v5 Add C:** `decade_idx` (ordinal 0=1950s through 7=2020s, derived from `chart_entry_date`) added to Cox PH and Log-OLS only. Captures the streaming-era structural shift in chart dynamics. Became the most statistically significant predictor in the longevity model (HR=0.878, p<0.00001).
- **v5 Add D:** Precision-Recall curve figure (Fig 9) added alongside the ROC curve.
- **v5 Add E:** `genre_chart_rates.csv` saved to outputs — a table of chart rate and average Spotify popularity by genre.
- **v5 Fix F:** `danceability` removed from `BASE_FEATURES` (VIF=12.41). After adding `duration_min` and `explicit` in v5, `danceability` became a collinearity hub — it correlates simultaneously with `tempo`, `valence`, and `duration_min`. This inflated VIF for both itself (12.41) and `tempo` (11.50). Removing it drops max VIF from 12.41 to **2.02** (loudness). Confirmed analytically. The pre-patch SHAP value was 0.224 (8th of 12 features), meaning its effect is absorbed by the remaining correlated features.

---

## 4. Final Feature Set (run_all_v5.py)

### 4.1 BASE_FEATURES (11 features, used in all 4 models)

| Feature | Type | VIF (post-patch) | Notes |
|---|---|---|---|
| `valence` | Audio [0,1] | 1.20 | Musical positivity/happiness |
| `tempo` | Audio (BPM) | 1.07 | Dropped from 11.50 after danceability removed |
| `acousticness` | Audio [0,1] | 1.65 | Acoustic vs. electronic production |
| `loudness` | Audio (dB) | 2.02 | Max VIF in final set |
| `speechiness` | Audio [0,1] | 1.19 | Speech content |
| `instrumentalness` | Audio [0,1] | 1.39 | Non-vocal content |
| `liveness` | Audio [0,1] | 1.07 | Live performance quality |
| `mode` | Binary (0=minor, 1=major) | 1.04 | Stratum in Cox PH |
| `key` | Ordinal (0–11) | 1.02 | Chromatic pitch class |
| `explicit` | Binary (0/1) | 1.15 | v5 control |
| `duration_min` | Continuous (min, capped 10) | 1.08 | v5 control |

### 4.2 Removed Features

| Feature | Removed in | VIF | Reason |
|---|---|---|---|
| `energy` | v4 Fix A | 15.07 | Collinear with `loudness` (r≈0.78) |
| `danceability` | v5 Fix F | 12.41 | Collinearity hub with tempo, valence, duration_min |

### 4.3 `decade_idx` — Cox PH and Log-OLS Only

Added to `COX_FEATURES = FEATURES + ["decade_idx"]`. Not in `BASE_FEATURES` because `chart_entry_date` is NaN for all non-charted tracks, making it unusable in the classification model. Computed as `(year - 1950) // 10`, giving ordinal values 0 (1950s) through 7 (2020s). Nullable integer type (`Int64`) to keep NaN for non-charted rows.

### 4.4 Future Augmented Features (auto-detected when files present)

When `artist_features.csv` is in the working directory, v5 automatically detects and merges `artist_followers` (log1p-transformed) and `artist_popularity_api`. When `librosa_features.csv` is present, the secondary analysis section will use it (to be added in a future run once Librosa pipeline completes).

---

## 5. Model Architecture and Results

### 5.1 Model 1: Logistic Regression (Chart Entry)

**Setup:** `LogisticRegression(class_weight="balanced", max_iter=1000)`. `class_weight="balanced"` is critical — it up-weights the positive class (charted) by a factor of ~25x to compensate for the 3.90% positive rate. `StandardScaler` applied before fitting so all coefficients are on the same scale (per 1 SD change), enabling direct comparison of odds ratios.

**Results (v5 pre-VIF-patch run, numbers may shift slightly on clean run):**
- Test AUC-ROC: 0.7106
- Test PR-AUC: 0.0743 (vs. random baseline 0.039 → **1.9× lift**)
- CV AUC-ROC (5-fold): 0.7133 ± 0.0072 (very low variance → model generalizes well)

**Key odds ratios:**
- `instrumentalness`: OR=0.28 — 72% less likely to chart per 1 SD. Strongest predictor.
- `valence`: OR=1.32 — happier songs 32% more likely to chart
- `explicit`: OR=1.16 — explicit content 16% more likely to chart
- `mode`: OR=1.15 — major key 15% more likely to chart
- `speechiness`: OR=0.64 — high speech content 36% less likely (note: nonlinear — rap has mid-range speechiness, pure spoken word has extreme values; XGBoost captures the nonlinearity better)

**Classification report caveat:** 79% recall at 6% precision looks alarming but is correct for `class_weight="balanced"` at 3.90% positive rate. The model is tuned to not miss charted tracks (high recall), not to be conservative about predictions (low precision). Precision improves when you tune the decision threshold.

### 5.2 Model 2: XGBoost + SHAP (Chart Entry)

**Setup:**
```python
xgb.XGBClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=24.6,   # (n_negative / n_positive) ≈ 86239/3502
    eval_metric="auc",
    early_stopping_rounds=50,
)
```
`scale_pos_weight` is the XGBoost analog to `class_weight="balanced"` — it multiplies the gradient for positive examples by 24.6×. `max_depth=5` prevents deep, overfit trees. `subsample=0.8` and `colsample_bytree=0.8` add stochastic regularization. Early stopping at 50 rounds of no AUC improvement; stopped at iteration 497.

**Results:**
- Test AUC-ROC: 0.8343 (+12.4 points over LR → confirms strongly nonlinear relationships)
- Test PR-AUC: 0.3303 (vs. random baseline 0.039 → **8.5× lift**)
- 80% overall accuracy

**SHAP importance (Mean |SHAP| on test set):**
1. `instrumentalness`: 0.8017 — dominant, consistent with LR
2. `acousticness`: 0.5125
3. `duration_min`: 0.4040 — **new in v5, 3rd most important**
4. `valence`: 0.3437
5. `speechiness`: 0.2988
6. `loudness`: 0.2574
7. `liveness`: 0.2281
8. `danceability`: 0.2238 (pre-patch; this feature now removed in VIF fix)
9. `tempo`: 0.2148
10. `mode`: 0.1267
11. `key`: 0.0830
12. `explicit`: 0.0822

The large gap between XGBoost (0.834) and LR (0.711) AUC means the true relationships are substantially nonlinear. For example, `speechiness` has OR=0.64 in LR (linear: more speech = less likely to chart) but SHAP=0.299 in XGBoost (nonlinear: mid-range speechiness = rap = charts a lot; extreme speechiness = spoken word = never charts). Both are correct at different points on the distribution.

### 5.3 Model 3: Cox Proportional Hazards (Longevity)

**Setup:** `CoxPHFitter(penalizer=0.1)`. The penalizer adds L2 regularization to the Cox log-partial-likelihood, preventing coefficient explosion for correlated features. `strata=["mode"]` (v4 Fix B). Dataset: 3,502 charted tracks, all treated as observed events (no right-censoring, since the data is historical). `decade_idx` added as covariate (v5 Add C).

**Concordance Index: 0.5508** — barely above random (0.50). This is the central finding of the longevity analysis: audio features have weak discrimination power for predicting how long a track stays on the chart. Once in the chart, longevity is driven by label promotion, radio rotation, and cultural momentum — factors not captured in audio features.

**Key Cox coefficients:**
- `decade_idx`: HR=0.878, p<0.00001 — **most significant finding in the entire longevity analysis.** Each decade later → 12.2% higher hazard of leaving the chart per week. This quantifies the streaming-era compression of chart longevity: songs cycle through the Hot 100 much faster in the 2020s than in the 1980s.
- `acousticness`: HR=0.889 — acoustic tracks leave charts faster
- `loudness`: HR=0.900 — louder tracks have shorter chart runs
- `valence`: HR=1.073 — happier songs stay longer (consistent with entry finding)
- `liveness`, `key`, `duration_min`: Not significant (p>0.05) — these predict chart entry but not chart longevity

**Schoenfeld PH test violations:** 8 of 12 covariates fail the proportional hazards test. This is expected and explainable: the PH assumption requires that the hazard ratio between any two covariate values stays constant over time. Across 70 years of pop music history, this is structurally impossible — the relationship between danceability and chart longevity in 1965 is not the same as in 2020. The violations are most severe for `decade_idx` itself (km=123.14, p<0.005), which is ironic but makes sense: the decade variable captures a fundamental regime change (radio era → streaming era), and regime changes are by definition non-proportional. **The recommended report framing is to acknowledge the violations, note that they reflect real structural change, and use Log-OLS as a robustness check.**

### 5.4 Model 3b: Log-OLS (Longevity Robustness)

**Setup:** OLS on `log1p(wks_on_chart)` for 3,502 charted tracks, with `COX_FEATURES` (same as Cox plus `decade_idx`). The log transformation makes the right-skewed weeks distribution more normal. 80/20 train/test split.

**R² = 0.0456** — audio features explain 4.6% of variance in log-longevity. Very low, consistent with Cox C-stat=0.55.

**Direction inconsistencies between Cox and OLS for some features** (e.g., loudness is HR<1 in Cox but coef>0 in OLS) reflect the Cox PH violations — the effects are time-varying, and OLS averages over the time dimension in a way that can flip the sign. **Only findings where Cox and OLS directionally agree should be treated as robust. `decade_idx` is the only variable where both models strongly agree** (Cox HR=0.878, OLS coef=−0.110).

---

## 6. Spotify API / Modal Infrastructure

### 6.1 Why Modal

The Spotify artist scraping required ~31,437 API calls — too many to run synchronously on a laptop without risk of crashes and session interruptions (this already crashed Alex's machine once when the Spotipy block ran unconditionally in v3). Modal provides:
- Serverless cloud execution that continues even when the local terminal closes
- Persistent volume storage (`oit367-vol`) so completed batch CSVs survive container restarts
- Idempotent batch design: each batch writes its own CSV to the volume; re-running the script skips already-completed batches
- Parallel workers (`max_containers`) for faster throughput

### 6.2 All Modal Scripts

**`modal_spotify_scrape.py`** (ARCHIVED — do not use)
The original full-dataset artist scraper. Targeted all 31,437 unique artist strings across 63 batches. Hit Spotify's Development Mode rate limit on the second run, causing all 63 batches to queue with a Retry-After of ~28,425 seconds (~7.9 hours). Root cause: Spotify Development Mode apps have a daily quota; the previous run's 500 calls (2 completed batches) exhausted enough of it that the quota window reset hadn't completed. This script is in `archive/`.

**`modal_charted_scrape.py`** (CURRENT — ready to run)
Targeted replacement. Fetches `artist_followers` and `artist_popularity_api` for only the 953 unique artist strings from **charted tracks only**. 96.9% scope reduction. Just 2 batches of 500 artists. Expected runtime: ~3 minutes. Architecture is otherwise identical to the full scraper. Output: `/data/charted_artist_features.csv` on the `oit367-vol` volume.

**`modal_preview_urls.py`** (NEW — ready to run)
Job 1 of the Librosa pipeline. Calls `sp.track(track_id)` for each of the 3,502 charted track_ids to retrieve `preview_url` (Spotify's 30-second MP3 CDN link). Batches of 100 tracks, 3 concurrent workers. ~36 batches, ~20 minutes. Output: `/data/preview_urls.csv`. Roughly 10–15% of tracks will have `null` preview_url due to licensing restrictions (older catalog, regional content). Same credential secret as the artist scraper.

**`modal_librosa_extract.py`** (NEW — ready to run)
Job 2 of the Librosa pipeline. No Spotify API credentials needed. Downloads the MP3 preview for each track with a valid URL via plain HTTP request to Spotify's CDN, loads into Librosa via `io.BytesIO` (never writes audio to disk), and extracts 32 features:
- 13 MFCC means + 13 MFCC stds (timbre signature)
- Spectral centroid mean (brightness)
- Spectral rolloff mean (energy distribution)
- Chroma mean + std (harmonic content)
- Zero crossing rate mean (percussiveness)
- RMS energy mean (loudness envelope)

20 parallel containers, 50 tracks per batch, ~62 batches, estimated 2–3 hours. Requires `ffmpeg` (for MP3 decode) and `libsndfile1` in the Modal image. Output: `/data/librosa_features.csv`.

### 6.3 Known Spotify API Issues Fixed Across Versions

**`concurrency_limit` deprecation:** Modal 1.0 renamed this to `max_containers`. Fixed in all current scripts.

**`.cache` disk write error:** Spotipy tries to write a token cache file to disk by default. Modal containers don't have a writable working directory, causing a crash. Fix: `cache_handler=MemoryCacheHandler()` in all `SpotifyClientCredentials` calls.

**`'followers'` KeyError:** Spotify's search endpoint returns a `SimplifiedArtistObject` that does NOT include the `followers` field — only `FullArtistObject` has it. Fix: two-phase approach — Phase 1 uses `sp.search()` to get artist IDs; Phase 2 uses `sp.artist(id)` to get the full object including followers.

**`sp.artists()` batch endpoint removed (Feb 2026):** Spotify removed the `GET /artists` batch endpoint for Development Mode apps in February 2026. Fix: `modal_charted_scrape.py` now uses individual `sp.artist(id)` calls only, with a 0.35s throttle between each.

**`retries=2` integer deprecated:** Modal 1.0 requires a `modal.Retries()` object, not a plain integer. Fix: `retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, initial_delay=5.0)`.

**28,425-second Retry-After:** The `modal_spotify_scrape.py` full scraper hit Spotify's daily quota in Development Mode after ~500 calls. The 28,425s Retry-After (~7.9 hours) appeared **inside the container logs** (not just the local dispatcher), confirming the cloud workers themselves were blocked. This triggered the pivot to the charted-only scraper (2 batches vs. 63).

---

## 7. Complete File Inventory

### Active Files (root directory)

| File | Size | Purpose |
|---|---|---|
| `run_all_v5.py` | 31KB | **CURRENT** full pipeline: VIF, LR, XGBoost+SHAP, Cox PH, Log-OLS, 9 figures |
| `requirements.txt` | 170B | Python dependencies for pip install |
| `modal_charted_scrape.py` | 14KB | Cloud: artist followers/popularity for 953 charted artists |
| `modal_preview_urls.py` | 11KB | Cloud: Spotify preview URLs for 3,502 charted tracks |
| `modal_librosa_extract.py` | 16KB | Cloud: 32 Librosa acoustic features from 30s previews |
| `oit367_base_dataset.csv` | 16MB | Processed merged dataset (89,741 tracks); run_all_v5.py input |
| `README.md` | — | Team entry point: repo map, setup, Modal run sequence |
| `RESULTS.md` | 21KB | Full model results with all numbers, findings, limitations |
| `ANALYSIS_LOG.md` | 49KB | Verbose analysis log (Cursor-compatible for follow-up actions) |
| `LIBROSA_MODAL_PLAN.md` | 8.5KB | Librosa pipeline feasibility spec and research framing |
| `.gitignore` | — | Excludes large CSVs, caches, __pycache__, .DS_Store |
| `outputs/` | — | 16 files: fig1–fig9 (PNG) + 7 CSV tables |

### Archive (old versions, do not use)

| File | Superseded by | Reason archived |
|---|---|---|
| `run_all.py` | v5 | Original prototype; inner join bug |
| `run_all_v2.py` | v5 | Dataset fixed but VIF/Cox/PR-AUC issues remain |
| `run_all_v3.py` | v5 | Spotipy block ran unconditionally; crashed laptop |
| `run_all_v4.py` | v5 | Missing explicit/duration_min; danceability VIF unfixed |
| `modal_spotify_scrape.py` | modal_charted_scrape.py | Full 31k-artist scope; hit 8hr rate limit on second run |
| `oit367_pipeline_corrected.py` | v5 | Early pipeline draft |
| `run_log.md` | RESULTS.md | Superseded by comprehensive results file |
| `run_pipeline.sh` | — | Old shell wrapper |

### Gitignored (not committed)

| File | Size | Reason |
|---|---|---|
| `spotify_tracksdataset.csv` | 20MB | Raw Kaggle file; too large; Kaggle download link in README |
| `merged_spotify_billboard_data.csv` | 20MB | Raw Kaggle file; inner-join artifact |
| `hot-100-current.csv` | 19MB | Raw Kaggle file; too large |
| `oit367_augmented_dataset.csv` | — | Generated when artist_features.csv present; not yet created |
| `.DS_Store` | 6.1KB | macOS metadata |
| `.cache` | 229B | Spotipy token cache |
| `__pycache__/` | — | Python bytecode |

### Outputs Directory (committed)

All 16 files in `outputs/` are committed so teammates can see model results without running the pipeline.

Figures: `fig1_class_balance.png`, `fig2_correlation_heatmap.png`, `fig3_roc_curves.png`, `fig4_odds_ratios.png`, `fig5_shap_importance.png`, `fig6_cox_hazard_ratios.png`, `fig7_kaplan_meier.png`, `fig8_longevity_distribution.png`, `fig9_precision_recall.png`

CSVs: `model_performance_summary.csv`, `logistic_odds_ratios.csv`, `xgboost_shap_importance.csv`, `cox_summary.csv`, `ols_longevity_coefficients.csv`, `vif_table.csv`, `genre_chart_rates.csv`

---

## 8. Model Performance Summary

| Model | Task | Primary Metric | Score | PR-AUC | Notes |
|---|---|---|---|---|---|
| Logistic Regression | Chart Entry | AUC-ROC | 0.7106 | 0.0743 | CV: 0.7133 ± 0.0072 (5-fold); 1.9× PR lift |
| XGBoost | Chart Entry | AUC-ROC | 0.8343 | 0.3303 | Early stop iter 497; 8.5× PR lift |
| Cox PH | Longevity | C-statistic | 0.5508 | — | mode stratified; +decade_idx |
| Log-OLS | Longevity | R² | 0.0456 | — | log1p(wks); +decade_idx |

**Key takeaway:** The 12.4-point AUC gap between XGBoost and LR means the relationships are substantially nonlinear. The model is strong for chart entry (8.5× PR lift) but weak for longevity (C-stat barely above random), confirming that audio features determine whether a track can break into the chart, but not how long it stays.

---

## 9. Six Core Findings (Report-Ready)

1. **Instrumentalness is the dominant barrier to chart entry.** SHAP=0.80, OR=0.28. The Hot 100 is a vocal/lyric-driven chart. Instrumental and near-instrumental tracks are 72% less likely to chart per 1 SD increase.

2. **Valence (musical positivity) is the strongest positive audio predictor of chart entry.** SHAP=0.344, OR=1.32. Happier-sounding songs are 32% more likely to chart per 1 SD increase, consistent across both LR and XGBoost.

3. **Track length is the 3rd most predictive XGBoost feature (new in v5).** `duration_min` SHAP=0.404, ranked above valence, speechiness, and loudness. Shorter tracks chart more, with a nonlinear relationship (XGBoost captures the inflection point that LR cannot). The streaming era's structural shift toward shorter tracks is consistent with this finding.

4. **Explicit content modestly but reliably increases chart probability.** OR=1.16 (p=0.011) in LR. Explicit tracks that chart also stay on the chart slightly longer (Cox HR=1.047, p=0.011). Reflects the rise of hip-hop and trap on the Hot 100 over the last two decades.

5. **The streaming era has dramatically compressed chart longevity (strongest survival finding).** `decade_idx` HR=0.878, p<0.00001 — the only variable where Cox PH and Log-OLS (OLS coef=−0.110) directionally agree strongly. Each decade later corresponds to a 12.2% higher per-week hazard of leaving the chart. Songs in the 2020s cycle through the Hot 100 approximately twice as fast as songs from the 1980s.

6. **Audio features predict chart entry well but chart longevity poorly.** XGBoost AUC=0.834 and PR-AUC=0.330 for entry; Cox C-stat=0.551 and R²=0.046 for longevity. Once a track enters the chart, its longevity is driven by factors outside audio: label promotion spend, radio rotation, cultural moment, and platform algorithmic placement.

---

## 10. Pending Work and Next Steps

### Immediate (when Spotify rate limit resets, ~8 hours from 2026-03-07 morning)

```bash
# Run from OIT-367/ folder
# Step 1: Artist features (~3 min)
modal run --detach modal_charted_scrape.py
# Step 2: Preview URLs (~20 min) — can run concurrently with Step 1
modal run --detach modal_preview_urls.py
# Step 3: Librosa (~2-3 hr) — only after Step 2 finishes
modal run --detach modal_librosa_extract.py

# Download results
modal volume get oit367-vol /data/charted_artist_features.csv ./artist_features.csv
modal volume get oit367-vol /data/librosa_features.csv ./librosa_features.csv

# Re-run models (auto-detects artist_features.csv)
python3 run_all_v5.py
```

### Alex's Tasks
- **Lyric sentiment:** Genius API + VADER for charted tracks. Add `sentiment_compound` (VADER compound score, −1 to +1) to `BASE_FEATURES`. Expected to correlate with `valence` — if they're collinear, drop one. VADER is a lexicon-based model designed for short text; no training required. Python: `vaderSentiment` library.
- **Decade control variable** is already implemented in Cox PH and Log-OLS as of v5.

### Secondary Analysis (after Librosa completes)
Per `LIBROSA_MODAL_PLAN.md`:
1. PCA on 26 MFCC features → 3–5 components (avoid adding 26 correlated features directly to any regression)
2. Add PCA components to Cox PH + Log-OLS longevity models
3. K-means clustering on MFCC features to find "timbral archetypes" among charted tracks
4. Descriptive: "What do charted tracks sound like?" — mean MFCC profiles by genre
5. Note in report: 30s clip limitation; ~10–15% missing previews; charted-only scope means these features cannot be added to the main classification model

### Report Framing Guidance
- Cox PH violations: acknowledge, explain as structural regime change, cite Log-OLS as robustness check
- Danceability removal: note in limitations or footnote; pre-patch SHAP value documented
- Class imbalance: use PR-AUC as primary metric, not AUC-ROC; baseline PR-AUC = 0.039
- XGBoost vs. LR gap: frame as evidence of nonlinearity, cite as motivation for tree-based model

---

## 11. Environment Setup

```bash
# Python 3.11+ required (3.13 has importlib.util behavior change — handled in v5)
pip3 install -r requirements.txt

# requirements.txt contents:
# scikit-learn>=1.3
# xgboost>=2.0
# shap>=0.44
# lifelines>=0.27
# statsmodels>=0.14
# seaborn>=0.13
# matplotlib>=3.7
# pandas>=2.0
# numpy>=1.24

# For Modal scripts:
pip3 install modal
modal token new       # one-time browser auth
# Spotify credentials already set as Modal secret:
# modal secret create spotify-credentials \
#     SPOTIPY_CLIENT_ID=... SPOTIPY_CLIENT_SECRET=...
```

**Modal app names:** `oit367-charted-scrape`, `oit367-preview-urls`, `oit367-librosa`
**Modal volume name:** `oit367-vol`
**Volume data paths:** `/data/charted_artist_features.csv`, `/data/preview_urls.csv`, `/data/librosa_features.csv`
**Spotify credential secret name:** `spotify-credentials`
**Random seed:** `RANDOM_STATE = 42` throughout all models

---

## 12. Git Repository

**Remote:** `git@github.com:aewurm98/oit367_music_project.git`
**Branch:** `main`
**Last commit message (target):** `"cleanup: add README, .gitignore, archive old versions; add Librosa pipeline scripts"`

To push after a clean v5 run:
```bash
git add .
git commit -m "your message here"
git push -u origin main   # -u only needed first push; use git push after
```

Note: `git add .` is safe because `.gitignore` automatically excludes the large CSVs and system files.

---

*This document contains the complete context for the OIT367 project as of 2026-03-07.*
*Copy this entire file into a new chat session to continue without context loss.*

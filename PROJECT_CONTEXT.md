# OIT367 Full Project Context
**Stanford GSB Winter 2026 | Wurm · Chen · Barli · Taruno**
**Document purpose: Complete handoff context for continuation in a new session**
**Last updated: 2026-03-08**

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

### `run_all_v5.py` — CURRENT VERSION (control variables + VIF patch + dataset augmentation)
Nine additions on top of v4:
- **v5 Add A:** `explicit` (bool→int, 0/1) added to `BASE_FEATURES`. Charted tracks are 11.5% explicit vs. 8.5% for non-charted. This is a valid exogenous predictor — radio/platform play-listing decisions directly respond to explicit content ratings.
- **v5 Add B:** `duration_min` = `duration_ms / 60000`, clipped at 10 minutes, added to `BASE_FEATURES`. The raw max is ~87 minutes (podcasts/audiobooks in the Kaggle data), so the cap removes extreme outliers without affecting typical tracks. Charted tracks average 3.64 min vs. 3.79 min for non-charted.
- **v5 Add C:** `decade_idx` (ordinal 0=1950s through 7=2020s, derived from `chart_entry_date`) added to Cox PH and Log-OLS only. Captures the streaming-era structural shift in chart dynamics. Became the most statistically significant predictor in the longevity model (HR=0.878, p<0.00001).
- **v5 Add D:** Precision-Recall curve figure (Fig 9) added alongside the ROC curve.
- **v5 Add E:** `genre_chart_rates.csv` saved to outputs — a table of chart rate and average Spotify popularity by genre.
- **v5 Fix F:** `danceability` removed from `BASE_FEATURES` (VIF=12.41). After adding `duration_min` and `explicit` in v5, `danceability` became a collinearity hub. Removing it dropped max VIF significantly. The pre-patch SHAP value was 0.224 (8th of 12 features), meaning its effect is absorbed by the remaining correlated features.
- **v5 Add H:** `lastfm_listeners_log` = `log1p(listeners_lastfm)` added to all 4 models via `artist_features.csv` auto-detection. Sourced from MusicBrainz `artists.csv` (1.47M artists). 65.8% match rate on normalized artist names. VIF=5.19. Became the **2nd most important SHAP feature** (Mean|SHAP|=1.336), above all audio features except instrumentalness.
- **v5 Add I:** `is_us_artist` binary (1=United States, 0=non-US or unknown) added to all 4 models. Sourced from `country_mb` field in `artists.csv`. Conservative imputation: unknown nationality → 0 (non-US), pre-filled at df level so feature passes the >50% non-null auto-detection threshold. VIF=1.47. OR=1.867 in LR.
- **v5 Add J:** Lyric sentiment features (`sentiment_compound`, `sentiment_pos`, `sentiment_neg`, `lyric_word_count`) added to Cox PH and Log-OLS only via `lyric_features.csv` auto-detection. Computed with VADER on the Billboard Top 100 1946–2022 lyrics dataset (41.9% charted-track match rate, n=1,456 lyric-matched subset). `sentiment_neg` HR=1.124, p=0.0008 is the strongest new longevity finding.

---

## 4. Final Feature Set (run_all_v5.py)

### 4.1 BASE_FEATURES (11 audio + 2 artist features, used in all 4 models)

| Feature | Type | VIF | Notes |
|---|---|---|---|
| `valence` | Audio [0,1] | 1.20 | Musical positivity/happiness |
| `tempo` | Audio (BPM) | 1.07 | VIF dropped from 11.50 after danceability removed |
| `acousticness` | Audio [0,1] | 1.65 | Acoustic vs. electronic production |
| `loudness` | Audio (dB) | 2.02 | — |
| `speechiness` | Audio [0,1] | 1.19 | Speech content |
| `instrumentalness` | Audio [0,1] | 1.39 | Non-vocal content |
| `liveness` | Audio [0,1] | 1.07 | Live performance quality |
| `mode` | Binary (0=minor, 1=major) | 1.04 | Stratum in Cox PH |
| `key` | Ordinal (0–11) | 1.02 | Chromatic pitch class |
| `explicit` | Binary (0/1) | 1.15 | v5 Add A control |
| `duration_min` | Continuous (min, capped 10) | 1.08 | v5 Add B control |
| `lastfm_listeners_log` | Continuous (log1p scale) | 5.19 | v5 Add H; Last.fm listener count; 65.8% artist coverage |
| `is_us_artist` | Binary (0/1, NaN→0) | 1.47 | v5 Add I; MusicBrainz country; NaN=unknown→conservative 0 |

Note: `artist_peak_popularity` and `artist_track_count` (also in `artist_features.csv`) have VIF=15.80 and 9.60 respectively — a pre-existing collinearity with the Spotify audio features. These are included because they are auto-detected at >50% coverage and contribute meaningfully to prediction, but should be noted in any regression interpretability discussion.

### 4.2 Removed Features

| Feature | Removed in | VIF | Reason |
|---|---|---|---|
| `energy` | v4 Fix A | 15.07 | Collinear with `loudness` (r≈0.78) |
| `danceability` | v5 Fix F | 12.41 | Collinearity hub with tempo, valence, duration_min |

### 4.3 `decade_idx` — Cox PH and Log-OLS Only

Added to `COX_FEATURES = FEATURES + ["decade_idx"] + LYRIC_FEATURES`. Not in `BASE_FEATURES` because `chart_entry_date` is NaN for all non-charted tracks, making it unusable in the classification model. Computed as `(year - 1950) // 10`, giving ordinal values 0 (1950s) through 7 (2020s). Nullable integer type (`Int64`) to keep NaN for non-charted rows.

### 4.4 Lyric Features — Cox PH and Log-OLS Only (LYRIC_FEATURES)

When `lyric_features.csv` is present, v5 auto-detects and merges 4 VADER sentiment features into the longevity models only:

| Feature | Coverage | Notes |
|---|---|---|
| `sentiment_compound` | 41.9% charted | VADER compound score [−1, +1] |
| `sentiment_pos` | 41.9% charted | Positive lexicon fraction |
| `sentiment_neg` | 41.9% charted | Negative lexicon fraction; HR=1.124, p=0.0008 |
| `lyric_word_count` | 41.9% charted | Reconstructed token count from raw lyrics CSV |

Threshold: >20% non-null coverage (looser than the 50% threshold for base features, since the lyrics CSV inherently covers only charted tracks). The lyric-matched subset reduces the longevity model's n from 3,502 to **1,456**. Unmatched charted tracks get NaN and are excluded from Cox/OLS fits.

### 4.5 Auto-Detected Artist Features (all 4 models)

`artist_features.csv` is present and committed. The auto-detection loop in `run_all_v5.py` checks `df[col].notna().mean() > 0.5` and currently detects these artist columns:
- `artist_peak_popularity` (from Spotify API)
- `artist_track_count` (from Spotify API)
- `lastfm_listeners_log` (from MusicBrainz/Last.fm `artists.csv`)
- `is_us_artist` (from MusicBrainz `country_mb`; NaN pre-filled to 0 at df level)

When `librosa_features.csv` is present, the secondary analysis section will use it (to be added in a future run once Librosa pipeline completes).

---

## 5. Model Architecture and Results

### 5.1 Model 1: Logistic Regression (Chart Entry)

**Setup:** `LogisticRegression(class_weight="balanced", max_iter=1000)`. `class_weight="balanced"` is critical — it up-weights the positive class (charted) by a factor of ~25x to compensate for the 3.90% positive rate. `StandardScaler` applied before fitting so all coefficients are on the same scale (per 1 SD change), enabling direct comparison of odds ratios.

**Results (v5 with full augmentation — artist features + lyric features):**
- Test AUC-ROC: **0.9179** (↑ from 0.8922 pre-augmentation)
- Test PR-AUC: **0.3214** (vs. random baseline 0.039 → **8.2× lift**)
- CV AUC-ROC (5-fold): **0.9127 ± 0.0040** (very low variance → model generalizes well)

**Key odds ratios (selected):**
- `instrumentalness`: OR=0.28 — 72% less likely to chart per 1 SD. Dominant audio predictor.
- `lastfm_listeners_log`: OR=2.582 — **strongest overall predictor**; high-profile artists dramatically more likely to chart
- `is_us_artist`: OR=1.867 — US artists 87% more likely to chart per 1 SD
- `valence`: OR=1.32 — happier songs 32% more likely to chart
- `explicit`: OR=1.16 — explicit content 16% more likely to chart
- `mode`: OR=1.15 — major key 15% more likely to chart
- `speechiness`: OR=0.64 — high speech content 36% less likely (nonlinear — rap has mid-range speechiness, pure spoken word has extreme values; XGBoost captures the nonlinearity better)

VIF note: `artist_peak_popularity`=15.80, `artist_track_count`=9.60 are high (pre-existing; these artist-level metrics co-move with audio-feature proxies). New augmentation features are well-behaved: `lastfm_listeners_log`=5.19, `is_us_artist`=1.47.

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

**Results (v5 with full augmentation):**
- Test AUC-ROC: **0.9736** (↑ from 0.9608 pre-augmentation; +7.9 points over LR → confirms strongly nonlinear relationships)
- Test PR-AUC: **0.6869** (vs. random baseline 0.039 → **17.6× lift**)

**SHAP importance (Mean |SHAP| on test set, post-augmentation):**
1. `instrumentalness`: 0.8017 — dominant audio predictor, consistent with LR
2. `lastfm_listeners_log`: 1.336 — **new #1 or #2 overall**; artist notoriety overwhelms audio features for chart prediction
3. `acousticness`: ~0.51
4. `duration_min`: ~0.40
5. `valence`: ~0.34
6. `speechiness`: ~0.30
7. `loudness`: ~0.26
8. `liveness`: ~0.23
9. `tempo`: ~0.21
10. `mode`: ~0.13
11. `key`: ~0.08
12. `explicit`: ~0.08
13. `is_us_artist`: 0.458

The gap between XGBoost (0.974) and LR (0.918) AUC confirms the relationships remain substantially nonlinear even after augmentation. The `lastfm_listeners_log` SHAP dominance reveals that popularity/reach of the artist is the strongest single predictor of chart entry — a finding that can be framed as "social proof / prior popularity" driving chart outcomes.

### 5.3 Model 3: Cox Proportional Hazards (Longevity)

**Setup:** `CoxPHFitter(penalizer=0.1)`. The penalizer adds L2 regularization to the Cox log-partial-likelihood, preventing coefficient explosion for correlated features. `strata=["mode"]` (v4 Fix B). Dataset: 3,502 charted tracks, all treated as observed events (no right-censoring, since the data is historical). `decade_idx` added as covariate (v5 Add C).

**Concordance Index: 0.7240** (↑ from 0.5770 pre-augmentation; massive improvement driven by artist features and lyric sentiment). Dataset: **1,456 charted tracks** with matched lyrics (lyric-matched subset; down from 3,502 full charted set; unmatched tracks excluded due to NaN sentiment).

**Key Cox coefficients (post-augmentation, 18 features):**
- `decade_idx`: HR=0.878, p<0.00001 — **most significant finding.** Each decade later → 12.2% higher hazard of leaving the chart per week. Quantifies streaming-era compression of chart longevity.
- `sentiment_neg`: HR=1.124, p=0.0008 — **new significant lyric finding.** Higher negative lyric content → faster exit from chart. Songs with more negative lyrics (hurt, hate, cry) cycle off the Hot 100 sooner.
- `acousticness`: HR=0.889 — acoustic tracks leave charts faster
- `loudness`: HR=0.900 — louder tracks have shorter chart runs
- `valence`: HR=1.073 — happier songs stay longer (consistent with entry finding)
- `liveness`, `key`, `duration_min`: Not significant (p>0.05) — these predict chart entry but not chart longevity
- `sentiment_compound`, `sentiment_pos`, `lyric_word_count`: Not independently significant after controlling for `sentiment_neg`

**Schoenfeld PH test violations:** 13 of 18 covariates fail the proportional hazards test. This is expected and explainable across 70 years of pop music history. The violations are most severe for `decade_idx` itself, which captures a fundamental regime change (radio era → streaming era). **The recommended report framing is to acknowledge the violations, note that they reflect real structural change, and use Log-OLS as a robustness check.**

### 5.4 Model 3b: Log-OLS (Longevity Robustness)

**Setup:** OLS on `log1p(wks_on_chart)` for 3,502 charted tracks, with `COX_FEATURES` (same as Cox plus `decade_idx`). The log transformation makes the right-skewed weeks distribution more normal. 80/20 train/test split.

**R² = 0.3469** (↑ from 0.0438 pre-augmentation; artist features and lyric sentiment explain 34.7% of variance in log-longevity, up from 4.4%).

**Direction inconsistencies between Cox and OLS for some features** (e.g., loudness is HR<1 in Cox but coef>0 in OLS) reflect the Cox PH violations — the effects are time-varying, and OLS averages over the time dimension in a way that can flip the sign. **Only findings where Cox and OLS directionally agree should be treated as robust. `decade_idx` is the strongest such variable** (Cox HR=0.878, OLS coef=−0.110). `sentiment_neg` direction is also consistent across both models.

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
| `run_all_v5.py` | ~33KB | **CURRENT** full pipeline: VIF, LR, XGBoost+SHAP, Cox PH, Log-OLS, 9 figures |
| `build_artist_features.py` | ~9KB | Builds `artist_features.csv` from Spotify API + Last.fm/MusicBrainz `artists.csv` |
| `build_lyric_features.py` | ~5KB | Builds `lyric_features.csv`; VADER sentiment on Billboard Top 100 lyrics CSV |
| `requirements.txt` | ~200B | Python dependencies for pip install |
| `modal_charted_scrape.py` | 14KB | Cloud: artist followers/popularity for 953 charted artists |
| `modal_preview_urls.py` | 11KB | Cloud: Spotify preview URLs for 3,502 charted tracks |
| `modal_librosa_extract.py` | 16KB | Cloud: 32 Librosa acoustic features from 30s previews |
| `oit367_base_dataset.csv` | 16MB | Processed merged dataset (89,741 tracks); run_all_v5.py input |
| `artist_features.csv` | ~1MB | Artist-level features (committed); auto-detected by run_all_v5.py |
| `README.md` | — | Team entry point: repo map, setup, Modal run sequence |
| `RESULTS.md` | ~25KB | Full model results with all numbers, findings, limitations |
| `ANALYSIS_LOG.md` | 49KB | Verbose analysis log (Cursor-compatible for follow-up actions) |
| `AUGMENTATION_PLAN.md` | ~8KB | Blueprint for Add H/I/J augmentation (now completed) |
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
| `artists.csv` | ~100MB | MusicBrainz/Last.fm dump (1.47M artists); source for Add H/I |
| `billboard_lyrics.csv` | ~15MB | Billboard Top 100 lyrics 1946–2022; source for Add J (renamed from "billboard_top_100_1946_2022_lyrics .csv") |
| `oit367_augmented_dataset.csv` | ~18MB | Generated by run_all_v5.py when artist_features.csv present |
| `lyric_features.csv` | ~0.5MB | Generated by build_lyric_features.py; 3,502 rows, 1,456 with VADER scores |
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
| Logistic Regression | Chart Entry | AUC-ROC | **0.9179** | **0.3214** | CV: 0.9127 ± 0.0040 (5-fold); 8.2× PR lift |
| XGBoost | Chart Entry | AUC-ROC | **0.9736** | **0.6869** | 17.6× PR lift; `lastfm_listeners_log` SHAP=1.336 |
| Cox PH | Longevity | C-statistic | **0.7240** | — | mode stratified; +decade_idx +sentiment; n=1,456 |
| Log-OLS | Longevity | R² | **0.3469** | — | log1p(wks); +decade_idx +sentiment |

**Pre-augmentation baseline (for reference):** LR AUC=0.8922, XGBoost AUC=0.9608, Cox C-stat=0.5770, OLS R²=0.0438.

**Key takeaway:** Artist-level features (Last.fm listeners, US nationality) dominate both classification models. The XGBoost AUC of 0.974 and PR-AUC of 0.687 (17.6× lift) represent very strong performance. The longevity models improved dramatically (Cox C-stat 0.55→0.72, OLS R² 0.04→0.35), driven primarily by artist features. The lyric sentiment finding (`sentiment_neg` HR=1.124) is the most novel result from the augmentation.

---

## 9. Seven Core Findings (Report-Ready)

1. **Artist notoriety is the single strongest predictor of chart entry.** `lastfm_listeners_log` SHAP=1.336 (ranked #2 in XGBoost, above all audio features except instrumentalness), OR=2.582 in LR. High-profile artists are dramatically more likely to chart. This finding suggests that prior popularity / social proof matters more than the audio content itself for chart entry.

2. **Instrumentalness is the dominant audio barrier to chart entry.** SHAP=0.80, OR=0.28. The Hot 100 is a vocal/lyric-driven chart. Instrumental and near-instrumental tracks are 72% less likely to chart per 1 SD increase.

3. **Valence (musical positivity) is the strongest positive audio predictor of chart entry.** SHAP≈0.344, OR=1.32. Happier-sounding songs are 32% more likely to chart per 1 SD increase, consistent across both LR and XGBoost.

4. **Track length predicts chart entry nonlinearly.** `duration_min` SHAP≈0.40, ranked in the top 4 XGBoost features. Shorter tracks chart more, with a nonlinear relationship (XGBoost captures the inflection point that LR cannot). The streaming era's structural shift toward shorter tracks is consistent with this finding.

5. **The streaming era has dramatically compressed chart longevity (strongest survival finding).** `decade_idx` HR=0.878, p<0.00001 — the variable where Cox PH and Log-OLS (OLS coef=−0.110) directionally agree most strongly. Each decade later corresponds to a 12.2% higher per-week hazard of leaving the chart. Songs in the 2020s cycle through the Hot 100 approximately twice as fast as songs from the 1980s.

6. **Negative lyric sentiment accelerates chart exit — a novel finding from lyric augmentation.** `sentiment_neg` HR=1.124, p=0.0008 in Cox PH; direction consistent in Log-OLS. Songs with higher negative lexical content (hurt, hate, loss) exit the chart faster. This lyric-level signal is independent of the audio-feature `valence` score (r=−0.13 correlation only).

7. **Combined audio + artist + lyric features predict chart entry very strongly but longevity moderately.** Post-augmentation: XGBoost AUC=0.974 and PR-AUC=0.687 (17.6× lift) for entry; Cox C-stat=0.724 and R²=0.347 for longevity. Longevity models improved dramatically with artist features, suggesting that sustained chart presence is partly explained by artist commercial leverage (label promotion, radio relationships) which correlates with Last.fm listener base size.

---

## 10. Pending Work and Next Steps

### Completed Since Last Context Update (2026-03-08)

The full AUGMENTATION_PLAN.md has been executed:
- ✓ **Add H:** `lastfm_listeners_log` from MusicBrainz `artists.csv` — complete, committed
- ✓ **Add I:** `is_us_artist` from MusicBrainz `country_mb` — complete, committed
- ✓ **Add J:** VADER lyric sentiment from Billboard lyrics CSV — complete, `lyric_features.csv` generated
- ✓ `artist_features.csv` regenerated with Last.fm columns, committed
- ✓ `build_artist_features.py` and `build_lyric_features.py` committed
- ✓ `requirements.txt` updated with `vaderSentiment>=3.3.2`
- ✓ All stale comments in `run_all_v5.py` and `RESULTS.md` updated

**Note:** `xgboost` must be pinned to `==2.1.3` due to SHAP TreeExplainer incompatibility with xgboost 3.x. This is documented in the `run_all_v5.py` docstring.

### To Reproduce the Full Augmented Run

```bash
# Install dependencies (pin xgboost for SHAP compatibility)
pip install -r requirements.txt --break-system-packages
pip install "xgboost==2.1.3" --break-system-packages

# Step 1: Build artist features (requires artists.csv in working dir)
python3 build_artist_features.py

# Step 2: Build lyric features (requires billboard_lyrics.csv in working dir)
python3 build_lyric_features.py

# Step 3: Run all models
python3 run_all_v5.py
```

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
# IMPORTANT: pin xgboost after requirements.txt install (xgboost 3.x breaks SHAP TreeExplainer)
pip3 install "xgboost==2.1.3"

# requirements.txt contents:
# scikit-learn>=1.3
# xgboost>=2.0          ← pin to ==2.1.3 after install (see above)
# shap>=0.44
# lifelines>=0.27
# statsmodels>=0.14
# seaborn>=0.13
# matplotlib>=3.7
# pandas>=2.0
# numpy>=1.24
# tqdm>=4.65
# spotipy>=2.23
# requests>=2.31
# vaderSentiment>=3.3.2 ← added for lyric sentiment (Add J)

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
**Last commit message:** `"augment: add lastfm/is_us_artist/lyric-sentiment features; update RESULTS.md"`

To push after a clean v5 run:
```bash
git add .
git commit -m "your message here"
git push -u origin main   # -u only needed first push; use git push after
```

Note: `git add .` is safe because `.gitignore` automatically excludes the large CSVs and system files.

---

*This document contains the complete context for the OIT367 project as of 2026-03-08.*
*Copy this entire file into a new chat session to continue without context loss.*

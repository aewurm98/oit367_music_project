# OIT367 — Billboard Hot 100 Prediction
**Stanford GSB Winter 2026 | Wurm · Chen · Barli · Taruno**

Predicting Billboard Hot 100 chart entry and longevity from Spotify audio features using logistic regression, XGBoost, Cox proportional hazards, and OLS. Dataset: 89,741 Spotify tracks, 3,502 charted (3.90% positive rate).

---

## Current Model Results → [`RESULTS.md`](RESULTS.md)

Full findings, odds ratios, SHAP importance rankings, Cox hazard ratios, and key takeaways for the report are documented in `RESULTS.md`. Read this first.

---

## 1. Repository Structure

```
OIT-367/
├── README.md                    ← You are here
├── RESULTS.md                   ← All model results and findings
├── ANALYSIS_LOG.md              ← Detailed analysis log (Cursor-compatible)
├── LIBROSA_MODAL_PLAN.md        ← Secondary analysis feasibility plan
│
├── run_all_v5.py                ← ✅ CURRENT: Full model pipeline
├── requirements.txt             ← Python dependencies
│
├── modal_charted_scrape.py      ← Cloud: artist followers + popularity (953 artists)
├── modal_preview_urls.py        ← Cloud: Spotify preview URLs for Librosa pipeline
├── modal_librosa_extract.py     ← Cloud: 30 acoustic features from 30s audio previews
│
├── oit367_base_dataset.csv      ← Processed dataset (89,741 tracks; run_all_v5.py input)
├── outputs/                     ← Model outputs: figures (fig1–fig9) + CSV tables
│
└── archive/                     ← Old versions (v1–v4 pipeline, deprecated scrapers)
```

**Raw data files are NOT committed** (75MB total; see §2 to download).
**Do not run `git add *.csv` without checking `.gitignore` first.**

---

## 2. Raw Data Files (Download Required)

The three raw Kaggle files are excluded from the repo. Download them and place in the root `OIT-367/` folder before running the pipeline.

| File | Source | Size |
|---|---|---|
| `spotify_tracksdataset.csv` | [Kaggle: Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) | ~20MB |
| `hot-100-current.csv` | [Kaggle: Billboard Hot 100](https://www.kaggle.com/datasets/dhruvildave/billboard-the-hot-100-songs) | ~19MB |
| `merged_spotify_billboard_data.csv` | Generated from the two above (see §4 note) | ~20MB |

> `oit367_base_dataset.csv` is already committed — it's the cleaned, merged dataset that `run_all_v5.py` reads directly. You only need the raw files if you want to rebuild the base dataset from scratch.

---

## 3. Setup

```bash
# Clone repo
git clone git@github.com:aewurm98/oit367_music_project.git
cd oit367_music_project

# Install dependencies
pip3 install -r requirements.txt
```

**Requirements:** Python 3.11+, packages in `requirements.txt`
(`scikit-learn`, `xgboost`, `shap`, `lifelines`, `statsmodels`, `seaborn`, `matplotlib`, `pandas`, `numpy`)

---

## 4. Running the Pipeline

```bash
python3 run_all_v5.py
```

The script:
1. Reads `oit367_base_dataset.csv` (builds it from raw files if not present)
2. Runs VIF check, Logistic Regression, XGBoost + SHAP, Cox PH, Log-OLS
3. Saves all figures and CSV tables to `outputs/`
4. Prints full results summary to terminal

**Total runtime:** ~60–90 seconds.

If `artist_features.csv` is present in the folder, it's auto-detected and merged as augmented features. Download from Modal after the scraper completes (see §6).

---

## 5. Key Model Features

| Feature | Type | Notes |
|---|---|---|
| `valence` | Audio | Musical positivity; strongest positive predictor (OR=1.32) |
| `instrumentalness` | Audio | Dominant barrier to chart entry (SHAP=0.80, OR=0.28) |
| `acousticness` | Audio | 2nd-ranked SHAP feature |
| `duration_min` | Control | Track length (min); 3rd-ranked SHAP feature |
| `explicit` | Control | Binary; explicit tracks 16% more likely to chart |
| `loudness`, `speechiness`, `liveness`, `tempo`, `key`, `mode` | Audio | — |
| `decade_idx` | Control | Cox PH + OLS only; quantifies streaming-era longevity shift |

**Removed for multicollinearity (VIF > 10):** `energy` (VIF=15.07 in v4), `danceability` (VIF=12.41 in v5). Both documented in `RESULTS.md §3.2`.

---

## 6. Cloud Data Augmentation (Modal)

Three additional feature sets are being gathered via Modal cloud jobs. Run these after the Spotify API rate limit resets (~8 hours from a rate-limit event).

### Step 1 — Artist features (~3 min)
```bash
modal run --detach modal_charted_scrape.py
modal volume get oit367-vol /data/charted_artist_features.csv ./artist_features.csv
```
Adds `artist_followers` and `artist_popularity_api` for the 953 unique charted artists.

### Step 2 — Preview URLs (~20 min)
```bash
modal run --detach modal_preview_urls.py
```
Fetches 30-second MP3 preview URLs for 3,502 charted tracks via `sp.track()`.

### Step 3 — Librosa acoustic features (~2–3 hr)
```bash
# After Step 2 completes:
modal run --detach modal_librosa_extract.py
modal volume get oit367-vol /data/librosa_features.csv ./librosa_features.csv
```
Extracts 32 features per track (13 MFCC means + 13 MFCC stds + 6 spectral).
Used for the secondary analysis section (timbre → longevity).

**Steps 1 and 2 can run concurrently** (different API endpoints). Step 3 requires Step 2.

---

## 7. Outputs Reference

All outputs are saved to `outputs/`. Committed to the repo so teammates can reference results without running the pipeline.

| File | Description |
|---|---|
| `fig1_class_balance.png` | Class distribution (3.90% charted) |
| `fig2_correlation_heatmap.png` | Feature correlation matrix |
| `fig3_roc_curves.png` | ROC: LR (0.711) vs XGBoost (0.834) |
| `fig4_odds_ratios.png` | Logistic regression forest plot |
| `fig5_shap_importance.png` | XGBoost SHAP bar chart |
| `fig6_cox_hazard_ratios.png` | Cox PH hazard ratios |
| `fig7_kaplan_meier.png` | KM survival curves by genre |
| `fig8_longevity_distribution.png` | Weeks on chart distribution |
| `fig9_precision_recall.png` | PR curves (primary metric at 3.9% positive rate) |
| `model_performance_summary.csv` | All 4 models, all metrics |
| `logistic_odds_ratios.csv` | LR coefficients + odds ratios |
| `xgboost_shap_importance.csv` | Mean \|SHAP\| per feature |
| `cox_summary.csv` | Full Cox PH table with CIs and p-values |
| `ols_longevity_coefficients.csv` | Log-OLS coefficients |
| `vif_table.csv` | VIF per feature (max=2.02 post-patch) |
| `genre_chart_rates.csv` | Chart rate and avg popularity by genre |

---

## 8. Team Task Assignments

| Task | Owner | Status |
|---|---|---|
| Model pipeline (LR, XGBoost, Cox PH) | Alex / Vivian | ✅ Complete (v5) |
| SHAP analysis | Alex | ✅ Complete |
| Lyric sentiment (Genius + VADER) | Alex | 🔄 In progress |
| Artist features (Modal scraper) | Alex | ⏳ Awaiting rate limit reset |
| Librosa audio features (Modal) | Alex | ⏳ Scripts ready; run after preview URLs |
| Cox PH longevity analysis | Ben / Vivian | ✅ Complete (v5) |
| Report write-up | All | 🔄 In progress |

---

## 9. Archive

Old pipeline versions and deprecated scripts are in `archive/`. Do not use these — they have known bugs documented in `RESULTS.md`.

| File | Superseded by | Issue |
|---|---|---|
| `run_all_v4.py` | `run_all_v5.py` | Missing explicit/duration_min controls; danceability VIF unfixed |
| `run_all_v3.py` | `run_all_v5.py` | Spotipy block ran unconditionally; crashed laptop |
| `run_all_v2.py` | `run_all_v5.py` | Missing VIF fixes |
| `run_all.py` | `run_all_v5.py` | Original prototype |
| `modal_spotify_scrape.py` | `modal_charted_scrape.py` | Targeted full 31k-artist dataset; hit Spotify Dev Mode rate limit (8hr stall) |
| `oit367_pipeline_corrected.py` | `run_all_v5.py` | Early pipeline draft |

---

*Last updated: 2026-03-07 | Pipeline version: `run_all_v5.py`*

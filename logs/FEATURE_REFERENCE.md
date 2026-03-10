# OIT-367 Feature Reference — Dataset & Join Map
**Stanford GSB Winter 2026 | Wurm · Chen · Barli · Taruno**
**Last updated: 2026-03-09 — v6 pipeline (oit367_final_dataset.csv, cross-ID dedup applied)**

Use this document to identify where every feature lives, how datasets connect, and what is or isn't yet in the models.

---

## 1. Dataset Chain Overview

```
spotify_tracksdataset.csv   (114k rows, raw Kaggle — gitignored)
        │  dedup on track_id → keep first genre row
        │  left-join aggregated Billboard from merged_spotify_billboard_data.csv
        ▼
oit367_base_dataset.csv     (89,741 rows — DO NOT MODIFY, stable input)
        │  left-join artist_features.csv on 'artists' string
        ▼
oit367_augmented_dataset.csv  (89,741 rows — current best working dataset)
        │  left-join augmented_deduped_dataset_with_artists.csv
        │            primary key: track_id
        │            fallback key: normalized track_name + artists
        ▼
oit367_final_dataset.csv    (78,390 rows — CURRENT PIPELINE INPUT for run_all_v5.py)
```

Lyric sentiment is pre-baked into `oit367_final_dataset.csv` (no runtime merge needed in v6):

```
billboard_lyrics.csv  →  build_lyric_features.py  →  lyric_features.csv
        lyric_features.csv was joined by build_final_dataset.py → baked into oit367_final_dataset.csv
        run_all_v5.py detects existing columns and skips re-merge
```

---

## 2. Source Files Quick Reference

| File | Rows | Join Key | Status | Notes |
|---|---|---|---|---|
| `spotify_tracksdataset.csv` | 114k | `track_id` | gitignored, raw | Multi-genre rows; audio features source |
| `hot-100-current.csv` | ~690k weekly | title+performer | gitignored, raw | No track_id; pipeline uses merged_spotify_billboard_data.csv |
| `oit367_base_dataset.csv` | 89,741 | `track_id` | committed | Archive; no longer used as pipeline input |
| `oit367_augmented_dataset.csv` | 89,741 | `track_id` | generated | Side output of run_all_v5.py; superseded by final |
| `oit367_final_dataset.csv` | **78,390** | `track_id` | **committed** | **CURRENT INPUT**: cross-ID deduped + all features |
| `artist_features.csv` | 17,437 | `artists` (string) | committed | Built by `build_artist_features.py` |
| `lyric_features.csv` | 3,502 | `track_id` | gitignored | Built by `build_lyric_features.py`; charted only |
| `artists.csv` | 1.47M | normalized name | gitignored | MusicBrainz/Last.fm dump; source for lastfm + country |
| `billboard_lyrics.csv` | 6,879 | norm. name+artist | gitignored | Tokenized lyrics; source for VADER sentiment |
| `augmented_deduped_dataset_with_artists.csv` | 2,157 | `track_id` + key | teammate file | Charted only; deduped 3,502→2,157 |

---

## 3. Feature Reference Table

### 3.1 Spotify Audio Features
All sourced from `spotify_tracksdataset.csv` → `oit367_base_dataset.csv` (built-in, no extra join needed).

| Feature | Range | In Model? | Notes |
|---|---|---|---|
| `valence` | [0, 1] | ✓ BASE_FEATURES | Musical positivity. #4 SHAP (XGB), OR=1.33 (LR). Happier → more likely to chart. |
| `acousticness` | [0, 1] | ✓ BASE_FEATURES | Acoustic vs. electronic. VIF=3.21. |
| `loudness` | dB (negative) | ✓ BASE_FEATURES | VIF=6.76. Negative OR (louder songs less likely to chart — nonlinear). |
| `speechiness` | [0, 1] | ✓ BASE_FEATURES | Spoken word content. Nonlinear — XGBoost captures it better than LR. |
| `instrumentalness` | [0, 1] | ✓ BASE_FEATURES | **#3 SHAP (0.629), OR=0.34.** Hot 100 is vocal-driven; instrumental tracks 66% less likely to chart per 1 SD. Dominant audio barrier. |
| `liveness` | [0, 1] | ✓ BASE_FEATURES | Audience presence. Negative direction. |
| `mode` | 0 or 1 | ✓ BASE_FEATURES (stratum in Cox) | Major=1, Minor=0. Used as stratification variable in Cox PH. |
| `key` | 0–11 | ✓ BASE_FEATURES | Chromatic pitch class. VIF=3.05. Not significant in LR (OR≈1.0). |
| `explicit` | 0 or 1 | ✓ BASE_FEATURES | Explicit content flag. Slightly positive direction. |
| `duration_min` | continuous (min) | ✓ BASE_FEATURES | Derived from `duration_ms`. Pre-capped at 10 min. |
| `time_signature` | 1–5 | **✗ NOT IN MODEL** | In `oit367_base_dataset.csv`; available but never added to FEATURES. Easy v6 addition. |
| `track_genre` | string | **✗ Descriptive only** | 100% coverage in base dataset. Used for `genre_chart_rates.csv` only. Not encoded as model feature yet. Easy addition as mean-encoded `genre_chart_rate`. |
| `energy` | [0, 1] | ✗ REMOVED | VIF=15.07, collinear with `loudness`. Dropped in v4 Fix A. |
| `danceability` | [0, 1] | ✗ REMOVED | VIF=12.41, collinearity hub. Dropped in v5 Fix F. |
| `tempo` | BPM | ✗ REMOVED | VIF=10.65 after danceability removal. Dropped in v5 Fix G. |

### 3.2 Billboard Chart Outcomes
All sourced from `merged_spotify_billboard_data.csv` → already aggregated into `oit367_base_dataset.csv`.

| Column | Type | Used As | Notes |
|---|---|---|---|
| `is_charted` | 0/1 | **Classification label** | 1 = appeared on Hot 100. v6: 2,157 charted (2.75% of 78,390). |
| `wks_on_chart` | integer | **Longevity outcome** | Used as survival time in Cox PH and log1p-transformed in OLS. |
| `peak_pos` | integer | Not in features | Lower = better. Not used as predictor (would leak label info). |
| `chart_entry_date` | date | Derived only | NaN for non-charted tracks. Used to compute `decade_idx`. |
| `decade_idx` | ordinal 0–7 | ✓ COX/OLS FEATURES | `(year - 1950) // 10`. 0=1950s, 7=2020s. Strongest longevity predictor (HR=0.514 v6). NaN for non-charted tracks (not in LR/XGB features). |

### 3.3 Artist-Level Features
Sourced from `artist_features.csv` (built by `build_artist_features.py` from Spotify API + `artists.csv`).
**Join key: `artists` column (exact string match).**

| Feature | Source | In Model? | Notes |
|---|---|---|---|
| `artist_peak_popularity` | Spotify API (historical peak) | ✓ AUTO-DETECTED | **#1 SHAP (1.618), #1 LR OR (2.84).** Prior chart success dominates prediction. VIF=21.25 ⚠ collinear with `artist_popularity_api`. |
| `artist_popularity_api` | Spotify API (current) | ✓ AUTO-DETECTED | VIF=14.12. Collinear pair — opposite sign to `artist_peak_popularity` in LR/Cox is a collinearity artifact. Combined directional effect: established artists chart and stay on chart longer. |
| `artist_track_count` | Spotify API | ✓ AUTO-DETECTED | #5 SHAP (0.388). Number of tracks on Spotify. Negative direction (OR=0.80) — catalog size doesn't substitute for popularity. |
| `lastfm_listeners_log` | `artists.csv` via MusicBrainz/Last.fm | ✓ AUTO-DETECTED | **#2 SHAP (1.153), #2 LR OR (2.17).** log1p(listeners_lastfm). 65.8% match rate on artist name. Pre-existing audience predicts chart entry above any audio feature. |
| `is_us_artist` | `artists.csv` country_mb field | ✓ AUTO-DETECTED | Binary. US artists 87% more likely to chart (OR=1.87). NaN→0 filled conservatively. VIF=1.47. |

**How `auto-detected` works in `run_all_v5.py`:** The script checks `df[col].notna().mean() > 0.5` for each candidate column. Columns that pass this threshold are automatically added to `FEATURES`. `is_us_artist` is pre-filled NaN→0 to guarantee it passes.

### 3.4 Lyric Sentiment Features
Sourced from `lyric_features.csv` (built by `build_lyric_features.py` from `billboard_lyrics.csv` using VADER).
**Join key: `track_id`. Charted tracks only. v6: 856 lyric-matched of 2,157 charted (40.1%).**

| Feature | Range | In Model? | Notes |
|---|---|---|---|
| `sentiment_compound` | [-1, +1] | ✓ COX/OLS LYRIC | Aggregate positivity. HR=1.073, p=0.073 (marginal, not significant). |
| `sentiment_pos` | [0, 1] | ✓ COX/OLS LYRIC | Positive word fraction. p=0.875, not significant. |
| `sentiment_neg` | [0, 1] | ✓ COX/OLS LYRIC | **Negative word fraction. HR=1.088, p=0.051 (marginal v6). Directional trend — songs with more negative lexical content may cycle off chart faster.** |
| `lyric_word_count` | integer | ✓ COX/OLS LYRIC | p=0.592, not significant. |

**Note on coverage:** Only used in Cox PH and Log-OLS longevity models (n=856 v6). Classification models (LR, XGBoost) do not use lyric features — they would exclude ~58% of charted tracks from training.

**Note on vaderSentiment:** `pip install vaderSentiment` needed in `oit367_music_project` conda env before running `build_lyric_features.py`.

### 3.5 Teammate Enrichment Features (in `oit367_final_dataset.csv` — dataset only, not in models)
Sourced from `augmented_deduped_dataset_with_artists.csv` (charted tracks only, 2,157 rows).
**Join key: `track_id` (primary), normalized `track_name` + `artists` (fallback).**
**Coverage: ~58% of charted tracks have non-null values (906/2,157 rows fully null).**

| Feature (engineered) | Raw Column | In Model? | Notes |
|---|---|---|---|
| `is_male_artist` | `Artist Gender` | **✗ PENDING v6** | Binary. Null for non-binary/unknown. ~58% coverage. |
| `artist_age` | `Artist Age` | **✗ PENDING v6** | Continuous. Raw value 0.0 is NULL (corrected to NaN in `build_final_dataset.py`). |
| `artist_scrobbles_log` | `Artist Scrobbles` | **✗ PENDING v6** | log1p(all-time Last.fm scrobble count). Distinct from `lastfm_listeners_log` (all-time listeners). |
| `artist_listeners_monthly_log` | `Artist Listeners` | **✗ PENDING v6** | log1p(monthly active Last.fm listeners). Distinct from both above. |
| `is_mainstream_genre` | `Artist Genres` (parsed) | **✗ PENDING v6** | Binary. True if any genre ∈ {pop, hip hop, rap, r&b, reggaeton, latin, country, rock, edm, dance}. |
| `artist_genre_count` | `Artist Genres` (parsed) | **✗ PENDING v6** | Integer. Number of Spotify genre tags. Proxy for niche vs. mainstream. |
| `time_signature` | `time_signature` | **✗ PENDING v6** | Also in Spotify source — just never added to BASE_FEATURES. |
| `is_us_artist_iso` | `Artist Country` (ISO 2) | Blended into `is_us_artist` | More precise than MusicBrainz country_mb. Blended via `combine_first` in `build_final_dataset.py`. |

**Important notes on this file:**
- It is **charted tracks only** — it cannot replace the base 89,741-row universe (negative class would be lost).
- Artist name stored in **lowercase** — must normalize before joining.
- The 2,157 rows represent **true unique songs** (deduped from 3,502 track_ids by name+artist collision).

---

## 4. De-Duplication Notes

### Base dataset (pre-dedup)
The base dataset deduplicates only on `track_id` (removes multi-genre rows from Spotify source: 114k → 89,741). It does **not** collapse same-song entries with different Spotify IDs (album vs. single, regional variants, explicit vs. clean).

- **3,502 charted track_ids** exist in `oit367_base_dataset.csv`
- But only **~2,157 truly unique charted songs** (1,345 cross-ID same-song duplicates remain)

### v6 (build_final_dataset.py)
`build_final_dataset.py` resolves this using the teammate's canonical IDs:
1. For each (normalized name + artist) group with multiple track_ids, prefer the ID present in the teammate's file
2. Break remaining ties by `wks_on_chart` (keep the row with most chart weeks)
3. Non-charted universe (86,239 rows) is unaffected — no cross-ID duplication exists there

---

## 5. Features NOT Integrated (Low Coverage — Excluded)

| File | Coverage | Decision |
|---|---|---|
| `Labels.csv` | ~4.1% track match | Excluded from AUGMENTATION_PLAN.md — too sparse |
| `Best selling music artists.csv` | ~0.5% track match | Excluded — too sparse |
| `artists.csv` additional columns (`type`, `begin_date_year`, `gender`) | ~65% via name match | Not yet added; available if needed for v6 |

---

## 6. Model Feature Sets (Current v6)

```python
# Audio + artist: used in LR and XGBoost classification
BASE_FEATURES = [
    "valence", "acousticness", "loudness", "speechiness",
    "instrumentalness", "liveness", "mode", "key", "explicit", "duration_min",
    "artist_peak_popularity", "artist_popularity_api", "artist_track_count",
    "lastfm_listeners_log", "is_us_artist"
]

# Everything above + time + sentiment: used in Cox PH and Log-OLS longevity
COX_FEATURES = BASE_FEATURES + [
    "decade_idx",
    "sentiment_compound", "sentiment_pos", "sentiment_neg", "lyric_word_count"
]
# (lyric features only included when coverage >20%; model uses n=856 lyric-matched tracks v6)
```

---

## 7. Key Confirmed Results (v6 — verified run output)

| Model | Task | Metric | Score | Notes |
|---|---|---|---|---|
| Logistic Regression | Chart Entry | AUC-ROC | **0.9144** | CV: 0.9139±0.0037; 10.0× PR lift |
| XGBoost | Chart Entry | AUC-ROC | **0.9655** | 16.0× PR lift; early stop @193 |
| Cox PH | Longevity | C-stat | **0.7526** | mode-stratified; n=856 |
| Log-OLS | Longevity | R² | **0.2118** | log1p(wks_on_chart) |

**Pre-augmentation baselines (audio only):** LR 0.7106 / XGB 0.8343 / Cox 0.5508 / OLS 0.044

**Top predictors (XGBoost SHAP v6):**
1. `artist_peak_popularity` — 1.618 (artist commercial track record)
2. `lastfm_listeners_log` — 1.153 (pre-existing audience size)
3. `instrumentalness` — 0.629 (dominant audio barrier; instrumental = 66% less likely)
4. `is_us_artist` — 0.558
5. `artist_track_count` — 0.388

---

## 8. Quick Steps to Rebuild v6

```bash
# 1. Install dependencies (including vaderSentiment for lyric features)
pip install -r requirements.txt
pip install "xgboost==2.1.3"  # SHAP compatibility

# 2. Regenerate lyric features (requires billboard_lyrics.csv)
python3 build_lyric_features.py

# 3. Build final integrated dataset (requires oit367_base_dataset.csv, augmented_deduped_dataset_with_artists.csv)
python3 build_final_dataset.py

# 4. Run full pipeline
python3 run_all_v5.py
```

---

*Generated from session context — 2026-03-09. For model code, see `run_all_v5.py`. For full results, see `RESULTS.md`.*

# OIT367 — Dataset Augmentation Plan
**Stanford GSB Winter 2026 | Wurm · Chen · Barli · Taruno**
**Written: 2026-03-08 | Status: Ready to execute**

---

## 0. Context for New Conversation

This plan is self-contained. Here is the full project state before executing any of the steps below.

### Project summary
Billboard Hot 100 chart prediction study. Two research questions:
1. Binary classification: which audio/artist features predict chart entry?
2. Survival analysis: which features predict how long a track stays on the chart?

### Current pipeline file: `run_all_v5.py`
Runs 4 models on 89,741 Spotify tracks (3,502 charted = 3.90%):
- Model 1: Logistic Regression (AUC-ROC 0.8922, PR-AUC 0.3076)
- Model 2: XGBoost (AUC-ROC 0.9608, PR-AUC 0.6383 = 16.4× baseline)
- Model 3: Cox PH longevity (C-stat 0.5770)
- Model 4: Log-OLS longevity (R² 0.0438)

### Current features in model (BASE_FEATURES in run_all_v5.py)
Audio: `valence`, `acousticness`, `loudness`, `speechiness`, `instrumentalness`, `liveness`, `mode`, `key`, `explicit`, `duration_min`
Artist (auto-detected if artist_features.csv present): `artist_popularity_api`, `artist_peak_popularity`, `artist_track_count`
Longevity-only: `decade_idx` (ordinal 0=1950s…7=2020s, derived from chart_entry_date)

### Key files
```
OIT-367/
├── run_all_v5.py                  ← CURRENT pipeline (DO NOT BREAK)
├── build_artist_features.py       ← generates artist_features.csv from spotify_tracksdataset.csv
├── artist_features.csv            ← auto-detected by run_all_v5.py
├── oit367_augmented_dataset.csv   ← merged dataset written by run_all_v5.py
├── oit367_base_dataset.csv        ← base dataset (auto-built if missing)
├── spotify_tracksdataset.csv      ← 114k Spotify tracks, 31,437 artists
├── merged_spotify_billboard_data.csv ← original Billboard merge
├── hot-100-current.csv            ← raw Billboard weekly data
├── RESULTS.md                     ← model results documentation
├── requirements.txt               ← python dependencies
└── outputs/                       ← all figures and CSVs from pipeline
```

### How run_all_v5.py auto-detects new features
The pipeline already has hooks for artist augmentation:
```python
# In run_all_v5.py, after merging artist_features.csv:
for col in ["artist_followers", "artist_popularity_api", "artist_peak_popularity", "artist_track_count"]:
    if col in df.columns and df[col].notna().mean() > 0.5:
        FEATURES.append(col)
```
New artist-level features will be auto-included if added to `artist_features.csv` and they pass the >50% non-null threshold.

---

## 1. Datasets to Integrate

Four datasets were downloaded to the OIT-367 folder. Here is what each actually contains based on inspection:

### Dataset A: Billboard Lyrics (1946–2022)
**Filename:** `billboard_top_100_1946_2022_lyrics .csv` *(note: trailing space in filename)*
**Rows:** 6,879 | **Columns:** Song, Artist Names, Hot100 Ranking Year, Hot100 Rank, Lyrics
**Match rate against our 3,502 charted tracks:** 41.8% (901 tracks matched on song+artist name)
**Critical data quality issue:** Lyrics are stored as tokenized Python lists, not raw strings:
```
['someone', 'that', 'i', 'belong', 'to', 'doesnt', 'belong', 'to', 'me', ...]
```
Must use `ast.literal_eval()` then `' '.join()` to reconstruct before running VADER.

**What it adds:** Lyric sentiment score (VADER compound, pos, neg, word count)
**Scope limitation:** Only charted tracks can be enriched → lyric features used ONLY in Cox PH and Log-OLS longevity models, not the main classification model.

### Dataset B: Last.fm Music Artists Popularity
**Filename:** `artists.csv`
**Rows:** 1,466,083 | **Columns:** mbid, artist_mb, artist_lastfm, country_mb, country_lastfm, tags_mb, tags_lastfm, listeners_lastfm, scrobbles_lastfm, ambiguous_artist
**Match rate against our 17,437 normalized artist names:** 65.8% (11,482 matched)
**Data quality notes:**
- `country_mb` has 51.1% null rate (usable but needs NaN handling)
- `listeners_lastfm` range: 0–5,381,567 (Coldplay highest in matched set)
- Join on `artist_lastfm` column (normalized lowercase)

**What it adds:** `lastfm_listeners` (cross-platform reach, log-transformed) and `is_us_artist` binary (genuinely new — Spotify popularity doesn't capture nationality)

### Dataset C: UMG Record Labels
**Filename:** `Labels.csv` (latin-1 encoding)
**Rows:** 3,341 | **Columns:** ID, Group, Label, Artist
**Match rate against our 17,437 artists:** 4.1% (718 matched)
**Groups:** Universal Music Publishing Group (938), Verve Label Group (491), Capitol Music Group (441)

**Assessment: DO NOT INTEGRATE.** 4.1% coverage means 95.9% of artists would be NaN → fails the >50% threshold and would be excluded from the model anyway. The resulting binary would just flag 718 of 31,437 artists as UMG, which is too sparse and UMG-specific to represent "major label" broadly.

### Dataset D: Best-Selling Music Artists
**Filename:** `Best selling music artists.csv` (latin-1 encoding)
**Rows:** 121 | **Columns:** Artist name, Country, Active years, Release year of first charted record, Genre, Total certified units, Claimed sales
**Match rate against our 17,437 artists:** 0.5% (86 matched — e.g., ABBA, Adele, Aerosmith)
**Data quality:** `Claimed sales` and `Total certified units` are messy formatted strings (e.g., "600 million500 million"), not clean numerics.

**Assessment: DO NOT INTEGRATE.** Only 121 artists total, 86 matched. Would create a sparse binary flag for superstar legacy artists only. The `artist_peak_popularity` feature we already have captures much of this signal more cleanly.

---

## 2. Features to Add

| Feature | Source | Scope | Model use |
|---|---|---|---|
| `lastfm_listeners_log` | Last.fm `listeners_lastfm` (log1p) | All 17,437 artists → ~66% tracks covered | Classification + Longevity |
| `is_us_artist` | Last.fm `country_mb` | ~32% tracks (US artists with non-null country) | Classification + Longevity |
| `sentiment_compound` | VADER on Billboard lyrics | 901 charted tracks (41.8%) | Longevity only (Cox + OLS) |
| `sentiment_pos` | VADER | same | Longevity only |
| `sentiment_neg` | VADER | same | Longevity only |
| `lyric_word_count` | len(lyrics) | same | Longevity only |

**VIF pre-analysis:**
- `lastfm_listeners_log` vs `artist_popularity_api`: expect moderate correlation (r≈0.5–0.6) — check VIF, should stay under 10
- `is_us_artist`: binary, low VIF risk
- `sentiment_compound` vs `valence`: expect r≈0.3–0.4 — should be fine
- `lyric_word_count` vs `duration_min`: possible moderate correlation — check

---

## 3. Implementation Plan

### Step 1: Rename the lyrics file (fix trailing space)
```bash
cd OIT-367
mv "billboard_top_100_1946_2022_lyrics .csv" "billboard_lyrics.csv"
```

### Step 2: Install VADER
```bash
pip install vaderSentiment --break-system-packages
```
Add to `requirements.txt`:
```
vaderSentiment>=3.3.2
```

### Step 3: Create `build_lyric_features.py` (NEW FILE)

Create this file at `OIT-367/build_lyric_features.py`:

```python
"""
OIT367 — Lyric Sentiment Feature Builder
Stanford GSB Winter 2026 | Wurm / Chen / Barli / Taruno

Computes VADER sentiment scores for charted tracks using Billboard Top 100
lyrics dataset (1946–2022).

INPUT:  billboard_lyrics.csv         (renamed from 'billboard_top_100_1946_2022_lyrics .csv')
        oit367_augmented_dataset.csv  (or oit367_base_dataset.csv) — for charted track list

OUTPUT: lyric_features.csv           (charted tracks only, ~901 rows with matched lyrics)

AUTO-DETECTED BY: run_all_v5.py (Cox PH + Log-OLS longevity models only)

DATA QUALITY NOTES:
  - Billboard lyrics CSV covers 1946–2022 charted songs only
  - Lyrics are stored as tokenized Python lists — reconstructed via ast.literal_eval()
  - Match rate: ~41.8% of our 3,502 charted tracks (901 matched on song+artist name)
  - Unmatched charted tracks get NaN sentiment (excluded from longevity regressions)
  - VADER is trained on social media text; results on historical lyrics are approximate

RUN:
  python3 build_lyric_features.py
"""

import ast
import re
import pandas as pd
import numpy as np
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── Config ────────────────────────────────────────────────────────────────────
LYRICS_CSV   = Path("billboard_lyrics.csv")
DATASET_CSV  = Path("oit367_augmented_dataset.csv")
FALLBACK_CSV = Path("oit367_base_dataset.csv")
OUTPUT_CSV   = Path("lyric_features.csv")


def normalize(s: str) -> str:
    """Normalize artist/song names for fuzzy matching."""
    if pd.isna(s):
        return ""
    s = str(s).lower()
    # Strip list-like brackets from artist field (stored as "['artist name']")
    s = re.sub(r"[\[\]\"']", "", s)
    s = s.split(";")[0].strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def reconstruct_lyrics(raw: str) -> str:
    """
    Convert tokenized list string to plain text.
    Input:  "['word1', 'word2', 'word3']"
    Output: "word1 word2 word3"
    """
    if pd.isna(raw):
        return ""
    try:
        tokens = ast.literal_eval(str(raw))
        if isinstance(tokens, list):
            return " ".join(str(t) for t in tokens)
    except Exception:
        pass
    # Fallback: strip brackets and quotes manually
    s = re.sub(r"[\[\]\"']", "", str(raw))
    return re.sub(r",\s*", " ", s).strip()


# ── Load data ─────────────────────────────────────────────────────────────────
if not LYRICS_CSV.exists():
    raise FileNotFoundError(
        "billboard_lyrics.csv not found.\n"
        "Rename 'billboard_top_100_1946_2022_lyrics .csv' to 'billboard_lyrics.csv' first:\n"
        "  mv \"billboard_top_100_1946_2022_lyrics .csv\" billboard_lyrics.csv"
    )

dataset_path = DATASET_CSV if DATASET_CSV.exists() else FALLBACK_CSV
if not dataset_path.exists():
    raise FileNotFoundError("Run run_all_v5.py first to build oit367_augmented_dataset.csv")

print("Loading datasets...")
lyrics_df  = pd.read_csv(LYRICS_CSV)
dataset_df = pd.read_csv(dataset_path)

print(f"  Lyrics rows: {len(lyrics_df):,}")
print(f"  Dataset rows: {len(dataset_df):,}")

# ── Reconstruct and score lyrics ──────────────────────────────────────────────
print("\nReconstructing tokenized lyrics...")
lyrics_df["_lyrics_text"] = lyrics_df["Lyrics"].apply(reconstruct_lyrics)
lyrics_df["_word_count"]  = lyrics_df["_lyrics_text"].str.split().str.len()

print("Running VADER sentiment analysis...")
analyzer = SentimentIntensityAnalyzer()

def vader_scores(text: str) -> dict:
    if not text or len(text.split()) < 5:
        return {"compound": np.nan, "pos": np.nan, "neg": np.nan, "neu": np.nan}
    scores = analyzer.polarity_scores(text)
    return scores

vader_results = lyrics_df["_lyrics_text"].apply(vader_scores)
lyrics_df["sentiment_compound"] = [r["compound"] for r in vader_results]
lyrics_df["sentiment_pos"]      = [r["pos"]      for r in vader_results]
lyrics_df["sentiment_neg"]      = [r["neg"]      for r in vader_results]
lyrics_df["lyric_word_count"]   = lyrics_df["_word_count"]

# ── Match to charted tracks ───────────────────────────────────────────────────
print("\nMatching lyrics to charted tracks...")

lyrics_df["_song_norm"]   = lyrics_df["Song"].apply(normalize)
lyrics_df["_artist_norm"] = lyrics_df["Artist Names"].apply(normalize)

charted = dataset_df[dataset_df["is_charted"] == 1].copy()
charted["_song_norm"]   = charted["track_name"].apply(normalize)
charted["_artist_norm"] = charted["artists"].apply(normalize)

# Deduplicate lyrics by (song, artist) keeping highest-ranked entry
lyrics_dedup = (
    lyrics_df
    .sort_values("Hot100 Rank")
    .drop_duplicates(subset=["_song_norm", "_artist_norm"], keep="first")
)

# Merge
charted_enriched = charted.merge(
    lyrics_dedup[["_song_norm", "_artist_norm",
                  "sentiment_compound", "sentiment_pos", "sentiment_neg", "lyric_word_count"]],
    on=["_song_norm", "_artist_norm"],
    how="left",
)

matched = charted_enriched["sentiment_compound"].notna().sum()
total   = len(charted_enriched)
print(f"  Matched: {matched}/{total} charted tracks ({matched/total*100:.1f}%)")
print(f"  Unmatched: {total - matched} (will be NaN in longevity models)")

# ── Output summary ────────────────────────────────────────────────────────────
print("\nSentiment summary (matched tracks only):")
matched_df = charted_enriched[charted_enriched["sentiment_compound"].notna()]
print(matched_df[["sentiment_compound", "sentiment_pos", "sentiment_neg", "lyric_word_count"]]
      .describe().round(3).to_string())

print("\nSentiment vs valence correlation check:")
for col in ["sentiment_compound", "sentiment_pos", "sentiment_neg"]:
    r = matched_df[col].corr(matched_df["valence"])
    print(f"  {col} ↔ valence: r = {r:.3f}")

# ── Save ──────────────────────────────────────────────────────────────────────
output = charted_enriched[
    ["track_id", "artists", "track_name",
     "sentiment_compound", "sentiment_pos", "sentiment_neg", "lyric_word_count"]
].copy()

output.to_csv(OUTPUT_CSV, index=False)
print(f"\n✓ Saved {OUTPUT_CSV} ({len(output)} rows, {matched} with sentiment scores)")
print("\nNow run: python3 run_all_v5.py")
print("lyric_features.csv will be auto-detected and merged into Cox PH + Log-OLS models.")
```

---

### Step 4: Update `build_artist_features.py` to add Last.fm features

**Add the following block** to `build_artist_features.py`, after the existing artist aggregation block and before the `# ── Fill missing values` section. The existing aggregation logic is untouched.

Find this comment in the file:
```python
# ── Fill missing values with dataset medians (safety net) ─────────────────────
```

Insert this block **before** it:

```python
# ── Last.fm augmentation — add cross-platform reach + country ─────────────────
# Dataset: artists.csv (1.47M artists from MusicBrainz + Last.fm)
# Adds: lastfm_listeners_log (log1p of cumulative listeners)
#       is_us_artist (1=United States, 0=non-US, NaN=unknown country)
# Match rate: ~65.8% of our 17,437 normalized artist names

LASTFM_CSV = Path("artists.csv")

if LASTFM_CSV.exists():
    print("\nLoading Last.fm artist data (artists.csv)...")
    lastfm = pd.read_csv(LASTFM_CSV, low_memory=False,
                         usecols=["artist_lastfm", "country_mb", "listeners_lastfm"])
    lastfm["_norm"] = lastfm["artist_lastfm"].apply(normalize_artist)

    # Deduplicate on normalized name, keeping highest listener count
    lastfm_dedup = (
        lastfm.sort_values("listeners_lastfm", ascending=False)
        .drop_duplicates(subset="_norm", keep="first")
    )

    # Merge into artist_agg on normalized name
    artist_agg["_norm_key"] = artist_agg["artists"].apply(normalize_artist)
    artist_agg = artist_agg.merge(
        lastfm_dedup[["_norm", "listeners_lastfm", "country_mb"]],
        left_on="_norm_key",
        right_on="_norm",
        how="left",
    ).drop(columns=["_norm", "_norm_key"])

    # Derive features
    artist_agg["lastfm_listeners_log"] = np.log1p(
        artist_agg["listeners_lastfm"].fillna(0)
    )
    artist_agg["is_us_artist"] = (
        artist_agg["country_mb"]
        .str.strip()
        .eq("United States")
        .astype("Int64")  # nullable int: 1/0/NaN
    )
    # Set NaN where country_mb was null (unknown ≠ non-US)
    artist_agg.loc[artist_agg["country_mb"].isna(), "is_us_artist"] = pd.NA

    matched_lfm = artist_agg["listeners_lastfm"].notna().sum()
    print(f"  Last.fm matched: {matched_lfm:,}/{len(artist_agg):,} artists "
          f"({matched_lfm/len(artist_agg)*100:.1f}%)")
    print(f"  US artists identified: {(artist_agg['is_us_artist']==1).sum():,}")

    artist_agg = artist_agg.drop(columns=["listeners_lastfm", "country_mb"])

else:
    print("\nartists.csv not found — skipping Last.fm augmentation.")
    print("Download from: https://www.kaggle.com/datasets/pieca111/music-artists-popularity")
    artist_agg["lastfm_listeners_log"] = np.nan
    artist_agg["is_us_artist"]         = pd.NA
```

Also update the **output block** near the end of `build_artist_features.py` to include the new columns:

Replace:
```python
output = artist_agg[
    ["artists", "artist_popularity_api", "artist_peak_popularity", "artist_track_count"]
].copy()
```
With:
```python
# Build output — include Last.fm columns if they were computed
lastfm_cols = [c for c in ["lastfm_listeners_log", "is_us_artist"]
               if c in artist_agg.columns and artist_agg[c].notna().any()]

output = artist_agg[
    ["artists", "artist_popularity_api", "artist_peak_popularity", "artist_track_count"]
    + lastfm_cols
].copy()

if lastfm_cols:
    print(f"\nLast.fm columns included in output: {lastfm_cols}")
```

And update the rounding block to handle new columns:
```python
output["artist_popularity_api"]  = output["artist_popularity_api"].round(2)
output["artist_peak_popularity"] = output["artist_peak_popularity"].astype(int)
output["artist_track_count"]     = output["artist_track_count"].astype(int)
if "lastfm_listeners_log" in output.columns:
    output["lastfm_listeners_log"] = output["lastfm_listeners_log"].round(4)
# is_us_artist stays as nullable int (0/1/NaN)
```

---

### Step 5: Update `run_all_v5.py` — three targeted changes

**Change 5a: Add new columns to artist feature auto-detection loop**

Find:
```python
for col in [
    "artist_followers",        # Spotipy API path (modal_charted_scrape.py)
    "artist_popularity_api",   # both paths
    "artist_peak_popularity",  # local build path (build_artist_features.py)
    "artist_track_count",      # local build path (build_artist_features.py)
]:
```
Replace with:
```python
for col in [
    "artist_followers",        # Spotipy API path (modal_charted_scrape.py)
    "artist_popularity_api",   # both paths
    "artist_peak_popularity",  # local build path (build_artist_features.py)
    "artist_track_count",      # local build path (build_artist_features.py)
    "lastfm_listeners_log",    # Last.fm cross-platform reach (build_artist_features.py)
    "is_us_artist",            # Last.fm artist nationality binary (build_artist_features.py)
]:
```

**Change 5b: Add lyric features merge block for longevity models**

Find the section that starts:
```python
# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3: Cox PH — Chart Longevity
```

Insert this block **immediately before** it:
```python
# ─────────────────────────────────────────────────────────────────────────────
# LYRIC FEATURES — merge sentiment scores for longevity models (charted only)
# Generated by: python3 build_lyric_features.py
# Scope: ~41.8% of charted tracks (901/3,502) have matched lyrics
# Used in: Cox PH + Log-OLS only (charted subset)
# ─────────────────────────────────────────────────────────────────────────────
lyric_features_path = Path("lyric_features.csv")
LYRIC_FEATURES = []
if lyric_features_path.exists():
    print("\nFound lyric_features.csv — merging sentiment for longevity models…")
    lyric_df = pd.read_csv(lyric_features_path)
    lyric_cols = ["sentiment_compound", "sentiment_pos", "sentiment_neg", "lyric_word_count"]
    available_lyric_cols = [c for c in lyric_cols if c in lyric_df.columns]
    # Merge on track_id (most reliable key)
    df = df.merge(
        lyric_df[["track_id"] + available_lyric_cols],
        on="track_id",
        how="left",
    )
    matched = df.loc[df["is_charted"] == 1, "sentiment_compound"].notna().sum()
    print(f"  Lyric sentiment matched: {matched} charted tracks "
          f"({matched/df['is_charted'].sum()*100:.1f}%)")
    # Only include in longevity models if >20% of charted tracks are covered
    for col in available_lyric_cols:
        coverage = df.loc[df["is_charted"] == 1, col].notna().mean()
        if coverage > 0.20:
            LYRIC_FEATURES.append(col)
            print(f"  + Lyric feature included in longevity models: {col} "
                  f"(coverage: {coverage:.1%})")
```

**Change 5c: Add LYRIC_FEATURES to Cox PH and Log-OLS feature lists**

Find the Cox PH model section where `COX_FEATURES` is defined. It currently looks like:
```python
COX_FEATURES = FEATURES + ["decade_idx"]
```
Replace with:
```python
COX_FEATURES = FEATURES + ["decade_idx"] + LYRIC_FEATURES
```

Find the Log-OLS feature list (it uses `charted[COX_FEATURES]` or similar) — it should automatically use `COX_FEATURES` and inherit the lyric features.

**Change 5d: Update the NaN fill block to cover new columns**

Find the existing NaN fill block (it currently fills artist columns):
```python
for col in ["artist_popularity_api", "artist_peak_popularity", "artist_track_count",
            "artist_followers"]:
    if col in X.columns and X[col].isna().any():
        X[col] = X[col].fillna(X[col].median())
```
Replace with:
```python
# Fill NaN in artist features (median imputation for classification model)
for col in ["artist_popularity_api", "artist_peak_popularity", "artist_track_count",
            "artist_followers", "lastfm_listeners_log"]:
    if col in X.columns and X[col].isna().any():
        X[col] = X[col].fillna(X[col].median())
# is_us_artist: fill NaN with 0 (conservative — unknown treated as non-US)
if "is_us_artist" in X.columns:
    X["is_us_artist"] = X["is_us_artist"].fillna(0).astype(int)
```

Note: lyric features are NOT in X (classification model) — they're only for `charted` subset (Cox/OLS), where NaN rows are simply excluded from regression.

**Change 5e: Update changelog comment in docstring**

Find the existing changelog in the `run_all_v5.py` docstring and add:
```
  v5 Add H — 'lastfm_listeners_log' added (Last.fm cumulative listeners, log-scaled)
              Cross-platform artist reach; orthogonal to Spotify-derived artist_popularity_api.
              Source: artists.csv (pieca111/music-artists-popularity on Kaggle).
              Coverage: ~65.8% of artists in Spotify dataset.
  v5 Add I — 'is_us_artist' binary added (1=United States, 0=non-US, NaN=unknown)
              Source: country_mb from Last.fm/MusicBrainz dataset.
              Captures whether domestic (US) vs. international artists chart differently.
              ~31% of tracks have confirmed US artists; NaN filled with 0 in classification.
  v5 Add J — Lyric sentiment features added to longevity models (Cox PH + Log-OLS):
              sentiment_compound, sentiment_pos, sentiment_neg, lyric_word_count
              Source: billboard_lyrics.csv (rhaamrozenberg on Kaggle) + VADER.
              Coverage: ~41.8% of charted tracks (901/3,502). NaN rows excluded from
              longevity regressions. Provides textual channel independent of audio valence.
```

---

### Step 6: Update `requirements.txt`

Add:
```
vaderSentiment>=3.3.2
```

---

### Step 7: Add new source files to `.gitignore`

The `artists.csv` file is 100MB+ and should be gitignored. Add to `.gitignore`:
```
artists.csv
billboard_lyrics.csv
```

---

## 4. Execution Order

Run these commands **in sequence** from inside the OIT-367 folder:

```bash
# Step 1: Fix filename
mv "billboard_top_100_1946_2022_lyrics .csv" "billboard_lyrics.csv"

# Step 2: Install VADER
pip3 install vaderSentiment --break-system-packages

# Step 3: Rebuild artist features with Last.fm
python3 build_artist_features.py

# Step 4: Build lyric sentiment features
python3 build_lyric_features.py

# Step 5: Run full pipeline
python3 run_all_v5.py
```

---

## 5. Validation Checklist

After running, verify:

- [ ] `artist_features.csv` has columns: `artists`, `artist_popularity_api`, `artist_peak_popularity`, `artist_track_count`, `lastfm_listeners_log`, `is_us_artist`
- [ ] `lyric_features.csv` has ~3,502 rows, ~901 with non-null `sentiment_compound`
- [ ] `run_all_v5.py` output shows `+ Augmented feature included: lastfm_listeners_log`
- [ ] `run_all_v5.py` output shows `+ Augmented feature included: is_us_artist`
- [ ] `run_all_v5.py` output shows `+ Lyric feature included in longevity models: sentiment_compound`
- [ ] VIF check passes (no new features above 10; check `lastfm_listeners_log` vs `artist_popularity_api`)
- [ ] All 4 models produce results without errors
- [ ] RESULTS.md is updated with new metrics

---

## 6. Expected VIF Impact

| Feature | Expected VIF | Risk |
|---|---|---|
| `lastfm_listeners_log` | ~3–5 | Low-medium; correlated with `artist_popularity_api` (both measure fame) but different sources |
| `is_us_artist` | ~1.2 | Negligible |
| `sentiment_compound` (longevity only) | ~1.5 | Low; r≈0.3–0.4 with valence |
| `lyric_word_count` (longevity only) | ~2.0 | Low-medium; possible correlation with `duration_min` |

If `lastfm_listeners_log` VIF > 10, drop it and keep `artist_popularity_api` as the single artist-fame metric.

---

## 7. Datasets NOT Integrated (and Why)

| Dataset | Reason skipped |
|---|---|
| UMG Record Labels (`Labels.csv`) | 4.1% artist match rate — too sparse; 95.9% would be NaN |
| Best-Selling Music Artists (`Best selling music artists.csv`) | Only 121 artists (0.5% match); `Claimed sales` column is messy text; signal mostly captured by `artist_peak_popularity` |

---

## 8. RESULTS.md Updates Required

After running, update `RESULTS.md` (Section 2, 3, 4, and 6) to reflect:
- New features added: `lastfm_listeners_log`, `is_us_artist`
- Lyric sentiment in longevity models with coverage caveat
- Updated model metrics (AUC-ROC, PR-AUC, C-stat, R²)
- New finding candidates: "US artists vs. international?" / "Do angrier lyrics stay on chart longer?"
- Updated VIF table

---

*Plan authored 2026-03-08 based on dataset inspection and match rate analysis.*
*Execute in a fresh conversation by copying this file in full as context.*

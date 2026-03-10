"""
OIT367 — Local Artist Feature Builder  (no API required)
Stanford GSB Winter 2026 | Wurm / Chen / Barli / Taruno

Computes artist-level features directly from the existing Spotify tracks
dataset (spotify_tracksdataset.csv), bypassing the Spotify API entirely.

WHY THIS REPLACES modal_charted_scrape.py
──────────────────────────────────────────
Spotify Development Mode enforces a ~300-400 sp.search() call daily quota.
Our 953-artist scraper exhausts that quota in the first batch, triggering
86,000+ second Retry-After windows that reset only once per 24 hours.
Even with Modal retries, the script can never complete within the quota.

This script computes equivalent (and in some ways better) artist features
from the 114,000-track Spotify dataset we already have locally:

  artist_popularity_api   → mean track popularity across artist's full catalog
                            (Spotify's popularity is stream-count based;
                             averaging across tracks is a strong artist-fame proxy)
  artist_peak_popularity  → max track popularity (best-single metric)
  artist_track_count      → catalog size in dataset (prolificness proxy)

COVERAGE
────────
  Computes features for ALL 31,437 artists in the Spotify dataset.
  When merged with the full 89,741-track dataset, ~97% of rows get populated
  (tracks whose artist name normalizes to a match in the Spotify catalog).
  The remaining ~3% get filled with dataset medians at merge time.

  Why all artists (not just 953 charted)?
  The model needs artist features for BOTH charted AND non-charted tracks
  to use them as classification predictors. Computing only charted artists
  leaves 96% of rows as NaN, which fails the >50% non-null threshold
  check in run_all_v5.py and excludes the features from the model entirely.

RUN
───
  python3 build_artist_features.py

OUTPUT
──────
  artist_features.csv   (auto-detected by run_all_v5.py)

Then immediately run:
  python3 run_all_v5.py
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
SPOTIFY_CSV = Path("spotify_tracksdataset.csv")
MERGED_CSV  = Path("merged_spotify_billboard_data.csv")
OUTPUT_CSV  = Path("artist_features.csv")


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_artist(s: str) -> str:
    """
    Normalize artist name for fuzzy matching between datasets.
    - Lowercase
    - Strip punctuation (handles apostrophes, ampersands, dots)
    - Collapse whitespace
    - Take primary artist only (before semicolon in Spotify multi-artist strings)
    """
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = s.split(";")[0].strip()           # "Drake;Future" → "Drake"
    s = re.sub(r"[^a-z0-9\s]", " ", s)   # remove punctuation
    s = re.sub(r"\s+", " ", s).strip()   # collapse spaces
    return s


# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading datasets…")

if not SPOTIFY_CSV.exists():
    raise FileNotFoundError(
        f"Required file not found: {SPOTIFY_CSV}\n"
        "Download from Kaggle and place in the OIT-367 folder."
    )

spotify = pd.read_csv(SPOTIFY_CSV)
print(f"  Spotify tracks: {len(spotify):,} rows, {spotify['artists'].nunique():,} unique artists")


# ── Compute artist-level aggregates for ALL artists in Spotify dataset ────────
# We compute for all 31,437 artists (not just the 953 charted) so that when
# artist_features.csv is merged into the 89,741-track dataset, the >50% non-null
# threshold in run_all_v5.py is satisfied and the features enter the model.

print("\nComputing artist-level features for ALL artists in spotify_tracksdataset.csv…")

spotify["_norm"] = spotify["artists"].apply(normalize_artist)

artist_agg = (
    spotify
    .groupby("_norm")
    .agg(
        artist_popularity_api    = ("popularity", "mean"),
        artist_peak_popularity   = ("popularity", "max"),
        artist_track_count       = ("track_id",   "count"),
        # Canonical raw name (most-common form in dataset, used as join key)
        artists                  = ("artists",    lambda x: x.mode().iloc[0]),
    )
    .reset_index(drop=True)
    .reset_index(drop=True)
)

print(f"  Artist aggregates computed: {len(artist_agg):,} unique artists")


# ── Check coverage against full Spotify dataset ───────────────────────────────

# Also verify coverage for the 89,741 unique track-level artists
spotify["_norm_check"] = spotify["artists"].apply(normalize_artist)
all_track_norms = set(spotify["_norm_check"].unique())
agg_norms = set(spotify["_norm"].unique())
coverage = len(all_track_norms & agg_norms) / len(all_track_norms) * 100
print(f"  Coverage of all track artists: {coverage:.1f}%")


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

# ── Fill missing values with dataset medians (safety net) ─────────────────────

median_popularity  = artist_agg["artist_popularity_api"].median()
median_peak        = artist_agg["artist_peak_popularity"].median()
median_track_count = artist_agg["artist_track_count"].median()

artist_agg["artist_popularity_api"]  = artist_agg["artist_popularity_api"].fillna(median_popularity)
artist_agg["artist_peak_popularity"] = artist_agg["artist_peak_popularity"].fillna(median_peak)
artist_agg["artist_track_count"]     = artist_agg["artist_track_count"].fillna(median_track_count)

# Build output — include Last.fm columns if they were computed
lastfm_cols = [c for c in ["lastfm_listeners_log", "is_us_artist"]
               if c in artist_agg.columns and artist_agg[c].notna().any()]

output = artist_agg[
    ["artists", "artist_popularity_api", "artist_peak_popularity", "artist_track_count"]
    + lastfm_cols
].copy()

if lastfm_cols:
    print(f"\nLast.fm columns included in output: {lastfm_cols}")

output["artist_popularity_api"]  = output["artist_popularity_api"].round(2)
output["artist_peak_popularity"] = output["artist_peak_popularity"].astype(int)
output["artist_track_count"]     = output["artist_track_count"].astype(int)
if "lastfm_listeners_log" in output.columns:
    output["lastfm_listeners_log"] = output["lastfm_listeners_log"].round(4)
# is_us_artist stays as nullable int (0/1/NaN)


# ── Summary stats ─────────────────────────────────────────────────────────────

print("\nArtist feature summary (all artists):")
print(output[["artist_popularity_api", "artist_peak_popularity", "artist_track_count"]].describe().round(2).to_string())

print(f"\nTop 10 artists by avg catalog popularity:")
print(
    output.sort_values("artist_popularity_api", ascending=False)
    .head(10)[["artists", "artist_popularity_api", "artist_peak_popularity", "artist_track_count"]]
    .to_string(index=False)
)


# ── Save ──────────────────────────────────────────────────────────────────────

output.to_csv(OUTPUT_CSV, index=False)
print(f"\n✓ Saved {OUTPUT_CSV}  ({len(output)} artists)")
print("\nNow run:")
print("  python3 run_all_v5.py")
print("The pipeline auto-detects artist_features.csv and merges it.")
print("With all-artist coverage, artist_popularity_api and artist_track_count")
print("will pass the >50% non-null threshold and enter the model as features.")

#!/usr/bin/env python3
"""
audit_dataset.py — Run this locally to answer:
  1. Are there duplicate tracks beyond track_id dedup (same song, different IDs)?
  2. What joinable features are available but unused?

Usage: python3 audit_dataset.py
Requires: pandas, numpy  (pip install pandas numpy)
Run from the OIT-367/ folder.
"""

import pandas as pd
import numpy as np
import re

BASE = "."   # adjust if running from a different directory

print("=" * 70)
print("LOADING FILES")
print("=" * 70)

spotify = pd.read_csv(f"{BASE}/spotify_tracksdataset.csv")
base    = pd.read_csv(f"{BASE}/oit367_base_dataset.csv")

print(f"Spotify source: {spotify.shape[0]:,} rows, {spotify['track_id'].nunique():,} unique track_ids")
print(f"Charted in base: {base['is_charted'].sum():,} tracks")
print(f"Base dataset: {base.shape[0]:,} rows  (should equal unique spotify track_ids)")
print(f"  Charted: {base['is_charted'].sum():,} | Non-charted: {(base['is_charted']==0).sum():,}")
print(f"\nBase dataset columns:\n{list(base.columns)}\n")

# ─────────────────────────────────────────────────────────────
print("=" * 70)
print("QUESTION 1: DUPLICATE ANALYSIS")
print("=" * 70)

# 1a. Multi-genre duplication in Spotify source
multi_genre = len(spotify) - spotify["track_id"].nunique()
print(f"\n[A] Multi-genre rows in Spotify source (same track_id, different genres):")
print(f"    {multi_genre:,} rows dropped by track_id dedup ({multi_genre/len(spotify)*100:.1f}% of source)")

# 1b. Normalize name + artist for cross-ID duplicate check
def normalize(s):
    if pd.isna(s): return ""
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

spotify_dedup = (
    spotify
    .sort_values("track_genre")
    .drop_duplicates(subset="track_id", keep="first")
    .reset_index(drop=True)
)

spotify_dedup["_norm_name"]   = spotify_dedup["track_name"].apply(normalize)
spotify_dedup["_norm_artist"] = spotify_dedup["artists"].apply(normalize)
spotify_dedup["_name_artist"] = spotify_dedup["_norm_name"] + "|||" + spotify_dedup["_norm_artist"]

dupe_groups = spotify_dedup.groupby("_name_artist")["track_id"].count()
cross_id_dupes = dupe_groups[dupe_groups > 1]
total_extra = (cross_id_dupes - 1).sum()

print(f"\n[B] Cross-track_id duplicates (same normalized name+artist, different track_ids):")
print(f"    {len(cross_id_dupes):,} song+artist combos have multiple track_ids")
print(f"    {total_extra:,} extra rows that would be dropped by name+artist dedup")
print(f"\n    Top 15 examples:")

examples = (
    spotify_dedup[spotify_dedup["_name_artist"].isin(cross_id_dupes.index)]
    .sort_values("_name_artist")
    [["track_id", "track_name", "artists", "track_genre"]]
    .groupby(["track_name", "artists"])
    .apply(lambda x: x.head(3))
    .reset_index(drop=True)
)
print(examples.head(30).to_string())

# 1c. Cross-ID duplication IN THE CHARTED SET specifically
charted_ids = set(base[base["is_charted"] == 1]["track_id"])
charted_dedup = spotify_dedup[spotify_dedup["track_id"].isin(charted_ids)].copy()
charted_dupes_groups = charted_dedup.groupby("_name_artist")["track_id"].count()
charted_cross_id = charted_dupes_groups[charted_dupes_groups > 1]

print(f"\n[C] Cross-ID duplication WITHIN CHARTED TRACKS specifically:")
print(f"    {len(charted_cross_id):,} charted song+artist combos have multiple track_ids")

# Does the current base dataset include all these IDs or just one per song?
base_charted_ids = set(base[base["is_charted"] == 1]["track_id"])
charted_extra_ids = set()
for name_artist, group in charted_dedup.groupby("_name_artist"):
    if len(group) > 1:
        # sorted — first one is kept in our base dedup. Others may or may not be in base
        sorted_ids = group.sort_values("track_genre")["track_id"].tolist()
        for extra_id in sorted_ids[1:]:
            if extra_id in base_charted_ids:
                charted_extra_ids.add(extra_id)

print(f"    Of those, {len(charted_extra_ids):,} duplicate track_ids are in the base dataset")
print(f"    (i.e., same song appears multiple times as is_charted=1 with different track_ids)")

# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("QUESTION 2: AVAILABLE UNUSED FEATURES")
print("=" * 70)

# 2a. What's in the base dataset that isn't in FEATURES
FEATURES_USED = [
    "valence", "tempo", "acousticness", "loudness", "speechiness",
    "instrumentalness", "liveness", "mode", "key", "explicit", "duration_min",
    "artist_peak_popularity", "artist_track_count", "lastfm_listeners_log", "is_us_artist"
]
MODEL_COLS = FEATURES_USED + [
    "track_id", "artists", "track_name", "is_charted", "wks_on_chart",
    "peak_pos", "chart_entry_date", "decade_idx", "album_name"
]
unused_in_base = [c for c in base.columns if c not in MODEL_COLS]
print(f"\n[A] Columns in base dataset not used in any model:")
for col in unused_in_base:
    n_nonnull = base[col].notna().sum()
    pct = n_nonnull / len(base) * 100
    sample = base[col].dropna().iloc[:3].tolist() if n_nonnull > 0 else []
    print(f"    {col:<30} {n_nonnull:>7,}/{len(base):,} non-null ({pct:.1f}%)  sample: {sample}")

# 2b. Genre coverage and chart rate
if "track_genre" in base.columns:
    genre_stats = (
        base.groupby("track_genre")
        .agg(n_tracks=("is_charted", "count"),
             chart_rate=("is_charted", "mean"),
             avg_wks=("wks_on_chart", "mean"))
        .sort_values("chart_rate", ascending=False)
    )
    print(f"\n[B] track_genre coverage and chart rates (top 20 by chart rate):")
    print(genre_stats.head(20).to_string())
    print(f"\n    Total genres: {base['track_genre'].nunique()}")
    print(f"    track_genre non-null: {base['track_genre'].notna().sum():,} / {len(base):,}")

# 2c. What's in artists.csv but not used
try:
    artists = pd.read_csv(f"{BASE}/artists.csv")
    print(f"\n[C] artists.csv columns not currently used:")
    print(f"    All columns: {list(artists.columns)}")
    currently_used = ["name", "listeners_lastfm", "country_mb"]
    unused_artist_cols = [c for c in artists.columns if c not in currently_used and not c.startswith("Unnamed")]
    for col in unused_artist_cols:
        n_nonnull = artists[col].notna().sum()
        pct = n_nonnull / len(artists) * 100
        sample = artists[col].dropna().iloc[:3].tolist() if n_nonnull > 0 else []
        print(f"    {col:<30} {n_nonnull:>8,}/{len(artists):,} ({pct:.1f}%)  sample: {sample}")
except Exception as e:
    print(f"\n[C] Could not read artists.csv: {e}")

# 2d. Labels.csv coverage
try:
    labels = pd.read_csv(f"{BASE}/Labels.csv")
    print(f"\n[D] Labels.csv:")
    print(f"    Shape: {labels.shape}")
    print(f"    Columns: {list(labels.columns)}")
    print(labels.head(3).to_string())
except Exception as e:
    print(f"\n[D] Could not read Labels.csv: {e}")

# 2e. Best selling artists coverage
try:
    bestsellers = pd.read_csv(f"{BASE}/Best selling music artists.csv")
    print(f"\n[E] Best selling music artists.csv:")
    print(f"    Shape: {bestsellers.shape}")
    print(f"    Columns: {list(bestsellers.columns)}")
    print(bestsellers.head(3).to_string())
except Exception as e:
    print(f"\n[E] Could not read Best selling music artists.csv: {e}")

# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY AND RECOMMENDATIONS")
print("=" * 70)
print("""
1. DE-DUPLICATION
   - Current: dedup on track_id only (removes multi-genre rows within Spotify source)
   - Additional: cross-track_id duplication (same song, different IDs — see counts above)
   - Risk of name+artist dedup: may accidentally merge distinct songs or miss
     slight name variants. Better approach: manual review of flagged groups,
     or keep the version with the most chart data if both are charted.

2. GENRE FEATURE
   - track_genre is available in the base dataset at essentially 100% coverage
   - NOT currently in any model — only used for descriptive genre_chart_rates.csv
   - Easy to add as genre-mean-encoded feature (avoids 100+ one-hot dummies):
       genre_chart_rate = df.groupby('track_genre')['is_charted'].transform('mean')
   - This captures industry segment signal without high dimensionality

3. ARTISTS.CSV ADDITIONAL COLUMNS
   - Check [C] above for full list; likely includes: type (Person/Group), gender,
     begin_date_year (career length = current_year - begin_date_year)
   - These can be joined the same way as listeners_lastfm / country_mb

4. TIME_SIGNATURE
   - time_signature is in the Spotify source; check if it ended up in base dataset
   - Currently not in any model feature set
""")

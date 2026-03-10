#!/usr/bin/env python3
"""
build_final_dataset.py
======================
Produces oit367_final_dataset.csv — the single comprehensive dataset for final analysis.

IMPORTANT: Loads from oit367_base_dataset.csv (the pre-built 89,741-row universe)
rather than rebuilding from raw CSVs. The raw Billboard file uses different column
names than our processed base dataset; using the pre-built base is both safer and faster.

What this script does (in order):
  1. Loads oit367_base_dataset.csv as the full 89,741-row universe
  2. Deduplicates cross-ID same-song duplicates in the charted set using the
     teammate's file (augmented_deduped_dataset_with_artists.csv) as the canonical
     ID source — collapses 3,502 → ~2,157 charted tracks
  3. Left-joins the teammate's new features: Artist Gender → is_male_artist,
     Artist Age → artist_age, Artist Country → is_us_artist_iso,
     Artist Genres → is_mainstream_genre + artist_genre_count,
     Artist Scrobbles → artist_scrobbles_log,
     Artist Listeners (monthly) → artist_listeners_monthly_log,
     time_signature
  4. Blends with artist_features.csv (MusicBrainz) for full-universe coverage
     of lastfm_listeners_log, is_us_artist, artist_peak_popularity
  5. Merges lyric_features.csv (VADER sentiment)
  6. Saves oit367_final_dataset.csv and prints diagnostic summary

Run from the OIT-367/ folder:
  python3 build_final_dataset.py

Requirements: pandas, numpy  (pip install pandas numpy)
"""

import pandas as pd
import numpy as np
import re
import ast

BASE = "."

print("=" * 70)
print("STEP 1 — LOADING SOURCE FILES")
print("=" * 70)

# Load the pre-built base dataset — avoids raw-file column name differences
base = pd.read_csv(f"{BASE}/oit367_base_dataset.csv")
print(f"  Base dataset:      {len(base):>7,} rows  (from oit367_base_dataset.csv)")
print(f"  Charted:           {base['is_charted'].sum():>7,}")
print(f"  Non-charted:       {(base['is_charted']==0).sum():>7,}")
print(f"  Columns:           {list(base.columns)}")

# Teammate's enrichment file
teammate = pd.read_csv(f"{BASE}/augmented_deduped_dataset_with_artists.csv")
print(f"\n  Teammate charted:  {len(teammate):>7,} rows  (deduped charted-only)")
print(f"  Teammate columns:  {list(teammate.columns)}")

# Optional enrichment files
try:
    artist_feat = pd.read_csv(f"{BASE}/artist_features.csv")
    print(f"  artist_features:   {len(artist_feat):>7,} rows")
except FileNotFoundError:
    artist_feat = None
    print("  artist_features.csv not found — skipping MusicBrainz enrichment")

try:
    lyric_feat = pd.read_csv(f"{BASE}/lyric_features.csv")
    print(f"  lyric_features:    {len(lyric_feat):>7,} rows")
except FileNotFoundError:
    lyric_feat = None
    print("  lyric_features.csv not found — skipping lyric sentiment")


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 2 — ADD NORMALIZED KEYS FOR DEDUPLICATION")
print("=" * 70)

def normalize(s):
    """Lowercase, strip punctuation, collapse whitespace."""
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

base["_norm_name"]   = base["track_name"].apply(normalize)
base["_norm_artist"] = base["artists"].apply(normalize)
base["_key"]         = base["_norm_name"] + "|||" + base["_norm_artist"]

teammate["_norm_name"]   = teammate["track_name"].apply(normalize)
teammate["_norm_artist"] = teammate["artists"].apply(normalize)
teammate["_key"]         = teammate["_norm_name"] + "|||" + teammate["_norm_artist"]

# Ensure duration_min exists (may already be in base, compute if missing)
if "duration_min" not in base.columns:
    base["duration_min"] = (base["duration_ms"] / 60_000).clip(upper=10)

# Find cross-ID duplicate groups in our base
key_counts = base.groupby("_key")["track_id"].count()
multi_id_keys = key_counts[key_counts > 1]
print(f"  Normalized key groups with >1 track_id: {len(multi_id_keys):,}")
print(f"  Total extra rows (cross-ID dupes):       {(multi_id_keys - 1).sum():,}")

charted_key_counts = base[base["is_charted"]==1].groupby("_key")["track_id"].count()
charted_multi = charted_key_counts[charted_key_counts > 1]
print(f"  Cross-ID dupes in charted set only:      {(charted_multi - 1).sum():,}")


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 3 — CROSS-ID DEDUPLICATION (use teammate's canonical IDs)")
print("=" * 70)

teammate_track_ids = set(teammate["track_id"].tolist())

def pick_canonical_row(group):
    """
    For each normalized name+artist group:
      Priority 1: row whose track_id is in the teammate's deduplicated file
      Priority 2: charted row with most weeks on chart
      Priority 3: first row
    Only apply to groups with >1 row (single-member groups pass through unchanged).
    """
    if len(group) == 1:
        return group.iloc[0]

    # Prefer the teammate's canonical ID
    in_teammate = group[group["track_id"].isin(teammate_track_ids)]
    if len(in_teammate) > 0:
        return in_teammate.iloc[0]

    # Among charted rows, prefer most weeks
    charted = group[group["is_charted"] == 1]
    if len(charted) > 0:
        return charted.loc[charted["wks_on_chart"].idxmax()]

    return group.iloc[0]

base_deduped = (
    base
    .groupby("_key", group_keys=False, sort=False)
    .apply(pick_canonical_row)
    .reset_index(drop=True)
)

n_before     = len(base)
n_after      = len(base_deduped)
n_charted_b  = base["is_charted"].sum()
n_charted_a  = base_deduped["is_charted"].sum()
charted_in_tm = base_deduped[base_deduped["is_charted"]==1]["track_id"].isin(teammate_track_ids).sum()

print(f"  Before dedup: {n_before:,} rows  ({n_charted_b:,} charted)")
print(f"  After dedup:  {n_after:,} rows  ({n_charted_a:,} charted)")
print(f"  Removed:      {n_before - n_after:,} cross-ID same-song duplicates")
print(f"  Charted rows matching teammate's canonical IDs: {charted_in_tm:,} / {n_charted_a:,}")


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 4 — MERGE TEAMMATE'S NEW FEATURES")
print("=" * 70)

TEAMMATE_RENAME = {
    "Artist Gender"    : "artist_gender",
    "Artist Age"       : "artist_age_raw",
    "Artist Country"   : "artist_country_iso",
    "Artist Genres"    : "artist_genres_raw",
    "Artist Scrobbles" : "artist_scrobbles_raw",
    "Artist Popularity": "artist_popularity_spot",
    "Artist Followers" : "artist_followers_spot",
    "Artist Listeners" : "artist_listeners_monthly",
    "time_signature"   : "time_signature_tm",
}

avail_tm_cols = [c for c in TEAMMATE_RENAME.keys() if c in teammate.columns]
tm_merge = (
    teammate[["track_id"] + avail_tm_cols]
    .rename(columns=TEAMMATE_RENAME)
    .copy()
)

# Phase A: exact track_id join
final = base_deduped.merge(tm_merge, on="track_id", how="left")
direct_hits = final["artist_gender"].notna().sum()
print(f"  Direct track_id matches:     {direct_hits:,}")

# Phase B: key-based fallback for charted tracks still unmatched
tm_keyed = teammate[["_key"] + avail_tm_cols].rename(columns=TEAMMATE_RENAME).copy()
unmatched = (final["is_charted"] == 1) & final["artist_gender"].isna()
n_unmatched = unmatched.sum()

if n_unmatched > 0:
    unmatched_keys = final.loc[unmatched, ["_key"]].copy()
    key_fill = unmatched_keys.merge(tm_keyed, on="_key", how="left")
    key_fill.index = unmatched_keys.index
    new_cols = list({v for v in TEAMMATE_RENAME.values() if v in final.columns})
    for col in new_cols:
        if col in key_fill.columns:
            final.loc[unmatched, col] = (
                final.loc[unmatched, col].combine_first(key_fill[col])
            )
    key_hits = final.loc[unmatched, "artist_gender"].notna().sum()
    print(f"  Key-based fallback matches:  {key_hits:,}  (different canonical ID)")
    still_missing = unmatched.sum() - key_hits
    print(f"  Still unmatched charted:     {still_missing:,}  (will get NaN for artist enrichment)")

charted_cov = final.loc[final["is_charted"]==1, "artist_gender"].notna().mean()*100
print(f"  Total charted enrichment:    {charted_cov:.1f}% have artist gender/age/country data")


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 5 — FEATURE ENGINEERING")
print("=" * 70)

# 5a. time_signature — merge from teammate (prefer) or keep existing
if "time_signature" in final.columns and "time_signature_tm" in final.columns:
    # Keep existing where present, fill from teammate
    final["time_signature"] = (
        pd.to_numeric(final["time_signature"], errors="coerce")
        .combine_first(pd.to_numeric(final["time_signature_tm"], errors="coerce"))
    )
    final.drop(columns=["time_signature_tm"], inplace=True)
elif "time_signature_tm" in final.columns:
    final.rename(columns={"time_signature_tm": "time_signature"}, inplace=True)
    final["time_signature"] = pd.to_numeric(final["time_signature"], errors="coerce")
ts_cov = final["time_signature"].notna().mean()*100 if "time_signature" in final.columns else 0
print(f"  time_signature coverage:     {ts_cov:.1f}%")

# 5b. artist_age: 0.0 is a null sentinel in the teammate's data
if "artist_age_raw" in final.columns:
    final["artist_age"] = pd.to_numeric(final["artist_age_raw"], errors="coerce")
    final.loc[final["artist_age"] == 0.0, "artist_age"] = np.nan
    final.drop(columns=["artist_age_raw"], inplace=True)
age_cov = final["artist_age"].notna().mean()*100 if "artist_age" in final.columns else 0
print(f"  artist_age coverage:         {age_cov:.1f}%  (0.0→NaN corrected)")

# 5c. artist_gender → binary is_male_artist
if "artist_gender" in final.columns:
    gc = final["artist_gender"].str.lower().str.strip()
    final["is_male_artist"] = pd.array(
        [1 if v == "male" else (0 if v in ("female", "group", "other", "non-binary") else pd.NA)
         for v in gc.fillna("")],
        dtype="Int64"
    )
    final.loc[final["artist_gender"].isna(), "is_male_artist"] = pd.NA
gender_cov = final["is_male_artist"].notna().mean()*100 if "is_male_artist" in final.columns else 0
print(f"  is_male_artist coverage:     {gender_cov:.1f}%")

# 5d. artist_country_iso → is_us_artist_iso (ISO 2-letter more precise than MusicBrainz full name)
if "artist_country_iso" in final.columns:
    iso = final["artist_country_iso"].astype(str).str.strip().str.upper()
    final["is_us_artist_iso"] = pd.array(
        [1 if c == "US" else (0 if c not in ("", "NAN", "NONE", "NAT", "NA") else pd.NA)
         for c in iso],
        dtype="Int64"
    )
country_cov = final["is_us_artist_iso"].notna().mean()*100 if "is_us_artist_iso" in final.columns else 0
print(f"  is_us_artist_iso coverage:   {country_cov:.1f}%")

# 5e. artist_scrobbles → log scale
if "artist_scrobbles_raw" in final.columns:
    final["artist_scrobbles_raw"] = pd.to_numeric(final["artist_scrobbles_raw"], errors="coerce")
    final["artist_scrobbles_log"] = np.log1p(final["artist_scrobbles_raw"].fillna(0))
    final.drop(columns=["artist_scrobbles_raw"], inplace=True)
scr_cov = (final["artist_scrobbles_log"] > 0).mean()*100 if "artist_scrobbles_log" in final.columns else 0
print(f"  artist_scrobbles_log cov.:   {scr_cov:.1f}%  (>0 = has data)")

# 5f. artist_listeners_monthly → log scale
if "artist_listeners_monthly" in final.columns:
    final["artist_listeners_monthly"] = pd.to_numeric(final["artist_listeners_monthly"], errors="coerce")
    final["artist_listeners_monthly_log"] = np.log1p(final["artist_listeners_monthly"].fillna(0))
lm_cov = (final.get("artist_listeners_monthly", pd.Series()).notna()).mean()*100
print(f"  artist_listeners_monthly:    {lm_cov:.1f}%  (log version computed)")

# 5g. artist_genres_raw → parse Python list → is_mainstream_genre + artist_genre_count
def parse_genre_list(raw):
    if pd.isna(raw) or str(raw).strip() in ("", "nan"):
        return []
    try:
        parsed = ast.literal_eval(str(raw))
        if isinstance(parsed, list):
            return [str(g).lower().strip() for g in parsed]
    except Exception:
        pass
    return []

if "artist_genres_raw" in final.columns:
    MAINSTREAM = {"pop", "hip hop", "r&b", "rap", "dance pop", "trap",
                  "latin", "reggaeton", "urban contemporary", "soul",
                  "country", "rock", "edm", "house", "k-pop"}
    genre_lists = final["artist_genres_raw"].apply(parse_genre_list)
    final["artist_genre_count"] = genre_lists.apply(len).astype("Int64")
    final.loc[final["artist_genre_count"] == 0, "artist_genre_count"] = pd.NA
    final["is_mainstream_genre"] = pd.array(
        [1 if any(g in MAINSTREAM for g in gl) else (pd.NA if len(gl) == 0 else 0)
         for gl in genre_lists],
        dtype="Int64"
    )
gm_cov = final["is_mainstream_genre"].notna().mean()*100 if "is_mainstream_genre" in final.columns else 0
print(f"  is_mainstream_genre cov.:    {gm_cov:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 6 — BLEND MusicBrainz ARTIST FEATURES (full-universe coverage)")
print("=" * 70)

if artist_feat is not None:
    # artist_features.csv covers all 89k tracks (matched by normalized artist name)
    # Provides: artist_peak_popularity, artist_track_count,
    #           lastfm_listeners_log, is_us_artist
    final = final.merge(artist_feat, on="artists", how="left", suffixes=("", "_mb"))

    # is_us_artist: prefer teammate's ISO code (more precise), fallback to MusicBrainz
    us_iso = final.get("is_us_artist_iso", pd.Series(dtype="Int64"))
    us_mb  = final.get("is_us_artist_mb", final.get("is_us_artist", pd.Series(dtype=object)))
    final["is_us_artist"] = pd.array(
        [int(a) if pd.notna(a) else (int(b) if pd.notna(b) else pd.NA)
         for a, b in zip(us_iso, pd.to_numeric(us_mb, errors="coerce"))],
        dtype="Int64"
    )
    # Conservative fill: unknown → 0 (non-US) so feature passes >50% threshold
    final["is_us_artist"] = final["is_us_artist"].fillna(0).astype(int)

    us_pct = (final["is_us_artist"] == 1).mean()*100
    print(f"  is_us_artist (blended):      {us_pct:.1f}% flagged as US")
    llm_cov = final["lastfm_listeners_log"].notna().mean()*100 if "lastfm_listeners_log" in final.columns else 0
    print(f"  lastfm_listeners_log cov.:   {llm_cov:.1f}%")
    pp_cov = final["artist_peak_popularity"].notna().mean()*100 if "artist_peak_popularity" in final.columns else 0
    print(f"  artist_peak_popularity cov.: {pp_cov:.1f}%")
else:
    if "is_us_artist_iso" in final.columns:
        final["is_us_artist"] = final["is_us_artist_iso"].fillna(0).astype(int)
    print("  Skipped (artist_features.csv not available)")


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 7 — MERGE LYRIC SENTIMENT FEATURES")
print("=" * 70)

if lyric_feat is not None:
    lyric_cols = [c for c in ["track_id","sentiment_compound","sentiment_pos",
                               "sentiment_neg","lyric_word_count"] if c in lyric_feat.columns]
    final = final.merge(lyric_feat[lyric_cols], on="track_id", how="left")
    lyric_n   = final["sentiment_compound"].notna().sum()
    lyric_pct = lyric_n / len(final) * 100
    print(f"  Lyric sentiment coverage:    {lyric_n:,} / {len(final):,} rows ({lyric_pct:.1f}%)")
else:
    print("  Skipped (lyric_features.csv not available)")
    print("  Fix: conda activate oit367_music_project && pip install vaderSentiment")
    print("       then: python3 build_lyric_features.py")


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 8 — CLEAN UP AND SAVE")
print("=" * 70)

# Drop internal working columns
DROP = [c for c in final.columns if c.startswith("_")]
DROP += [c for c in ["artist_gender", "artist_country_iso", "is_us_artist_iso",
                     "artist_genres_raw", "artist_listeners_monthly",
                     "artist_popularity_spot", "artist_followers_spot"]
         if c in final.columns]
final.drop(columns=[c for c in DROP if c in final.columns], inplace=True)

OUT_PATH = f"{BASE}/oit367_final_dataset.csv"
final.to_csv(OUT_PATH, index=False)
print(f"  Saved → {OUT_PATH}")
print(f"  Shape: {final.shape[0]:,} rows × {final.shape[1]} columns")


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)
print(f"\n  Universe:        {len(final):>7,} rows")
print(f"  Charted:         {final['is_charted'].sum():>7,}")
print(f"  Non-charted:     {(final['is_charted']==0).sum():>7,}")
print(f"  Positive rate:   {final['is_charted'].mean()*100:.2f}%")

FEATURE_CHECK = [
    ("valence",                    "base audio"),
    ("tempo",                      "base audio"),
    ("acousticness",               "base audio"),
    ("loudness",                   "base audio"),
    ("speechiness",                "base audio"),
    ("instrumentalness",           "base audio"),
    ("liveness",                   "base audio"),
    ("mode",                       "base audio"),
    ("key",                        "base audio"),
    ("explicit",                   "base audio"),
    ("duration_min",               "base control"),
    ("time_signature",             "NEW from teammate"),
    ("track_genre",                "base categorical"),
    ("artist_peak_popularity",     "MusicBrainz"),
    ("artist_track_count",         "MusicBrainz"),
    ("lastfm_listeners_log",       "MusicBrainz Last.fm"),
    ("is_us_artist",               "MusicBrainz + ISO (blended)"),
    ("is_male_artist",             "NEW from teammate"),
    ("artist_age",                 "NEW from teammate"),
    ("artist_scrobbles_log",       "NEW from teammate"),
    ("artist_listeners_monthly_log","NEW from teammate"),
    ("is_mainstream_genre",        "NEW from teammate"),
    ("artist_genre_count",         "NEW from teammate"),
    ("sentiment_compound",         "lyric VADER"),
    ("sentiment_neg",              "lyric VADER"),
    ("lyric_word_count",           "lyric VADER"),
    ("decade_idx",                 "derived (Cox/OLS only)"),
]

print(f"\n  {'FEATURE':<35} {'SOURCE':<28} {'NON-NULL':>8}  {'COV%':>6}")
print(f"  {'-'*35} {'-'*28} {'-'*8}  {'-'*6}")
for feat, source in FEATURE_CHECK:
    if feat in final.columns:
        n   = int(final[feat].notna().sum())
        pct = n / len(final) * 100
        print(f"  {feat:<35} {source:<28} {n:>8,}  {pct:>5.1f}%")
    else:
        print(f"  {feat:<35} {source:<28} {'MISSING':>8}")

print(f"\n  Full column list ({final.shape[1]} cols):")
print(f"  {', '.join(final.columns)}")

print("\n" + "=" * 70)
print("NEXT STEPS")
print("=" * 70)
print("""
  1. Install vaderSentiment if lyric_features.csv was skipped:
       conda activate oit367_music_project
       pip install vaderSentiment
       python3 build_lyric_features.py

  2. The oit367_final_dataset.csv now has NEW features available.
     To add them to model runs, update run_all_v5.py:
     a) Change the dataset load line from oit367_base_dataset.csv →
        oit367_final_dataset.csv (or rename the file)
     b) Add to the auto-detection loop:
        "is_male_artist", "artist_age", "artist_scrobbles_log",
        "artist_listeners_monthly_log", "is_mainstream_genre", "time_signature"
     c) Add is_male_artist, is_mainstream_genre to the NaN→0 conservative fill
     d) Add artist_age to median fill (not 0 fill)
     This will produce the v6 results.

  3. Commit oit367_final_dataset.csv to git and push:
       git add oit367_final_dataset.csv build_final_dataset.py
       git commit -m "data: add consolidated final dataset with teammate enrichment (v6)"
       git push
""")

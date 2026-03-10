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

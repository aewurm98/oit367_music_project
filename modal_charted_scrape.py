"""
OIT367 — Modal Cloud Spotify Artist Scraper  (CHARTED-ONLY edition)
Stanford GSB Winter 2026 | Wurm / Chen / Barli / Taruno

Fetches artist_followers and artist_popularity_api for the ~953 unique
artists that appear on the Billboard Hot 100 in our dataset.

This is a targeted replacement for modal_spotify_scrape.py.
Scope reduction: 31,437 artists → 953 artists (96.9% fewer API calls)
Estimated run time: ~3 minutes (2 batches of 500, 2 workers)

──────────────────────────────────────────────────────────────
WHEN TO RUN THIS
──────────────────────────────────────────────────────────────

Spotify's Development Mode enforces a rolling rate-limit window.
If you hit the 429 "Retry-After: 28000+ s" error, wait until the
window resets before running (typically ~8 hours from the last
exhaustion event).

To check if the rate limit has reset: try a quick manual Spotify API
call or simply run the script — if it progresses past the first batch
within 30 seconds, you're good.

──────────────────────────────────────────────────────────────
CANCEL THE STUCK FULL SCRAPER FIRST
──────────────────────────────────────────────────────────────

    modal app stop oit367-spotify

Then run this targeted version:

    modal run --detach modal_charted_scrape.py

Monitor at: https://modal.com/apps

──────────────────────────────────────────────────────────────
AFTER COMPLETION — download results:
──────────────────────────────────────────────────────────────

    modal volume get oit367-vol /data/charted_artist_features.csv ./artist_features.csv

Then re-run the model pipeline:

    python3 run_all_v5.py

The pipeline auto-detects artist_features.csv and merges it.

──────────────────────────────────────────────────────────────
DESIGN NOTES
──────────────────────────────────────────────────────────────

Scope        : Only the ~953 unique artist strings from is_charted==1 tracks
               vs. 31,437 in the full dataset. Rate-limit risk drops ~97%.
Parallelism  : 2 concurrent workers, 500 artists/batch → 2 batches
               Phase 1 search: 0.35s × 500 / 2 workers → ~1.5 min total
               Phase 2 fetch : individual sp.artist() calls (Dev Mode safe)
Resilience   : Idempotent — completed batch CSVs survive restarts.
               Re-run at any time; finished batches are skipped.
Volume key   : /data/charted_artist_features.csv (separate from full-scrape)
"""

import modal
import sys

# ── Modal app & infrastructure ────────────────────────────────────────────────
app = modal.App("oit367-charted-scrape")

vol = modal.Volume.from_name("oit367-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("spotipy>=2.23", "pandas>=2.0")
)

VOLUME_DATA_DIR = "/data"
BATCH_SIZE      = 500   # 953 artists → 2 batches


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_primary_artist(artist_str: str) -> str:
    """
    Normalize the 'artists' column value to a single search name.
    Handles list-like strings: "['Drake']", "['Taylor Swift', 'Ed Sheeran']"
    """
    import ast, re
    s = str(artist_str).strip()
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list) and parsed:
            return str(parsed[0]).strip()
    except Exception:
        pass
    s = re.sub(r"^[\[\(\"']|[\]\)\"']$", "", s)
    return s.split(",")[0].strip()


# ── Remote function: scrape one batch ─────────────────────────────────────────

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("spotify-credentials")],
    volumes={VOLUME_DATA_DIR: vol},
    timeout=3600,
    max_containers=2,
    retries=modal.Retries(
        max_retries=3,
        backoff_coefficient=2.0,
        initial_delay=10.0,
    ),
)
def scrape_charted_batch(batch: list, batch_id: int) -> int:
    """
    Scrape Spotify artist features for one batch of charted artists.

    Two-phase approach:
      Phase 1 — sp.search() per artist → collect Spotify artist IDs
      Phase 2 — sp.artist(id) per ID  → get followers + popularity
                 (sp.artists() batch endpoint removed in Dev Mode Feb 2026)

    Parameters
    ----------
    batch    : list of dicts with keys 'raw' (join key) and 'primary' (search name)
    batch_id : int, used for output CSV naming

    Returns
    -------
    int : number of artist records written
    """
    import os, time, random
    import pandas as pd
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    from spotipy.cache_handler import MemoryCacheHandler
    from pathlib import Path

    out_path = Path(VOLUME_DATA_DIR) / f"charted_batch_{batch_id:04d}.csv"

    # Idempotent: skip if already completed
    if out_path.exists():
        existing = pd.read_csv(out_path)
        print(f"[batch {batch_id:04d}] Already done ({len(existing)} records). Skipping.")
        return len(existing)

    sp = spotipy.Spotify(
        auth_manager=SpotifyClientCredentials(
            client_id=os.environ["SPOTIPY_CLIENT_ID"],
            client_secret=os.environ["SPOTIPY_CLIENT_SECRET"],
            cache_handler=MemoryCacheHandler(),
        ),
        requests_timeout=15,
        retries=5,
    )

    def safe_search(search_name: str) -> str | None:
        """Search for an artist; return their Spotify ID, or None on failure."""
        for attempt in range(6):
            try:
                r = sp.search(q=f"artist:{search_name}", type="artist", limit=1)
                items = r.get("artists", {}).get("items", [])
                return items[0]["id"] if items else None
            except Exception as exc:
                err = str(exc)
                if "429" in err or "rate" in err.lower() or "timeout" in err.lower():
                    wait = min(120, (2 ** attempt) * 5 + random.random() * 2)
                    print(f"  429 on search '{search_name}' attempt {attempt+1}. "
                          f"Sleeping {wait:.0f}s …")
                    time.sleep(wait)
                else:
                    print(f"  Search error '{search_name}': {err[:80]}")
                    return None
        print(f"  Giving up on '{search_name}' after 6 attempts.")
        return None

    def fetch_artist(artist_id: str) -> dict:
        """Fetch full ArtistObject (with followers) via sp.artist(id)."""
        for attempt in range(6):
            try:
                obj = sp.artist(artist_id)
                return {
                    "artist_followers":      obj.get("followers", {}).get("total"),
                    "artist_popularity_api": obj.get("popularity"),
                }
            except Exception as exc:
                err = str(exc)
                if "429" in err or "rate" in err.lower() or "timeout" in err.lower():
                    wait = min(120, (2 ** attempt) * 5 + random.random() * 2)
                    time.sleep(wait)
                else:
                    break
        return {}

    # ── Phase 1: search each artist → Spotify ID ─────────────────────────────
    id_map: dict[str, str | None] = {}
    for idx, item in enumerate(batch):
        raw_key     = item["raw"]
        search_name = item["primary"]
        id_map[raw_key] = safe_search(search_name)
        time.sleep(0.35)   # ~171 req/min per worker, safely under limit

        if (idx + 1) % 100 == 0:
            print(f"[batch {batch_id:04d}] Phase 1: {idx+1}/{len(batch)} searched "
                  f"({sum(v is not None for v in id_map.values())} IDs found)")

    found = sum(v is not None for v in id_map.values())
    print(f"[batch {batch_id:04d}] Phase 1 complete: {found}/{len(batch)} IDs resolved")

    # ── Phase 2: fetch full ArtistObject for each resolved ID ─────────────────
    # Using individual sp.artist() — compatible with Spotify Dev Mode (Feb 2026)
    artist_details: dict[str, dict] = {}
    valid_pairs = [(raw, sid) for raw, sid in id_map.items() if sid]

    for i, (raw_key, artist_id) in enumerate(valid_pairs):
        artist_details[raw_key] = fetch_artist(artist_id)
        time.sleep(0.35)
        if (i + 1) % 100 == 0:
            print(f"[batch {batch_id:04d}] Phase 2: {i+1}/{len(valid_pairs)} fetched")

    print(f"[batch {batch_id:04d}] Phase 2 complete: {len(artist_details)} artists fetched")

    # ── Write results ──────────────────────────────────────────────────────────
    results = []
    for item in batch:
        raw_key = item["raw"]
        details = artist_details.get(raw_key, {})
        results.append({
            "artists":               raw_key,
            "artist_followers":      details.get("artist_followers"),
            "artist_popularity_api": details.get("artist_popularity_api"),
        })

    df_batch = pd.DataFrame(results)
    df_batch.to_csv(out_path, index=False)
    vol.commit()

    n_found = df_batch["artist_followers"].notna().sum()
    print(f"[batch {batch_id:04d}] DONE — {len(df_batch)} records, "
          f"{n_found} with follower data ({n_found/len(df_batch):.0%} hit rate).")
    return len(df_batch)


# ── Remote function: merge all batch CSVs ────────────────────────────────────

@app.function(
    image=image,
    volumes={VOLUME_DATA_DIR: vol},
    timeout=120,
)
def merge_charted_batches() -> int:
    """Concatenate charted_batch_*.csv → charted_artist_features.csv"""
    import pandas as pd
    from pathlib import Path

    vol.reload()
    data_dir    = Path(VOLUME_DATA_DIR)
    batch_files = sorted(data_dir.glob("charted_batch_*.csv"))

    if not batch_files:
        print("ERROR: No charted batch files found on volume.")
        return 0

    print(f"Merging {len(batch_files)} batch file(s)…")
    dfs = [pd.read_csv(f) for f in batch_files]
    merged = pd.concat(dfs, ignore_index=True).drop_duplicates(
        subset="artists", keep="first"
    )

    out = data_dir / "charted_artist_features.csv"
    merged.to_csv(out, index=False)
    vol.commit()

    n_data = merged["artist_followers"].notna().sum()
    print(f"Merged → {len(merged)} unique artist records")
    print(f"  With follower data  : {n_data} ({n_data/len(merged):.1%})")
    print(f"  Saved to volume     : /data/charted_artist_features.csv")
    return len(merged)


# ── Local entrypoint ──────────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    """
    1. Read oit367_base_dataset.csv locally.
    2. Filter to is_charted==1 rows → extract unique artist strings.
    3. Build search names, batch into groups of 500.
    4. Dispatch to Modal; merge when done.
    """
    import pandas as pd

    csv_path = "oit367_base_dataset.csv"
    try:
        df = pd.read_csv(csv_path, usecols=["artists", "is_charted"])
    except FileNotFoundError:
        print(f"ERROR: '{csv_path}' not found.")
        print("Run python3 run_all_v5.py first to generate the base dataset.")
        sys.exit(1)

    # Filter to charted tracks only
    charted_df   = df[df["is_charted"] == 1]
    raw_artists  = charted_df["artists"].dropna().unique().tolist()
    artist_records = [
        {"raw": raw, "primary": extract_primary_artist(raw)}
        for raw in raw_artists
    ]

    print(f"Charted artist strings           : {len(artist_records):,}")
    print(f"Batch size                       : {BATCH_SIZE}")

    batches   = [
        artist_records[i : i + BATCH_SIZE]
        for i in range(0, len(artist_records), BATCH_SIZE)
    ]
    n_batches = len(batches)
    est_min   = (len(artist_records) * 0.35) / 60 / 2   # 2 workers, phase 1 dominates
    print(f"Number of batches                : {n_batches}")
    print(f"Estimated run time (2 workers)   : ~{est_min:.0f} minutes")
    print(f"\nDispatching {n_batches} batch(es) to Modal…")
    print("(Monitor at https://modal.com/apps)\n")

    results       = list(scrape_charted_batch.starmap(
        [(batch, i) for i, batch in enumerate(batches)]
    ))
    total_scraped = sum(results)
    print(f"\nAll batches complete. Total artist records: {total_scraped:,}")

    print("\nMerging batch files…")
    n_merged = merge_charted_batches.remote()
    print(f"Merge complete: {n_merged:,} unique artists in charted_artist_features.csv")

    print("\n" + "=" * 60)
    print("DONE — Download results with:")
    print()
    print("  modal volume get oit367-vol /data/charted_artist_features.csv ./artist_features.csv")
    print()
    print("Then run the augmented pipeline:")
    print()
    print("  python3 run_all_v5.py")
    print("=" * 60)

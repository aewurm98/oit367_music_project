"""
OIT367 — Modal Cloud Spotify Artist Scraper
Stanford GSB Winter 2026 | Wurm / Chen / Barli / Taruno

Fetches artist_followers and artist_popularity_api for all ~15k unique
artists in oit367_base_dataset.csv using the Spotify Web API.

Runs entirely in Modal's cloud — your laptop can close while it runs.
Results persist in a Modal Volume (oit367-vol) and are downloaded when done.

──────────────────────────────────────────────────────────────
ONE-TIME SETUP (run these commands in your terminal once):
──────────────────────────────────────────────────────────────

1.  pip3 install modal

2.  modal token new
    # Opens browser — log in with GitHub/Google. One-time auth.

3.  modal secret create spotify-credentials \\
        SPOTIPY_CLIENT_ID=YOUR_CLIENT_ID_HERE \\
        SPOTIPY_CLIENT_SECRET=YOUR_CLIENT_SECRET_HERE
    # Paste your actual credentials. No quotes needed.

──────────────────────────────────────────────────────────────
RUNNING THE SCRAPER:
──────────────────────────────────────────────────────────────

    cd path/to/OIT-367
    modal run modal_spotify_scrape.py

    # To run in background (laptop can close):
    modal run --detach modal_spotify_scrape.py
    # Monitor at: https://modal.com/apps

──────────────────────────────────────────────────────────────
AFTER IT COMPLETES — download results:
──────────────────────────────────────────────────────────────

    modal volume get oit367-vol /data/artist_features.csv ./artist_features.csv

    # Then re-run the model pipeline (auto-detects new columns):
    python3 run_all_v4.py

──────────────────────────────────────────────────────────────
DESIGN NOTES:
──────────────────────────────────────────────────────────────

Parallelism  : 2 concurrent workers (max_containers=2)
               → ~340 search req/min total, safely under Spotify's ~360/min limit

Batch size   : 500 artists per batch → ~63 batches for 31k artists
               Phase 1 (search per artist): 0.35s throttle → ~3 min/batch
               Phase 2 (sp.artists batch, 50 at a time): ~10 calls, near-instant
               Both workers: ~94 min total

Resilience   : retries=2 on each function; each batch is idempotent
               (batch CSV written atomically at end, not mid-run)
               Re-run the script at any time — completed batches are skipped

Resume       : batch_{id}.csv files on the volume survive restarts.
               The merge step is run last, after all batches complete.
"""

import modal
import sys

# ── Modal app & infrastructure ────────────────────────────────────────────────
app = modal.App("oit367-spotify")

vol = modal.Volume.from_name("oit367-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("spotipy>=2.23", "pandas>=2.0", "tqdm>=4.65")
)

VOLUME_DATA_DIR = "/data"
BATCH_SIZE = 500


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_primary_artist(artist_str: str) -> str:
    """
    Spotify's 'artists' column often stores a Python-list-like string:
        "['Drake']"
        "['Taylor Swift', 'Ed Sheeran']"
        "Taylor Swift"            (plain string in some exports)

    Returns the first artist name as a clean string for the API search.
    """
    import ast
    import re

    s = str(artist_str).strip()
    # Try parsing as a Python literal (handles ['Name1', 'Name2'] format)
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list) and parsed:
            return str(parsed[0]).strip()
    except Exception:
        pass
    # Fallback: strip brackets/quotes, take everything before first comma
    s = re.sub(r"^[\[\(\"']|[\]\)\"']$", "", s)
    return s.split(",")[0].strip()


# ── Remote function: scrape one batch ─────────────────────────────────────────

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("spotify-credentials")],
    volumes={VOLUME_DATA_DIR: vol},
    timeout=3600,      # 1 hour max per worker
    max_containers=2,  # FIX 1: was concurrency_limit=2 (renamed in Modal 1.0)
    # FIX 4: retries= now requires a modal.Retries object, not a plain integer
    retries=modal.Retries(
        max_retries=2,
        backoff_coefficient=2.0,
        initial_delay=5.0,
    ),
)
def scrape_batch(batch: list, batch_id: int) -> int:
    """
    Scrape Spotify artist features for one batch of artists.

    Parameters
    ----------
    batch     : list of dicts, each with keys 'raw' and 'primary'
                  raw     — original artists string from the dataset (join key)
                  primary — normalized first artist name (used for API search)
    batch_id  : int, used for naming the output CSV

    Returns
    -------
    int : number of artist records written
    """
    import os
    import time
    import random
    import pandas as pd
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    from spotipy.cache_handler import MemoryCacheHandler  # FIX 2: avoid .cache disk writes
    from pathlib import Path

    out_path = Path(VOLUME_DATA_DIR) / f"batch_{batch_id:04d}.csv"

    # Skip if this batch was already completed in a prior run (idempotent)
    if out_path.exists():
        existing = pd.read_csv(out_path)
        print(f"[batch {batch_id:04d}] Already done ({len(existing)} records). Skipping.")
        return len(existing)

    # FIX 2: MemoryCacheHandler prevents spotipy from trying to read/write a
    # .cache token file on disk — Modal containers have no writable cwd for this.
    sp = spotipy.Spotify(
        auth_manager=SpotifyClientCredentials(
            client_id=os.environ["SPOTIPY_CLIENT_ID"],
            client_secret=os.environ["SPOTIPY_CLIENT_SECRET"],
            cache_handler=MemoryCacheHandler(),
        ),
        requests_timeout=10,
        retries=3,
    )

    def safe_search(search_name: str) -> str | None:
        """Search for an artist by name and return their Spotify ID, or None."""
        for attempt in range(5):
            try:
                r = sp.search(q=f"artist:{search_name}", type="artist", limit=1)
                items = r.get("artists", {}).get("items", [])
                return items[0]["id"] if items else None
            except Exception as exc:
                err = str(exc)
                if "429" in err or "rate" in err.lower() or "timeout" in err.lower():
                    wait = (2 ** attempt) + random.random()
                    print(f"  Rate-limited on '{search_name}' (attempt {attempt+1}). "
                          f"Waiting {wait:.1f}s …")
                    time.sleep(wait)
                else:
                    print(f"  Search error on '{search_name}': {err[:80]}")
                    return None
        return None

    # ── Phase 1: search each artist → collect Spotify IDs ────────────────────
    # FIX 3: Spotify's search endpoint returns a SimplifiedArtistObject that
    # does NOT include the 'followers' field. We collect IDs here, then fetch
    # full ArtistObjects (with followers) in Phase 2 via sp.artists(ids).
    id_map: dict[str, str | None] = {}   # raw_key → spotify_id or None
    for idx, item in enumerate(batch):
        raw_key     = item["raw"]
        search_name = item["primary"]
        id_map[raw_key] = safe_search(search_name)
        time.sleep(0.35)   # proactive throttle: ~171 req/min per worker

        if (idx + 1) % 50 == 0:
            print(f"[batch {batch_id:04d}] phase 1 checkpoint: {idx+1}/{len(batch)} searched")

    # ── Phase 2: fetch full ArtistObjects (includes followers) ───────────────
    # FIX 5: Spotify removed the batch GET /artists endpoint for apps in
    # Development Mode (February 2026). All new Spotify apps default to
    # Development Mode. We try the batch endpoint first (faster for Extended
    # Quota Mode apps), and fall back to individual sp.artist(id) calls if it
    # returns a 403/404, which works in both modes.
    artist_details: dict[str, dict] = {}
    valid_pairs = [(raw, sid) for raw, sid in id_map.items() if sid]

    def fetch_one(raw_key: str, artist_id: str) -> None:
        """Fetch a single artist's full object and store in artist_details."""
        for attempt in range(5):
            try:
                obj = sp.artist(artist_id)
                artist_details[raw_key] = {
                    "artist_followers":      obj.get("followers", {}).get("total"),
                    "artist_popularity_api": obj.get("popularity"),
                }
                return
            except Exception as exc:
                err = str(exc)
                if "429" in err or "rate" in err.lower() or "timeout" in err.lower():
                    wait = (2 ** attempt) + random.random()
                    time.sleep(wait)
                else:
                    break   # non-rate error, leave as None

    # Try batch first; if it fails (Dev Mode), fall back to individual calls
    use_batch = True
    for chunk_start in range(0, len(valid_pairs), 50):
        chunk          = valid_pairs[chunk_start : chunk_start + 50]
        raw_keys_chunk = [x[0] for x in chunk]
        ids_chunk      = [x[1] for x in chunk]

        if use_batch:
            try:
                response = sp.artists(ids_chunk)
                for raw_key, artist_obj in zip(raw_keys_chunk, response["artists"]):
                    if artist_obj:
                        artist_details[raw_key] = {
                            "artist_followers":      artist_obj.get("followers", {}).get("total"),
                            "artist_popularity_api": artist_obj.get("popularity"),
                        }
                time.sleep(0.2)
                continue   # batch succeeded, move to next chunk
            except Exception as exc:
                err = str(exc)
                if "403" in err or "404" in err or "forbidden" in err.lower():
                    # Batch endpoint not available (Development Mode app)
                    print("  Batch endpoint unavailable (Dev Mode) — switching to "
                          "individual sp.artist() calls for remaining chunks.")
                    use_batch = False
                elif "429" in err or "rate" in err.lower():
                    time.sleep(5 + random.random())
                    use_batch = False   # be conservative after a rate limit
                else:
                    print(f"  Batch fetch error: {err[:80]} — switching to individual calls.")
                    use_batch = False

        # Individual fallback (works in all Spotify quota modes)
        for raw_key, artist_id in zip(raw_keys_chunk, ids_chunk):
            fetch_one(raw_key, artist_id)
            time.sleep(0.35)   # match Phase 1 throttle

    # ── Build results and write ───────────────────────────────────────────────
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
    print(f"[batch {batch_id:04d}] Done — {len(df_batch)} records, "
          f"{n_found} with follower data.")
    return len(df_batch)


# ── Remote function: merge all batch CSVs ────────────────────────────────────

@app.function(
    image=image,
    volumes={VOLUME_DATA_DIR: vol},
    timeout=300,
)
def merge_batches() -> int:
    """
    Read all batch_NNNN.csv files from the volume, concatenate them,
    deduplicate on the 'artists' join key, and write artist_features.csv.
    Returns the number of unique artist records.
    """
    import pandas as pd
    from pathlib import Path

    vol.reload()   # ensure we see latest committed files

    data_dir   = Path(VOLUME_DATA_DIR)
    batch_files = sorted(data_dir.glob("batch_*.csv"))

    if not batch_files:
        print("ERROR: No batch files found on volume.")
        return 0

    print(f"Merging {len(batch_files)} batch files …")
    dfs = []
    for f in batch_files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"  Warning: could not read {f.name}: {e}")

    merged = pd.concat(dfs, ignore_index=True)
    # Keep first occurrence if an artist appears in multiple batches
    merged = merged.drop_duplicates(subset="artists", keep="first")

    out = data_dir / "artist_features.csv"
    merged.to_csv(out, index=False)
    vol.commit()

    n_with_data = merged["artist_followers"].notna().sum()
    print(f"Merged → {len(merged)} unique artist records")
    print(f"  With follower data : {n_with_data} ({n_with_data/len(merged):.1%})")
    print(f"  Saved to volume    : /data/artist_features.csv")
    return len(merged)


# ── Local entrypoint (runs on your machine, orchestrates Modal) ───────────────

@app.local_entrypoint()
def main():
    """
    1. Reads oit367_base_dataset.csv locally to get unique artist strings.
    2. Normalizes to primary artist names for the Spotify search.
    3. Splits into batches of BATCH_SIZE and dispatches to Modal workers.
    4. Runs merge after all batches complete.
    5. Prints download instructions.
    """
    import pandas as pd

    # ── Load dataset ──────────────────────────────────────────────────────────
    csv_path = "oit367_base_dataset.csv"
    try:
        df = pd.read_csv(csv_path, usecols=["artists"])
    except FileNotFoundError:
        print(f"ERROR: '{csv_path}' not found.")
        print("Make sure you run this script from the OIT-367 folder and that")
        print("oit367_base_dataset.csv exists (run run_all_v4.py first if needed).")
        sys.exit(1)

    # ── Build artist list ─────────────────────────────────────────────────────
    raw_artists = df["artists"].dropna().unique().tolist()
    artist_records = [
        {"raw": raw, "primary": extract_primary_artist(raw)}
        for raw in raw_artists
    ]

    print(f"Unique artist strings in dataset : {len(artist_records):,}")
    print(f"Batch size                       : {BATCH_SIZE}")

    # ── Split into batches ────────────────────────────────────────────────────
    batches = [
        artist_records[i : i + BATCH_SIZE]
        for i in range(0, len(artist_records), BATCH_SIZE)
    ]
    n_batches = len(batches)
    # Phase 1 (search): 0.35s × n_artists / 2 workers
    # Phase 2 (batch fetch): negligible (~10 calls per batch)
    est_minutes = (len(artist_records) * 0.35) / 60 / 2
    print(f"Number of batches                : {n_batches}")
    print(f"Estimated run time (2 workers)   : ~{est_minutes:.0f} minutes (search phase)")
    print(f"\nDispatching {n_batches} batches to Modal …")
    print("(Monitor progress at https://modal.com/apps)\n")

    # ── Run all batches in parallel ───────────────────────────────────────────
    # max_containers=2 on scrape_batch ensures at most 2 run simultaneously
    results = list(
        scrape_batch.starmap(
            [(batch, i) for i, batch in enumerate(batches)]
        )
    )
    total_scraped = sum(results)
    print(f"\nAll batches complete. Total artist records: {total_scraped:,}")

    # ── Merge all batch CSVs into one file ────────────────────────────────────
    print("\nMerging batch files …")
    n_merged = merge_batches.remote()
    print(f"Merge complete: {n_merged:,} unique artists in artist_features.csv")

    # ── Download instructions ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DONE — Download your results with:")
    print()
    print("  modal volume get oit367-vol /data/artist_features.csv ./artist_features.csv")
    print()
    print("Then run the augmented model pipeline:")
    print()
    print("  python3 run_all_v4.py")
    print()
    print("The pipeline auto-detects artist_followers and artist_popularity_api")
    print("columns and adds them to all models.")
    print("=" * 60)

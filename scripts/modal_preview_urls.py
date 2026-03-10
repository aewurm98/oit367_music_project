"""
OIT367 — Modal: Fetch Spotify Preview URLs  (Job 1 of 2 for Librosa pipeline)
Stanford GSB Winter 2026 | Wurm / Chen / Barli / Taruno

Fetches the 30-second preview_url for each charted track using sp.track(track_id).
Uses oit367_base_dataset.csv (3,502 charted, pre-dedup). Preview URLs point to
Spotify's CDN (no API call needed to download them) and are consumed by
modal_librosa_extract.py.

──────────────────────────────────────────────────────────────
RUN ORDER (after Spotify rate limit resets)
──────────────────────────────────────────────────────────────

  Step 1 — Artist features (already written, ~3 min):
    modal run --detach modal_charted_scrape.py

  Step 2 — Preview URLs (this script, ~20 min):
    modal run --detach modal_preview_urls.py

  Step 3 — Librosa extraction (modal_librosa_extract.py, ~2-3 hr):
    # Wait for Step 2 to finish, then:
    modal run --detach modal_librosa_extract.py

  Download all results when complete:
    modal volume get oit367-vol /data/charted_artist_features.csv ./artist_features.csv
    modal volume get oit367-vol /data/preview_urls.csv ./preview_urls.csv
    modal volume get oit367-vol /data/librosa_features.csv ./librosa_features.csv

──────────────────────────────────────────────────────────────
WHY THIS IS A SEPARATE JOB FROM LIBROSA
──────────────────────────────────────────────────────────────

sp.track() requires the Spotify Web API (rate-limited, credentials needed).
Downloading the actual MP3 preview is a plain HTTPS request to Spotify's CDN —
no credentials, no rate limit. Separating the two jobs means:
  - The Librosa job can be scaled to 20 containers with no API constraint.
  - If preview URLs expire before the Librosa job runs, re-run this script
    (fast, idempotent) to refresh them without re-running Librosa.

──────────────────────────────────────────────────────────────
COVERAGE EXPECTATION
──────────────────────────────────────────────────────────────

~85–90% of tracks have a non-null preview_url. Tracks without previews
are typically due to licensing restrictions (older catalog, some regional
content). Expect ~2,900–3,150 usable URLs from the 3,502 charted tracks.
"""

import modal
import sys

# ── Modal app & infrastructure ────────────────────────────────────────────────
app = modal.App("oit367-preview-urls")

vol = modal.Volume.from_name("oit367-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("spotipy>=2.23", "pandas>=2.0")
)

VOLUME_DATA_DIR = "/data"
BATCH_SIZE      = 100    # 3,502 tracks → 36 batches; sp.track() is fast


# ── Remote function: fetch preview URLs for one batch ────────────────────────

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("spotify-credentials")],
    volumes={VOLUME_DATA_DIR: vol},
    timeout=600,         # 10 min per batch of 100 is very generous
    max_containers=3,    # 3 × 100 = 300 concurrent tracks; ~8 API calls/sec
    retries=modal.Retries(
        max_retries=3,
        backoff_coefficient=2.0,
        initial_delay=10.0,
    ),
)
def fetch_preview_batch(batch: list, batch_id: int) -> int:
    """
    Call sp.track(track_id) for each track in the batch to retrieve
    preview_url. Writes a batch CSV to the Modal volume.

    Parameters
    ----------
    batch    : list of dicts with keys 'track_id', 'track_name', 'artists'
    batch_id : int, used for output CSV naming

    Returns
    -------
    int : number of tracks with a non-null preview_url
    """
    import os, time, random
    import pandas as pd
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    from spotipy.cache_handler import MemoryCacheHandler
    from pathlib import Path

    out_path = Path(VOLUME_DATA_DIR) / f"preview_batch_{batch_id:04d}.csv"

    # Idempotent: skip if already completed
    if out_path.exists():
        existing = pd.read_csv(out_path)
        n_found  = existing["preview_url"].notna().sum()
        print(f"[batch {batch_id:04d}] Already done "
              f"({n_found}/{len(existing)} with preview). Skipping.")
        return n_found

    sp = spotipy.Spotify(
        auth_manager=SpotifyClientCredentials(
            client_id=os.environ["SPOTIPY_CLIENT_ID"],
            client_secret=os.environ["SPOTIPY_CLIENT_SECRET"],
            cache_handler=MemoryCacheHandler(),
        ),
        requests_timeout=15,
        retries=5,
    )

    def fetch_one(track_id: str) -> str | None:
        """Return preview_url for a track_id, or None on failure."""
        for attempt in range(6):
            try:
                obj = sp.track(track_id)
                return obj.get("preview_url")   # None if no preview available
            except Exception as exc:
                err = str(exc)
                if "429" in err or "rate" in err.lower() or "timeout" in err.lower():
                    wait = min(120, (2 ** attempt) * 5 + random.random() * 2)
                    print(f"  429 on track {track_id} attempt {attempt+1}. "
                          f"Sleeping {wait:.0f}s …")
                    time.sleep(wait)
                else:
                    print(f"  Error on track {track_id}: {err[:80]}")
                    return None
        return None

    results = []
    for idx, item in enumerate(batch):
        track_id   = item["track_id"]
        preview_url = fetch_one(track_id)
        results.append({
            "track_id":    track_id,
            "track_name":  item.get("track_name", ""),
            "artists":     item.get("artists", ""),
            "preview_url": preview_url,
        })
        time.sleep(0.35)   # proactive throttle

        if (idx + 1) % 25 == 0:
            n_so_far = sum(1 for r in results if r["preview_url"])
            print(f"[batch {batch_id:04d}] {idx+1}/{len(batch)} fetched "
                  f"({n_so_far} with preview so far)")

    df_out = pd.DataFrame(results)
    df_out.to_csv(out_path, index=False)
    vol.commit()

    n_found = df_out["preview_url"].notna().sum()
    print(f"[batch {batch_id:04d}] DONE — "
          f"{n_found}/{len(df_out)} tracks have a preview URL "
          f"({n_found/len(df_out):.0%} hit rate).")
    return n_found


# ── Remote function: merge all batch CSVs ────────────────────────────────────

@app.function(
    image=image,
    volumes={VOLUME_DATA_DIR: vol},
    timeout=120,
)
def merge_preview_batches() -> int:
    """Concatenate preview_batch_*.csv → preview_urls.csv on the volume."""
    import pandas as pd
    from pathlib import Path

    vol.reload()
    data_dir    = Path(VOLUME_DATA_DIR)
    batch_files = sorted(data_dir.glob("preview_batch_*.csv"))

    if not batch_files:
        print("ERROR: No preview batch files found.")
        return 0

    print(f"Merging {len(batch_files)} batch file(s)…")
    merged = pd.concat(
        [pd.read_csv(f) for f in batch_files], ignore_index=True
    ).drop_duplicates(subset="track_id", keep="first")

    out = data_dir / "preview_urls.csv"
    merged.to_csv(out, index=False)
    vol.commit()

    n_found = merged["preview_url"].notna().sum()
    print(f"Merged → {len(merged):,} tracks total")
    print(f"  With preview_url : {n_found:,} ({n_found/len(merged):.1%})")
    print(f"  Missing preview  : {len(merged)-n_found:,} (licensing restrictions)")
    print(f"  Saved to volume  : /data/preview_urls.csv")
    return n_found


# ── Local entrypoint ──────────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    """
    1. Read oit367_base_dataset.csv, filter to is_charted==1.
    2. Build batches of BATCH_SIZE tracks.
    3. Dispatch to Modal; merge when all complete.
    """
    import pandas as pd

    csv_path = "oit367_base_dataset.csv"
    try:
        df = pd.read_csv(
            csv_path,
            usecols=["track_id", "track_name", "artists", "is_charted"]
        )
    except FileNotFoundError:
        print(f"ERROR: '{csv_path}' not found. Run python3 run_all_v5.py first.")
        sys.exit(1)

    charted = df[df["is_charted"] == 1].copy()
    records = charted[["track_id", "track_name", "artists"]].to_dict("records")

    batches   = [records[i : i + BATCH_SIZE] for i in range(0, len(records), BATCH_SIZE)]
    n_batches = len(batches)
    est_min   = (len(records) * 0.35) / 60 / 3   # 3 workers

    print(f"Charted tracks to fetch preview URLs for : {len(records):,}")
    print(f"Batch size                               : {BATCH_SIZE}")
    print(f"Number of batches                        : {n_batches}")
    print(f"Estimated run time (3 workers)           : ~{est_min:.0f} minutes")
    print(f"\nDispatching {n_batches} batches to Modal…")
    print("(Monitor at https://modal.com/apps)\n")

    results  = list(fetch_preview_batch.starmap(
        [(batch, i) for i, batch in enumerate(batches)]
    ))
    total_found = sum(results)
    print(f"\nAll batches complete. Tracks with preview_url: {total_found:,}")

    print("\nMerging batch files…")
    n_merged = merge_preview_batches.remote()
    print(f"Merge complete: {n_merged:,} preview URLs in /data/preview_urls.csv")

    print("\n" + "=" * 60)
    print("DONE — Download with:")
    print()
    print("  modal volume get oit367-vol /data/preview_urls.csv ./preview_urls.csv")
    print()
    print("Then run Librosa extraction (Job 2):")
    print()
    print("  modal run --detach modal_librosa_extract.py")
    print("=" * 60)

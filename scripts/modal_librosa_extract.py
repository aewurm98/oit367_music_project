"""
OIT367 — Modal: Librosa Audio Feature Extraction  (Job 2 of 2)
Stanford GSB Winter 2026 | Wurm / Chen / Barli / Taruno

Downloads 30-second Spotify preview MP3s and extracts 30 acoustic features
per track using Librosa. Runs entirely in Modal's cloud — no Spotify API
credentials needed (preview URLs are plain CDN links).

──────────────────────────────────────────────────────────────
PREREQUISITES
──────────────────────────────────────────────────────────────

  Run modal_preview_urls.py first to populate /data/preview_urls.csv
  on the Modal volume, then run this script:

    modal run --detach modal_librosa_extract.py

  Monitor at: https://modal.com/apps

──────────────────────────────────────────────────────────────
FEATURES EXTRACTED (30 per track)
──────────────────────────────────────────────────────────────

  MFCCs (×13 mean + ×13 std)   — timbre / production quality signature
  Spectral centroid (mean)     — brightness; higher = brighter/thinner sound
  Spectral rolloff (mean)      — energy distribution; proxy for genre
  Chroma mean + std            — harmonic content / key stability
  Zero crossing rate (mean)    — noisiness / percussiveness
  RMS energy (mean)            — loudness envelope (finer-grained than API loudness)

  All features computed on the 30-second preview clip at 22050 Hz mono.

──────────────────────────────────────────────────────────────
DOWNLOAD & MERGE AFTER COMPLETION
──────────────────────────────────────────────────────────────

    modal volume get oit367-vol /data/librosa_features.csv ./librosa_features.csv

  The file auto-merges into run_all_v5.py output if placed in the OIT-367
  folder. Secondary analysis (Cox PH + PCA on MFCCs) can then be run.

──────────────────────────────────────────────────────────────
DESIGN NOTES
──────────────────────────────────────────────────────────────

  Parallelism  : 20 containers × 50 tracks/batch → up to 1,000 concurrent
  No API calls : pure HTTP downloads from Spotify CDN; no rate-limit risk
  MP3 decode   : librosa + ffmpeg (via apt); loaded into memory via BytesIO
  Resilience   : idempotent batch CSVs; re-run at any time to resume
  Est. runtime : ~2-3 hours for ~3,000-3,150 tracks with valid previews
  Est. cost    : < $1 USD on Modal serverless CPU pricing
"""

import modal
import sys

# ── Modal app & infrastructure ────────────────────────────────────────────────
app = modal.App("oit367-librosa")

vol = modal.Volume.from_name("oit367-vol", create_if_missing=True)

# ffmpeg is required to decode MP3 preview files.
# libsndfile1 is the soundfile backend (used for non-MP3 formats).
# librosa falls back to audioread (which wraps ffmpeg) for MP3 decoding.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "librosa>=0.10",
        "soundfile>=0.12",
        "requests>=2.31",
        "pandas>=2.0",
        "numpy>=1.24",
    )
)

VOLUME_DATA_DIR = "/data"
BATCH_SIZE      = 50     # ~3,100 valid previews → ~62 batches
N_MFCC          = 13     # number of MFCC coefficients to compute


# ── Feature extraction helper (runs inside container) ─────────────────────────

def extract_features(preview_url: str, track_id: str) -> dict | None:
    """
    Download a 30-second Spotify preview MP3 and extract 30 acoustic features.

    Parameters
    ----------
    preview_url : str  — Spotify CDN URL (e.g. https://p.scdn.co/mp3-preview/...)
    track_id    : str  — for logging only

    Returns
    -------
    dict of 30 features, or None if download/processing fails.
    """
    import io
    import requests
    import numpy as np
    import librosa

    # ── Download preview MP3 into memory ──────────────────────────────────────
    try:
        resp = requests.get(preview_url, timeout=20)
        resp.raise_for_status()
        audio_bytes = resp.content
    except Exception as exc:
        print(f"  Download failed for {track_id}: {str(exc)[:80]}")
        return None

    if len(audio_bytes) < 10_000:
        # Suspiciously small — likely an error page or empty response
        print(f"  Skipping {track_id}: response too small ({len(audio_bytes)} bytes)")
        return None

    # ── Load audio via librosa (MP3 decoded via ffmpeg / audioread) ───────────
    try:
        y, sr = librosa.load(
            io.BytesIO(audio_bytes),
            sr=22050,     # standard resampling; 22050 Hz is librosa default
            mono=True,    # collapse stereo → mono
            duration=30,  # safety cap; previews are already 30s
        )
    except Exception as exc:
        print(f"  Librosa load failed for {track_id}: {str(exc)[:80]}")
        return None

    if len(y) < sr * 5:
        # Less than 5 seconds of audio — skip
        print(f"  Skipping {track_id}: audio too short ({len(y)/sr:.1f}s)")
        return None

    # ── Extract features ──────────────────────────────────────────────────────
    try:
        mfccs      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        chroma     = librosa.feature.chroma_stft(y=y, sr=sr)
        centroid   = librosa.feature.spectral_centroid(y=y, sr=sr)
        rolloff    = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
        zcr        = librosa.feature.zero_crossing_rate(y)
        rms        = librosa.feature.rms(y=y)

        feats = {}

        # 13 MFCC means + 13 MFCC stds = 26 features
        # MFCCs capture timbre: the "color" of the sound independent of pitch/rhythm.
        # Mean captures the overall timbral character; std captures variation over time.
        for i in range(N_MFCC):
            feats[f"mfcc_{i+1:02d}_mean"] = float(np.mean(mfccs[i]))
            feats[f"mfcc_{i+1:02d}_std"]  = float(np.std(mfccs[i]))

        # Spectral centroid — weighted mean of frequencies (brightness)
        feats["spectral_centroid_mean"] = float(np.mean(centroid))

        # Spectral rolloff — frequency below which 85% of energy is contained
        feats["spectral_rolloff_mean"]  = float(np.mean(rolloff))

        # Chroma — distribution of energy across 12 pitch classes (harmony)
        feats["chroma_mean"] = float(np.mean(chroma))
        feats["chroma_std"]  = float(np.std(chroma))

        # Zero crossing rate — rate of sign changes (percussiveness/noisiness)
        feats["zcr_mean"] = float(np.mean(zcr))

        # RMS energy — root mean square amplitude (loudness envelope)
        feats["rms_mean"] = float(np.mean(rms))

        return feats   # 30 features total

    except Exception as exc:
        print(f"  Feature extraction failed for {track_id}: {str(exc)[:80]}")
        return None


# ── Remote function: process one batch ───────────────────────────────────────

@app.function(
    image=image,
    volumes={VOLUME_DATA_DIR: vol},
    timeout=600,           # 10 min per batch of 50; typical is ~2-3 min
    max_containers=20,     # 20 × 50 = 1,000 tracks processed concurrently
    retries=modal.Retries(
        max_retries=3,
        backoff_coefficient=2.0,
        initial_delay=5.0,
    ),
)
def extract_librosa_batch(batch: list, batch_id: int) -> int:
    """
    Download and process one batch of preview URLs.

    Parameters
    ----------
    batch    : list of dicts with keys 'track_id', 'preview_url',
               'track_name', 'artists'
    batch_id : int, used for output CSV naming

    Returns
    -------
    int : number of tracks successfully processed
    """
    import pandas as pd
    from pathlib import Path

    out_path = Path(VOLUME_DATA_DIR) / f"librosa_batch_{batch_id:04d}.csv"

    # Idempotent: skip if already completed
    if out_path.exists():
        existing = pd.read_csv(out_path)
        n_ok = existing["mfcc_01_mean"].notna().sum()
        print(f"[batch {batch_id:04d}] Already done ({n_ok}/{len(existing)} processed). Skipping.")
        return n_ok

    print(f"[batch {batch_id:04d}] Processing {len(batch)} tracks…")
    results = []
    n_ok    = 0

    for idx, item in enumerate(batch):
        track_id    = item["track_id"]
        preview_url = item.get("preview_url")

        row = {
            "track_id":   track_id,
            "track_name": item.get("track_name", ""),
            "artists":    item.get("artists", ""),
        }

        if not preview_url or pd.isna(preview_url):
            # No preview available — fill features with NaN
            # (these rows are excluded from analysis but tracked for coverage stats)
            pass
        else:
            feats = extract_features(preview_url, track_id)
            if feats:
                row.update(feats)
                n_ok += 1

        results.append(row)

        if (idx + 1) % 10 == 0:
            print(f"[batch {batch_id:04d}] {idx+1}/{len(batch)} done ({n_ok} successful)")

    df_out = pd.DataFrame(results)

    # Ensure all feature columns exist even for failed rows (fills with NaN)
    mfcc_cols     = [f"mfcc_{i+1:02d}_{s}" for i in range(N_MFCC) for s in ("mean","std")]
    spectral_cols = ["spectral_centroid_mean", "spectral_rolloff_mean",
                     "chroma_mean", "chroma_std", "zcr_mean", "rms_mean"]
    for col in mfcc_cols + spectral_cols:
        if col not in df_out.columns:
            df_out[col] = float("nan")

    df_out.to_csv(out_path, index=False)
    vol.commit()
    print(f"[batch {batch_id:04d}] DONE — {n_ok}/{len(batch)} tracks processed successfully.")
    return n_ok


# ── Remote function: merge all batch CSVs ────────────────────────────────────

@app.function(
    image=image,
    volumes={VOLUME_DATA_DIR: vol},
    timeout=300,
)
def merge_librosa_batches() -> int:
    """Concatenate librosa_batch_*.csv → librosa_features.csv on the volume."""
    import pandas as pd
    from pathlib import Path

    vol.reload()
    data_dir    = Path(VOLUME_DATA_DIR)
    batch_files = sorted(data_dir.glob("librosa_batch_*.csv"))

    if not batch_files:
        print("ERROR: No librosa batch files found on volume.")
        return 0

    print(f"Merging {len(batch_files)} batch file(s)…")
    merged = pd.concat(
        [pd.read_csv(f) for f in batch_files], ignore_index=True
    ).drop_duplicates(subset="track_id", keep="first")

    out = data_dir / "librosa_features.csv"
    merged.to_csv(out, index=False)
    vol.commit()

    mfcc_col    = "mfcc_01_mean"
    n_processed = merged[mfcc_col].notna().sum() if mfcc_col in merged.columns else 0
    print(f"Merged → {len(merged):,} total tracks")
    print(f"  Successfully processed : {n_processed:,} ({n_processed/len(merged):.1%})")
    print(f"  Missing preview/failed : {len(merged)-n_processed:,}")
    print(f"  Features per track     : 30 (13 MFCC means + 13 stds + 4 spectral)")
    print(f"  Saved to volume        : /data/librosa_features.csv")
    return n_processed


# ── Local entrypoint ──────────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    """
    1. Read preview_urls.csv from the Modal volume (written by modal_preview_urls.py).
    2. Filter to rows with a non-null preview_url.
    3. Batch and dispatch to 20 parallel containers.
    4. Merge all batch CSVs into librosa_features.csv.
    """
    import pandas as pd
    import tempfile, subprocess, os

    # ── Pull preview_urls.csv from the volume to read it locally ─────────────
    print("Reading preview_urls.csv from Modal volume…")
    try:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name

        result = subprocess.run(
            ["modal", "volume", "get", "oit367-vol",
             "/data/preview_urls.csv", tmp_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print("ERROR: Could not read preview_urls.csv from volume.")
            print("Make sure modal_preview_urls.py has completed first.")
            print(result.stderr)
            sys.exit(1)

        preview_df = pd.read_csv(tmp_path)
        os.unlink(tmp_path)

    except FileNotFoundError:
        print("ERROR: 'modal' CLI not found. Install with: pip3 install modal")
        sys.exit(1)

    print(f"preview_urls.csv: {len(preview_df):,} total tracks")
    valid = preview_df[preview_df["preview_url"].notna()].copy()
    print(f"  With valid preview_url : {len(valid):,}")
    print(f"  Missing preview_url    : {len(preview_df)-len(valid):,} (skipped)")

    if len(valid) == 0:
        print("ERROR: No tracks with preview URLs. Run modal_preview_urls.py first.")
        sys.exit(1)

    records   = valid.to_dict("records")
    batches   = [records[i : i + BATCH_SIZE] for i in range(0, len(records), BATCH_SIZE)]
    n_batches = len(batches)
    # Estimate: ~3s per track (download + librosa) / 20 parallel workers
    est_min   = (len(records) * 3) / 60 / 20

    print(f"\nBatch size             : {BATCH_SIZE}")
    print(f"Number of batches      : {n_batches}")
    print(f"Max parallel containers: 20")
    print(f"Estimated run time     : ~{est_min:.0f}–{est_min*2:.0f} minutes")
    print(f"\nDispatching {n_batches} batches to Modal…")
    print("(Monitor at https://modal.com/apps)\n")

    results      = list(extract_librosa_batch.starmap(
        [(batch, i) for i, batch in enumerate(batches)]
    ))
    total_ok = sum(results)
    print(f"\nAll batches complete. Successfully processed: {total_ok:,} tracks")

    print("\nMerging batch files…")
    n_merged = merge_librosa_batches.remote()
    print(f"Merge complete: {n_merged:,} tracks with Librosa features")

    print("\n" + "=" * 60)
    print("DONE — Download results with:")
    print()
    print("  modal volume get oit367-vol /data/librosa_features.csv ./librosa_features.csv")
    print()
    print("Feature columns (30 per track):")
    print("  mfcc_01_mean … mfcc_13_mean  (13 MFCC means — timbre)")
    print("  mfcc_01_std  … mfcc_13_std   (13 MFCC stds  — timbral variation)")
    print("  spectral_centroid_mean        (brightness)")
    print("  spectral_rolloff_mean         (energy distribution)")
    print("  chroma_mean, chroma_std       (harmonic content)")
    print("  zcr_mean                      (percussiveness)")
    print("  rms_mean                      (loudness envelope)")
    print()
    print("Recommended secondary analysis in run_all_v5.py:")
    print("  1. PCA on 26 MFCC features → 3–5 components")
    print("  2. Add PCA components to Cox PH + Log-OLS longevity models")
    print("  3. K-means clustering on MFCCs to find timbral archetypes")
    print("=" * 60)

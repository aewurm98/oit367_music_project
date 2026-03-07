# Librosa Audio Feature Extraction — Modal Feasibility Plan
**OIT367 | Stanford GSB Winter 2026 | Wurm / Chen / Barli / Taruno**
**Status: VIABLE for secondary analysis; target completion within 24–36 hours**

---

## Summary Verdict

Running Librosa on the **charted subset (3,502 tracks)** via Modal is technically
feasible and can be completed in under 24 hours. It requires two sequential Modal
jobs: one to fetch Spotify preview URLs, one to download + process audio.

The full non-charted dataset (86,239 tracks) is not recommended within your timeline —
audio download alone would take 6–12 hours and ~85 GB of storage.

**Recommended scope: charted tracks only.** Librosa features would serve as a
secondary analysis section in your report (e.g., "Do chart-successful songs share
structural audio signatures beyond Spotify's own metrics?").

---

## What Librosa Adds vs. What Spotify Already Provides

| Feature | Spotify API | Librosa |
|---------|-------------|---------|
| Tempo / BPM | ✓ (audio feature) | ✓ (more precise) |
| Key / Mode | ✓ | ✓ (via chroma) |
| Danceability | ✓ (proprietary) | ✗ (no direct analog) |
| **MFCCs (timbre)** | ✗ | ✓ (13–20 coefficients) |
| **Spectral centroid** | ✗ | ✓ (brightness) |
| **Spectral rolloff** | ✗ | ✓ (energy distribution) |
| **Zero crossing rate** | ✗ | ✓ (noisiness/percussiveness) |
| **Chroma features** | ✗ | ✓ (harmonic content) |
| RMS energy | ≈ loudness | ✓ (finer-grained) |

**Primary research value:** MFCCs capture *timbre* — the "sound quality" or
production style of a track that Spotify's features do not measure. The hypothesis:
charted tracks may share timbral signatures (e.g., compressed, radio-ready production)
that predict chart success independent of rhythm/harmony features.

---

## Data Source: Spotify Preview URLs

Spotify provides a **30-second preview clip** for most tracks via the API:
```
sp.track(track_id) → {"preview_url": "https://p.scdn.co/mp3-preview/..."}
```

- **Coverage:** ~85–90% of tracks have a non-null preview_url
  (some tracks lack previews due to licensing restrictions)
- **Format:** MP3, 128 kbps, 30 seconds → ~480 KB per file
- **Total data:** 3,502 × 480 KB ≈ **1.7 GB**
- **API calls needed:** 3,502 individual `sp.track()` calls
- **Spotify rate limit risk:** LOW — 3,502 calls at 0.35s each = ~20 minutes,
  well within Development Mode quota for a fresh rate-limit window

---

## Pipeline Architecture (Two Modal Jobs)

### Job 1: `modal_preview_urls.py` (runs on your laptop, ~20 min)

Fetches preview_url for each of the 3,502 charted track_ids.

```
Input : oit367_base_dataset.csv (filter is_charted==1)
Output: /data/preview_urls.csv  [track_id, preview_url]
Volume: oit367-vol
```

Can be run locally (not Modal) or as a simple Modal function.
Shares the same Spotify credential secret as the artist scraper.

### Job 2: `modal_librosa_extract.py` (~2–4 hours on Modal)

For each track with a valid preview_url:
1. Download the MP3 into container memory (no disk persistence needed)
2. Load into Librosa via `io.BytesIO` (avoid temp file writes)
3. Extract features
4. Write results to volume in batches

```
Input : preview_urls.csv from the volume
Output: /data/librosa_features.csv
Parallelism: 10–20 containers (audio processing is CPU-bound, no GPU needed)
Batch size : 50 tracks per container
```

**Key Modal settings for Job 2:**
```python
@app.function(
    image=image,          # includes librosa, requests, numpy
    volumes={"/data": vol},
    timeout=600,          # 10 min per batch of 50
    max_containers=20,    # ~700 concurrent tracks processed
    retries=modal.Retries(max_retries=3, backoff_coefficient=2.0, initial_delay=5.0),
)
```

---

## Features Extracted (per track)

```python
import librosa, numpy as np, io, requests

def extract_features(preview_url: str) -> dict:
    audio_bytes = requests.get(preview_url, timeout=15).content
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, mono=True)

    mfccs      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma     = librosa.feature.chroma_stft(y=y, sr=sr)
    centroid   = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff    = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr        = librosa.feature.zero_crossing_rate(y)
    rms        = librosa.feature.rms(y=y)

    return {
        **{f"mfcc_{i+1}_mean": mfccs[i].mean() for i in range(13)},
        **{f"mfcc_{i+1}_std":  mfccs[i].std()  for i in range(13)},
        "chroma_mean":    chroma.mean(),
        "chroma_std":     chroma.std(),
        "spectral_centroid_mean": centroid.mean(),
        "spectral_rolloff_mean":  rolloff.mean(),
        "zcr_mean":               zcr.mean(),
        "rms_mean":               rms.mean(),
    }
```

**Total features per track: 30** (13 MFCC means + 13 MFCC stds + 4 spectral)

---

## Timeline Estimate

| Step | Time | Notes |
|------|------|-------|
| Job 1: Fetch preview URLs | ~25 min | 3,502 sp.track() calls at 0.35s |
| Job 2: Download + process | ~2–3 hours | 20 containers × 50 tracks |
| Merge + download | ~5 min | modal volume get |
| Add to run_all_v5.py | ~15 min | auto-detect, same pattern as artist_features |
| **Total** | **~3–4 hours** | After Spotify rate limit resets |

---

## Technical Dependencies

Add to Modal image for Job 2:
```python
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "ffmpeg")   # required for librosa MP3 support
    .pip_install("librosa>=0.10", "soundfile>=0.12",
                 "requests>=2.31", "pandas>=2.0", "numpy>=1.24")
)
```

**Note:** `ffmpeg` is needed to decode MP3 preview files. `soundfile` alone cannot
read MP3. The apt_install step adds ~120 MB to the container image but is cached
after the first build.

---

## Known Limitations for Your Report

1. **30-second clips only:** Librosa features are computed on the preview excerpt,
   not the full track. Intros and outros may differ from the 30s middle section.
   This is a standard limitation in music information retrieval (MIR) research
   and should be acknowledged as a robustness caveat.

2. **~10–15% missing previews:** Some tracks, particularly older or regionally
   restricted songs, lack preview_url. Your charted sample shrinks to ~3,000–3,150
   tracks with audio data.

3. **Charted-only scope:** Without Librosa features for non-charted tracks,
   you cannot add these features to the main classification model. Use them as:
   - Descriptive statistics: "What do charted tracks sound like?"
   - Cluster analysis: timbral clusters among charted tracks
   - Within-charted longevity prediction: do timbral features predict weeks on chart?

4. **MFCCs are high-dimensional:** 13 MFCC means + 13 stds = 26 correlated features.
   Run PCA to reduce to 3–5 components before adding to any regression. Report
   variance explained.

---

## Recommended Report Framing

> **Section 4.3 — Secondary Analysis: Audio Timbre and Chart Longevity**
>
> "To complement Spotify's proprietary audio features, we extracted 30 acoustic
> features from 30-second preview clips using Librosa (McFee et al., 2015) for
> the 3,502 charted tracks. MFCC principal components were used to characterize
> timbral similarity among charted tracks. We tested whether timbral features
> predicted weeks on chart (Cox PH + Log-OLS), controlling for the audio features
> used in our primary models."

---

## Action Items to Execute This Plan

1. **Wait for Spotify rate limit to reset** (~8 hours from this morning's run)
   Confirm reset by running: `modal run modal_charted_scrape.py` and checking
   that the first batch progresses normally within 30 seconds.

2. **Run artist scraper first** (2 batches, ~3 min):
   ```bash
   modal app stop oit367-spotify
   modal run --detach modal_charted_scrape.py
   ```

3. **After artist scraper completes, run preview URL fetcher** (Job 1):
   Script: `modal_preview_urls.py` (to be written; ~50 lines)

4. **Run Librosa extractor** (Job 2):
   Script: `modal_librosa_extract.py` (to be written; ~150 lines)
   ```bash
   modal run --detach modal_librosa_extract.py
   ```

5. **Download and merge into run_all_v5.py:**
   ```bash
   modal volume get oit367-vol /data/librosa_features.csv ./librosa_features.csv
   ```
   Add auto-detection block in v5 (same pattern as artist_features.csv merge).

---

## Build These Scripts When Ready

Ask Claude to write `modal_preview_urls.py` and `modal_librosa_extract.py`
when the Spotify rate limit has reset and the artist scraper has completed.
The architecture described above is ready to implement.

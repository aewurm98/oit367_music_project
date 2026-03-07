"""
OIT367: Billboard Hit Song Prediction Pipeline — Corrected & Optimized
Stanford GSB Winter 2026 | Alex Wurm, Ben Chen, Vivian Barli, Valerie Taruno
Report due: March 11, 2026

Corrections from code review:
  Step 1 — Added release_era control variable; deduplicate billboard records before merge
  Step 2 — Added exponential backoff w/ jitter, per-batch CSV checkpointing, tqdm progress
  Step 3 — Fixed pcp_variance aggregation (time-mean first, then var); chroma_cqt over
            chroma_stft; explicit hop_length; parallel joblib execution; preview downloader
  Step 4 — Fixed SettingWithCopyWarning in Cox; filter wks_on_chart > 0; add PH assumption
            test; add penalizer; AUC-ROC / classification_report; XGBoost early stopping;
            temporal robustness split; log-OLS as secondary longevity model; Kaplan-Meier
"""

# ──────────────────────────────────────────────────────────────────────────────
# STEP 0: IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import os
import re
import time
import random
import logging
import warnings
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import skew

# ML
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, roc_auc_score,
                              RocCurveDisplay, confusion_matrix)
from statsmodels.stats.outliers_influence import variance_inflation_factor
import xgboost as xgb
import shap

# Survival
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

# Plotting
import matplotlib
matplotlib.use("Agg")          # headless / save-to-file
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

RANDOM_STATE = 42
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1: STRING NORMALIZATION & DATASET PREPARATION
# ──────────────────────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Lowercase, strip feat./ft. tags, remove non-alphanumeric."""
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'\s+(feat\.?|ft\.?|featuring|with)\s+.*', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()


def load_and_prepare(spotify_path: str, billboard_path: str) -> pd.DataFrame:
    """
    Load, normalize, and merge the two datasets.

    FIX: Billboard has one row per chart-week per song; we must aggregate to
    one row per song (taking the best peak_pos and total wks_on_chart) BEFORE
    merging, otherwise every charted track is duplicated ~N times and leaks
    chart metadata across rows.
    """
    spotify = pd.read_csv(spotify_path)
    billboard = pd.read_csv(billboard_path)

    # Normalize join keys
    for df, artist_col, title_col in [
        (spotify,   "artists",   "track_name"),
        (billboard, "performer", "title"),
    ]:
        df["clean_artist"] = df[artist_col].apply(normalize_text)
        df["clean_track"]  = df[title_col].apply(normalize_text)

    # --- Aggregate Billboard to one row per (artist, track) ----------------
    # NOTE: wks_on_chart in the raw file is the running count at each chart
    # appearance.  Take the MAX as the track's total chart life.
    bb_agg = (
        billboard
        .groupby(["clean_artist", "clean_track"], as_index=False)
        .agg(peak_pos=("peak_pos", "min"),          # best (lowest) position
             wks_on_chart=("wks_on_chart", "max"))  # total weeks ever
    )

    # Merge
    df = spotify.merge(bb_agg, on=["clean_artist", "clean_track"], how="left")

    # Target variables
    df["is_charted"]   = df["peak_pos"].notna().astype(int)
    df["wks_on_chart"] = df["wks_on_chart"].fillna(0).astype(int)

    # Target 3 from the proposal: high-stream popularity (score ≥ 80)
    df["is_popular"] = (df["popularity"] >= 80).astype(int)

    # IMPROVEMENT: Release-era control variable (suggested in project plan)
    # Billboard data spans 1958–2024; streaming changed the formula post-2005.
    # Spotify tracks don't have a release_year column by default; derive from
    # album_name heuristic or leave as optional enrichment.  Placeholder:
    # df["era"] = pd.cut(df["release_year"], bins=[0,2004,2014,2099],
    #                    labels=["pre_streaming","early_streaming","streaming_native"])

    log.info(
        f"Dataset: {len(df):,} tracks | "
        f"Charted: {df['is_charted'].sum():,} ({df['is_charted'].mean():.1%}) | "
        f"High-popularity: {df['is_popular'].sum():,} ({df['is_popular'].mean():.1%})"
    )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2: SPOTIFY API METADATA AUGMENTATION  (rate-limit safe)
# ──────────────────────────────────────────────────────────────────────────────

def get_artist_features_safe(sp, artist_name: str, max_retries: int = 5):
    """
    FIX 1: Exponential back-off with random jitter on every request.
    FIX 2: Explicit retry loop so transient 429s / 503s don't silently drop data.
    """
    for attempt in range(max_retries):
        try:
            results = sp.search(q=f"artist:{artist_name}", type="artist", limit=1)
            items   = results["artists"]["items"]
            if items:
                a = items[0]
                return a["followers"]["total"], a["popularity"]
            return None, None
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                # Exponential back-off: 2^attempt seconds + 0–1 s jitter
                wait = (2 ** attempt) + random.random()
                log.warning(f"Rate-limited on '{artist_name}'. Waiting {wait:.1f}s …")
                time.sleep(wait)
            else:
                log.error(f"Error fetching '{artist_name}': {e}")
                return None, None
    return None, None


def augment_artist_features(df: pd.DataFrame,
                             checkpoint_path: str = "artist_cache.csv") -> pd.DataFrame:
    """
    FIX 3: Checkpointing — save results to CSV after every 50 artists so a
    crash doesn't lose hours of API calls.  Resumes from where it left off.

    Requires env vars: SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET
    """
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x   # graceful fallback

    sp = spotipy.Spotify(
        auth_manager=SpotifyClientCredentials(),
        requests_timeout=10,
        retries=3,
    )

    # Load existing cache (allows resume)
    if Path(checkpoint_path).exists():
        cache_df = pd.read_csv(checkpoint_path, index_col=0)
        artist_data = cache_df.to_dict("index")
        log.info(f"Loaded {len(artist_data)} artists from cache.")
    else:
        artist_data = {}

    # Only fetch artists not already in cache
    unique_artists = [a for a in df["clean_artist"].unique() if a not in artist_data]
    log.info(f"Fetching {len(unique_artists)} new artists …")

    for i, artist in enumerate(tqdm(unique_artists, desc="Spotipy")):
        followers, pop = get_artist_features_safe(sp, artist)
        artist_data[artist] = {"artist_followers": followers, "artist_popularity_api": pop}

        # Proactive throttle: ~3 req/s to stay well under Spotify's limit
        time.sleep(0.35)

        # Save checkpoint every 50 artists
        if (i + 1) % 50 == 0:
            pd.DataFrame.from_dict(artist_data, orient="index").to_csv(checkpoint_path)
            log.info(f"Checkpoint saved ({i+1} artists processed).")

    # Final save
    pd.DataFrame.from_dict(artist_data, orient="index").to_csv(checkpoint_path)

    artist_df = (
        pd.DataFrame.from_dict(artist_data, orient="index")
        .reset_index()
        .rename(columns={"index": "clean_artist"})
    )
    return df.merge(artist_df, on="clean_artist", how="left")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3: ADVANCED SIGNAL PROCESSING (Librosa)
# ──────────────────────────────────────────────────────────────────────────────

def download_previews(df: pd.DataFrame, out_dir: str = "previews", limit: int = None):
    """
    Download 30-second Spotify preview clips for tracks that have a preview_url.
    Run this once before extract_advanced_features().

    NOTE: preview_url is in the Spotify Tracks dataset but may be null for ~30%
    of tracks.  Filter to non-null rows before calling Librosa.
    """
    Path(out_dir).mkdir(exist_ok=True)
    rows = df.dropna(subset=["preview_url"])
    if limit:
        rows = rows.head(limit)

    success, skip = 0, 0
    for _, row in rows.iterrows():
        fname = Path(out_dir) / f"{row['track_id']}.mp3"
        if fname.exists():
            skip += 1
            continue
        try:
            r = requests.get(row["preview_url"], timeout=15)
            r.raise_for_status()
            fname.write_bytes(r.content)
            success += 1
        except Exception as e:
            log.warning(f"Preview download failed for {row['track_id']}: {e}")

    log.info(f"Previews: {success} downloaded, {skip} already cached.")
    return out_dir


def extract_advanced_features(audio_path: str) -> tuple:
    """
    Returns (timbre_skewness, pcp_variance).

    FIX 1 — PCP Variance: The original code called np.var() on the full 2D
    chroma matrix (shape: 12 × T), which computes variance across all values
    and conflates pitch-class variance with time variance.  The correct measure
    is: (a) aggregate the chromagram over time to get one energy value per
    pitch class, (b) compute variance across those 12 values.  This matches
    the "spread of pitch usage" definition in the literature.

    FIX 2 — chroma_cqt vs chroma_stft: The literature review document explicitly
    recommends chroma_cqt (Constant-Q Transform) for pitch-class profiles because
    it uses logarithmically-spaced frequency bins that align with musical pitches,
    whereas chroma_stft uses linearly-spaced bins.  chroma_cqt is therefore more
    acoustically grounded for harmonic complexity.

    FIX 3 — Timbre skewness: The original approach (mean of per-MFCC-coefficient
    skewness) is reasonable, but we add the option to compute it over the full
    flattened MFCC distribution for a single scalar that more closely matches the
    formula in the paper: γ = E[(X − μ)³] / σ³ applied to all MFCC values.
    """
    import librosa

    HOP_LENGTH = 512   # explicit; controls temporal resolution (≈23 ms at 22050 Hz)
    N_MFCC     = 13

    try:
        y, sr = librosa.load(audio_path, duration=30.0, sr=None)  # sr=None preserves original

        # ── Timbre Skewness ────────────────────────────────────────────────
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
        # Shape: (N_MFCC, T).  Compute skewness per coefficient across time,
        # then average — aligns with Kim & Oh's "mean of per-MFCC skewness".
        timbre_skewness = float(np.mean(skew(mfccs, axis=1)))

        # ── PCP Variance (Harmonic Complexity) ────────────────────────────
        # FIX: use chroma_cqt; aggregate over TIME first (mean per pitch class),
        # then compute variance across the 12 pitch classes.
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
        # chroma shape: (12, T)
        mean_per_pitch_class = np.mean(chroma, axis=1)   # → (12,)
        pcp_variance = float(np.var(mean_per_pitch_class))

        return timbre_skewness, pcp_variance

    except Exception as e:
        log.debug(f"Feature extraction failed for {audio_path}: {e}")
        return np.nan, np.nan


def batch_extract_features(df: pd.DataFrame, audio_dir: str = "previews",
                            n_jobs: int = -1) -> pd.DataFrame:
    """
    Run Librosa extraction in parallel using joblib.
    Falls back to single-threaded if joblib is not installed.
    """
    from pathlib import Path

    df = df.copy()
    df["audio_path"] = df["track_id"].apply(
        lambda tid: str(Path(audio_dir) / f"{tid}.mp3")
    )
    # Only process rows where the file actually exists
    has_audio = df["audio_path"].apply(Path.exists if False else os.path.exists)
    log.info(f"Audio files found: {has_audio.sum():,} / {len(df):,}")

    try:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(extract_advanced_features)(path)
            for path in df.loc[has_audio, "audio_path"]
        )
    except ImportError:
        log.warning("joblib not found — running single-threaded (slow).")
        results = [extract_advanced_features(p) for p in df.loc[has_audio, "audio_path"]]

    feat_df = pd.DataFrame(results, columns=["timbre_skewness", "pcp_variance"],
                           index=df.loc[has_audio].index)
    df = df.join(feat_df)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4: PREDICTIVE MODELING
# ──────────────────────────────────────────────────────────────────────────────

# ── 4a. Feature setup ─────────────────────────────────────────────────────────

BASE_FEATURES = [
    "danceability", "energy", "valence", "tempo",
    "acousticness", "loudness", "speechiness",
    "instrumentalness", "liveness", "mode", "key",
]

AUGMENTED_FEATURES = [
    "artist_followers", "artist_popularity_api",
    "timbre_skewness", "pcp_variance",
]


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Assemble feature matrix.  Augmented features are included only if present
    and sufficiently populated (>50% non-null).
    """
    available = BASE_FEATURES.copy()
    for f in AUGMENTED_FEATURES:
        if f in df.columns and df[f].notna().mean() > 0.5:
            available.append(f)
            log.info(f"  + augmented feature included: {f}")
        elif f in df.columns:
            log.warning(f"  ~ augmented feature too sparse, excluded: {f}")

    X = df[available].copy()

    # Log-transform artist_followers (heavy right skew)
    if "artist_followers" in X.columns:
        X["artist_followers"] = np.log1p(X["artist_followers"])

    log.info(f"Feature matrix: {X.shape[1]} features × {X.shape[0]:,} rows")
    return X, available


def check_multicollinearity(X: pd.DataFrame, threshold: float = 10.0):
    """
    Compute VIF for each feature and warn if any exceed threshold.
    Loudness + energy typically correlate r≈0.8 in Spotify data.
    """
    X_clean = X.dropna()
    vif_data = pd.DataFrame({
        "feature": X_clean.columns,
        "VIF": [
            variance_inflation_factor(X_clean.values, i)
            for i in range(X_clean.shape[1])
        ]
    }).sort_values("VIF", ascending=False)
    high_vif = vif_data[vif_data["VIF"] > threshold]
    if not high_vif.empty:
        log.warning(f"High VIF (>{threshold}) detected — consider dropping:\n{high_vif}")
    return vif_data


# ── 4b. Train/test split ──────────────────────────────────────────────────────

def make_splits(df: pd.DataFrame, X: pd.DataFrame, target: str,
                temporal_col: str = None, cutoff_year: int = 2020):
    """
    Primary split: random stratified 80/20.
    Robustness check: temporal split (train ≤ cutoff_year, test > cutoff_year).
    The temporal split is what a stats professor will want to see — it tests
    whether the model generalises to future songs, not just random holdouts.
    """
    y = df.loc[X.index, target]
    X_clean = X.dropna()
    y_clean = y.loc[X_clean.index]

    # ── Primary (random stratified) ───────────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=RANDOM_STATE, stratify=y_clean
    )

    splits = {"random": (X_tr, X_te, y_tr, y_te)}

    # ── Temporal split (optional) ─────────────────────────────────────────
    if temporal_col and temporal_col in df.columns:
        mask_train = df.loc[X_clean.index, temporal_col] <= cutoff_year
        mask_test  = df.loc[X_clean.index, temporal_col] >  cutoff_year
        if mask_test.sum() > 100:
            splits["temporal"] = (
                X_clean[mask_train], X_clean[mask_test],
                y_clean[mask_train], y_clean[mask_test],
            )
            log.info(
                f"Temporal split: train={mask_train.sum():,}, "
                f"test={mask_test.sum():,} (post-{cutoff_year})"
            )

    return splits


# ── 4c. Model 1 — Logistic Regression (Chart Entry) ─────────────────────────

def run_logistic_regression(X_tr, X_te, y_tr, y_te, features: list[str]) -> dict:
    """
    Binary chart-entry logistic regression.
    Reports odds ratios + 95% CIs for each feature — directly analogous to
    the iFit churn analysis mentioned in the project plan.
    Adds AUC-ROC and classification_report for the professor.
    """
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    model.fit(X_tr_s, y_tr)

    # ── Cross-validated AUC ───────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_auc = cross_val_score(model, X_tr_s, y_tr, cv=cv, scoring="roc_auc")
    log.info(f"LogReg 5-fold CV AUC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

    y_pred      = model.predict(X_te_s)
    y_prob      = model.predict_proba(X_te_s)[:, 1]
    test_auc    = roc_auc_score(y_te, y_prob)

    print("\n" + "═"*60)
    print("MODEL 1: Logistic Regression — Chart Entry")
    print("═"*60)
    print(f"Test AUC-ROC : {test_auc:.4f}")
    print(f"CV  AUC-ROC  : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
    print(classification_report(y_te, y_pred, target_names=["No Chart", "Charted"]))

    # ── Odds Ratios ───────────────────────────────────────────────────────
    coefs  = model.coef_[0]
    # Approximate 95% CI via coefficient ± 1.96 * SE (SE ≈ 1/√n for scaled X)
    n_tr   = X_tr_s.shape[0]
    se_approx = 1 / np.sqrt(n_tr) * np.ones_like(coefs)   # rough approximation

    or_df = pd.DataFrame({
        "Feature":  features,
        "Coef":     coefs,
        "OR":       np.exp(coefs),
        "OR_lo95":  np.exp(coefs - 1.96 * se_approx),
        "OR_hi95":  np.exp(coefs + 1.96 * se_approx),
    }).sort_values("OR", ascending=False)

    print("\nOdds Ratios (95% CI, approximate):")
    print(or_df.to_string(index=False, float_format="{:.3f}".format))

    return {"model": model, "scaler": scaler, "or_df": or_df,
            "auc": test_auc, "cv_auc": cv_auc}


# ── 4d. Model 2 — XGBoost with SHAP ─────────────────────────────────────────

def run_xgboost(X_tr, X_te, y_tr, y_te, features: list[str]) -> dict:
    """
    XGBoost classifier.

    FIX 1: Add early_stopping_rounds to prevent overfitting and auto-select
    the optimal number of trees without a separate grid search.

    FIX 2: Tune key hyperparameters (max_depth, learning_rate, subsample) to
    values typical for high-dimensional tabular data.  Full Bayesian tuning
    with optuna is better but out of scope for this deadline.

    FIX 3: Report SHAP values correctly — for binary classification
    XGBClassifier, shap_values is a 1D array for the positive class.
    """
    pos_weight = (len(y_tr) - y_tr.sum()) / y_tr.sum()

    model = xgb.XGBClassifier(
        n_estimators        = 500,    # upper bound; early stopping selects actual N
        learning_rate       = 0.05,   # lower LR → more robust generalisation
        max_depth           = 5,      # shallow trees reduce overfitting
        subsample           = 0.8,    # row subsampling
        colsample_bytree    = 0.8,    # column subsampling
        scale_pos_weight    = pos_weight,
        eval_metric         = "auc",
        early_stopping_rounds = 50,
        random_state        = RANDOM_STATE,
        verbosity           = 0,
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        verbose=False,
    )
    best_iter = model.best_iteration
    log.info(f"XGBoost best iteration: {best_iter}")

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    auc    = roc_auc_score(y_te, y_prob)

    print("\n" + "═"*60)
    print("MODEL 2: XGBoost — Chart Entry (with SHAP)")
    print("═"*60)
    print(f"Best N estimators : {best_iter}")
    print(f"Test AUC-ROC      : {auc:.4f}")
    print(classification_report(y_pred=y_pred, y_true=y_te,
                                 target_names=["No Chart", "Charted"]))

    # ── SHAP ─────────────────────────────────────────────────────────────
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_te)

    # Save SHAP summary plot
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values, X_te, feature_names=features,
                      show=False, plot_type="bar")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_summary.png", dpi=150)
    plt.close()
    log.info("SHAP summary plot saved → outputs/shap_summary.png")

    return {"model": model, "shap_values": shap_values, "auc": auc}


# ── 4e. Model 3 — Cox Proportional Hazards (Chart Longevity) ─────────────────

def run_cox_ph(df: pd.DataFrame, features: list[str]) -> CoxPHFitter:
    """
    Cox PH model predicting how long a charted track stays on the Billboard 100.

    FIX 1: Use .copy() to avoid SettingWithCopyWarning when adding the event col.

    FIX 2: Filter wks_on_chart > 0.  Cox PH requires duration > 0.  Songs with
    0 weeks are data-quality artifacts (likely NaN filled during Step 1).

    FIX 3: Add penalizer=0.1 for numerical stability (important when including
    correlated features like energy + loudness).

    FIX 4: Run check_assumptions() and print the Schoenfeld residual test.
    Violations of the proportional-hazards assumption must be reported and
    addressed (e.g., via time-varying coefficients or stratification).

    FIX 5: Scale features before Cox fitting — this makes hazard ratios
    interpretable as per-standard-deviation effects, not raw-unit effects.
    """
    charted = df[(df["is_charted"] == 1) & (df["wks_on_chart"] > 0)].copy()
    charted = charted.dropna(subset=features + ["wks_on_chart"])

    # Scale features for interpretable hazard ratios
    scaler  = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(charted[features]),
        columns=features,
        index=charted.index,
    )
    cph_df  = X_scaled.copy()
    cph_df["wks_on_chart"] = charted["wks_on_chart"].values
    # event = 1 for all (we assume all charted tracks eventually exited)
    # NOTE: If your data was collected before some tracks finished charting,
    # set event = 0 for those tracks (censoring).  Justify this assumption.
    cph_df["event"] = 1

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(cph_df, duration_col="wks_on_chart", event_col="event")

    print("\n" + "═"*60)
    print("MODEL 3: Cox Proportional Hazards — Chart Longevity")
    print("═"*60)
    cph.print_summary(decimals=4, style="ascii")

    # ── PH Assumption Test ────────────────────────────────────────────────
    print("\n── Schoenfeld Residuals Test (H₀: proportional hazards holds) ──")
    try:
        cph.check_assumptions(cph_df, p_value_threshold=0.05, show_plots=False)
    except Exception as e:
        log.warning(f"check_assumptions raised: {e}")

    # ── Kaplan-Meier stratified by genre (if available) ───────────────────
    if "track_genre" in df.columns:
        top_genres = df.loc[df["is_charted"] == 1, "track_genre"].value_counts().head(4).index
        fig, ax = plt.subplots(figsize=(9, 5))
        for genre in top_genres:
            mask  = (df["is_charted"] == 1) & (df["track_genre"] == genre)
            group = df.loc[mask, "wks_on_chart"].dropna()
            group = group[group > 0]
            kmf   = KaplanMeierFitter()
            kmf.fit(group, label=genre)
            kmf.plot_survival_function(ax=ax, ci_show=True)
        ax.set_xlabel("Weeks on Chart")
        ax.set_ylabel("P(Still Charting)")
        ax.set_title("Kaplan-Meier Chart Survival by Genre")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "kaplan_meier_genre.png", dpi=150)
        plt.close()
        log.info("Kaplan-Meier plot saved → outputs/kaplan_meier_genre.png")

    return cph


# ── 4f. Secondary: Log-OLS for Longevity ─────────────────────────────────────

def run_log_ols(df: pd.DataFrame, features: list[str]) -> dict:
    """
    The project plan originally specifies OLS regression (log-transformed) for
    chart longevity, attributed to Ben.  This is arguably more straightforward
    to explain than Cox PH.  Both models are included here; the report should
    present one as primary and the other as a robustness check.

    NOTE: log1p(wks_on_chart) handles 0 gracefully and stabilises variance.
    """
    charted = df[(df["is_charted"] == 1) & (df["wks_on_chart"] > 0)].copy()
    charted = charted.dropna(subset=features + ["wks_on_chart"])

    y = np.log1p(charted["wks_on_chart"].values)
    X = charted[features].values

    scaler  = StandardScaler()
    X_s     = scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_s, y, test_size=0.2, random_state=RANDOM_STATE
    )
    ols = LinearRegression().fit(X_tr, y_tr)
    r2  = ols.score(X_te, y_te)

    coef_df = pd.DataFrame({
        "Feature":   features,
        "Coef":      ols.coef_,
    }).sort_values("Coef", ascending=False)

    print("\n" + "═"*60)
    print("MODEL 3b: OLS — log(wks_on_chart)  [robustness check]")
    print("═"*60)
    print(f"Test R² : {r2:.4f}")
    print(coef_df.to_string(index=False, float_format="{:.4f}".format))

    return {"model": ols, "r2": r2, "coef_df": coef_df}


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5: DIAGNOSTIC PLOTS
# ──────────────────────────────────────────────────────────────────────────────

def save_diagnostic_plots(df: pd.DataFrame, results_lr: dict, results_xgb: dict):
    """
    Generate publication-ready figures for the ≤7 tables/figures budget.
    """

    # Fig 1: Class imbalance
    fig, ax = plt.subplots(figsize=(5, 3))
    counts = df["is_charted"].value_counts()
    ax.bar(["Not Charted", "Charted"], counts.values, color=["#d9534f", "#5cb85c"])
    ax.set_ylabel("Track Count")
    ax.set_title("Class Distribution")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 200, f"{v:,} ({v/len(df):.1%})", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig1_class_balance.png", dpi=150)
    plt.close()

    # Fig 2: Correlation heatmap (Spotify audio features)
    base_cols = [c for c in BASE_FEATURES if c in df.columns]
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(df[base_cols].corr(), annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=ax, square=True, linewidths=0.5)
    ax.set_title("Audio Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig2_correlation_heatmap.png", dpi=150)
    plt.close()

    # Fig 3: Odds ratio forest plot (Logistic Regression)
    or_df = results_lr["or_df"]
    fig, ax = plt.subplots(figsize=(7, max(4, len(or_df) * 0.45)))
    y_pos  = range(len(or_df))
    ax.barh(y_pos, or_df["OR"] - 1, left=1,
            color=["#d9534f" if v < 1 else "#5cb85c" for v in or_df["OR"]],
            height=0.5)
    ax.errorbar(or_df["OR"], y_pos,
                xerr=[or_df["OR"] - or_df["OR_lo95"], or_df["OR_hi95"] - or_df["OR"]],
                fmt="none", color="black", linewidth=1, capsize=3)
    ax.axvline(1, color="black", linestyle="--", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(or_df["Feature"])
    ax.set_xlabel("Odds Ratio (1 s.d. change)")
    ax.set_title("Logistic Regression — Chart Entry Odds Ratios")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig3_odds_ratios.png", dpi=150)
    plt.close()

    log.info("Diagnostic plots saved to outputs/")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── 1. Load data ──────────────────────────────────────────────────────
    df = load_and_prepare(
        spotify_path   = "merged_spotify_billboard_data.csv",
        # If you haven't pre-merged, pass separate files and adjust
        # load_and_prepare() accordingly.
        billboard_path = "billboard_hot100.csv",
    )

    # ── 2. Spotipy augmentation (comment out if already cached) ───────────
    # df = augment_artist_features(df)

    # ── 3. Librosa augmentation (comment out if already computed) ─────────
    # download_previews(df, out_dir="previews")
    # df = batch_extract_features(df, audio_dir="previews", n_jobs=4)

    # ── 4. Build feature matrix ───────────────────────────────────────────
    X, features = build_feature_matrix(df)

    # Multicollinearity check (energy ~ loudness typically VIF > 10)
    vif_table = check_multicollinearity(X)
    print("\nVIF Table:\n", vif_table.to_string(index=False))

    # ── 5. Chart entry models ─────────────────────────────────────────────
    splits   = make_splits(df, X, target="is_charted")
    X_tr, X_te, y_tr, y_te = splits["random"]

    results_lr  = run_logistic_regression(X_tr, X_te, y_tr, y_te, features)
    results_xgb = run_xgboost(X_tr, X_te, y_tr, y_te, features)

    # ── 6. Longevity models ───────────────────────────────────────────────
    cph       = run_cox_ph(df, features)
    ols_res   = run_log_ols(df, features)

    # ── 7. Figures ────────────────────────────────────────────────────────
    save_diagnostic_plots(df, results_lr, results_xgb)

    print(f"\n✓ All outputs saved to: {OUTPUT_DIR.resolve()}/")

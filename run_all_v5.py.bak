"""
OIT367 — Full Pipeline Runner  v5  (control variables release)
Stanford GSB Winter 2026 | Wurm / Chen / Barli / Taruno

Changes from v4:
  Fix A — Removed 'energy' from BASE_FEATURES (VIF=15.07, collinear with loudness)
  Fix B — Cox PH now stratifies on 'mode' (was failing Schoenfeld test as covariate)
  Fix C — PR-AUC added to Logistic Regression and XGBoost outputs
  Fix D — model_performance_summary.csv includes pr_auc column
  Fix E — Spotipy block properly disabled (was running unconditionally in v3)

  v5 Add A — 'explicit' added to BASE_FEATURES (binary; charted tracks 11.5% vs 8.5%)
  v5 Add B — 'duration_min' added to BASE_FEATURES (track length in minutes, capped 10 min)
  v5 Add C — 'decade' ordinal added to Cox PH + Log-OLS longevity models
              (captures streaming-era vs. traditional-radio-era effects;
               only available for charted tracks which have chart_entry_date)
  v5 Add D — Precision-Recall curve figure added (Fig 9) alongside ROC
  v5 Add E — Genre-stratified chart rate summary table saved to outputs/
  v5 Fix F — Removed 'danceability' from BASE_FEATURES (VIF=12.41 with duration_min added).
              Danceability is the collinearity hub in v5: it correlates simultaneously
              with tempo, valence, and duration_min, inflating VIF for both itself
              (12.41) and tempo (11.50). Removing danceability drops max VIF.
              Effect is likely captured by the retained correlated features.
              SHAP from the v5 pre-patch run: danceability Mean|SHAP|=0.224 (8th of 12).

  v5 Add H — 'lastfm_listeners_log' added (Last.fm cumulative listeners, log-scaled)
              Cross-platform artist reach; orthogonal to Spotify-derived artist_popularity_api.
              Source: artists.csv (pieca111/music-artists-popularity on Kaggle).
              Coverage: ~65.8% of artists in Spotify dataset.
  v5 Add I — 'is_us_artist' binary added (1=United States, 0=non-US, NaN=unknown)
              Source: country_mb from Last.fm/MusicBrainz dataset.
              Captures whether domestic (US) vs. international artists chart differently.
              ~31% of tracks have confirmed US artists; NaN filled with 0 in classification.
  v5 Add J — Lyric sentiment features added to longevity models (Cox PH + Log-OLS):
              sentiment_compound, sentiment_pos, sentiment_neg, lyric_word_count
              Source: billboard_lyrics.csv (rhaamrozenberg on Kaggle) + VADER.
              Coverage: ~41.8% of charted tracks (901/3,502). NaN rows excluded from
              longevity regressions. Provides textual channel independent of audio valence.

  v5 Fix G — Removed 'tempo' from BASE_FEATURES (VIF=10.65 after Fix F).
              After removing danceability, tempo becomes the next VIF hub (10.65),
              correlating with duration_min (6.62) through genre/era effects
              (e.g., dance/disco tracks tend to be mid-length AND have specific BPM ranges;
               hip-hop tracks skew slower AND shorter). Removing tempo drops max VIF
              to 6.47 (loudness). Tempo's rhythmic signal is partially captured by
              loudness and speechiness; decade_idx captures the BPM drift over eras.
              SHAP from prior runs: tempo was 10th of 12 (low marginal value for LR).

SETUP (run once):
    pip3 install -r requirements.txt
    # Important: use xgboost==2.1.3 (xgboost 3.x breaks shap TreeExplainer)

Run from inside your OIT-367 folder:
    python3 run_all_v5.py

For local artist augmentation (no API required — immediate):
    python3 build_artist_features.py   ← generates artist_features.csv
    python3 run_all_v5.py              ← auto-detects and merges it

For Spotipy augmentation (requires Spotify API quota reset — ~24h wait):
    modal run --detach modal_charted_scrape.py
    modal volume get oit367-vol /data/charted_artist_features.csv ./artist_features.csv
"""

# ── Preflight: verify all required packages ───────────────────────────────────
import importlib.util
import sys

REQUIRED = {
    "sklearn":     "scikit-learn>=1.3",
    "xgboost":     "xgboost>=2.0",
    "shap":        "shap>=0.44",
    "lifelines":   "lifelines>=0.27",
    "statsmodels": "statsmodels>=0.14",
    "seaborn":     "seaborn>=0.13",
    "matplotlib":  "matplotlib>=3.7",
    "pandas":      "pandas>=2.0",
    "numpy":       "numpy>=1.24",
}
missing = [pip for mod, pip in REQUIRED.items() if importlib.util.find_spec(mod) is None]
if missing:
    print("=" * 60)
    print("ERROR: Missing required packages. Run:\n")
    print(f"  pip3 install {' '.join(missing)}\n")
    print("Or install everything: pip3 install -r requirements.txt")
    print("=" * 60)
    sys.exit(1)

# ── Standard imports ──────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import warnings, re
warnings.filterwarnings("ignore")
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
)
from statsmodels.stats.outliers_influence import variance_inflation_factor
import xgboost as xgb
import shap
from lifelines import CoxPHFitter, KaplanMeierFitter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA PREP  (skipped if oit367_base_dataset.csv already exists)
# ─────────────────────────────────────────────────────────────────────────────
BASE_CSV = Path("oit367_base_dataset.csv")

if not BASE_CSV.exists():
    print("Building base dataset from raw files...")

    def normalize_text(text):
        if pd.isna(text): return ""
        text = str(text).lower().strip()
        text = re.sub(r'\s+(feat\.?|ft\.?|featuring|with)\s+.*', '', text)
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.strip()

    for fname in ["spotify_tracksdataset.csv", "merged_spotify_billboard_data.csv"]:
        if not Path(fname).exists():
            print(f"ERROR: Required file not found: {fname}")
            sys.exit(1)

    spotify   = pd.read_csv("spotify_tracksdataset.csv")
    bb_weekly = pd.read_csv("merged_spotify_billboard_data.csv")

    spotify_dedup = (
        spotify
        .sort_values("track_genre")
        .drop_duplicates(subset="track_id", keep="first")
        .reset_index(drop=True)
    )

    bb_weekly["chart_week"] = pd.to_datetime(bb_weekly["chart_week"], errors="coerce")
    bb_agg = (
        bb_weekly
        .groupby("track_id", as_index=False)
        .agg(
            peak_pos        =("peak_pos",    "min"),
            wks_on_chart    =("wks_on_chart","max"),
            chart_entry_date=("chart_week",  "min"),
        )
    )

    df = spotify_dedup.merge(bb_agg, on="track_id", how="left")
    df["is_charted"]   = df["peak_pos"].notna().astype(int)
    df["wks_on_chart"] = df["wks_on_chart"].fillna(0).astype(int)
    df["is_popular"]   = (df["popularity"] >= 80).astype(int)
    df.to_csv(BASE_CSV, index=False)
    print(f"  Saved {len(df):,} rows → {BASE_CSV}")

df = pd.read_csv(BASE_CSV)
print(f"\nDataset: {len(df):,} tracks  |  "
      f"Charted: {df['is_charted'].sum():,} ({df['is_charted'].mean():.2%})  |  "
      f"Not charted: {(df['is_charted']==0).sum():,} ({(df['is_charted']==0).mean():.2%})")

# ─────────────────────────────────────────────────────────────────────────────
# v5 Add A/B: Derived control columns
# ─────────────────────────────────────────────────────────────────────────────

# explicit: convert bool → int (0/1) so it works as a numeric feature
df["explicit"] = df["explicit"].astype(int)

# duration_min: track length in minutes, capped at 10 min to remove outliers
# (max raw value is ~87 min — clearly audio books / podcast episodes)
# StandardScaler handles the scale; capping reduces leverage of extreme outliers.
df["duration_min"] = (df["duration_ms"] / 60_000).clip(upper=10.0)

print(f"\nv5 control variable check:")
print(f"  explicit  — charted mean: {df[df['is_charted']==1]['explicit'].mean():.3f}, "
      f"non-charted mean: {df[df['is_charted']==0]['explicit'].mean():.3f}")
print(f"  duration_min — charted mean: {df[df['is_charted']==1]['duration_min'].mean():.2f} min, "
      f"non-charted mean: {df[df['is_charted']==0]['duration_min'].mean():.2f} min")

# ─────────────────────────────────────────────────────────────────────────────
# SPOTIPY AUGMENTATION — merge artist_features.csv if available
# Generate artist_features.csv by running: modal run modal_charted_scrape.py
# then: modal volume get oit367-vol /data/charted_artist_features.csv ./artist_features.csv
# ─────────────────────────────────────────────────────────────────────────────
artist_features_path = Path("artist_features.csv")
if artist_features_path.exists():
    print("\nFound artist_features.csv — merging Spotipy augmentation…")
    artist_df = pd.read_csv(artist_features_path)
    df = df.merge(artist_df, on="artists", how="left")
    # log-transform follower count only when present (Spotipy API path)
    if "artist_followers" in df.columns:
        df["artist_followers"] = np.log1p(df["artist_followers"].fillna(0))
    # is_us_artist: conservative pre-fill — unknown nationality treated as non-US (0)
    # This ensures the column passes the >50% non-null threshold in the feature loop below.
    # Rationale: the model treats "we don't know their country" the same as "non-US",
    # which is a deliberate conservative assumption documented in v5 Add I.
    if "is_us_artist" in df.columns:
        df["is_us_artist"] = df["is_us_artist"].fillna(0).astype(int)
    df.to_csv("oit367_augmented_dataset.csv", index=False)
    print("  Saved oit367_augmented_dataset.csv with artist features.")

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SETUP
# ─────────────────────────────────────────────────────────────────────────────
# v5 adds 'explicit' and 'duration_min' as control variables.
#
# Design decisions:
#   explicit     — binary predictor; radio/platform play-listing decisions
#                  directly depend on explicit content rating.
#   duration_min — track length; streaming-era playlists favor shorter songs;
#                  longer tracks may indicate classical/ambient/non-chart genres.
#   Decade       — NOT in BASE_FEATURES because chart_entry_date is only
#                  available for charted tracks; would create NaN for 96% of rows
#                  in the classification model. Added only to Cox PH / Log-OLS.
#   Genre dummies — The Kaggle dataset has 114 genres, each with ~1,000 tracks
#                  (engineered sample). Uniform distribution means genre dummies
#                  add noise to the classification model; effect is already
#                  captured in audio features (tempo, danceability, etc.).
#                  Genre is analysed in Fig 7 (KM) and the chart-rate table.
BASE_FEATURES = [
    # Audio features (from v4, minus three VIF-flagged removals):
    #   energy       removed in Fix A (v4): VIF=15.07, collinear with loudness
    #   danceability removed in Fix F (v5): VIF=12.41, collinear hub
    #   tempo        removed in Fix G (v5): VIF=10.65, secondary hub post Fix F;
    #                correlates with duration_min + loudness through genre/era effects
    "valence",
    "acousticness", "loudness", "speechiness",
    "instrumentalness", "liveness", "mode", "key",
    # v5 control variables:
    "explicit",      # v5 Add A: binary (0=clean, 1=explicit)
    "duration_min",  # v5 Add B: track length in minutes, capped at 10
]
FEATURES = BASE_FEATURES.copy()
# Auto-detect artist augmentation columns from either source:
#   modal_charted_scrape.py  → artist_followers, artist_popularity_api (Spotify API)
#   build_artist_features.py → artist_popularity_api, artist_peak_popularity,
#                              artist_track_count (computed locally, no API)
for col in [
    "artist_followers",        # Spotipy API path (modal_charted_scrape.py)
    "artist_popularity_api",   # both paths
    "artist_peak_popularity",  # local build path (build_artist_features.py)
    "artist_track_count",      # local build path (build_artist_features.py)
    "lastfm_listeners_log",    # Last.fm cross-platform reach (build_artist_features.py)
    "is_us_artist",            # Last.fm artist nationality binary (build_artist_features.py)
]:
    if col in df.columns and df[col].notna().mean() > 0.5:
        FEATURES.append(col)
        print(f"  + Augmented feature included: {col}")

X       = df[FEATURES].copy()
# Fill NaN in artist features (median imputation for classification model)
# (~3% of tracks whose artist name could not be matched in artist_features.csv)
for col in ["artist_popularity_api", "artist_peak_popularity", "artist_track_count",
            "artist_followers", "lastfm_listeners_log", "is_us_artist"]:
    if col in X.columns and X[col].isna().any():
        X[col] = X[col].fillna(X[col].median())

y_chart = df["is_charted"]

# ─────────────────────────────────────────────────────────────────────────────
# v5 Add E: Genre chart-rate summary table
# ─────────────────────────────────────────────────────────────────────────────
genre_summary = (
    df.groupby("track_genre")
    .agg(
        total_tracks  =("is_charted", "count"),
        charted_tracks=("is_charted", "sum"),
        chart_rate    =("is_charted", "mean"),
        avg_popularity=("popularity", "mean"),
    )
    .sort_values("chart_rate", ascending=False)
    .round(4)
    .reset_index()
)
genre_summary.to_csv(OUT / "genre_chart_rates.csv", index=False)
print(f"\nTop 10 genres by chart rate:")
print(genre_summary.head(10).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# VIF CHECK
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60 + "\nVIF CHECK\n" + "="*60)
vif = pd.DataFrame({
    "Feature": FEATURES,
    "VIF":     [variance_inflation_factor(X.values, i) for i in range(len(FEATURES))]
}).sort_values("VIF", ascending=False).round(2)
print(vif.to_string(index=False))
high_vif = vif[vif["VIF"] > 10]
if not high_vif.empty:
    print(f"\n  ⚠  High VIF (>10): {high_vif['Feature'].tolist()}")
else:
    print(f"\n  ✓  All features VIF ≤ 10 — no problematic multicollinearity")
vif.to_csv(OUT / "vif_table.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / TEST SPLIT  (stratified 80/20)
# ─────────────────────────────────────────────────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y_chart, test_size=0.2, random_state=RANDOM_STATE, stratify=y_chart)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)
print(f"\nSplit — Train: {len(X_tr):,}  |  Test: {len(X_te):,}")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1: LOGISTIC REGRESSION  (Chart Entry — Vivian)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60 + "\nMODEL 1: Logistic Regression — Chart Entry\n" + "="*60)
lr     = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
lr.fit(X_tr_s, y_tr)
cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_auc = cross_val_score(lr, X_tr_s, y_tr, cv=cv, scoring="roc_auc")
y_prob_lr = lr.predict_proba(X_te_s)[:, 1]
y_pred_lr = lr.predict(X_te_s)
auc_lr    = roc_auc_score(y_te, y_prob_lr)
pr_auc_lr = average_precision_score(y_te, y_prob_lr)
print(f"Test AUC-ROC : {auc_lr:.4f}")
print(f"Test PR-AUC  : {pr_auc_lr:.4f}  (random baseline = {y_te.mean():.4f})")
print(f"CV   AUC-ROC : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}  (5-fold)")
print(classification_report(y_te, y_pred_lr, target_names=["No Chart", "Charted"]))
coefs = lr.coef_[0]
or_df = pd.DataFrame({
    "Feature": FEATURES,
    "Coef":    coefs.round(4),
    "OR":      np.exp(coefs).round(4),
}).sort_values("OR", ascending=False)
print("Odds Ratios (per 1 SD change):")
print(or_df.to_string(index=False))
or_df.to_csv(OUT / "logistic_odds_ratios.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2: XGBOOST + SHAP  (Chart Entry — Alex)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60 + "\nMODEL 2: XGBoost — Chart Entry\n" + "="*60)
pos_weight = (len(y_tr) - y_tr.sum()) / y_tr.sum()
xgb_m = xgb.XGBClassifier(
    n_estimators          = 500,
    learning_rate         = 0.05,
    max_depth             = 5,
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    scale_pos_weight      = pos_weight,
    eval_metric           = "auc",
    early_stopping_rounds = 50,
    random_state          = RANDOM_STATE,
    verbosity             = 0,
)
xgb_m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
y_prob_xgb = xgb_m.predict_proba(X_te)[:, 1]
y_pred_xgb = xgb_m.predict(X_te)
auc_xgb    = roc_auc_score(y_te, y_prob_xgb)
pr_auc_xgb = average_precision_score(y_te, y_prob_xgb)
print(f"Best N estimators : {xgb_m.best_iteration}")
print(f"Test AUC-ROC      : {auc_xgb:.4f}")
print(f"Test PR-AUC       : {pr_auc_xgb:.4f}  (random baseline = {y_te.mean():.4f})")
print(classification_report(y_te, y_pred_xgb, target_names=["No Chart", "Charted"]))
explainer = shap.TreeExplainer(xgb_m)
shap_vals = explainer.shap_values(X_te)
fi = pd.DataFrame({
    "Feature":   FEATURES,
    "Mean_SHAP": np.abs(shap_vals).mean(axis=0),
}).sort_values("Mean_SHAP", ascending=False).round(6)
print("SHAP Feature Importance:")
print(fi.to_string(index=False))
fi.to_csv(OUT / "xgboost_shap_importance.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# LYRIC FEATURES — merge sentiment scores for longevity models (charted only)
# Generated by: python3 build_lyric_features.py
# Scope: ~41.8% of charted tracks (901/3,502) have matched lyrics
# Used in: Cox PH + Log-OLS only (charted subset)
# ─────────────────────────────────────────────────────────────────────────────
lyric_features_path = Path("lyric_features.csv")
LYRIC_FEATURES = []
if lyric_features_path.exists():
    print("\nFound lyric_features.csv — merging sentiment for longevity models…")
    lyric_df = pd.read_csv(lyric_features_path)
    lyric_cols = ["sentiment_compound", "sentiment_pos", "sentiment_neg", "lyric_word_count"]
    available_lyric_cols = [c for c in lyric_cols if c in lyric_df.columns]
    # Merge on track_id (most reliable key)
    df = df.merge(
        lyric_df[["track_id"] + available_lyric_cols],
        on="track_id",
        how="left",
    )
    matched = df.loc[df["is_charted"] == 1, "sentiment_compound"].notna().sum()
    print(f"  Lyric sentiment matched: {matched} charted tracks "
          f"({matched/df['is_charted'].sum()*100:.1f}%)")
    # Only include in longevity models if >20% of charted tracks are covered
    for col in available_lyric_cols:
        coverage = df.loc[df["is_charted"] == 1, col].notna().mean()
        if coverage > 0.20:
            LYRIC_FEATURES.append(col)
            print(f"  + Lyric feature included in longevity models: {col} "
                  f"(coverage: {coverage:.1%})")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3: COX PROPORTIONAL HAZARDS  (Longevity — Ben / Vivian)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60 + "\nMODEL 3: Cox PH — Chart Longevity\n" + "="*60)

# v5 Add C: Build decade ordinal for charted tracks only.
# chart_entry_date is available for all is_charted==1 rows.
# decade_idx: 0=1950s, 1=1960s, ..., 7=2020s (ordinal, captures era trend)
df["chart_entry_date"] = pd.to_datetime(df["chart_entry_date"], errors="coerce")
df["decade_idx"] = (
    (df["chart_entry_date"].dt.year.clip(lower=1950, upper=2029) - 1950) // 10
).astype("Int64")  # nullable int so non-charted tracks stay NaN

# Build Cox dataset (charted tracks only)
COX_FEATURES = FEATURES + ["decade_idx"] + LYRIC_FEATURES   # decade + lyrics only meaningful for charted
charted = (
    df[(df["is_charted"] == 1) & (df["wks_on_chart"] > 0)]
    .dropna(subset=COX_FEATURES + ["wks_on_chart"])
    .copy()
)
charted["decade_idx"] = charted["decade_idx"].astype(float)

cox_sc = StandardScaler()
X_cox  = pd.DataFrame(
    cox_sc.fit_transform(charted[COX_FEATURES]),
    columns=COX_FEATURES, index=charted.index,
)
cph_df = X_cox.copy()
cph_df["wks_on_chart"] = charted["wks_on_chart"].values
cph_df["event"]        = 1

print(f"  Charted tracks in Cox sample: {len(cph_df):,}")
print(f"  Decade distribution:\n{charted['decade_idx'].value_counts().sort_index().to_string()}")
print("  Note: 'mode' is a stratum (Fix B); 'decade_idx' is a new covariate (v5 Add C)")

cph = CoxPHFitter(penalizer=0.1)
cph.fit(
    cph_df,
    duration_col="wks_on_chart",
    event_col="event",
    strata=["mode"],
)
cph.print_summary(decimals=4, style="ascii")
print(f"\nConcordance Index (C-stat): {cph.concordance_index_:.4f}")
cph.summary.to_csv(OUT / "cox_summary.csv")
print("\n── Schoenfeld Residuals Test (H₀: proportional hazards holds) ──")
print("  (mode is a stratum and excluded from this test)")
try:
    cph.check_assumptions(cph_df, p_value_threshold=0.05, show_plots=False)
except Exception as e:
    print(f"  check_assumptions: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3b: LOG-OLS  (Longevity robustness — Ben)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60 + "\nMODEL 3b: Log-OLS — Longevity (robustness check)\n" + "="*60)
y_log  = np.log1p(charted["wks_on_chart"].values)
ols_sc = StandardScaler()
X_ols  = ols_sc.fit_transform(charted[COX_FEATURES])
Xo_tr, Xo_te, yo_tr, yo_te = train_test_split(
    X_ols, y_log, test_size=0.2, random_state=RANDOM_STATE)
ols    = LinearRegression().fit(Xo_tr, yo_tr)
r2     = ols.score(Xo_te, yo_te)
ols_coef = pd.DataFrame({
    "Feature": COX_FEATURES, "Coef": ols.coef_
}).sort_values("Coef", ascending=False)
print(f"Test R²: {r2:.4f}")
print(ols_coef.round(4).to_string(index=False))
ols_coef.to_csv(OUT / "ols_longevity_coefficients.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURES  (9 publication-ready plots; Fig 9 is new in v5)
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating 9 figures...")
sns.set_style("whitegrid")
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})

# Fig 1: Class balance
fig, ax = plt.subplots(figsize=(5, 3.5))
counts  = y_chart.value_counts().sort_index()
bars    = ax.bar(["Not Charted", "Charted"], counts.values,
                 color=["#c0392b", "#27ae60"], width=0.5, edgecolor="white")
for bar, v in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, v + 600,
            f"{v:,}\n({v/len(df):.1%})", ha="center", va="bottom",
            fontsize=9, fontweight="bold")
ax.set_ylabel("Track Count")
ax.set_title("Class Distribution: Billboard Chart Entry", fontweight="bold")
ax.set_ylim(0, max(counts) * 1.15)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(OUT / "fig1_class_balance.png", dpi=150, bbox_inches="tight")
plt.close()

# Fig 2: Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df[FEATURES].corr(), annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, ax=ax, square=True, linewidths=0.5,
            vmin=-1, vmax=1, cbar_kws={"shrink": 0.8})
ax.set_title("Audio Feature Correlation Matrix (v5: +explicit, +duration_min)",
             fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(OUT / "fig2_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

# Fig 3: ROC curves (LR vs XGBoost)
fig, ax = plt.subplots(figsize=(6, 5))
for name, probs, color in [
    ("Logistic Reg (baseline)", y_prob_lr,  "#2980b9"),
    ("XGBoost",                 y_prob_xgb, "#e67e22"),
]:
    fpr, tpr, _ = roc_curve(y_te, probs)
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f"{name}  (AUC={roc_auc_score(y_te, probs):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random  (AUC=0.500)")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — Chart Entry Prediction", fontweight="bold")
ax.legend(loc="lower right")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(OUT / "fig3_roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# Fig 4: Odds ratio forest plot
or_sorted = or_df.sort_values("OR").reset_index(drop=True)
fig, ax   = plt.subplots(figsize=(7, max(4.5, len(or_sorted) * 0.5)))
colors    = ["#c0392b" if v < 1 else "#27ae60" for v in or_sorted["OR"]]
ax.barh(range(len(or_sorted)), or_sorted["OR"] - 1, left=1,
        color=colors, height=0.55, alpha=0.85)
ax.axvline(1, color="black", lw=1, linestyle="--")
ax.set_yticks(range(len(or_sorted)))
ax.set_yticklabels(or_sorted["Feature"])
ax.set_xlabel("Odds Ratio  (per 1 SD increase)")
ax.set_title("Logistic Regression — Chart Entry Odds Ratios\n"
             "(standardized features; per 1 SD change)", fontweight="bold")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(OUT / "fig4_odds_ratios.png", dpi=150, bbox_inches="tight")
plt.close()

# Fig 5: SHAP bar
fig, ax = plt.subplots(figsize=(7, 5))
shap.summary_plot(shap_vals, X_te, feature_names=FEATURES,
                  show=False, plot_type="bar", color="#2980b9")
plt.title("XGBoost Feature Importance (Mean |SHAP|)", fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "fig5_shap_importance.png", dpi=150, bbox_inches="tight")
plt.close()

# Fig 6: Cox hazard ratios
fig, ax = plt.subplots(figsize=(7, max(5, len(COX_FEATURES) * 0.45)))
cph.plot(ax=ax)
ax.set_title("Cox PH — Hazard Ratios (per 1 SD)\n"
             "Chart Longevity on Billboard Hot 100\n"
             "(stratified on mode; decade_idx added in v5)", fontweight="bold")
ax.axvline(0, color="black", lw=0.8, linestyle="--")
plt.tight_layout()
plt.savefig(OUT / "fig6_cox_hazard_ratios.png", dpi=150, bbox_inches="tight")
plt.close()

# Fig 7: Kaplan-Meier by genre (top 5 genres among charted tracks)
top_genres = (
    df[df["is_charted"] == 1]["track_genre"]
    .value_counts().head(5).index.tolist()
)
palette = ["#2980b9", "#e67e22", "#27ae60", "#8e44ad", "#c0392b"]
fig, ax = plt.subplots(figsize=(9, 5.5))
for genre, color in zip(top_genres, palette):
    times = df.loc[
        (df["is_charted"] == 1) & (df["track_genre"] == genre),
        "wks_on_chart",
    ].dropna()
    times = times[times > 0]
    if len(times) < 5:
        continue
    KaplanMeierFitter().fit(
        times, label=f"{genre}  (n={len(times)})"
    ).plot_survival_function(ax=ax, ci_show=True, color=color)
ax.set_xlabel("Weeks on Billboard Hot 100")
ax.set_ylabel("P(Still Charting)")
ax.set_title("Kaplan-Meier Survival Curves by Genre", fontweight="bold")
ax.legend(loc="upper right", fontsize=8)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(OUT / "fig7_kaplan_meier.png", dpi=150, bbox_inches="tight")
plt.close()

# Fig 8: Longevity distribution
fig, ax = plt.subplots(figsize=(7, 4))
wks = df.loc[df["is_charted"] == 1, "wks_on_chart"]
ax.hist(wks, bins=40, color="#2980b9", edgecolor="white", alpha=0.85)
ax.axvline(wks.median(), color="#e67e22", lw=2, linestyle="--",
           label=f"Median = {wks.median():.0f} wks")
ax.axvline(wks.mean(),   color="#c0392b", lw=2, linestyle=":",
           label=f"Mean = {wks.mean():.1f} wks")
ax.set_xlabel("Weeks on Chart"); ax.set_ylabel("Track Count")
ax.set_title("Distribution of Chart Longevity  (Charted Tracks Only)", fontweight="bold")
ax.legend()
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(OUT / "fig8_longevity_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Fig 9: Precision-Recall curves (v5 Add D) ─────────────────────────────────
# At 3.9% positive rate, the PR curve is far more informative than ROC.
# The PR random baseline is simply the positive rate (= 0.039).
# A model with lift over random should stay above this baseline across recall.
fig, ax = plt.subplots(figsize=(7, 5))
for name, probs, color in [
    ("Logistic Reg (baseline)", y_prob_lr,  "#2980b9"),
    ("XGBoost",                 y_prob_xgb, "#e67e22"),
]:
    prec, rec, _ = precision_recall_curve(y_te, probs)
    ap = average_precision_score(y_te, probs)
    ax.plot(rec, prec, color=color, lw=2, label=f"{name}  (AP={ap:.3f})")
ax.axhline(y_te.mean(), color="gray", lw=1.2, linestyle="--",
           label=f"Random baseline  (AP={y_te.mean():.3f})")
ax.set_xlabel("Recall (Sensitivity)")
ax.set_ylabel("Precision (PPV)")
ax.set_title("Precision-Recall Curves — Chart Entry Prediction\n"
             "(note: 3.9% positive rate; AP = area under PR curve)", fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(OUT / "fig9_precision_recall.png", dpi=150, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
summary = pd.DataFrame({
    "Model":     ["Logistic Regression", "XGBoost", "Cox PH", "Log-OLS"],
    "Task":      ["Chart Entry (binary)", "Chart Entry (binary)",
                  "Longevity (survival)", "Longevity (OLS)"],
    "Metric":    ["AUC-ROC", "AUC-ROC", "C-statistic", "R²"],
    "Score":     [round(auc_lr, 4), round(auc_xgb, 4),
                  round(cph.concordance_index_, 4), round(r2, 4)],
    "PR_AUC":    [round(pr_auc_lr, 4), round(pr_auc_xgb, 4), None, None],
    "CV / note": [
        f"{cv_auc.mean():.4f} ± {cv_auc.std():.4f}  (5-fold)",
        f"early stop @ iter {xgb_m.best_iteration}",
        "penalizer=0.1; mode stratified; +decade_idx +sentiment",
        "outcome=log1p(wks); +decade_idx +sentiment",
    ],
})
print("\n" + "="*60 + "\nMODEL PERFORMANCE SUMMARY\n" + "="*60)
print(summary.to_string(index=False))
summary.to_csv(OUT / "model_performance_summary.csv", index=False)
print(f"\n{'='*60}")
print(f"All outputs → {OUT.resolve()}/")
print(f"  CSVs   : vif_table, logistic_odds_ratios, xgboost_shap_importance,")
print(f"           cox_summary, ols_longevity_coefficients, model_performance_summary,")
print(f"           genre_chart_rates  (new in v5)")
print(f"  Figures: fig1–fig9  (PNG, 150 dpi; fig9=Precision-Recall is new in v5)")
print(f"{'='*60}")
print(f"\nv5 changes applied:")
print(f"  ✓ Fix A–E from v4 (VIF, Cox stratification, PR-AUC, etc.)")
print(f"  ✓ v5 Add A — 'explicit' added to BASE_FEATURES")
print(f"  ✓ v5 Add B — 'duration_min' added to BASE_FEATURES (capped at 10 min)")
print(f"  ✓ v5 Add C — 'decade_idx' added to Cox PH + Log-OLS longevity models")
print(f"  ✓ v5 Add D — Precision-Recall curve saved as fig9_precision_recall.png")
print(f"  ✓ v5 Add E — genre_chart_rates.csv saved to outputs/")
print(f"  ✓ v5 Fix F — 'danceability' removed from BASE_FEATURES (VIF=12.41)")
print(f"  ✓ v5 Fix G — 'tempo' removed from BASE_FEATURES (VIF=10.65 after Fix F)")
print(f"  ✓ v5 Add H — 'lastfm_listeners_log' added (Last.fm listeners, log-scaled; 65.8% coverage)")
print(f"  ✓ v5 Add I — 'is_us_artist' binary added (MusicBrainz country; NaN→0 conservative fill)")
print(f"  ✓ v5 Add J — lyric sentiment added to longevity models (VADER; 41.9% charted coverage)")
print(f"\nFor Spotipy augmentation (charted artists only, ~3 min):")
print(f"  modal app stop oit367-spotify          # cancel the stuck full scraper")
print(f"  modal run --detach modal_charted_scrape.py")
print(f"  modal volume get oit367-vol /data/charted_artist_features.csv ./artist_features.csv")
print(f"  python3 run_all_v5.py")

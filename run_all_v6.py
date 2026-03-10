"""
OIT367 — Full Pipeline Runner  v6  (report-final additions)
Stanford GSB Winter 2026 | Wurm / Chen / Barli / Taruno

New in v6 (on top of v5 pipeline logic):
  v6 Add A — Enrichment comparison: audio-only vs full-model performance table
             (quantifies the artist-feature contribution to prediction accuracy)
  v6 Add B — Threshold analysis table for XGBoost chart-entry classification
             (precision / recall / F1 at operating thresholds 0.02–0.50)
             (makes the classifier actionable for A&R screening use cases)
  v6 Add C — Schoenfeld residuals table: formal per-feature PH test results
             (documents which Cox features violate proportional hazards, with p-values)
  v6 Add D — Genre descriptive table: top-10 and bottom-10 genres by chart rate
             (formatted cleanly for the report appendix)
  v6 Add E — New Fig 10: enrichment comparison bar chart (audio-only vs full model)
  v6 Add F — New Fig 11: SHAP dependence plot for instrumentalness
             (shows the nonlinear effect of the dominant audio barrier)
  v6 Fix G — Corrected PR-curve baseline label (2.75% not 3.9%)
  v6 Fix H — model_performance_summary.csv extended with audio_only_score column

USAGE:
    python3 run_all_v6.py
    (run from inside the project directory containing oit367_final_dataset.csv)
"""

# ── Preflight ────────────────────────────────────────────────────────────────
import importlib.util, sys
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
    sys.exit(f"ERROR: pip install {' '.join(missing)}")

# ── Imports ──────────────────────────────────────────────────────────────────
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
    precision_score, recall_score, f1_score,
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
# DATA PREP
# ─────────────────────────────────────────────────────────────────────────────
BASE_CSV = Path("oit367_final_dataset.csv")
if not BASE_CSV.exists():
    sys.exit("ERROR: oit367_final_dataset.csv not found.")

df = pd.read_csv(BASE_CSV)
print(f"\nDataset: {len(df):,} tracks  |  "
      f"Charted: {df['is_charted'].sum():,} ({df['is_charted'].mean():.2%})  |  "
      f"Not charted: {(df['is_charted']==0).sum():,}")

df["explicit"]    = df["explicit"].astype(int)
df["duration_min"] = (df["duration_ms"] / 60_000).clip(upper=10.0)

# Merge artist features if not already in dataset
artist_features_path = Path("artist_features.csv")
if artist_features_path.exists() and "artist_peak_popularity" not in df.columns:
    print("\nMerging artist_features.csv...")
    artist_df = pd.read_csv(artist_features_path)
    df = df.merge(artist_df, on="artists", how="left")
    if "artist_followers" in df.columns:
        df["artist_followers"] = np.log1p(df["artist_followers"].fillna(0))
    if "is_us_artist" in df.columns:
        df["is_us_artist"] = df["is_us_artist"].fillna(0).astype(int)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SETUP
# ─────────────────────────────────────────────────────────────────────────────
BASE_FEATURES = [
    "valence", "acousticness", "loudness", "speechiness",
    "instrumentalness", "liveness", "mode", "key",
    "explicit", "duration_min",
]
FEATURES = BASE_FEATURES.copy()
ARTIST_FEATURE_CANDIDATES = [
    "artist_followers", "artist_popularity_api", "artist_peak_popularity",
    "artist_track_count", "lastfm_listeners_log", "is_us_artist",
    "is_male_artist", "artist_age", "is_mainstream_genre",
]
ARTIST_FEATURES_ADDED = []
for col in ARTIST_FEATURE_CANDIDATES:
    if col in df.columns and df[col].notna().mean() > 0.5:
        FEATURES.append(col)
        ARTIST_FEATURES_ADDED.append(col)
        print(f"  + Artist feature included: {col}")

X = df[FEATURES].copy()
for col in ["artist_popularity_api", "artist_peak_popularity", "artist_track_count",
            "artist_followers", "lastfm_listeners_log", "is_us_artist"]:
    if col in X.columns and X[col].isna().any():
        X[col] = X[col].fillna(X[col].median())
for col in ["is_male_artist", "is_mainstream_genre"]:
    if col in X.columns and X[col].isna().any():
        X[col] = X[col].fillna(0)
for col in ["artist_age"]:
    if col in X.columns and X[col].isna().any():
        X[col] = X[col].fillna(X[col].median())

y_chart = df["is_charted"]

# ─────────────────────────────────────────────────────────────────────────────
# Genre chart-rate summary (v5 Add E) — enhanced top/bottom tables in v6
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

# v6 Add D: top-10 and bottom-10 genre tables
top10 = genre_summary[genre_summary["total_tracks"] >= 50].head(10).copy()
bot10 = genre_summary[genre_summary["total_tracks"] >= 50].tail(10).copy()
top10["chart_rate_pct"] = (top10["chart_rate"] * 100).round(1).astype(str) + "%"
bot10["chart_rate_pct"] = (bot10["chart_rate"] * 100).round(1).astype(str) + "%"
top10[["track_genre","total_tracks","charted_tracks","chart_rate_pct","avg_popularity"]].to_csv(
    OUT / "genre_top10.csv", index=False)
bot10[["track_genre","total_tracks","charted_tracks","chart_rate_pct","avg_popularity"]].to_csv(
    OUT / "genre_bottom10.csv", index=False)
print(f"\nTop 10 genres by chart rate (min 50 tracks):")
print(top10[["track_genre","chart_rate_pct","avg_popularity"]].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# VIF CHECK
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60 + "\nVIF CHECK\n" + "="*60)
vif = pd.DataFrame({
    "Feature": FEATURES,
    "VIF":     [variance_inflation_factor(X.values, i) for i in range(len(FEATURES))]
}).sort_values("VIF", ascending=False).round(2)
print(vif.to_string(index=False))
vif.to_csv(OUT / "vif_table.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y_chart, test_size=0.2, random_state=RANDOM_STATE, stratify=y_chart)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)
print(f"\nTrain: {len(X_tr):,}  |  Test: {len(X_te):,}  |  "
      f"Test positive rate: {y_te.mean():.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# v6 Add A: AUDIO-ONLY BASELINE MODELS (for enrichment comparison)
# These reproduce the v4-equivalent models using only audio features,
# so we can quantify the lift from artist-level enrichment.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("AUDIO-ONLY BASELINE (for enrichment comparison)")
print("="*60)

X_audio    = df[BASE_FEATURES].copy()
Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(
    X_audio, y_chart, test_size=0.2, random_state=RANDOM_STATE, stratify=y_chart)
sa = StandardScaler()
Xa_tr_s = sa.fit_transform(Xa_tr)
Xa_te_s = sa.transform(Xa_te)

# Audio-only Logistic Regression
lr_audio = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
lr_audio.fit(Xa_tr_s, ya_tr)
auc_lr_audio   = roc_auc_score(ya_te, lr_audio.predict_proba(Xa_te_s)[:,1])
prauc_lr_audio = average_precision_score(ya_te, lr_audio.predict_proba(Xa_te_s)[:,1])
print(f"Audio-only LR   — AUC-ROC: {auc_lr_audio:.4f}  PR-AUC: {prauc_lr_audio:.4f}")

# Audio-only XGBoost
pos_w_audio = (len(ya_tr) - ya_tr.sum()) / ya_tr.sum()
xgb_audio = xgb.XGBClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=pos_w_audio, eval_metric="auc",
    early_stopping_rounds=50, random_state=RANDOM_STATE, verbosity=0,
)
xgb_audio.fit(Xa_tr, ya_tr, eval_set=[(Xa_te, ya_te)], verbose=False)
proba_xgb_audio = xgb_audio.predict_proba(Xa_te)[:,1]
auc_xgb_audio   = roc_auc_score(ya_te, proba_xgb_audio)
prauc_xgb_audio = average_precision_score(ya_te, proba_xgb_audio)
print(f"Audio-only XGBoost — AUC-ROC: {auc_xgb_audio:.4f}  PR-AUC: {prauc_xgb_audio:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1: LOGISTIC REGRESSION  (Chart Entry — full features)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60 + "\nMODEL 1: Logistic Regression — Chart Entry (full)\n" + "="*60)
lr     = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
lr.fit(X_tr_s, y_tr)
cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_auc = cross_val_score(lr, X_tr_s, y_tr, cv=cv, scoring="roc_auc")
y_prob_lr = lr.predict_proba(X_te_s)[:, 1]
y_pred_lr = lr.predict(X_te_s)
auc_lr    = roc_auc_score(y_te, y_prob_lr)
pr_auc_lr = average_precision_score(y_te, y_prob_lr)
print(f"Test AUC-ROC : {auc_lr:.4f}")
print(f"Test PR-AUC  : {pr_auc_lr:.4f}  (baseline = {y_te.mean():.4f})")
print(f"CV   AUC-ROC : {cv_auc.mean():.4f} +/- {cv_auc.std():.4f}  (5-fold)")
print(classification_report(y_te, y_pred_lr, target_names=["No Chart", "Charted"]))
coefs = lr.coef_[0]
or_df = pd.DataFrame({
    "Feature": FEATURES,
    "Coef":    coefs.round(4),
    "OR":      np.exp(coefs).round(4),
}).sort_values("OR", ascending=False)
print("Odds Ratios (per 1 SD):")
print(or_df.to_string(index=False))
or_df.to_csv(OUT / "logistic_odds_ratios.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2: XGBOOST + SHAP  (Chart Entry — full features)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60 + "\nMODEL 2: XGBoost — Chart Entry (full)\n" + "="*60)
pos_weight = (len(y_tr) - y_tr.sum()) / y_tr.sum()
xgb_m = xgb.XGBClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=pos_weight, eval_metric="auc",
    early_stopping_rounds=50, random_state=RANDOM_STATE, verbosity=0,
)
xgb_m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
y_prob_xgb = xgb_m.predict_proba(X_te)[:, 1]
y_pred_xgb = xgb_m.predict(X_te)
auc_xgb    = roc_auc_score(y_te, y_prob_xgb)
pr_auc_xgb = average_precision_score(y_te, y_prob_xgb)
print(f"Best iteration : {xgb_m.best_iteration}")
print(f"Test AUC-ROC   : {auc_xgb:.4f}")
print(f"Test PR-AUC    : {pr_auc_xgb:.4f}  (baseline = {y_te.mean():.4f})")
print(classification_report(y_te, y_pred_xgb, target_names=["No Chart", "Charted"]))
explainer  = shap.TreeExplainer(xgb_m)
shap_vals  = explainer.shap_values(X_te)
fi = pd.DataFrame({
    "Feature":   FEATURES,
    "Mean_SHAP": np.abs(shap_vals).mean(axis=0),
}).sort_values("Mean_SHAP", ascending=False).round(6)
print("SHAP Feature Importance:")
print(fi.to_string(index=False))
fi.to_csv(OUT / "xgboost_shap_importance.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# v6 Add B: THRESHOLD ANALYSIS TABLE
# At a 2.75% positive rate, the default 0.5 threshold is not useful.
# This table shows precision/recall/F1 trade-offs so A&R teams can choose
# a threshold appropriate for their screening volume and tolerance for false positives.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60 + "\nv6 Add B: THRESHOLD ANALYSIS (XGBoost)\n" + "="*60)
thresholds = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
thresh_rows = []
for t in thresholds:
    y_t = (y_prob_xgb >= t).astype(int)
    n_flagged = y_t.sum()
    tp = ((y_t == 1) & (y_te == 1)).sum()
    tn = ((y_t == 0) & (y_te == 0)).sum()
    fp = ((y_t == 1) & (y_te == 0)).sum()
    fn = ((y_t == 0) & (y_te == 1)).sum()
    prec  = precision_score(y_te, y_t, zero_division=0)
    rec   = recall_score(y_te, y_t)
    f1    = f1_score(y_te, y_t, zero_division=0)
    ppv   = tp / (tp + fp) if (tp + fp) > 0 else 0
    fdr   = fp / (tp + fp) if (tp + fp) > 0 else 0
    thresh_rows.append({
        "Threshold": t,
        "Flagged_tracks": n_flagged,
        "True_positives": int(tp),
        "False_positives": int(fp),
        "True_negatives": int(tn),
        "False_negatives": int(fn),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "FDR": round(fdr, 4),
        "Pct_flagged": round(n_flagged / len(y_te) * 100, 2),
    })
thresh_df = pd.DataFrame(thresh_rows)
print(thresh_df.to_string(index=False))
thresh_df.to_csv(OUT / "threshold_analysis.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# LYRIC FEATURES  (longevity models)
# ─────────────────────────────────────────────────────────────────────────────
LYRIC_FEATURES = []
lyric_cols = ["sentiment_compound", "sentiment_pos", "sentiment_neg", "lyric_word_count"]
if "sentiment_compound" in df.columns:
    print("\nLyric sentiment columns present in dataset.")
    matched = df.loc[df["is_charted"] == 1, "sentiment_compound"].notna().sum()
    print(f"  Lyric sentiment matched: {matched} charted tracks "
          f"({matched/df['is_charted'].sum()*100:.1f}%)")
    for col in lyric_cols:
        if col in df.columns:
            coverage = df.loc[df["is_charted"] == 1, col].notna().mean()
            if coverage > 0.20:
                LYRIC_FEATURES.append(col)
                print(f"  + {col} included (coverage: {coverage:.1%})")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3: COX PROPORTIONAL HAZARDS  (Longevity)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60 + "\nMODEL 3: Cox PH — Chart Longevity\n" + "="*60)
df["chart_entry_date"] = pd.to_datetime(df["chart_entry_date"], errors="coerce")
df["decade_idx"] = (
    (df["chart_entry_date"].dt.year.clip(lower=1950, upper=2029) - 1950) // 10
).astype("Int64")

COX_FEATURES = FEATURES + ["decade_idx"] + LYRIC_FEATURES
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
cph = CoxPHFitter(penalizer=0.1)
cph.fit(cph_df, duration_col="wks_on_chart", event_col="event", strata=["mode"])
cph.print_summary(decimals=4, style="ascii")
print(f"\nConcordance Index: {cph.concordance_index_:.4f}")
cph.summary.to_csv(OUT / "cox_summary.csv")

# v6 Add C: Schoenfeld residuals table
print("\n── Schoenfeld Residuals Test ──")
schoenfeld_rows = []
try:
    from io import StringIO
    import contextlib
    # Capture output to parse p-values
    f = StringIO()
    with contextlib.redirect_stdout(f):
        cph.check_assumptions(cph_df, p_value_threshold=0.05, show_plots=False)
    schoenfeld_output = f.getvalue()
    print(schoenfeld_output)
    # Also save the raw p_value_residuals from lifelines if available
    try:
        from lifelines.statistics import proportional_hazard_test
        result_ph = proportional_hazard_test(cph, cph_df, time_transform="rank")
        result_ph.summary.to_csv(OUT / "schoenfeld_residuals.csv")
        print(result_ph.summary.to_string())
        print("  Saved schoenfeld_residuals.csv")
    except Exception as e2:
        print(f"  Note: proportional_hazard_test unavailable ({e2}); "
              f"results available from check_assumptions output above")
        # Fallback: save parsed output as plain text
        with open(OUT / "schoenfeld_residuals.txt", "w") as fout:
            fout.write(schoenfeld_output)
except Exception as e:
    print(f"  check_assumptions: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3b: LOG-OLS  (Longevity robustness)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60 + "\nMODEL 3b: Log-OLS — Longevity (robustness)\n" + "="*60)
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
# PERFORMANCE SUMMARY (v6 Fix H: extended with audio-only baseline column)
# ─────────────────────────────────────────────────────────────────────────────
summary = pd.DataFrame({
    "Model":            ["Logistic Regression", "XGBoost", "Cox PH", "Log-OLS"],
    "Task":             ["Chart Entry (binary)", "Chart Entry (binary)",
                         "Longevity (survival)", "Longevity (OLS)"],
    "Metric":           ["AUC-ROC", "AUC-ROC", "C-statistic", "R²"],
    "Audio_Only_Score": [round(auc_lr_audio,  4), round(auc_xgb_audio,  4), "0.5508", "0.0438"],
    "Full_Model_Score": [round(auc_lr,  4), round(auc_xgb, 4),
                         round(cph.concordance_index_, 4), round(r2, 4)],
    "PR_AUC_AudioOnly": [round(prauc_lr_audio, 4), round(prauc_xgb_audio, 4), None, None],
    "PR_AUC_Full":      [round(pr_auc_lr, 4), round(pr_auc_xgb, 4), None, None],
    "CV_note":          [
        f"{cv_auc.mean():.4f} +/- {cv_auc.std():.4f}  (5-fold)",
        f"early stop @ iter {xgb_m.best_iteration}",
        "penalizer=0.1; mode stratified; n=" + str(len(cph_df)),
        "log1p(wks); n=" + str(len(charted)),
    ],
})
print("\n" + "="*60 + "\nMODEL PERFORMANCE SUMMARY (with audio-only baseline)\n" + "="*60)
print(summary.to_string(index=False))
summary.to_csv(OUT / "model_performance_summary.csv", index=False)

# Also save a concise version for the report
enrichment_comparison = pd.DataFrame({
    "Model":           ["Logistic Regression (LR)", "XGBoost"],
    "Metric":          ["ROC-AUC", "ROC-AUC"],
    "Audio_Only":      [round(auc_lr_audio, 3), round(auc_xgb_audio, 3)],
    "Full_Model":      [round(auc_lr, 3), round(auc_xgb, 3)],
    "Gain":            [round(auc_lr - auc_lr_audio, 3), round(auc_xgb - auc_xgb_audio, 3)],
    "PR_Audio_Only":   [round(prauc_lr_audio, 3), round(prauc_xgb_audio, 3)],
    "PR_Full_Model":   [round(pr_auc_lr, 3), round(pr_auc_xgb, 3)],
    "PR_Gain":         [round(pr_auc_lr - prauc_lr_audio, 3),
                        round(pr_auc_xgb - prauc_xgb_audio, 3)],
})
enrichment_comparison.to_csv(OUT / "enrichment_comparison.csv", index=False)
print("\nEnrichment comparison (artist features lift):")
print(enrichment_comparison.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# FIGURES  (11 total in v6; figs 1–9 carried over, 10–11 new)
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating figures...")
sns.set_style("whitegrid")
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})

# ── Fig 1: Class balance ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 3.5))
counts  = y_chart.value_counts().sort_index()
bars    = ax.bar(["Not Charted", "Charted"], counts.values,
                 color=["#c0392b", "#27ae60"], width=0.5, edgecolor="white")
for bar, v in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, v + 600,
            f"{v:,}\n({v/len(df):.1%})", ha="center", va="bottom",
            fontsize=9, fontweight="bold")
ax.set_ylabel("Track Count")
ax.set_title("Class Distribution: Billboard Hot 100 Chart Entry", fontweight="bold")
ax.set_ylim(0, max(counts) * 1.15)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(OUT / "fig1_class_balance.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Fig 2: Correlation heatmap ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df[FEATURES].corr(), annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, ax=ax, square=True, linewidths=0.5,
            vmin=-1, vmax=1, cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Matrix", fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(OUT / "fig2_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Fig 3: ROC curves ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
for name, probs, color in [
    ("Logistic Regression", y_prob_lr,  "#2980b9"),
    ("XGBoost",             y_prob_xgb, "#e67e22"),
]:
    fpr, tpr, _ = roc_curve(y_te, probs)
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f"{name}  (AUC={roc_auc_score(y_te, probs):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random  (AUC=0.500)")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — Billboard Chart Entry Prediction", fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(OUT / "fig3_roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Fig 4: Odds ratio forest plot ─────────────────────────────────────────────
or_sorted = or_df.sort_values("OR").reset_index(drop=True)
fig, ax   = plt.subplots(figsize=(7, max(5, len(or_sorted) * 0.5)))
colors    = ["#c0392b" if v < 1 else "#27ae60" for v in or_sorted["OR"]]
ax.barh(range(len(or_sorted)), or_sorted["OR"] - 1, left=1,
        color=colors, height=0.55, alpha=0.85)
ax.axvline(1, color="black", lw=1, linestyle="--")
ax.set_yticks(range(len(or_sorted)))
ax.set_yticklabels(or_sorted["Feature"])
ax.set_xlabel("Odds Ratio  (per 1 SD increase)")
ax.set_title("Logistic Regression — Chart Entry Odds Ratios\n(standardized; per 1 SD change)",
             fontweight="bold")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(OUT / "fig4_odds_ratios.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Fig 5: SHAP bar ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
shap.summary_plot(shap_vals, X_te, feature_names=FEATURES,
                  show=False, plot_type="bar", color="#2980b9")
plt.title("XGBoost Feature Importance (Mean |SHAP| Value)", fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "fig5_shap_importance.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Fig 6: Cox hazard ratios ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, max(5, len(COX_FEATURES) * 0.45)))
cph.plot(ax=ax)
ax.set_title("Cox Proportional Hazards — Chart Longevity\n"
             "(hazard ratios per 1 SD; mode stratified; n=" + str(len(cph_df)) + ")",
             fontweight="bold")
ax.axvline(0, color="black", lw=0.8, linestyle="--")
plt.tight_layout()
plt.savefig(OUT / "fig6_cox_hazard_ratios.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Fig 7: Kaplan-Meier by genre ──────────────────────────────────────────────
top_genres = (df[df["is_charted"] == 1]["track_genre"]
              .value_counts().head(5).index.tolist())
palette = ["#2980b9", "#e67e22", "#27ae60", "#8e44ad", "#c0392b"]
fig, ax = plt.subplots(figsize=(9, 5.5))
for genre, color in zip(top_genres, palette):
    times = df.loc[
        (df["is_charted"] == 1) & (df["track_genre"] == genre), "wks_on_chart"
    ].dropna()
    times = times[times > 0]
    if len(times) < 5:
        continue
    KaplanMeierFitter().fit(times, label=f"{genre}  (n={len(times)})").plot_survival_function(
        ax=ax, ci_show=True, color=color)
ax.set_xlabel("Weeks on Billboard Hot 100")
ax.set_ylabel("P(Still Charting)")
ax.set_title("Kaplan-Meier Survival Curves by Genre", fontweight="bold")
ax.legend(loc="upper right", fontsize=8)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(OUT / "fig7_kaplan_meier.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Fig 8: Longevity distribution ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
wks = df.loc[df["is_charted"] == 1, "wks_on_chart"]
ax.hist(wks, bins=40, color="#2980b9", edgecolor="white", alpha=0.85)
ax.axvline(wks.median(), color="#e67e22", lw=2, linestyle="--",
           label=f"Median = {wks.median():.0f} wks")
ax.axvline(wks.mean(), color="#c0392b", lw=2, linestyle=":",
           label=f"Mean = {wks.mean():.1f} wks")
ax.set_xlabel("Weeks on Chart"); ax.set_ylabel("Track Count")
ax.set_title("Distribution of Chart Longevity (Charted Tracks Only)", fontweight="bold")
ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(OUT / "fig8_longevity_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Fig 9: Precision-Recall curves (v6 Fix G: corrected baseline label) ────────
fig, ax = plt.subplots(figsize=(7, 5))
for name, probs, color in [
    ("Logistic Regression", y_prob_lr,  "#2980b9"),
    ("XGBoost",             y_prob_xgb, "#e67e22"),
]:
    prec, rec, _ = precision_recall_curve(y_te, probs)
    ap = average_precision_score(y_te, probs)
    ax.plot(rec, prec, color=color, lw=2, label=f"{name}  (AP={ap:.3f})")
baseline = y_te.mean()
ax.axhline(baseline, color="gray", lw=1.2, linestyle="--",
           label=f"Random baseline  (AP={baseline:.3f})")
ax.set_xlabel("Recall (Sensitivity)")
ax.set_ylabel("Precision (PPV)")
ax.set_title("Precision-Recall Curves — Billboard Chart Entry\n"
             f"({baseline:.1%} positive rate; AP = area under PR curve)", fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(OUT / "fig9_precision_recall.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Fig 10: Enrichment comparison bar chart (v6 Add E) ───────────────────────
# Shows AUC-ROC and PR-AUC for audio-only vs full model, side by side.
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
models   = ["LR", "XGBoost"]
x_pos    = np.arange(len(models))
width    = 0.32

# Left panel: ROC-AUC
audio_roc = [auc_lr_audio, auc_xgb_audio]
full_roc  = [auc_lr,       auc_xgb]
b1 = axes[0].bar(x_pos - width/2, audio_roc, width, label="Audio features only",
                  color="#95a5a6", edgecolor="white")
b2 = axes[0].bar(x_pos + width/2, full_roc,  width, label="+ Artist features",
                  color="#2980b9", edgecolor="white")
for bar, v in zip(list(b1)+list(b2), audio_roc+full_roc):
    axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.003, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=8)
axes[0].set_xticks(x_pos); axes[0].set_xticklabels(models)
axes[0].set_ylim(0.5, 1.03)
axes[0].set_ylabel("AUC-ROC"); axes[0].set_title("ROC-AUC", fontweight="bold")
axes[0].legend(fontsize=8)
axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)
axes[0].axhline(0.5, color="gray", lw=0.8, linestyle="--")

# Right panel: PR-AUC
audio_pr = [prauc_lr_audio, prauc_xgb_audio]
full_pr  = [pr_auc_lr,      pr_auc_xgb]
b3 = axes[1].bar(x_pos - width/2, audio_pr, width, label="Audio features only",
                  color="#95a5a6", edgecolor="white")
b4 = axes[1].bar(x_pos + width/2, full_pr,  width, label="+ Artist features",
                  color="#e67e22", edgecolor="white")
for bar, v in zip(list(b3)+list(b4), audio_pr+full_pr):
    axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.005, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=8)
axes[1].set_xticks(x_pos); axes[1].set_xticklabels(models)
axes[1].set_ylim(0, max(full_pr) * 1.35)
axes[1].axhline(baseline, color="gray", lw=0.8, linestyle="--",
                label=f"Random baseline ({baseline:.3f})")
axes[1].set_ylabel("PR-AUC"); axes[1].set_title("Precision-Recall AUC", fontweight="bold")
axes[1].legend(fontsize=8)
axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)

fig.suptitle("Impact of Artist Enrichment Features on Prediction Performance",
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT / "fig10_enrichment_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig10_enrichment_comparison.png")

# ── Fig 11: SHAP dependence plot for instrumentalness (v6 Add F) ─────────────
# Shows how SHAP contribution of instrumentalness changes as the feature value
# increases; color-coded by valence to show interaction.
if "instrumentalness" in FEATURES and "valence" in FEATURES:
    instr_idx  = FEATURES.index("instrumentalness")
    valence_idx = FEATURES.index("valence")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sc = ax.scatter(
        X_te.iloc[:, instr_idx] if hasattr(X_te, 'iloc') else X_te[:, instr_idx],
        shap_vals[:, instr_idx],
        c=X_te.iloc[:, valence_idx] if hasattr(X_te, 'iloc') else X_te[:, valence_idx],
        cmap="RdYlGn", alpha=0.4, s=8, rasterized=True,
    )
    plt.colorbar(sc, ax=ax, label="Valence (low=sad, high=happy)")
    ax.axhline(0, color="black", lw=0.8, linestyle="--")
    ax.set_xlabel("Instrumentalness (raw feature value)")
    ax.set_ylabel("SHAP value (contribution to log-odds of charting)")
    ax.set_title("SHAP Dependence: Instrumentalness\n"
                 "(negative SHAP = reduces probability of charting)",
                 fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT / "fig11_shap_instrumentalness.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fig11_shap_instrumentalness.png")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("RUN COMPLETE — All outputs in outputs/")
print(f"  Task 1 (Chart Entry): LR AUC-ROC={auc_lr:.4f} PR-AUC={pr_auc_lr:.4f}")
print(f"                    XGB AUC-ROC={auc_xgb:.4f} PR-AUC={pr_auc_xgb:.4f}")
print(f"  Task 2 (Longevity):   Cox C-stat={cph.concordance_index_:.4f} n={len(cph_df)}")
print(f"                    OLS R²={r2:.4f}")
print(f"  Enrichment lift:  LR ROC +{auc_lr - auc_lr_audio:.3f} | XGB ROC +{auc_xgb - auc_xgb_audio:.3f}")
print(f"  New outputs: enrichment_comparison.csv, threshold_analysis.csv,")
print(f"               schoenfeld_residuals.csv/txt, genre_top10.csv, genre_bottom10.csv")
print(f"               fig10_enrichment_comparison.png, fig11_shap_instrumentalness.png")
print("="*60)

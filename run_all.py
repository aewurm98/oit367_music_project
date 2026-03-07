"""
OIT367 — Run this script directly in your terminal to execute the full pipeline.

Prerequisites (install once):
    pip install scikit-learn xgboost shap lifelines statsmodels seaborn tqdm spotipy

Usage:
    python3 run_all.py

Outputs go to: ./outputs/
"""

import pandas as pd
import numpy as np
import warnings, re
warnings.filterwarnings("ignore")
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
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
# DATA PREP  (skip if oit367_base_dataset.csv already exists)
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

    spotify   = pd.read_csv("spotify_tracksdataset.csv")
    bb_weekly = pd.read_csv("merged_spotify_billboard_data.csv")  # inner-joined weekly file

    # Deduplicate Spotify (114k → 89,741 unique track_ids)
    spotify_dedup = (spotify.sort_values('track_genre')
                     .drop_duplicates(subset='track_id', keep='first')
                     .reset_index(drop=True))

    # Aggregate Billboard to one row per track
    bb_weekly['chart_week'] = pd.to_datetime(bb_weekly['chart_week'])
    bb_agg = (bb_weekly.groupby('track_id', as_index=False)
              .agg(peak_pos=('peak_pos','min'),
                   wks_on_chart=('wks_on_chart','max'),
                   chart_entry_date=('chart_week','min')))

    # Left join
    df = spotify_dedup.merge(bb_agg, on='track_id', how='left')
    df['is_charted']   = df['peak_pos'].notna().astype(int)
    df['wks_on_chart'] = df['wks_on_chart'].fillna(0).astype(int)
    df['is_popular']   = (df['popularity'] >= 80).astype(int)
    df.to_csv(BASE_CSV, index=False)
    print(f"  Saved {len(df):,} rows → {BASE_CSV}")
else:
    print(f"Loading existing dataset: {BASE_CSV}")

df = pd.read_csv(BASE_CSV)
print(f"Tracks: {len(df):,} | Charted: {df['is_charted'].sum():,} ({df['is_charted'].mean():.2%})")

FEATURES = [
    'danceability','energy','valence','tempo',
    'acousticness','loudness','speechiness',
    'instrumentalness','liveness','mode','key'
]

# If Spotipy augmentation has been run, add those features
AUGMENTED = [f for f in ['artist_followers','artist_popularity_api'] if f in df.columns]
if AUGMENTED:
    df['artist_followers'] = np.log1p(df['artist_followers'].fillna(0))
    FEATURES += AUGMENTED
    print(f"  + Augmented features included: {AUGMENTED}")

X       = df[FEATURES].copy()
y_chart = df['is_charted']

# ─────────────────────────────────────────────────────────────────────────────
# VIF
# ─────────────────────────────────────────────────────────────────────────────
vif = pd.DataFrame({
    'Feature': FEATURES,
    'VIF':     [variance_inflation_factor(X.values, i) for i in range(len(FEATURES))]
}).sort_values('VIF', ascending=False).round(2)
print("\nVIF Table:\n", vif.to_string(index=False))
vif.to_csv(OUT/'vif_table.csv', index=False)

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y_chart, test_size=0.2, random_state=RANDOM_STATE, stratify=y_chart)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)
print(f"\nTrain: {len(X_tr):,} | Test: {len(X_te):,}")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1: LOGISTIC REGRESSION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60 + "\nMODEL 1: Logistic Regression — Chart Entry\n" + "="*60)
lr     = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE)
lr.fit(X_tr_s, y_tr)
cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_auc = cross_val_score(lr, X_tr_s, y_tr, cv=cv, scoring='roc_auc')
y_prob_lr = lr.predict_proba(X_te_s)[:,1]
y_pred_lr = lr.predict(X_te_s)
auc_lr    = roc_auc_score(y_te, y_prob_lr)
print(f"Test AUC-ROC : {auc_lr:.4f}")
print(f"CV   AUC-ROC : {cv_auc.mean():.4f} +/- {cv_auc.std():.4f}")
print(classification_report(y_te, y_pred_lr, target_names=['No Chart','Charted']))
coefs = lr.coef_[0]
or_df = pd.DataFrame({'Feature':FEATURES,'Coef':coefs.round(4),'OR':np.exp(coefs).round(4)}
                     ).sort_values('OR', ascending=False)
print("Odds Ratios (per 1 SD):\n", or_df.to_string(index=False))
or_df.to_csv(OUT/'logistic_odds_ratios.csv', index=False)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2: XGBOOST + SHAP
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60 + "\nMODEL 2: XGBoost — Chart Entry\n" + "="*60)
pw    = (len(y_tr) - y_tr.sum()) / y_tr.sum()
xgb_m = xgb.XGBClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pw,
    eval_metric='auc', early_stopping_rounds=50,
    random_state=RANDOM_STATE, verbosity=0)
xgb_m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
y_prob_xgb = xgb_m.predict_proba(X_te)[:,1]
y_pred_xgb = xgb_m.predict(X_te)
auc_xgb    = roc_auc_score(y_te, y_prob_xgb)
print(f"Best N estimators: {xgb_m.best_iteration}")
print(f"Test AUC-ROC     : {auc_xgb:.4f}")
print(classification_report(y_te, y_pred_xgb, target_names=['No Chart','Charted']))
explainer  = shap.TreeExplainer(xgb_m)
shap_vals  = explainer.shap_values(X_te)
fi = pd.DataFrame({'Feature':FEATURES,'Mean_SHAP':np.abs(shap_vals).mean(axis=0)}
                  ).sort_values('Mean_SHAP', ascending=False).round(6)
print("SHAP Importance:\n", fi.to_string(index=False))
fi.to_csv(OUT/'xgboost_shap_importance.csv', index=False)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3: COX PROPORTIONAL HAZARDS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60 + "\nMODEL 3: Cox PH — Chart Longevity\n" + "="*60)
charted  = df[(df['is_charted']==1) & (df['wks_on_chart']>0)].dropna(subset=FEATURES+['wks_on_chart']).copy()
cox_sc   = StandardScaler()
X_cox    = pd.DataFrame(cox_sc.fit_transform(charted[FEATURES]), columns=FEATURES, index=charted.index)
cph_df   = X_cox.copy()
cph_df['wks_on_chart'] = charted['wks_on_chart'].values
cph_df['event']        = 1
cph      = CoxPHFitter(penalizer=0.1)
cph.fit(cph_df, duration_col='wks_on_chart', event_col='event')
cph.print_summary(decimals=4, style='ascii')
print(f"\nConcordance Index (C-stat): {cph.concordance_index_:.4f}")
cph.summary.to_csv(OUT/'cox_summary.csv')

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3b: LOG-OLS LONGEVITY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60 + "\nMODEL 3b: Log-OLS — Longevity (robustness)\n" + "="*60)
y_log   = np.log1p(charted['wks_on_chart'].values)
ols_sc  = StandardScaler()
X_ols   = ols_sc.fit_transform(charted[FEATURES])
Xo_tr, Xo_te, yo_tr, yo_te = train_test_split(X_ols, y_log, test_size=0.2, random_state=RANDOM_STATE)
ols     = LinearRegression().fit(Xo_tr, yo_tr)
r2      = ols.score(Xo_te, yo_te)
ols_coef = pd.DataFrame({'Feature':FEATURES,'Coef':ols.coef_}).sort_values('Coef',ascending=False)
print(f"Test R2: {r2:.4f}")
print(ols_coef.round(4).to_string(index=False))
ols_coef.to_csv(OUT/'ols_longevity_coefficients.csv', index=False)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating figures...")
sns.set_style("whitegrid")

# Fig 1: Class balance
fig, ax = plt.subplots(figsize=(5,3.5))
counts  = y_chart.value_counts().sort_index()
bars    = ax.bar(['Not Charted','Charted'], counts.values, color=['#c0392b','#27ae60'], width=0.5)
for bar, v in zip(bars, counts.values):
    ax.text(bar.get_x()+bar.get_width()/2, v+600, f'{v:,}\n({v/len(df):.1%})',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_ylabel('Track Count'); ax.set_title('Class Distribution: Billboard Chart Entry', fontweight='bold')
ax.set_ylim(0, max(counts)*1.15)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(); plt.savefig(OUT/'fig1_class_balance.png', dpi=150, bbox_inches='tight'); plt.close()

# Fig 2: Correlation heatmap
fig, ax = plt.subplots(figsize=(9,7))
sns.heatmap(df[FEATURES].corr(), annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            ax=ax, square=True, linewidths=0.5, cbar_kws={'shrink':0.8})
ax.set_title('Audio Feature Correlation Matrix', fontweight='bold', pad=12)
plt.tight_layout(); plt.savefig(OUT/'fig2_correlation_heatmap.png', dpi=150, bbox_inches='tight'); plt.close()

# Fig 3: ROC curves
fig, ax = plt.subplots(figsize=(6,5))
for name, probs, color in [('Logistic Reg',y_prob_lr,'#2980b9'),
                            ('XGBoost',y_prob_xgb,'#e67e22')]:
    fpr, tpr, _ = roc_curve(y_te, probs)
    ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={roc_auc_score(y_te,probs):.3f})')
ax.plot([0,1],[0,1],'k--', lw=1, label='Random (AUC=0.500)')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — Chart Entry Prediction', fontweight='bold')
ax.legend(loc='lower right')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(); plt.savefig(OUT/'fig3_roc_curves.png', dpi=150, bbox_inches='tight'); plt.close()

# Fig 4: Odds ratio forest plot
or_sorted = or_df.sort_values('OR')
fig, ax   = plt.subplots(figsize=(7, max(4.5, len(or_sorted)*0.5)))
colors    = ['#c0392b' if v<1 else '#27ae60' for v in or_sorted['OR']]
ax.barh(range(len(or_sorted)), or_sorted['OR']-1, left=1, color=colors, height=0.55, alpha=0.85)
ax.axvline(1, color='black', lw=1, linestyle='--')
ax.set_yticks(range(len(or_sorted))); ax.set_yticklabels(or_sorted['Feature'])
ax.set_xlabel('Odds Ratio (per 1 SD increase)')
ax.set_title('Logistic Regression — Chart Entry Odds Ratios\n(scaled; per 1 SD change)', fontweight='bold')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(); plt.savefig(OUT/'fig4_odds_ratios.png', dpi=150, bbox_inches='tight'); plt.close()

# Fig 5: SHAP bar
fig, ax = plt.subplots(figsize=(7,5))
shap.summary_plot(shap_vals, X_te, feature_names=FEATURES, show=False, plot_type='bar', color='#2980b9')
plt.title('XGBoost Feature Importance (Mean |SHAP|)', fontweight='bold')
plt.tight_layout(); plt.savefig(OUT/'fig5_shap_importance.png', dpi=150, bbox_inches='tight'); plt.close()

# Fig 6: Cox hazard ratios
fig, ax = plt.subplots(figsize=(7,5))
cph.plot(ax=ax)
ax.set_title('Cox PH — Hazard Ratios (per 1 SD)\nChart Longevity', fontweight='bold')
ax.axvline(0, color='black', lw=0.8, linestyle='--')
plt.tight_layout(); plt.savefig(OUT/'fig6_cox_hazard_ratios.png', dpi=150, bbox_inches='tight'); plt.close()

# Fig 7: Kaplan-Meier by genre
top_genres = df[df['is_charted']==1]['track_genre'].value_counts().head(5).index.tolist()
palette    = ['#2980b9','#e67e22','#27ae60','#8e44ad','#c0392b']
fig, ax    = plt.subplots(figsize=(9,5.5))
for genre, color in zip(top_genres, palette):
    times = df.loc[(df['is_charted']==1)&(df['track_genre']==genre),'wks_on_chart']
    times = times[times>0].dropna()
    if len(times) < 5: continue
    KaplanMeierFitter().fit(times, label=f'{genre} (n={len(times)})').plot_survival_function(
        ax=ax, ci_show=True, color=color)
ax.set_xlabel('Weeks on Billboard Hot 100'); ax.set_ylabel('P(Still Charting)')
ax.set_title('Kaplan-Meier Survival Curves by Genre', fontweight='bold')
ax.legend(loc='upper right', fontsize=8)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(); plt.savefig(OUT/'fig7_kaplan_meier.png', dpi=150, bbox_inches='tight'); plt.close()

# Fig 8: Longevity distribution
fig, ax = plt.subplots(figsize=(7,4))
wks = df.loc[df['is_charted']==1,'wks_on_chart']
ax.hist(wks, bins=40, color='#2980b9', edgecolor='white', alpha=0.85)
ax.axvline(wks.median(), color='#e67e22', lw=2, linestyle='--', label=f'Median={wks.median():.0f} wks')
ax.axvline(wks.mean(),   color='#c0392b', lw=2, linestyle=':',  label=f'Mean={wks.mean():.1f} wks')
ax.set_xlabel('Weeks on Chart'); ax.set_ylabel('Track Count')
ax.set_title('Distribution of Chart Longevity (Charted Tracks)', fontweight='bold')
ax.legend(); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(); plt.savefig(OUT/'fig8_longevity_distribution.png', dpi=150, bbox_inches='tight'); plt.close()

# Summary
summary = pd.DataFrame({
    'Model':  ['Logistic Regression','XGBoost','Cox PH','Log-OLS'],
    'Task':   ['Chart Entry','Chart Entry','Longevity (survival)','Longevity (OLS)'],
    'Metric': ['AUC-ROC','AUC-ROC','C-statistic','R2'],
    'Score':  [round(auc_lr,4), round(auc_xgb,4), round(cph.concordance_index_,4), round(r2,4)],
    'CV':     [f"{cv_auc.mean():.4f}+/-{cv_auc.std():.4f}",
               f"early_stop@{xgb_m.best_iteration}", 'penalizer=0.1','log1p(wks)'],
})
print("\n" + "="*60 + "\nMODEL PERFORMANCE SUMMARY\n" + "="*60)
print(summary.to_string(index=False))
summary.to_csv(OUT/'model_performance_summary.csv', index=False)
print(f"\nAll outputs saved to: {OUT.resolve()}/")

# ─────────────────────────────────────────────────────────────────────────────
# SPOTIPY AUGMENTATION (run separately after setting env vars)
# ─────────────────────────────────────────────────────────────────────────────
# export SPOTIPY_CLIENT_ID=your_id
# export SPOTIPY_CLIENT_SECRET=your_secret
# Then uncomment and run:
#
import spotipy, time, random
from spotipy.oauth2 import SpotifyClientCredentials
from tqdm import tqdm

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(), requests_timeout=10)
unique_artists = df['artists'].unique()
artist_data = {}
cache_path = Path("artist_cache.csv")
if cache_path.exists():
    artist_data = pd.read_csv(cache_path, index_col=0).to_dict("index")
pending = [a for a in unique_artists if a not in artist_data]
for i, artist in enumerate(tqdm(pending)):
    for attempt in range(5):
        try:
            r = sp.search(q=f"artist:{artist}", type="artist", limit=1)
            items = r["artists"]["items"]
            artist_data[artist] = {"artist_followers": items[0]["followers"]["total"] if items else None,
                                   "artist_popularity_api": items[0]["popularity"] if items else None}
            break
        except Exception:
            time.sleep((2**attempt) + random.random())
    time.sleep(0.35)
    if (i+1) % 50 == 0:
        pd.DataFrame.from_dict(artist_data, orient="index").to_csv(cache_path)
artist_df = pd.DataFrame.from_dict(artist_data, orient="index").reset_index().rename(columns={"index":"artists"})
df = df.merge(artist_df, on="artists", how="left")
df['artist_followers'] = np.log1p(df['artist_followers'].fillna(0))
df.to_csv("oit367_augmented_dataset.csv", index=False)
print("Augmented dataset saved.")

# Analysis Status: Complete and Not-Yet-Done
**OIT367 Final Report | Stanford GSB Winter 2026 | Wurm · Chen · Barli · Taruno**
**Reference: run_all_v5.py on oit367_final_dataset.csv (78,390 rows, v6)**

---

## Part 1: Analyses Complete and Report-Ready

All of the following are computed, saved to `outputs/`, and represent the primary findings for the report.

### Dataset and Descriptive Statistics

**[DONE] Dataset construction pipeline (v6)**
78,390 unique tracks (2,157 charted / 76,233 not charted). Left join confirmed; cross-ID deduplication applied. Class imbalance documented (2.75% positive rate). `fig1_class_balance.png` generated. Baseline PR-AUC = 0.0275.

**[DONE] Genre chart rate table (`genre_chart_rates.csv`)**
All 100+ Spotify genre tags ranked by chart rate. Top genres by rate: rock (32.5%), country (26.5%), dance (24.5%), hard-rock (19.3%). Bottom genres include hip-hop (6.9%), r-n-b (5.1%), classical-adjacent tags near 0%. The wide dispersion reflects that some genres are over-represented in the Billboard Hot 100 by construction (radio-oriented formats).

**[DONE] VIF table (`vif_table.csv`)**
All 15 classification features checked. Problematic: `artist_peak_popularity` (21.25), `artist_popularity_api` (14.12). Acceptable for all other features (max 6.72 for `loudness`). `energy`, `danceability`, and `tempo` were removed in earlier versions for VIF > 10.

**[DONE] Correlation heatmap (`fig2_correlation_heatmap.png`)**
Pairwise Pearson correlations among all features. Confirms `energy`-`loudness` co-movement (r ≈ 0.78), `artist_peak_popularity`-`artist_popularity_api` correlation, and low cross-correlations among audio features after removing the collinear trio.

### Classification: Logistic Regression

**[DONE] Model fit and performance (`logistic_odds_ratios.csv`, `model_performance_summary.csv`)**
- ROC-AUC = 0.9144 (5-fold CV: 0.9139 +/- 0.0037)
- PR-AUC = 0.2750 (10.0x above random baseline)
- All 15 feature odds ratios computed with correct StandardScaler preprocessing

**[DONE] ROC curve (`fig3_roc_curves.png`)**
Side-by-side ROC for LR and XGBoost.

**[DONE] Precision-recall curve (`fig9_precision_recall.png`)**
Side-by-side PR curves for LR and XGBoost. This is the primary classification visualization given class imbalance.

**[DONE] Odds ratio forest plot (`fig4_odds_ratios.png`)**
Forest plot of per-1-SD odds ratios with 95% CIs. Dominant positive predictors: `artist_peak_popularity` (OR=2.84), `lastfm_listeners_log` (OR=2.17), `is_us_artist` (OR=1.91). Dominant negative: `instrumentalness` (OR=0.34).

### Classification: XGBoost + SHAP

**[DONE] Model fit and performance**
- ROC-AUC = 0.9655 (early stop at iteration 193)
- PR-AUC = 0.4402 (16.0x above random baseline)

**[DONE] SHAP importance (`xgboost_shap_importance.csv`, `fig5_shap_importance.png`)**
All 15 features ranked by mean |SHAP|. Top 5: `artist_peak_popularity` (1.618), `lastfm_listeners_log` (1.153), `instrumentalness` (0.629), `is_us_artist` (0.558), `artist_track_count` (0.388). Rankings stable vs. v5 model.

### Survival Analysis: Cox PH

**[DONE] Cox PH model fit (`cox_summary.csv`, `fig6_cox_hazard_ratios.png`)**
- Concordance index = 0.7526 (vs. 0.5508 audio-only baseline)
- n = 856 lyric-matched charted tracks
- `strata=["mode"]`, `penalizer=0.1`
- Schoenfeld violations: 4/19 features fail (down from 13/19 in v5 after cross-ID dedup)

**[DONE] Kaplan-Meier curves (`fig7_kaplan_meier.png`)**
Survival function plot for charted tracks. Can be stratified by mode or genre for illustrative comparison.

**[DONE] Longevity distribution (`fig8_longevity_distribution.png`)**
Histogram of `wks_on_chart` for charted tracks (range: 1 to 90+ weeks). Shows the right-skewed distribution motivating log transformation in OLS.

### Survival Robustness: Log-OLS

**[DONE] OLS on log1p(wks_on_chart) (`ols_longevity_coefficients.csv`)**
- R² = 0.2118 (n = 856, same lyric-matched subset)
- `decade_idx` directional agreement with Cox confirmed (both indicate streaming era compression of median chart runs, with OLS mean skewed up by mega-hits)
- `sentiment_neg` directional agreement with Cox confirmed

---

## Part 2: Analyses Not Yet Complete -- Important to Address Before Submission

The following gaps are ordered by priority. Some are quick computations; others require judgment calls.

---

### HIGH PRIORITY: Required for Methodological Completeness

**[NOT DONE] Formal pre- vs. post-enrichment performance comparison table**

What exists: RESULTS.md documents baseline audio-only scores and final scores in prose. What is missing: a clean 4-row table showing baseline (audio only) vs. full model (audio + artist + lyric) for all four models. This table is the clearest evidence of the artist enrichment contribution and should appear in the Methods or Results section.

Suggested table format:

| Model | Audio-Only ROC-AUC | Full-Model ROC-AUC | Gain |
|---|---|---|---|
| Logistic Regression | 0.711 | 0.914 | +0.203 |
| XGBoost | 0.834 | 0.966 | +0.132 |
| Cox PH (C-stat) | 0.551 | 0.753 | +0.202 |
| Log-OLS (R²) | 0.044 | 0.212 | +0.168 |

This can be generated from `model_performance_summary.csv` and the baseline numbers in RESULTS.md. No re-running required.

---

**[NOT DONE] Schoenfeld residuals discussion with formal table**

What exists: the analysis log and RESULTS.md note that 4/19 Cox features fail the Schoenfeld test. What is missing: a summary table of which features fail and p-values, plus a sentence explaining why the OLS robustness check partially mitigates the concern. This is important for the Methods section. The violation details are in the Cox output but have not been formatted for the report.

Affected features (v6): `decade_idx`, `acousticness`, `duration_min`, `sentiment_pos`. For `decade_idx` specifically, the OLS and Cox findings are directionally consistent despite the PH violation, which is a meaningful robustness statement.

---

**[PARTIALLY DONE] VIF table formatting for main body vs. appendix**

What exists: `vif_table.csv` with all 15 features. What is missing: a decision on whether to include this in the main body (slot 7 of 7) or appendix. Given the collinearity discussion around `artist_peak_popularity` and `artist_popularity_api`, including the VIF table in the main body or immediately adjacent to the odds ratio table strengthens credibility. Recommend condensing to a 3-column table (Feature, VIF, Flag) and noting that features with VIF > 10 are excluded or flagged.

---

### MEDIUM PRIORITY: Strengthens the Report's Main Claims

**[NOT DONE] SHAP dependence plots for instrumentalness and valence**

What exists: the summary SHAP importance bar chart (fig5). What is missing: SHAP dependence plots that show how the predicted probability of charting changes as a function of `instrumentalness` and `valence`. These two plots would (a) visually confirm the nonlinear relationship for `speechiness` (where LR and XGBoost disagree), and (b) show the valence effect is monotonic rather than artifact-driven. These are one-line additions to `run_all_v5.py` using `shap.dependence_plot()`. Worth adding if time permits -- but not required for the report's core argument.

---

**[NOT DONE] Genre-level descriptive comparison (top 10 and bottom 10)**

What exists: `genre_chart_rates.csv` with all genres. What is missing: a clean, labeled table of the top 10 and bottom 10 genres by chart rate for inclusion in the report appendix. This table motivates why genre encoding was considered (and can mention that mean-encoded genre rate was explored but not included in the final models due to collinearity and genre-assignment inconsistency). Takes 5 minutes to prepare manually.

---

**[NOT DONE] Threshold optimization table for classification (precision vs. recall tradeoff)**

What exists: PR curves (fig9). What is missing: a small table showing how precision and recall change at several decision thresholds (e.g., 0.05, 0.10, 0.20, 0.30) for XGBoost. This is the key practical output for A&R teams: at what threshold does the model become useful for screening? The current classification report uses the default threshold (0.5), which is inappropriate for a 2.75% base rate. Including a threshold analysis in the appendix directly addresses the "usefulness and practicality" grading criterion.

---

**[NOT DONE] Descriptive analysis of teammate enrichment features**

What exists: `oit367_final_dataset.csv` contains `is_male_artist`, `artist_age`, and `is_mainstream_genre` for approximately 58% of charted tracks. What is missing: a descriptive table comparing these characteristics between charted tracks with available data (n ≈ 1,251) and the lyric-matched longevity subset (n = 856). These features were excluded from the formal models because coverage across the full 78,390-row universe falls far below the 50% threshold required for reliable auto-detection. However, a descriptive comparison in the appendix would provide additional context about the charted-track population. This requires only a groupby summary on the existing dataset -- no re-running.

---

### LOW PRIORITY: Nice-to-Have, Not Report-Blocking

**[NOT DONE] Calibration plot for XGBoost**

XGBoost at `scale_pos_weight=24.6` is tuned for discriminative performance (AUC), not calibration. A reliability diagram showing predicted probability vs. observed frequency would clarify whether the model's raw scores can be interpreted as probabilities. Relevant for the "could this tool be deployed?" framing in the Conclusions. Low priority given the report's focus on prediction and interpretation rather than deployment.

**[NOT DONE] Key-as-circular-feature encoding (sin/cos)**

Musical keys are circular: key=11 (B) is adjacent to key=0 (C). The current model treats key as a linear ordinal 0-11, which imposes a false ordering. A sin/cos circular encoding would better represent the geometry. The existing key VIF (3.04) and near-zero OR (1.01) suggest key is not a meaningful predictor in this dataset regardless of encoding. Not worth re-running models for the report.

**[NOT DONE] Librosa audio features from 30-second preview clips**

`LIBROSA_MODAL_PLAN.md` documents a secondary analysis pipeline using Librosa to extract spectral features (MFCCs, spectral centroid, zero crossing rate, chroma) from 30-second Spotify preview clips via Modal cloud infrastructure. `modal_librosa_extract.py` is written and committed. Status: preview URLs were fetched for charted tracks (`modal_preview_urls.py`), but the Librosa extraction job has not been confirmed complete and `librosa_features.csv` is not present in the outputs. No Librosa-derived figures appear in `outputs/`. If the extraction completed, adding these features to a v7 comparison would be a strong bonus-point finding ("entrepreneurial data collection"). If not completed, this is out of scope for the March 11 deadline.

---

## Part 3: Figure and Table Budget Recommendation

With 9 figures and 7 CSV tables generated, the team must select 7 items for the main body and assign the rest to the appendix. Recommended allocation:

### Main Body (7 slots)

1. **Table: Model performance summary** (from `model_performance_summary.csv`, extended with baseline column -- see "Not Done" item above). Essential anchor for Results section.
2. **Fig3: ROC curves** (LR and XGBoost). Standard evidence of classification validity.
3. **Fig9: Precision-recall curves** (LR and XGBoost). Primary metric visualization; cannot omit for imbalanced data.
4. **Fig5: SHAP importance** (XGBoost, all 15 features). Key interpretability figure.
5. **Fig4: Odds ratios** (logistic regression forest plot). Interpretable regression output.
6. **Fig7: Kaplan-Meier** survival curves. Visual entry point for the longevity section.
7. **Table: Condensed Cox PH results** (top significant covariates from `cox_summary.csv`: decade_idx, artist_peak_popularity, artist_track_count, artist_popularity_api, loudness, with p-values and HRs). Alternatively, replace with VIF table if the collinearity argument is made prominently.

### Appendix

Fig1 (class balance), Fig2 (correlation heatmap), Fig6 (Cox hazard ratio plot), Fig8 (longevity distribution), full Cox coefficient table, OLS coefficient table, VIF table, genre chart rate table, full SHAP table.

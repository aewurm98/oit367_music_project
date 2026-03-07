❯ python3 run_all_v3.py
Building base dataset from raw files...
Saved 89,741 rows → oit367_base_dataset.csv

Dataset: 89,741 tracks | Charted: 3,502 (3.90%) | Not charted: 86,239 (96.10%)

============================================================
VIF CHECK
============================================================
Feature VIF
tempo 15.33
energy 15.07
danceability 12.28
loudness 7.12
valence 6.41
acousticness 3.73
key 3.19
mode 2.77
liveness 2.59
instrumentalness 1.85
speechiness 1.78

⚠ High VIF (>10): ['tempo', 'energy', 'danceability']

Split — Train: 71,792 | Test: 17,949

============================================================
MODEL 1: Logistic Regression — Chart Entry
============================================================
Test AUC-ROC : 0.7066
CV AUC-ROC : 0.7150 ± 0.0044 (5-fold)
precision recall f1-score support

    No Chart       0.98      0.54      0.70     17249
     Charted       0.06      0.78      0.12       700

    accuracy                           0.55     17949

macro avg 0.52 0.66 0.41 17949
weighted avg 0.95 0.55 0.68 17949

Odds Ratios (per 1 SD change):
Feature Coef OR
valence 0.3575 1.4298
loudness 0.1672 1.1820
mode 0.1284 1.1370
key 0.0422 1.0431
danceability -0.0389 0.9618
tempo -0.0448 0.9562
liveness -0.2535 0.7761
speechiness -0.3429 0.7097
energy -0.4447 0.6410
acousticness -0.4615 0.6303
instrumentalness -1.2179 0.2958

============================================================
MODEL 2: XGBoost — Chart Entry
============================================================
Best N estimators : 498
Test AUC-ROC : 0.8215
precision recall f1-score support

    No Chart       0.98      0.81      0.89     17249
     Charted       0.12      0.66      0.21       700

    accuracy                           0.80     17949

macro avg 0.55 0.73 0.55 17949
weighted avg 0.95 0.80 0.86 17949

SHAP Feature Importance:
Feature Mean_SHAP
instrumentalness 0.830214
acousticness 0.527616
valence 0.344810
speechiness 0.275638
loudness 0.245082
energy 0.242866
danceability 0.233190
liveness 0.228953
tempo 0.228742
mode 0.113726
key 0.089470

============================================================
MODEL 3: Cox PH — Chart Longevity
============================================================
<lifelines.CoxPHFitter: fitted with 3502 total observations, 0 right-censored observations>
duration col = 'wks_on_chart'
event col = 'event'
penalizer = 0.1
l1 ratio = 0.0
baseline estimation = breslow
number of observations = 3502
number of events observed = 3502
partial log-likelihood = -25009.3117
time fit was run = 2026-03-07 02:32:59 UTC

---

                    coef exp(coef)  se(coef)  coef lower 95%  coef upper 95% exp(coef) lower 95% exp(coef) upper 95%

covariate
danceability -0.0967 0.9078 0.0197 -0.1353 -0.0581 0.8735 0.9435
energy 0.0957 1.1004 0.0265 0.0438 0.1476 1.0448 1.1591
valence 0.0716 1.0742 0.0196 0.0331 0.1101 1.0336 1.1164
tempo 0.0287 1.0291 0.0163 -0.0033 0.0607 0.9967 1.0626
acousticness -0.0734 0.9292 0.0210 -0.1146 -0.0323 0.8917 0.9682
loudness -0.1838 0.8321 0.0224 -0.2276 -0.1400 0.7964 0.8694
speechiness 0.0559 1.0575 0.0167 0.0232 0.0886 1.0235 1.0927
instrumentalness 0.0381 1.0389 0.0167 0.0054 0.0709 1.0054 1.0734
liveness 0.0021 1.0021 0.0173 -0.0318 0.0361 0.9687 1.0367
mode 0.0240 1.0243 0.0165 -0.0083 0.0562 0.9918 1.0578
key 0.0092 1.0092 0.0165 -0.0232 0.0415 0.9771 1.0424

                  cmp to       z      p  -log2(p)

covariate
danceability 0.0000 -4.9126 <5e-05 20.0856
energy 0.0000 3.6139 0.0003 11.6950
valence 0.0000 3.6436 0.0003 11.8607
tempo 0.0000 1.7590 0.0786 3.6697
acousticness 0.0000 -3.4996 0.0005 11.0676
loudness 0.0000 -8.2210 <5e-05 52.1383
speechiness 0.0000 3.3536 0.0008 10.2920
instrumentalness 0.0000 2.2828 0.0224 5.4778
liveness 0.0000 0.1226 0.9024 0.1481
mode 0.0000 1.4569 0.1451 2.7846
key 0.0000 0.5557 0.5784 0.7898

---

Concordance = 0.5494
Partial AIC = 50040.6234
log-likelihood ratio test = 147.6466 on 11 df
-log2(p) of ll-ratio test = 84.1980

Concordance Index (C-stat): 0.5494

── Schoenfeld Residuals Test (H₀: proportional hazards holds) ──
The `p_value_threshold` is set at 0.05. Even under the null hypothesis of no violations, some
covariates will be below the threshold by chance. This is compounded when there are many covariates.
Similarly, when there are lots of observations, even minor deviances from the proportional hazard
assumption will be flagged.

With that in mind, it's best to use a combination of statistical tests and visual tests to determine
the most serious violations. Produce visual plots using `check_assumptions(..., show_plots=True)`
and looking for non-constant lines. See link [A] below for a full example.

<lifelines.StatisticalResult: proportional_hazard_test>
null_distribution = chi squared
degrees_of_freedom = 1
model = <lifelines.CoxPHFitter: fitted with 3502 total observations, 0 right-censored observations>
test_name = proportional_hazard_test

---

                       test_statistic      p  -log2(p)

acousticness km 19.34 <0.005 16.48
rank 19.66 <0.005 16.72
danceability km 0.78 0.38 1.40
rank 1.11 0.29 1.77
energy km 40.59 <0.005 32.31
rank 40.00 <0.005 31.87
instrumentalness km 0.48 0.49 1.04
rank 0.36 0.55 0.87
key km 2.33 0.13 2.98
rank 3.11 0.08 3.69
liveness km 0.47 0.49 1.02
rank 0.41 0.52 0.93
loudness km 33.75 <0.005 27.25
rank 38.16 <0.005 30.51
mode km 29.29 <0.005 23.93
rank 28.44 <0.005 23.31
speechiness km 11.12 <0.005 10.19
rank 12.14 <0.005 10.99
tempo km 1.99 0.16 2.66
rank 1.64 0.20 2.32
valence km 8.87 <0.005 8.43
rank 10.37 <0.005 9.61

1. Variable 'energy' failed the non-proportional test: p-value is <5e-05.

   Advice 1: the functional form of the variable 'energy' might be incorrect. That is, there may be
   non-linear terms missing. The proportional hazard test used is very sensitive to incorrect
   functional forms. See documentation in link [D] below on how to specify a functional form.

   Advice 2: try binning the variable 'energy' using pd.cut, and then specify it in
   `strata=['energy', ...]` in the call in `.fit`. See documentation in link [B] below.

   Advice 3: try adding an interaction term with your time variable. See documentation in link [C]
   below.

2. Variable 'valence' failed the non-proportional test: p-value is 0.0013.

   Advice 1: the functional form of the variable 'valence' might be incorrect. That is, there may be
   non-linear terms missing. The proportional hazard test used is very sensitive to incorrect
   functional forms. See documentation in link [D] below on how to specify a functional form.

   Advice 2: try binning the variable 'valence' using pd.cut, and then specify it in
   `strata=['valence', ...]` in the call in `.fit`. See documentation in link [B] below.

   Advice 3: try adding an interaction term with your time variable. See documentation in link [C]
   below.

3. Variable 'acousticness' failed the non-proportional test: p-value is <5e-05.

   Advice 1: the functional form of the variable 'acousticness' might be incorrect. That is, there
   may be non-linear terms missing. The proportional hazard test used is very sensitive to incorrect
   functional forms. See documentation in link [D] below on how to specify a functional form.

   Advice 2: try binning the variable 'acousticness' using pd.cut, and then specify it in
   `strata=['acousticness', ...]` in the call in `.fit`. See documentation in link [B] below.

   Advice 3: try adding an interaction term with your time variable. See documentation in link [C]
   below.

4. Variable 'loudness' failed the non-proportional test: p-value is <5e-05.

   Advice 1: the functional form of the variable 'loudness' might be incorrect. That is, there may
   be non-linear terms missing. The proportional hazard test used is very sensitive to incorrect
   functional forms. See documentation in link [D] below on how to specify a functional form.

   Advice 2: try binning the variable 'loudness' using pd.cut, and then specify it in
   `strata=['loudness', ...]` in the call in `.fit`. See documentation in link [B] below.

   Advice 3: try adding an interaction term with your time variable. See documentation in link [C]
   below.

5. Variable 'speechiness' failed the non-proportional test: p-value is 0.0005.

   Advice 1: the functional form of the variable 'speechiness' might be incorrect. That is, there
   may be non-linear terms missing. The proportional hazard test used is very sensitive to incorrect
   functional forms. See documentation in link [D] below on how to specify a functional form.

   Advice 2: try binning the variable 'speechiness' using pd.cut, and then specify it in
   `strata=['speechiness', ...]` in the call in `.fit`. See documentation in link [B] below.

   Advice 3: try adding an interaction term with your time variable. See documentation in link [C]
   below.

6. Variable 'mode' failed the non-proportional test: p-value is <5e-05.

   Advice: with so few unique values (only 2), you can include `strata=['mode', ...]` in the call in
   `.fit`. See documentation in link [E] below.

---

[A] https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html
[B] https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html#Bin-variable-and-stratify-on-it
[C] https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html#Introduce-time-varying-covariates
[D] https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html#Modify-the-functional-form
[E] https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html#Stratification

============================================================
MODEL 3b: Log-OLS — Longevity (robustness check)
============================================================
Test R²: 0.0078
Feature Coef
danceability 0.0595
loudness 0.0539
valence 0.0319
energy 0.0052
mode 0.0026
acousticness -0.0035
liveness -0.0101
instrumentalness -0.0238
tempo -0.0383
key -0.0386
speechiness -0.0771

Generating 8 figures...

============================================================
MODEL PERFORMANCE SUMMARY
============================================================
Model Task Metric Score CV / note
Logistic Regression Chart Entry (binary) AUC-ROC 0.7066 0.7150 ± 0.0044 (5-fold)
XGBoost Chart Entry (binary) AUC-ROC 0.8215 early stop @ iter 498
Cox PH Longevity (survival) C-statistic 0.5494 penalizer=0.1
Log-OLS Longevity (OLS) R² 0.0078 outcome = log1p(wks_on_chart)

============================================================
All outputs → /Users/alexwurm/Documents/Stanford/Classwork/OIT-367/outputs/
CSVs : vif_table, logistic_odds_ratios, xgboost_shap_importance,
cox_summary, ols_longevity_coefficients, model_performance_summary
Figures: fig1–fig8 (PNG, 150 dpi)
============================================================

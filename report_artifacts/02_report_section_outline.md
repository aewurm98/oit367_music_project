# Report Section Outline
**OIT367 Final Report: Predicting Billboard Hot 100 Chart Entry and Longevity**
**Stanford GSB, Winter 2026 | Wurm · Chen · Barli · Taruno**

---

## Formatting Constraints (Per Course Instructions)

- Maximum 2,000 words in report body (tables, figures, and appendix do not count)
- Maximum 7 tables and figures combined in the main body
- Page 1: one-page executive summary with team names
- Required sections: Introduction, Data, Methods, Results, Conclusions, Appendix (optional)

**Recommended main-body figure/table budget (7 total):**

| Slot | Content | Rationale |
|---|---|---|
| 1 | Model performance summary table | Anchors all four models; readers need this immediately |
| 2 | Fig3: ROC curves (LR + XGB) | Standard classification validity evidence |
| 3 | Fig9: Precision-recall curves (LR + XGB) | Primary metric at 2.75% positive rate; cannot omit |
| 4 | Fig5: SHAP importance (XGBoost) | Key interpretability figure; shows feature hierarchy clearly |
| 5 | Fig4: Odds ratios forest plot (LR) | Interpretable coefficients; complements SHAP |
| 6 | Fig7: Kaplan-Meier curves | Visual entry point for survival section |
| 7 | VIF table or condensed LR + Cox comparison table | Demonstrates methodological rigor |

Reserve for appendix: fig1 (class balance), fig2 (correlation heatmap), fig6 (Cox hazard ratios), fig8 (longevity distribution), full cox_summary.csv, genre_chart_rates.csv, OLS coefficients.

---

## Page 1: Executive Summary

**Content:**
- One paragraph: the business question. A&R teams, labels, and streaming platforms need to identify potential hits before they chart, and understand what differentiates long-charting tracks from one-week entries. This study builds predictive models using publicly available Spotify audio features and artist-level commercial signals.
- One paragraph: data. 78,390 unique Spotify tracks, of which 2,157 appeared on the Billboard Hot 100 between 1958 and 2024 (2.75% positive rate). Enriched with Last.fm listener counts and VADER lyric sentiment scores.
- One paragraph: key findings. (1) Artist commercial track record (peak popularity and audience size) outweighs all audio features for chart entry prediction. (2) Instrumentalness is the dominant audio barrier -- vocal content is essential for the Hot 100. (3) Happier songs are more likely to chart, but the streaming era has compressed the average chart lifespan. (4) XGBoost achieves ROC-AUC = 0.966 and PR-AUC = 0.440 (16x above random); Cox PH concordance index = 0.753.
- One paragraph: recommendations for stakeholders -- record labels evaluating new signings, streaming platforms building editorial playlists, and artists choosing between vocal and instrumental releases.

**Word target: 250 words.**

---

## Section 1: Introduction (~300 words)

**Purpose:** Establish the business context, motivate the research questions, and state what the paper does and does not address.

**Content:**

*Background and motivation.* The Billboard Hot 100 aggregates radio airplay, digital downloads, and streaming activity into a single weekly ranking. Predicting which songs reach the chart -- and for how long -- has direct commercial value for record labels (A&R investment decisions), streaming platforms (editorial and algorithmic playlist curation), and artists (release strategy). Prior academic work on hit prediction has largely relied on audio features alone (e.g., Pachet and Roy 2008; Ni et al. 2011); this study extends that line of work by incorporating artist-level commercial signals and lyric sentiment.

*Research questions.* This paper addresses two related questions:
1. Can observable pre-release features predict whether a new Spotify track will appear on the Billboard Hot 100? (Binary classification)
2. Among tracks that chart, what features predict longer chart longevity? (Survival analysis)

*Scope.* The analysis uses historical chart data (1958-2024) and cannot directly predict future chart performance for tracks not yet released. The models are trained on observable song and artist characteristics at the time of release; they do not incorporate post-release streaming volume, which would constitute label information.

*Why these models?* Logistic regression provides interpretable odds ratios for business reporting. XGBoost captures nonlinear feature interactions that logistic regression misses. Cox proportional hazards is the appropriate survival model for right-censored chart data (some tracks may still be charting at the data cutoff). OLS on log-transformed weeks serves as a distributional robustness check.

**Note:** Avoid using "predict" to imply causal inference. These are associative models trained on historical patterns.

---

## Section 2: Data (~350 words)

**Purpose:** Describe the data sources, construction pipeline, and key characteristics. Justify methodological choices in the dataset build.

**Content:**

*Sources.* Two public Kaggle datasets form the base corpus: (1) Spotify Tracks Dataset (~114k rows), containing audio features for tracks cataloged on Spotify, and (2) Billboard Hot 100 weekly snapshots (~690k rows, 1958-2024), joined via a pre-matched intermediate file that resolves track names to Spotify IDs. Three external sources augment the base: Spotify API artist profiles (peak popularity, catalog size), MusicBrainz/Last.fm (listener counts and artist country), and a Billboard lyrics dataset processed with the VADER sentiment analyzer.

*Dataset construction.* The Spotify source contains duplicate rows when Spotify assigns a track to multiple genres. The pipeline deduplicates to one row per `track_id` (retaining one genre label per track for survival analysis stratification), then performs a left join with aggregated Billboard data -- retaining all 78,390 Spotify tracks and assigning `is_charted = 1` only to the 2,157 that appeared on the Hot 100. A left join is critical; an inner join (used in early pipeline iterations) would retain only charted tracks, making classification impossible.

*Cross-ID deduplication.* The raw Billboard join produced 3,502 charted track IDs, but ~1,345 of these represent the same song under different Spotify IDs (album vs. single releases, regional variants, explicit vs. clean editions). These near-duplicates with identical audio features would artificially inflate model performance. The final dataset collapses these to 2,157 truly unique charted songs.

*Class imbalance.* 2,157 of 78,390 tracks are charted (2.75% positive rate). This imbalance is real and meaningful -- the overwhelming majority of released music never reaches the Hot 100. Models are tuned accordingly (see Methods), and PR-AUC is reported as the primary classification metric. At 2.75% positive rate, a random classifier achieves PR-AUC = 0.0275.

*Descriptive statistics.* Include a brief table or inline statistics: median track duration, fraction explicit, most common genre, distribution of weeks on chart (median X weeks, range 1-90+). Reference Fig8 (longevity distribution) in appendix.

*Lyric sentiment coverage note.* VADER sentiment features are available for 856 of 2,157 charted tracks (40.1%) via title-and-artist matching. This subset is used exclusively in the survival models; including it in the classification models would exclude 59.9% of positive-class training examples.

---

## Section 3: Methods (~350 words)

**Purpose:** Describe the four models, justify each choice, and explain how methodological challenges were addressed.

**Content:**

*Feature engineering.* All continuous features are standardized with `StandardScaler` before logistic regression so coefficients are on a per-standard-deviation scale and odds ratios are directly comparable. `duration_ms` is converted to `duration_min` and capped at 10 minutes to exclude outlier podcasts and audiobooks. `lastfm_listeners_log` and `artist_scrobbles_log` use `log1p` transformation to reduce right skew. `is_us_artist` is conservative: unknown nationality is coded as 0 (non-US).

*Multicollinearity check.* Variance inflation factors (VIF) are computed for all features before fitting logistic regression. `energy` (VIF=15.07) and `danceability` (VIF=12.41) are removed. The final VIF table is included in the appendix. Two artist popularity metrics (`artist_peak_popularity` VIF=21.25, `artist_popularity_api` VIF=14.12) remain in the model despite elevated VIFs because their combined directional signal is substantively meaningful; individual coefficient estimates for this pair are treated as unreliable and are not interpreted independently.

*Model 1: Logistic Regression.* `class_weight="balanced"` scales the minority class loss by approximately 25x to compensate for the 2.75% positive rate. 5-fold cross-validation confirms stability (AUC-ROC 0.9139 +/- 0.0037).

*Model 2: XGBoost + SHAP.* `scale_pos_weight = n_negative / n_positive = 24.6`. 500 estimators, learning rate 0.05, max depth 5, early stopping at iteration 193. SHAP (SHapley Additive exPlanations) values decompose each prediction into per-feature contributions, enabling feature importance ranking that accounts for nonlinear interactions and feature correlations -- limitations of standard gain-based importance metrics.

*Model 3: Cox Proportional Hazards.* `wks_on_chart` is the survival time; all observations are treated as right-censored (no confirmed "end" is recorded for tracks still on chart at data cutoff). `strata=["mode"]` because `mode` fails the Schoenfeld residuals test (p < 0.05), violating the proportional hazards assumption. Stratification accommodates this by estimating separate baseline hazard functions for major vs. minor key songs. Ridge penalizer = 0.1 stabilizes coefficient estimates. Concordance index is the primary fit metric (analogous to AUC-ROC for survival models).

*Model 4: Log-OLS.* OLS regression on `log1p(wks_on_chart)` using the same feature set as Cox PH. Serves as a distributional robustness check -- if Cox and OLS yield consistent directional findings for key coefficients, the conclusions are not an artifact of the proportional hazards assumption.

---

## Section 4: Results (~400 words)

**Purpose:** Report model performance, key findings, and statistical significance. Address generalizability.

**Subsections:**

### 4.1 Chart Entry: Classification Performance

Report ROC-AUC and PR-AUC for both models in the performance table (Table 1). Emphasize PR-AUC as the primary metric given class imbalance. Reference Fig3 (ROC curves) and Fig9 (precision-recall curves).

Key performance numbers:
- LR: ROC-AUC = 0.914, PR-AUC = 0.275 (10.0x above random baseline of 0.0275)
- XGBoost: ROC-AUC = 0.966, PR-AUC = 0.440 (16.0x above random)
- LR 5-fold CV: 0.914 +/- 0.004 (low variance confirms generalizability)

The 5.1-point ROC-AUC gap between XGBoost and logistic regression confirms substantial nonlinearity in the chart entry problem that a linear model cannot capture.

**Pre- vs. post-enrichment comparison.** Audio features alone yield LR ROC-AUC = 0.711 and XGBoost ROC-AUC = 0.834. Adding artist commercial features raises these to 0.914 and 0.966, respectively. This improvement directly quantifies the marginal value of the artist enrichment.

### 4.2 Feature Importance and Interpretation

Reference Fig4 (odds ratios) and Fig5 (SHAP importance). Describe the top findings:

*Artist track record and audience size dominate all audio features.* `artist_peak_popularity` is the top predictor in both models (SHAP = 1.618, OR = 2.84 per 1 SD). `lastfm_listeners_log` ranks second (SHAP = 1.153, OR = 2.17). The top two predictors are both artist-level commercial variables; the first audio feature ranks third.

*Instrumentalness is the dominant audio barrier.* SHAP = 0.629, OR = 0.34 per 1 SD -- instrumental tracks are 66% less likely to chart per one-standard-deviation increase in instrumentalness. The Hot 100 is a vocal and lyric-driven chart.

*Valence is the strongest positive audio predictor.* OR = 1.44 per 1 SD. Happier-sounding songs are more likely to appear on the chart.

*US artist origin is a meaningful signal.* OR = 1.91 (logistic regression), SHAP rank 4 (XGBoost). US-based artists are approximately 91% more likely to chart per 1 SD, reflecting the Hot 100's structural bias toward domestic radio markets.

**Speechiness nonlinearity note.** Logistic regression assigns OR = 0.71 to `speechiness`, but XGBoost ranks it 9th by SHAP (above `duration_min` and `loudness`). This divergence reflects a nonlinear relationship: moderate speechiness (typical rap/hip-hop) is associated with charting, while extreme speechiness (spoken word) is not.

### 4.3 Chart Longevity: Survival Analysis

Report Cox PH concordance index = 0.753. Reference Fig7 (Kaplan-Meier). Describe the top findings:

*Decade index is the strongest longevity predictor.* HR = 0.514 (p < 0.00001). Each decade later in chart history corresponds to a 49% increase in per-week exit hazard -- songs in the streaming era rotate off the chart faster on average. This finding is robust to the OLS specification: although OLS shows a positive coefficient for `decade_idx` (longer total weeks in recent decades for mega-hits), the Cox per-week hazard is the correct framing for the median song.

*Current artist popularity sustains longevity.* `artist_popularity_api` HR = 1.085 (p = 0.030). Active label promotion and algorithmic playlist placement associated with higher current popularity extend chart runs.

*Negative lyric sentiment is directionally associated with shorter chart runs.* `sentiment_neg` HR = 1.088, p = 0.051 (borderline significance, n = 856). Consistent in sign across Cox and OLS. Report as a directional trend, not a confirmed finding; the 40.1% lyric match rate limits statistical power.

*Schoenfeld violations.* Four of 19 Cox features fail the Schoenfeld proportional hazards test: `decade_idx`, `acousticness`, `duration_min`, and `sentiment_pos`. This means the proportional hazards assumption does not hold uniformly for these covariates. OLS robustness confirms `decade_idx` findings directionally; specific coefficient magnitudes from Cox should be interpreted with caution for these four features.

### 4.4 Generalizability

The 5-fold CV result (AUC-ROC 0.914 +/- 0.004 for logistic regression) confirms that classification findings are not an artifact of the particular train/test split. However, both training and evaluation data are drawn from the same historical distribution (1958-2024); performance on future tracks may differ as chart dynamics continue to evolve. The streaming era structural shift -- captured by `decade_idx` -- suggests that models trained primarily on pre-streaming data may understate the importance of playlist placement and short-form attention dynamics for contemporary releases.

---

## Section 5: Conclusions (~200 words)

**Purpose:** Synthesize findings into actionable takeaways. Address limitations.

**Content:**

*Principal findings (three to four sentences).* Artist commercial track record and pre-existing audience size explain chart entry better than any combination of audio features. Instrumentalness is the only audio feature with a large and consistent directional effect: vocal content is nearly a prerequisite for the Hot 100. The streaming era has fundamentally changed chart longevity dynamics -- average rotation has accelerated even as occasional mega-hits accumulate unprecedented week counts.

*Recommendations.* For A&R teams: the models confirm that investing in artists with established audiences provides more predictive return than optimizing audio production characteristics. For playlist editors: instrumentalness and valence are the most actionable audio signals for editorial sorting. For researchers: future work should incorporate streaming velocity (first-week Spotify streams) and social media engagement metrics, which likely carry stronger predictive signal than static audio features.

*Limitations.* The Billboard Hot 100 has a structural US and English-language bias not fully captured by `is_us_artist`. Lyric sentiment analysis covers only 40.1% of charted tracks. The two artist popularity metrics are highly collinear; causal interpretation of their individual effects is not supported by this design.

---

## Appendix (Optional)

Recommended appendix items:
- Fig1: Class balance bar chart (2.75% positive rate visualization)
- Fig2: Correlation heatmap (supports VIF discussion)
- Fig6: Cox PH hazard ratio plot (full coefficient detail)
- Fig8: Weeks-on-chart distribution histogram
- Full Cox PH coefficient table (`cox_summary.csv` formatted)
- Genre chart rate table: top and bottom 10 genres by chart rate
- OLS longevity coefficients (`ols_longevity_coefficients.csv`)
- `vif_table.csv` if not included in main body
- Full SHAP importance table (all 15 features ranked)
- Brief note on pipeline version history (inner join bug, cross-ID dedup rationale)

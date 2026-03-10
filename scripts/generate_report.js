const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, VerticalAlign, PageNumber, PageBreak, LevelFormat,
  TabStopType, TabStopPosition
} = require('docx');
const fs = require('fs');
const path = require('path');

const IMG = '/sessions/hopeful-optimistic-shannon/project_run/outputs';

// ── Helpers ────────────────────────────────────────────────────────────────
const W = 9360; // content width DXA (8.5" - 2" margins)

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };
const noBorder = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const noBorders = { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder };

function txt(text, opts = {}) {
  return new TextRun({ text, font: "Arial", size: opts.size || 20, bold: opts.bold || false,
    italics: opts.italics || false, color: opts.color || undefined });
}
function para(children, opts = {}) {
  return new Paragraph({
    children: Array.isArray(children) ? children : [children],
    alignment: opts.align || AlignmentType.LEFT,
    spacing: { before: opts.before ?? 80, after: opts.after ?? 80 },
    ...opts
  });
}
function heading1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    children: [new TextRun({ text, font: "Arial", size: 28, bold: true, color: "1F3864" })],
    spacing: { before: 240, after: 120 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "2E75B6", space: 4 } },
  });
}
function heading2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    children: [new TextRun({ text, font: "Arial", size: 22, bold: true, color: "2E75B6" })],
    spacing: { before: 180, after: 80 },
  });
}
function figCaption(text) {
  return new Paragraph({
    children: [new TextRun({ text, font: "Arial", size: 18, italics: true, color: "595959" })],
    alignment: AlignmentType.CENTER,
    spacing: { before: 60, after: 180 },
  });
}
function tableCaption(text) {
  return new Paragraph({
    children: [new TextRun({ text, font: "Arial", size: 18, bold: true, color: "1F3864" })],
    spacing: { before: 180, after: 60 },
  });
}
function img(filename, emuW, emuH) {
  const data = fs.readFileSync(path.join(IMG, filename));
  return new Paragraph({
    children: [new ImageRun({
      type: "png", data,
      transformation: { width: emuW / 9144, height: emuH / 9144 },
      altText: { title: filename, description: filename, name: filename }
    })],
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 60 },
  });
}

function hdrCell(text, w, opts = {}) {
  return new TableCell({
    borders,
    width: { size: w, type: WidthType.DXA },
    shading: { fill: "2E75B6", type: ShadingType.CLEAR },
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      children: [new TextRun({ text, font: "Arial", size: opts.size || 18, bold: true, color: "FFFFFF" })],
      alignment: opts.align || AlignmentType.LEFT,
      spacing: { before: 0, after: 0 },
    })]
  });
}
function dataCell(text, w, opts = {}) {
  return new TableCell({
    borders,
    width: { size: w, type: WidthType.DXA },
    shading: { fill: opts.fill || "FFFFFF", type: ShadingType.CLEAR },
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      children: [new TextRun({ text, font: "Arial", size: 18, bold: opts.bold || false, italics: opts.italics || false, color: opts.color || "000000" })],
      alignment: opts.align || AlignmentType.LEFT,
      spacing: { before: 0, after: 0 },
    })]
  });
}

// ── Executive Summary ──────────────────────────────────────────────────────
const execSummary = [
  new Paragraph({
    children: [new TextRun({ text: "Predicting Billboard Hot 100 Chart Entry and Longevity", font: "Arial", size: 40, bold: true, color: "1F3864" })],
    alignment: AlignmentType.CENTER,
    spacing: { before: 480, after: 120 },
  }),
  new Paragraph({
    children: [new TextRun({ text: "OIT367: Business Intelligence from Big Data  |  Stanford GSB  |  Winter 2026", font: "Arial", size: 22, italics: true, color: "595959" })],
    alignment: AlignmentType.CENTER,
    spacing: { before: 0, after: 80 },
  }),
  new Paragraph({
    children: [new TextRun({ text: "Alex Wurm  |  Ben Chen  |  Vivian Barli  |  Valerie Taruno", font: "Arial", size: 22, bold: true, color: "2E75B6" })],
    alignment: AlignmentType.CENTER,
    spacing: { before: 0, after: 400 },
  }),
  new Paragraph({
    children: [new TextRun({ text: "EXECUTIVE SUMMARY", font: "Arial", size: 26, bold: true, color: "1F3864" })],
    alignment: AlignmentType.CENTER,
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "2E75B6", space: 4 } },
    spacing: { before: 0, after: 200 },
  }),
  para([
    txt("Business Problem. ", { bold: true }),
    txt("The Billboard Hot 100 aggregates radio airplay, digital downloads, and streaming activity into a single weekly ranking. For record labels, streaming platforms, and artists, predicting which songs will reach the chart and how long they will remain there carries direct commercial value. Prior research has largely relied on audio features alone; this study asks whether combining Spotify audio features with artist-level commercial signals meaningfully improves predictive performance.")
  ], { before: 0, after: 120 }),
  para([
    txt("Data. ", { bold: true }),
    txt("We assembled 78,390 unique Spotify tracks from two public Kaggle datasets: the Spotify Tracks Dataset and weekly Billboard Hot 100 snapshots spanning 1958 to 2024. Of these, 2,157 tracks (2.75%) appeared on the Hot 100. Three external enrichment sources were added: Spotify API artist profiles, MusicBrainz/Last.fm listener counts, and VADER lyric sentiment scores (available for 856 charted tracks, 40.1%).")
  ], { before: 0, after: 120 }),
  para([
    txt("Principal Findings. ", { bold: true }),
    txt("Artist commercial track record and pre-existing audience size are the dominant predictors of chart entry, outperforming all audio features. XGBoost with full enrichment achieves AUC-ROC = 0.966 and PR-AUC = 0.441, representing a 16-fold improvement over random chance. Instrumentalness is the primary audio barrier: vocal content is nearly a prerequisite for the Hot 100 (OR = 0.34 per 1 SD). For longevity, the streaming era has structurally compressed chart rotation (HR = 0.514 per decade, p < 0.001), with mega-hits masking this effect in simple averages.")
  ], { before: 0, after: 120 }),
  para([
    txt("Recommendations. ", { bold: true }),
    txt("Label A&R teams should treat the XGBoost model at a threshold of 0.10 to 0.30 as a screening tool: the model flags 14 to 20 percent of candidate tracks while recovering 96 to 98 percent of eventual hits, enabling prioritized human review rather than wholesale prediction. Audio production optimization has far lower return than artist roster and audience development. Streaming platform editors can use instrumentalness and valence as the most actionable audio signals for editorial sorting.")
  ], { before: 0, after: 0 }),
  new Paragraph({ children: [new PageBreak()] }),
];

// ── Introduction ───────────────────────────────────────────────────────────
const intro = [
  heading1("1. Introduction"),
  para([
    txt("The music industry generated $28.6 billion in global recorded revenue in 2023, with streaming accounting for 67 percent of that total. In this environment, the Billboard Hot 100 serves as the primary commercial benchmark in the United States, synthesizing radio, download, and streaming signals into a single weekly chart. Label A&R divisions and streaming platform editors make resource allocation decisions based in part on which emerging artists and tracks are likely to reach this benchmark.")
  ], { before: 0 }),
  para([txt("Academic research on hit prediction dates to at least Pachet and Roy (2008), who showed that audio feature classifiers trained on chart data could distinguish hits from non-hits at above-chance accuracy. Subsequent work has largely corroborated this finding while highlighting its limitations: audio features explain some of the variance in chart success, but the lion's share is explained by factors exogenous to the recording itself, such as marketing spend, label affiliation, and the pre-existing popularity of the artist.")]),
  para([txt("This study extends that literature in two directions. First, we combine audio features with artist-level commercial signals (historical chart performance, catalog size, and cross-platform audience reach via Last.fm) to quantify the marginal predictive contribution of each signal class. Second, we address not just chart entry but chart longevity, using survival analysis to model how long a charted track remains on the Hot 100 as a function of its audio and artist characteristics.")]),
  para([txt("Two research questions guide the analysis:")]),
  new Paragraph({
    numbering: { reference: "numbers", level: 0 },
    children: [txt("Can observable pre-release features predict whether a Spotify track will appear on the Billboard Hot 100? (Binary classification)")],
    spacing: { before: 60, after: 40 },
  }),
  new Paragraph({
    numbering: { reference: "numbers", level: 0 },
    children: [txt("Among charted tracks, what features predict how long a track remains on the chart? (Survival analysis)")],
    spacing: { before: 40, after: 120 },
  }),
  para([txt("Four working hypotheses follow from these questions. H1: Artist-level commercial features — specifically pre-existing audience reach (Last.fm listeners) and historical chart performance (peak popularity) — will provide meaningful incremental predictive lift above audio features alone, because promotional infrastructure and star power are primary determinants of chart outcomes independent of a song's intrinsic audio properties. H2: Instrumentalness will be the dominant audio barrier to chart entry, because the Hot 100 is structurally oriented toward vocal and lyric-driven radio programming. H3: Chart rotation speed has increased in the streaming era relative to the radio era, because digital platform abundance accelerates song cycling for the median track. H4: Among charted tracks, songs by commercially active artists will sustain longer chart runs, reflecting ongoing editorial and promotional support beyond the initial release window.")]),
  para([txt("All models are trained on historical patterns from 1958 to 2024 and are associative rather than causal. The analysis uses a stratified 80/20 train-test split with 5-fold cross-validation for logistic regression stability and is implemented in Python 3.11 (scikit-learn, XGBoost 2.1.3, SHAP, lifelines; random seed = 42 throughout). The full reproducible pipeline is contained in run_all_v6.py.")]),
];

// ── Data ───────────────────────────────────────────────────────────────────
const dataSection = [
  heading1("2. Data"),
  heading2("2.1 Sources and Selection Rationale"),
  para([
    txt("The base corpus combines two public Kaggle datasets selected for comprehensive temporal coverage and exact Spotify track ID linkage. The "),
    txt("Spotify Tracks Dataset", { italics: true }),
    txt(" [1] contains approximately 114,000 rows representing unique track-genre pairs with Spotify audio features for each track. The "),
    txt("Billboard Hot 100 (1958–2024)", { italics: true }),
    txt(" dataset [2] contains weekly chart snapshots aggregated by an intermediate pre-matched file that resolves artist and title strings to Spotify track IDs. These two sources were selected because they are the only publicly available datasets that provide both granular Spotify audio features and full Billboard chart history with a common track-level identifier. Other sources evaluated — including a best-selling artists catalog [5] and a Universal Music Group label assignment file — were tested but excluded due to sparse coverage (under 5% track match rate against the 78,390-track universe)."),
  ]),
  para([
    txt("Three external sources augment the base. The "),
    txt("Spotify Web Developer API", { italics: true }),
    txt(" [6] supplies artist-level commercial signals (historical peak popularity score, current popularity score, and catalog track count) for matched artists. The "),
    txt("Music Artists Popularity", { italics: true }),
    txt(" dataset [4], a MusicBrainz and Last.fm data extract covering 1.47 million artists, provides cross-platform listener counts and country of origin; Last.fm listeners are log-transformed to correct for heavy right skew. The "),
    txt("Lyrics for Billboard Top 100 Songs 1946–2022", { italics: true }),
    txt(" dataset [3] supplies track lyrics processed with the VADER sentiment analyzer (Hutto & Gilbert, 2014) for compound, positive, and negative sentiment scores; this dataset covers 856 of 2,157 charted tracks (40.1%) after cross-ID deduplication. Librosa spectral feature extraction was designed but not executed within the project timeline due to API rate limit constraints on preview URL retrieval."),
  ]),
  heading2("2.2 Dataset Construction"),
  para([
    txt("The Spotify source contains duplicate rows when a track is assigned to multiple genres. The pipeline deduplicates to one row per track ID, retaining one genre label. It then performs a "),
    txt("left join", { italics: true }),
    txt(" with aggregated Billboard data, retaining all 78,390 unique Spotify tracks and assigning is_charted = 1 only to the 2,157 that appeared on the Hot 100. A left join is critical; an inner join used in an earlier pipeline iteration retained only charted tracks, producing a 100 percent positive rate and making binary classification impossible.")
  ]),
  para([txt("A second deduplication step collapsed 3,502 initially matched charted track IDs to 2,157 truly unique songs, removing album versus single release pairs and regional variants that share identical audio features. Retaining these near-duplicates inflated model performance artificially by creating trivially similar training and test instances.")]),
  heading2("2.3 Class Imbalance and Evaluation Metric"),
  para([txt("2,157 of 78,390 tracks are charted (2.75% positive rate). This imbalance is real and meaningful; the overwhelming majority of released music never reaches the Hot 100. At a 2.75% positive rate, a random classifier achieves PR-AUC = 0.0275. For this reason, Precision-Recall AUC (PR-AUC) is reported as the primary classification metric alongside ROC-AUC throughout this paper. A model that achieves PR-AUC = 0.275 against a 0.0275 baseline represents a 10-fold lift over random.")]),
  heading2("2.4 Feature Selection"),
  para([txt("The candidate feature pool initially included 18 variables. Three were removed iteratively for variance inflation factor (VIF) exceeding 10: energy (VIF = 15.1, collinear with loudness), danceability (VIF = 12.4, collinearity hub with tempo and valence), and tempo (VIF = 10.7 after danceability removal). Four additional features available in the dataset were excluded for insufficient universe coverage: is_male_artist, artist_age, and is_mainstream_genre are populated for only 58% of charted tracks and essentially 0% of non-charted tracks, failing the >50% non-null threshold required for classification model inclusion. Time signature (VIF = 19.1) was excluded for multicollinearity. The final classification feature set contains 15 variables: 10 audio features from Spotify (valence, acousticness, loudness, speechiness, instrumentalness, liveness, mode, key, explicit, track duration) and 5 artist-level features (historical peak popularity, current popularity, catalog size, Last.fm log-listeners, US artist indicator). Full feature definitions are provided in Appendix F. The longevity models add decade_idx (an ordinal index from 0 = 1950s to 7 = 2020s, derived from each charted track's first chart appearance date) and four VADER lyric sentiment features, available for 856 of 2,157 charted tracks.")]),
];

// ── Methods ────────────────────────────────────────────────────────────────
const methods = [
  heading1("3. Methods"),
  heading2("3.1 Model Selection"),
  para([txt("Model selection was driven by the distinct statistical structure of each research question. For chart entry (RQ1), the outcome is binary with severe class imbalance (2.75% positive rate), ruling out OLS-family approaches. Logistic regression was selected as the interpretable baseline: standardized coefficients yield per-standard-deviation odds ratios that are directly comparable across all features and straightforward to communicate to non-technical stakeholders. XGBoost was selected as the high-performance nonlinear alternative because gradient-boosted trees handle feature interactions, nonlinearities, and skewed input distributions without requiring distributional assumptions, and the accompanying SHAP framework provides post-hoc feature importance that is consistent with the model's actual predictions rather than a linear proxy. Alternatives such as random forests were considered but not reported separately; preliminary runs showed negligible performance difference from XGBoost with higher computational cost. Neural approaches were not pursued given the dataset size (n = 78,390) and the interpretability requirements of the use case.")]),
  para([txt("For chart longevity (RQ2), the outcome is a duration with right-censoring: the Billboard dataset records a running week counter that may stop before a track's true chart exit. Cox proportional hazards is the standard method for right-censored survival data and was selected as the primary longevity model. Log-OLS on log1p-transformed weeks was included as a robustness check. If the directional findings are consistent across both specifications, they are not artifacts of the proportional hazards assumption. An accelerated failure time (AFT) model would be an alternative; given that four covariates fail the Schoenfeld test and are documented in Appendix B, an AFT model or a time-varying covariate extension of Cox PH is the recommended next step for future work.")]),
  heading2("3.2 Feature Engineering and Preprocessing"),
  para([txt("Continuous audio features arrive on heterogeneous scales (valence and acousticness are bounded [0, 1]; loudness is in decibels; duration is in milliseconds). All continuous features are standardized using sklearn's StandardScaler before logistic regression fitting, so all reported odds ratios are per-one-standard-deviation change and are directly comparable across features. XGBoost is scale-invariant by construction but receives the same standardized inputs for consistency. Track duration is converted from milliseconds to minutes and capped at 10 minutes to exclude podcast episodes and audiobooks present in the Kaggle Spotify source; this cap affects fewer than 0.1% of tracks.")]),
  para([
    txt("Last.fm listener counts span several orders of magnitude (right-skewed). These are log-transformed as log1p(listeners) before model inclusion, compressing the scale and reducing leverage from a small number of globally dominant artists. Similarly, "),
    txt("decade_idx", { italics: true }),
    txt(" is constructed as an ordinal integer: floor((chart_entry_year − 1950) / 10), yielding values 0 (1950s) through 7 (2020s). This variable is only defined for charted tracks (non-charted tracks have no chart_entry_date) and is therefore restricted to the Cox PH and Log-OLS longevity models. The is_us_artist binary is conservatively imputed: artists with unknown nationality (not present in the MusicBrainz country field) are coded 0 (non-US) rather than excluded, ensuring 100% feature coverage at the cost of a downward bias on the US artist effect.")
  ]),
  heading2("3.3 Research Question 1: Chart Entry Classification"),
  para([txt("Both classifiers are trained on a stratified 80/20 random split (train n = 62,712; test n = 15,678; random seed = 42). Stratification preserves the 2.75% positive rate in both partitions. Class imbalance is addressed in logistic regression via class_weight=balanced (positive class up-weighted by a factor of ~35x) and in XGBoost via scale_pos_weight = 24.6 (the ratio of negative to positive training examples). Five-fold stratified cross-validation is used for logistic regression to confirm out-of-sample stability; mean CV AUC-ROC = 0.914 ± 0.004. XGBoost uses early stopping (patience = 50 rounds, monitored on a 10% held-out validation split drawn from the training set) to prevent overfitting; training stops at iteration 189.")]),
  para([txt("SHAP (SHapley Additive exPlanations) values are computed on the held-out test set using the TreeExplainer. Mean absolute SHAP value per feature is reported as the feature importance metric because it accounts for nonlinear interactions and co-occurrences that a permutation importance measure would miss. Audio-only baseline models are trained with identical hyperparameters restricted to the 10 audio features, providing a controlled experimental comparison of the marginal predictive value of artist enrichment features.")]),
  heading2("3.4 Research Question 2: Chart Longevity Survival Analysis"),
  para([txt("Chart longevity is the number of weeks a charted track remains on the Hot 100. This is modeled as a survival outcome: wks_on_chart is the survival time, all observations are treated as right-censored (no confirmed removal is recorded in the dataset for tracks still charting at the data cutoff), and the event indicator is set to 1 for all observations. The analysis is restricted to the 856 charted tracks with lyric sentiment data, enabling a consistent comparison between Cox PH and Log-OLS.")]),
  para([txt("Cox proportional hazards (Cox PH) is the primary longevity model. It estimates the per-week hazard of exiting the chart as a function of standardized covariates. The proportional hazards assumption is verified for each covariate using Schoenfeld residuals (Appendix B). Mode (major vs. minor key) fails this test and is handled by stratification: separate baseline hazards are estimated for major and minor key tracks rather than imposing a proportional effect. A ridge penalizer of 0.1 is applied for coefficient stability given the modest sample size (n = 856). The concordance index (C-statistic) is the primary fit metric; it is equivalent to the probability that the model assigns a higher risk score to a track that exits the chart sooner, analogous to ROC-AUC for survival data.")]),
  para([txt("Log-OLS regresses log1p(wks_on_chart) on the same feature set. The log transformation is motivated by the heavily right-skewed distribution of chart weeks (median = 8 weeks; mean = 17.6 weeks; max = 90+ weeks), which violates OLS homoscedasticity assumptions on the raw scale. Both models use the same 856 lyric-matched tracks to allow direct comparison of coefficient directions.")]),
];

// ── Results ────────────────────────────────────────────────────────────────

// Table 1: Model performance summary
const perfTable = [
  tableCaption("Table 1. Model Performance: Audio-Only Baseline vs. Full Model (Audio + Artist Features)"),
  new Table({
    width: { size: W, type: WidthType.DXA },
    columnWidths: [1900, 1500, 1000, 1200, 1200, 1200, 1360],
    rows: [
      new TableRow({
        children: [
          hdrCell("Model", 1900),
          hdrCell("Task", 1500),
          hdrCell("Metric", 1000),
          hdrCell("Audio Only", 1200, { align: AlignmentType.CENTER }),
          hdrCell("Full Model", 1200, { align: AlignmentType.CENTER }),
          hdrCell("PR-AUC (Full)", 1200, { align: AlignmentType.CENTER }),
          hdrCell("CV / Notes", 1360),
        ],
        tableHeader: true,
      }),
      new TableRow({
        children: [
          dataCell("Logistic Regression", 1900),
          dataCell("Chart Entry", 1500),
          dataCell("AUC-ROC", 1000),
          dataCell("0.737", 1200, { align: AlignmentType.CENTER }),
          dataCell("0.914", 1200, { align: AlignmentType.CENTER, bold: true }),
          dataCell("0.275", 1200, { align: AlignmentType.CENTER }),
          dataCell("0.914 ± 0.004 (5-fold CV)", 1360),
        ],
      }),
      new TableRow({
        children: [
          dataCell("XGBoost", 1900),
          dataCell("Chart Entry", 1500),
          dataCell("AUC-ROC", 1000),
          dataCell("0.773", 1200, { align: AlignmentType.CENTER }),
          dataCell("0.966", 1200, { align: AlignmentType.CENTER, bold: true }),
          dataCell("0.441", 1200, { align: AlignmentType.CENTER }),
          dataCell("Early stop @ iter 189", 1360),
        ],
      }),
      new TableRow({
        children: [
          dataCell("Cox PH", 1900, { fill: "F2F2F2" }),
          dataCell("Longevity", 1500, { fill: "F2F2F2" }),
          dataCell("C-statistic", 1000, { fill: "F2F2F2" }),
          dataCell("0.551", 1200, { align: AlignmentType.CENTER, fill: "F2F2F2" }),
          dataCell("0.753", 1200, { align: AlignmentType.CENTER, bold: true, fill: "F2F2F2" }),
          dataCell("—", 1200, { align: AlignmentType.CENTER, fill: "F2F2F2" }),
          dataCell("Mode stratified; n = 856", 1360, { fill: "F2F2F2" }),
        ],
      }),
      new TableRow({
        children: [
          dataCell("Log-OLS (robustness)", 1900),
          dataCell("Longevity", 1500),
          dataCell("R²", 1000),
          dataCell("0.044", 1200, { align: AlignmentType.CENTER }),
          dataCell("0.212", 1200, { align: AlignmentType.CENTER, bold: true }),
          dataCell("—", 1200, { align: AlignmentType.CENTER }),
          dataCell("log1p(wks); n = 856", 1360),
        ],
      }),
    ],
  }),
  para([txt("Audio-only baselines estimated using identical hyperparameters on 10 Spotify audio features only. Cox PH audio-only baseline from prior pipeline run. PR-AUC random baseline = 0.0275 (2.75% positive rate).", { italics: true, size: 18 })], { before: 60, after: 180 }),
];

// Table 2: Threshold analysis
const thresholdTable = [
  tableCaption("Table 2. XGBoost Decision Threshold Analysis (Test Set, n = 15,678; 431 charted)"),
  new Table({
    width: { size: W, type: WidthType.DXA },
    columnWidths: [1200, 1600, 1300, 1300, 1300, 2660],
    rows: [
      new TableRow({
        children: [
          hdrCell("Threshold", 1200, { align: AlignmentType.CENTER }),
          hdrCell("Tracks Flagged (%)", 1600, { align: AlignmentType.CENTER }),
          hdrCell("Precision", 1300, { align: AlignmentType.CENTER }),
          hdrCell("Recall", 1300, { align: AlignmentType.CENTER }),
          hdrCell("F1", 1300, { align: AlignmentType.CENTER }),
          hdrCell("Practical Interpretation", 2660),
        ],
        tableHeader: true,
      }),
      ...[
        ["0.05", "4,044 (25.8%)", "10.5%", "98.1%", "0.189", "High recall; screen large catalog quickly"],
        ["0.10", "3,155 (20.1%)", "13.3%", "97.7%", "0.235", "Broad screening; misses ~10 hits per 1,000"],
        ["0.20", "2,506 (16.0%)", "16.5%", "96.1%", "0.282", "Balanced; recommended for A&R shortlisting"],
        ["0.30", "2,143 (13.7%)", "19.2%", "95.6%", "0.320", "Tighter list; ~4% of hits missed"],
        ["0.50", "1,684 (10.7%)", "23.5%", "91.9%", "0.375", "Conservative; misses ~8% of hits"],
      ].map((row, i) => new TableRow({
        children: row.map((v, j) => {
          const w = [1200, 1600, 1300, 1300, 1300, 2660][j];
          const fill = i % 2 === 0 ? "FFFFFF" : "F2F2F2";
          const align = j < 5 ? AlignmentType.CENTER : AlignmentType.LEFT;
          return dataCell(v, w, { fill, align });
        })
      })),
    ],
  }),
  para([txt("Each threshold corresponds to a different operating point for a label A&R screening tool. At a 0.20 threshold, the model flags 16% of new tracks and recovers 96% of eventual chart entries.", { italics: true, size: 18 })], { before: 60, after: 180 }),
];

const results = [
  heading1("4. Results"),
  heading2("4.1 Research Question 1: Chart Entry Classification"),
  ...perfTable,
  para([txt("Table 1 shows model performance for both audio-only and full-model specifications. Artist enrichment features provide large and consistent improvements across all four models. For logistic regression, ROC-AUC improves from 0.737 (audio only) to 0.914 (+0.177), while PR-AUC improves from 0.065 to 0.275 (10-fold lift over random). For XGBoost, ROC-AUC improves from 0.773 to 0.966, with PR-AUC reaching 0.441 (16-fold lift). The five-fold cross-validated ROC-AUC for logistic regression is 0.914 ± 0.004, confirming that these results generalize to unseen data.")]),
  img("fig10_enrichment_comparison.png", 5943600, 2779554),
  figCaption("Figure 1. ROC-AUC and PR-AUC for audio-only versus full model specifications. Artist enrichment features account for the majority of predictive performance across both classification models."),
  img("fig3_roc_curves.png", 5943600, 4914900),
  figCaption("Figure 2. ROC curves for logistic regression (AUC = 0.914) and XGBoost (AUC = 0.966) on the held-out test set."),
  img("fig9_precision_recall.png", 5943600, 4178914),
  figCaption("Figure 3. Precision-recall curves for both classifiers. The random baseline (gray dashed) lies at PR-AUC = 0.0275, corresponding to the 2.75% positive rate. XGBoost achieves 16-fold lift."),
  para([txt("The 5.1-point gap between XGBoost and logistic regression on ROC-AUC confirms meaningful nonlinearity in the chart entry decision surface. SHAP values identify the feature hierarchy driving XGBoost predictions:")]),
  img("fig5_shap_importance.png", 5943600, 5557391),
  figCaption("Figure 4. XGBoost SHAP feature importance (mean |SHAP| value on the test set). Artist-level commercial features dominate audio features."),
  para([txt("The top two predictors by mean |SHAP| value are both artist-level commercial variables: historical peak popularity (SHAP = 1.535) and Last.fm log-listener count (SHAP = 1.205). These outweigh all audio features combined. Among audio predictors, instrumentalness is the dominant signal (SHAP = 0.636, OR = 0.34 per 1 SD in logistic regression): instrumental tracks are 66 percent less likely to chart per one-standard-deviation increase. Valence is the strongest positive audio predictor (OR = 1.44). The US artist binary (SHAP = 0.538, OR = 1.91) reflects the Hot 100's structural orientation toward domestic radio markets.")]),
  para([txt("Table 2 translates model performance into a practical screening tool for label A&R applications.")]),
  ...thresholdTable,
  heading2("4.2 Research Question 2: Chart Longevity"),
  para([txt("The Cox proportional hazards model achieves a concordance index of 0.753 on 856 lyric-matched charted tracks, representing a substantial improvement over the audio-only baseline (C = 0.551) and indicating that the model meaningfully discriminates between short-charting and long-charting tracks.")]),
  img("fig7_kaplan_meier.png", 5943600, 3596502),
  figCaption("Figure 5. Kaplan-Meier survival curves by genre. The probability of remaining on the chart declines sharply in the first 10 weeks, with genre-level differences reflecting format and rotational differences across radio markets."),
  para([txt("The strongest and most significant longevity predictor is decade index (HR = 0.514, p < 0.001): each additional decade of chart history corresponds to a 49% increase in the per-week exit hazard, confirming that streaming-era tracks rotate off the chart faster than pre-streaming era tracks on a per-song basis. This finding is robust to the OLS specification (see Appendix). Note that OLS shows a positive coefficient for decade index because streaming-era mega-hits accumulate unprecedented total week counts, pulling the mean upward; the Cox per-week hazard correctly captures the median song's faster rotation.")]),
  para([txt("Among artist-level features, historical peak popularity (HR = 0.835, p < 0.001) is associated with longer chart runs, while larger catalog size (HR = 1.115, p = 0.005) and higher current API popularity (HR = 1.085, p = 0.030) also reach significance. The latter suggests that active label promotion sustains chart position beyond what the song's audio characteristics alone would predict. Loudness is the only audio feature reaching significance (HR = 0.900, p = 0.012). Instrumentalness shows a marginal positive hazard ratio (HR = 1.074, p = 0.043): the few instrumental tracks that do chart tend to exit faster than vocal counterparts.")]),
  para([txt("Four covariates fail the Schoenfeld proportional hazards test: decade index, acousticness, duration, and positive lyric sentiment. For the most critical predictor (decade index), the directional findings in Cox PH and Log-OLS are consistent, which provides robustness evidence. Sentiment findings should be interpreted cautiously given this violation and the 40.1% lyric coverage rate.")]),
];

// ── Conclusions ────────────────────────────────────────────────────────────
const conclusions = [
  heading1("5. Conclusions"),
  para([txt("This study addressed two related prediction problems using a dataset of 78,390 Spotify tracks linked to Billboard Hot 100 history. Four principal findings emerge.")]),
  new Paragraph({
    numbering: { reference: "numbers", level: 0 },
    children: [txt("Artist commercial track record and pre-existing audience size explain chart entry better than any audio feature. The top two SHAP predictors are both artist-level commercial variables. Audio features contribute modestly once artist context is known. A&R investment in artist development and audience building has substantially higher predicted return than audio production optimization.")],
    spacing: { before: 80, after: 80 },
  }),
  new Paragraph({
    numbering: { reference: "numbers", level: 0 },
    children: [txt("Instrumentalness is the dominant audio barrier: vocal content is nearly a prerequisite for the Hot 100. This is the one audio feature with a large, consistent, and statistically robust effect across both classification models.")],
    spacing: { before: 80, after: 80 },
  }),
  new Paragraph({
    numbering: { reference: "numbers", level: 0 },
    children: [txt("The streaming era has structurally accelerated chart rotation. The per-week exit hazard for a track charting in the 2020s is approximately twice that of a track charting in the 1950s, holding all other features constant.")],
    spacing: { before: 80, after: 80 },
  }),
  new Paragraph({
    numbering: { reference: "numbers", level: 0 },
    children: [txt("XGBoost at a 0.20 decision threshold provides a practical A&R screening tool, recovering 96% of eventual chart entries while flagging only 16% of candidate tracks for human review.")],
    spacing: { before: 80, after: 160 },
  }),
  para([txt("Returning to the four working hypotheses: H1 is strongly confirmed — artist enrichment features (peak popularity SHAP = 1.618, Last.fm listeners SHAP = 1.153) dominate all audio features and provide a 16-fold PR-AUC lift over the audio-only baseline for XGBoost. H2 is confirmed — instrumentalness is the dominant audio predictor across both classifiers (SHAP = 0.629, OR = 0.34). H3 is confirmed by the Cox model — each additional decade corresponds to a 49% increase in per-week exit hazard (HR = 0.514, p < 0.001). H4 receives partial support — artist_popularity_api (current) reaches significance in Cox (HR = 1.085, p = 0.030), consistent with ongoing promotional support sustaining chart runs, though the effect is modest relative to the decade index.")]),
  para([
    txt("Limitations. ", { bold: true }),
    txt("The most significant methodological limitation is temporal feature misalignment in the artist-level predictors. The three Spotify API features — artist_peak_popularity, artist_popularity_api, and artist_track_count — were scraped in early 2026 and reflect each artist's career through that date. The Last.fm listener count (lastfm_listeners_log) comes from a dataset published circa 2020. Both are applied as static features to chart events spanning 1958 to 2024. For a track that charted in 1975, the model therefore has access to information about that artist's career trajectory through 2026 when predicting whether the 1975 track charted — information that was not available at the time. If an artist had modest commercial standing in 1975 but became globally prominent by 2000, the model uses their 2026 peak popularity to retroactively explain their 1975 chart entry. This is a form of temporal data leakage that inflates the apparent predictive contribution of artist features and is one reason the artist enrichment lift (16-fold PR-AUC improvement over audio-only) should be interpreted as an upper bound rather than a point estimate of prospective performance.")
  ]),
  para([txt("A related issue is artist-level correlation across the random train/test split. Because artist features are single values per artist that do not vary across tracks, every track by the same artist carries identical feature values. A random 80/20 split places multiple tracks from the same artist in both training and test partitions. The model can effectively learn per-artist decision boundaries in training and reproduce them at test time without genuinely generalizing to unseen artists. This inflates holdout metrics beyond what would be observed for a deployment scenario involving artists not previously seen by the model — again, precisely the highest-value use case for an A&R screening tool.")]),
  para([txt("The degree of practical impact from these two issues depends on the deployment context. For A&R screening of tracks by established artists with existing Spotify and Last.fm presence, artist_popularity_api and lastfm_listeners_log are legitimately available at prediction time, and the temporal misalignment is less severe because established artists' current and historical popularity are strongly correlated. The concern is highest for genuinely emerging artists — those with little or no prior commercial history — where current-era popularity scores carry no valid predictive signal and per-artist memorization cannot occur. A production deployment would require point-in-time artist feature reconstruction (indexing popularity scores to the date of each chart event) and an artist-disjoint train/test split to produce unbiased estimates for new-artist generalization.")]),
  para([txt("Additional limitations: Spotify's Developer Mode rate limit prevented full API-based artist feature collection for non-charted tracks, requiring a workaround that introduces data source asymmetry between charted and non-charted artists (see Section 2.1). Five artist demographic features were excluded due to coverage below 1% of the non-charted universe. The two Spotify popularity metrics are highly collinear (VIF > 14), making individual coefficient signs unreliable. Lyric sentiment covers only 40.1% of charted tracks. Four covariates fail the Schoenfeld proportional hazards test. All reported associations are observational and should not be interpreted causally.")]),
  para([
    txt("Unexecuted Analyses. ", { bold: true }),
    txt("Three analyses were scoped and designed but not completed within the project timeline. (1) Librosa spectral feature extraction: pipeline scripts were written to fetch 30-second Spotify preview URLs (modal_preview_urls.py) and extract 13 MFCCs, spectral centroid, and rolloff per track (modal_librosa_extract.py), scoped to the 2,157 charted tracks. Execution was blocked by the same Spotify Developer Mode rate limit that constrained artist scraping. Had these features been collected, they would have enabled a direct test of whether low-level timbre and production quality predict chart success independently of Spotify's pre-computed audio features. (2) Temporal robustness validation: a methodologically stronger holdout would train on pre-2020 tracks and evaluate on 2020 to 2024 tracks, directly testing whether the model generalizes to the contemporary streaming era or merely interpolates within the historical distribution. Given the streaming-era structural shift identified in H3, this check is particularly important and was not completed. (3) Spotify popularity classification (RQ3): an original research question — whether audio and artist features predict whether a track reaches Spotify popularity score >= 80 — was deprioritized in favor of deeper treatment of the Billboard-based questions. The outcome variable is defined and available in the dataset.")
  ]),
  para([
    txt("Future Work. ", { bold: true }),
    txt("Addressing the temporal leakage issue is the most important methodological improvement for a follow-on study. This requires two changes: constructing point-in-time artist features indexed to each track's chart_entry_date (or release date), and using an artist-disjoint split where no artist appearing in the training set has tracks in the test set. Both changes would likely reduce reported AUC figures but would yield more honest estimates of prospective generalization performance. The most tractable data extension is completing the Librosa spectral extraction pipeline once the API rate limit resets. Incorporating first-week streaming velocity and social media signals (TikTok video count, Shazam search volume) would address the most commercially relevant predictors of contemporary chart trajectory. For the longevity model, a time-varying covariate extension of Cox PH or an accelerated failure time (AFT) model would resolve the four Schoenfeld violations, and a temporal train/test split would address the robustness gap. The Spotify popularity classification (RQ3) and artist demographic analysis remain natural extensions given the existing dataset infrastructure.")
  ]),
];

// ── Appendix ───────────────────────────────────────────────────────────────
const appendix = [
  new Paragraph({ children: [new PageBreak()] }),
  heading1("Appendix"),
  heading2("A. VIF Table"),
  new Table({
    width: { size: 4000, type: WidthType.DXA },
    columnWidths: [2800, 1200],
    rows: [
      new TableRow({ children: [hdrCell("Feature", 2800), hdrCell("VIF", 1200, { align: AlignmentType.CENTER })], tableHeader: true }),
      ...[ ["artist_peak_popularity", "21.25"], ["artist_popularity_api", "14.12"],
           ["loudness", "6.72"], ["duration_min", "6.28"], ["lastfm_listeners_log", "5.18"],
           ["valence", "3.75"], ["acousticness", "3.17"], ["key", "3.04"], ["mode", "2.65"],
           ["liveness", "2.31"], ["speechiness", "1.89"], ["instrumentalness", "1.87"],
           ["artist_track_count", "1.77"], ["is_us_artist", "1.40"], ["explicit", "1.27"],
      ].map((row, i) => new TableRow({
        children: [
          dataCell(row[0], 2800, { fill: i % 2 === 0 ? "FFFFFF" : "F2F2F2" }),
          dataCell(row[1], 1200, { align: AlignmentType.CENTER, fill: i % 2 === 0 ? "FFFFFF" : "F2F2F2",
            bold: parseFloat(row[1]) > 10, color: parseFloat(row[1]) > 10 ? "C0392B" : "000000" }),
        ]
      })),
    ],
  }),
  para([txt("Features with VIF > 10 flagged in red. energy (VIF = 15.1), danceability (VIF = 12.4), and tempo (VIF = 10.7) were removed from the final model.", { italics: true, size: 18 })], { before: 60, after: 180 }),
  heading2("B. Schoenfeld Residuals — Cox PH (Rank Test)"),
  new Table({
    width: { size: 5200, type: WidthType.DXA },
    columnWidths: [2800, 1400, 1000],
    rows: [
      new TableRow({ children: [hdrCell("Feature", 2800), hdrCell("Test Stat", 1400, { align: AlignmentType.CENTER }), hdrCell("p-value", 1000, { align: AlignmentType.CENTER })], tableHeader: true }),
      ...[ ["decade_idx", "17.39", "<0.001"], ["duration_min", "5.32", "0.021"],
           ["sentiment_pos", "0.73", "0.393"], ["acousticness", "1.66", "0.197"],
           ["artist_track_count", "0.59", "0.443"], ["artist_peak_popularity", "0.07", "0.789"],
           ["loudness", "1.61", "0.204"], ["valence", "0.02", "0.901"],
           ["instrumentalness", "0.09", "0.762"], ["speechiness", "0.12", "0.731"],
      ].map((row, i) => new TableRow({
        children: [
          dataCell(row[0], 2800, { fill: i % 2 === 0 ? "FFFFFF" : "F2F2F2" }),
          dataCell(row[1], 1400, { align: AlignmentType.CENTER, fill: i % 2 === 0 ? "FFFFFF" : "F2F2F2" }),
          dataCell(row[2], 1000, { align: AlignmentType.CENTER, fill: i % 2 === 0 ? "FFFFFF" : "F2F2F2",
            bold: parseFloat(row[2]) < 0.05 || row[2].startsWith("<"),
            color: (parseFloat(row[2]) < 0.05 || row[2].startsWith("<")) ? "C0392B" : "000000" }),
        ]
      })),
    ],
  }),
  para([txt("Violations (p < 0.05) flagged in red. Results based on rank transformation. Mode is excluded (handled by stratification).", { italics: true, size: 18 })], { before: 60, after: 180 }),
  heading2("C. Logistic Regression Odds Ratios (All Features)"),
  new Table({
    width: { size: 5600, type: WidthType.DXA },
    columnWidths: [2400, 1600, 1600],
    rows: [
      new TableRow({ children: [hdrCell("Feature", 2400), hdrCell("Coefficient", 1600, { align: AlignmentType.CENTER }), hdrCell("Odds Ratio", 1600, { align: AlignmentType.CENTER })], tableHeader: true }),
      ...[ ["artist_peak_popularity", "+1.045", "2.84"], ["lastfm_listeners_log", "+0.773", "2.17"],
           ["is_us_artist", "+0.647", "1.91"], ["valence", "+0.362", "1.44"],
           ["explicit", "+0.217", "1.24"], ["mode", "+0.117", "1.12"],
           ["key", "+0.013", "1.01"], ["duration_min", "-0.034", "0.97"],
           ["artist_popularity_api", "-0.132", "0.88 *"], ["artist_track_count", "-0.217", "0.80"],
           ["loudness", "-0.221", "0.80"], ["acousticness", "-0.260", "0.77"],
           ["liveness", "-0.262", "0.77"], ["speechiness", "-0.347", "0.71"],
           ["instrumentalness", "-1.081", "0.34"],
      ].map((row, i) => new TableRow({
        children: [
          dataCell(row[0], 2400, { fill: i % 2 === 0 ? "FFFFFF" : "F2F2F2" }),
          dataCell(row[1], 1600, { align: AlignmentType.CENTER, fill: i % 2 === 0 ? "FFFFFF" : "F2F2F2" }),
          dataCell(row[2], 1600, { align: AlignmentType.CENTER, fill: i % 2 === 0 ? "FFFFFF" : "F2F2F2" }),
        ]
      })),
    ],
  }),
  para([txt("* artist_popularity_api sign reversal is a collinearity artifact due to co-occurrence with artist_peak_popularity (VIF = 14.12). Only the combined directional signal (established artists more likely to chart) is interpretable.", { italics: true, size: 18 })], { before: 60, after: 180 }),
  heading2("D. Cox PH Significant Coefficients"),
  new Table({
    width: { size: 7200, type: WidthType.DXA },
    columnWidths: [2400, 1000, 1000, 1000, 1800],
    rows: [
      new TableRow({ children: [hdrCell("Feature", 2400), hdrCell("Coef", 1000, { align: AlignmentType.CENTER }), hdrCell("HR", 1000, { align: AlignmentType.CENTER }), hdrCell("p-value", 1000, { align: AlignmentType.CENTER }), hdrCell("Interpretation", 1800)], tableHeader: true }),
      ...[ ["decade_idx", "-0.666", "0.514", "<0.001", "49% higher exit hazard per decade"],
           ["artist_peak_popularity", "-0.180", "0.835", "<0.001", "Higher peak = longer runs"],
           ["artist_track_count", "+0.109", "1.115", "0.005", "Larger catalog = shorter runs"],
           ["artist_popularity_api", "+0.082", "1.085", "0.030", "Current popularity sustains chart"],
           ["loudness", "-0.106", "0.900", "0.012", "Louder = longer chart runs"],
           ["instrumentalness", "+0.071", "1.074", "0.043", "Instrumental = exits faster"],
      ].map((row, i) => new TableRow({
        children: [
          dataCell(row[0], 2400, { fill: i % 2 === 0 ? "FFFFFF" : "F2F2F2" }),
          dataCell(row[1], 1000, { align: AlignmentType.CENTER, fill: i % 2 === 0 ? "FFFFFF" : "F2F2F2" }),
          dataCell(row[2], 1000, { align: AlignmentType.CENTER, fill: i % 2 === 0 ? "FFFFFF" : "F2F2F2", bold: true }),
          dataCell(row[3], 1000, { align: AlignmentType.CENTER, fill: i % 2 === 0 ? "FFFFFF" : "F2F2F2" }),
          dataCell(row[4], 1800, { fill: i % 2 === 0 ? "FFFFFF" : "F2F2F2" }),
        ]
      })),
    ],
  }),
  para([txt("HR = hazard ratio. HR > 1 indicates faster exit from chart (shorter run); HR < 1 indicates slower exit (longer run). Mode handled by stratification.", { italics: true, size: 18 })], { before: 60, after: 180 }),
  heading2("E. Additional Figures"),
  // (figures follow below — see existing code)
];

// ── Appendix: Feature Glossary ─────────────────────────────────────────────
const appendixGlossary = [
  heading2("F. Feature Glossary"),
  para([txt("All features listed below are those used in at least one model in the final v6 pipeline. Audio features are sourced from the Spotify Tracks Dataset; artist-level features from the Spotify Web Developer API and the Music Artists Popularity dataset; temporal and sentiment features are derived variables.", { italics: true, size: 18 })], { before: 0, after: 100 }),
  new Table({
    width: { size: W, type: WidthType.DXA },
    columnWidths: [2100, 1100, 1300, 4860],
    rows: [
      new TableRow({
        children: [
          hdrCell("Feature", 2100),
          hdrCell("Type / Range", 1100),
          hdrCell("Model(s) Used", 1300),
          hdrCell("Definition", 4860),
        ],
        tableHeader: true,
      }),
      ...[
        ["valence", "[0, 1]", "All", "Spotify's measure of musical positiveness. High values correspond to happy, euphoric, or uplifting tracks; low values to sad, angry, or tense tracks."],
        ["acousticness", "[0, 1]", "All", "Confidence score (0–1) that the track was recorded acoustically rather than electronically produced. Derived from Spotify's audio analysis."],
        ["loudness", "dB (~−60 to 0)", "All", "Overall average loudness of the track in decibels (LUFS). Values closer to 0 indicate louder tracks. Negative by convention."],
        ["speechiness", "[0, 1]", "All", "Presence of spoken words. Values above 0.66 indicate speech-dominated content; below 0.33 indicate music with no speech; 0.33–0.66 indicate mixed content (e.g., rap)."],
        ["instrumentalness", "[0, 1]", "All", "Probability that the track contains no vocal content. Values above 0.5 are classified as instrumental. The dominant audio barrier to chart entry (OR = 0.34, SHAP = 0.629)."],
        ["liveness", "[0, 1]", "All", "Probability that the track was recorded in front of a live audience. Higher values suggest a live performance recording."],
        ["mode", "0 or 1", "All (stratum in Cox)", "Musical modality. 1 = major key; 0 = minor key. Used as a stratification variable in Cox PH because it fails the Schoenfeld proportional hazards test."],
        ["key", "0–11", "All", "Pitch class of the track using standard Pitch Class notation (0 = C, 1 = C#/D♭, …, 11 = B). Treated as a linear numeric feature; circular embedding was not applied."],
        ["explicit", "0 or 1", "All", "Whether the track has an explicit content flag on Spotify. 1 = explicit; 0 = clean. Explicit tracks are 24% more likely to chart per the logistic regression odds ratio."],
        ["duration_min", "minutes (capped 10)", "All", "Track duration in minutes, derived from duration_ms / 60000. Capped at 10 minutes to exclude podcast episodes and audiobook entries present in the Kaggle source."],
        ["artist_peak_popularity", "0–100", "All", "Historical peak Spotify popularity score for the track's primary artist, from the Spotify API. The strongest predictor in all models (SHAP = 1.618, OR = 2.84 per 1 SD)."],
        ["artist_popularity_api", "0–100", "All", "Current Spotify popularity score for the artist at the time of data collection. Highly collinear with artist_peak_popularity (VIF = 14.1); individual coefficient signs should not be interpreted in isolation."],
        ["artist_track_count", "integer", "All", "Number of tracks on Spotify attributed to the primary artist. Proxy for catalog size and career stage."],
        ["lastfm_listeners_log", "log1p scale", "All", "log1p-transformed cross-platform monthly listener count from the Last.fm / MusicBrainz dataset (1.47M artists). Covers 65.8% of artists by name match. The second-strongest predictor (SHAP = 1.153, OR = 2.17 per 1 SD)."],
        ["is_us_artist", "0 or 1", "All", "Binary indicator for US-based artist, derived from the MusicBrainz country field. Artists with unknown nationality are coded 0 (conservative imputation). US artists are 91% more likely to chart (OR = 1.91)."],
        ["decade_idx", "0–7", "Cox PH, OLS only", "Ordinal decade index derived from each charted track's first chart appearance: floor((chart_entry_year − 1950) / 10). 0 = 1950s, 1 = 1960s, …, 7 = 2020s. Undefined (NaN) for non-charted tracks. The strongest longevity predictor (HR = 0.514, p < 0.001)."],
        ["sentiment_compound", "[−1, +1]", "Cox PH, OLS only", "VADER aggregate sentiment score for the track's Billboard lyrics. Computed as the normalized, weighted sum of positive, negative, and neutral lexicon scores. Coverage: 40.1% of charted tracks (n = 856)."],
        ["sentiment_pos", "[0, 1]", "Cox PH, OLS only", "Fraction of VADER-identified positive words in the track's lyrics. Not statistically significant in v6 longevity models (p = 0.366). Fails Schoenfeld PH test (p = 0.393 rank test)."],
        ["sentiment_neg", "[0, 1]", "Cox PH, OLS only", "Fraction of VADER-identified negative words in the track's lyrics. HR = 1.088, p = 0.051 (marginal); directionally consistent with shorter chart runs for tracks with more negative lexical content."],
        ["lyric_word_count", "integer", "Cox PH, OLS only", "Total word count of the track's Billboard lyrics. Not statistically significant in v6 longevity models (p = 0.592)."],
      ].map((row, i) => new TableRow({
        children: [
          dataCell(row[0], 2100, { fill: i % 2 === 0 ? "FFFFFF" : "F2F2F2", italics: true }),
          dataCell(row[1], 1100, { fill: i % 2 === 0 ? "FFFFFF" : "F2F2F2", align: AlignmentType.CENTER }),
          dataCell(row[2], 1300, { fill: i % 2 === 0 ? "FFFFFF" : "F2F2F2", align: AlignmentType.CENTER }),
          dataCell(row[3], 4860, { fill: i % 2 === 0 ? "FFFFFF" : "F2F2F2" }),
        ]
      })),
    ],
  }),
  para([txt("Features removed for multicollinearity (VIF > 10): energy (VIF = 15.1), danceability (VIF = 12.4), tempo (VIF = 10.7). Features in dataset but excluded for insufficient coverage: is_male_artist, artist_age, is_mainstream_genre (<58% of charted tracks; <1% of full universe). Full VIF table in Appendix A.", { italics: true, size: 18 })], { before: 60, after: 180 }),
];

// ── Appendix: Bibliography ─────────────────────────────────────────────────
const appendixBiblio = [
  heading2("G. References"),
  para([txt("The following sources were used in this analysis. All Kaggle datasets are public and available without registration for download.", { italics: true, size: 18 })], { before: 0, after: 100 }),
  ...[
    ["[1]", "Maharshipandya. (2023). Spotify Tracks Dataset. Kaggle. https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset"],
    ["[2]", "Earhart, E. (2024). Billboard Hot 100 (1958–2024). Kaggle. https://www.kaggle.com/datasets/elizabethearhart/billboard-hot-1001958-2024"],
    ["[3]", "Rozenberg, R. (2022). Lyrics for Billboard Top 100 Songs 1946–2022. Kaggle. https://www.kaggle.com/datasets/rhaamrozenberg/billboards-top-100-song-1946-to-2022-lyrics"],
    ["[4]", "pieca111. (2020). Music Artists Popularity (MusicBrainz / Last.fm). Kaggle. https://www.kaggle.com/datasets/pieca111/music-artists-popularity"],
    ["[5]", "Spotify. (2024). Spotify Web Developer API — Artist and Track Endpoints. https://developer.spotify.com/documentation/web-api"],
    ["[6]", "Hutto, C. J., & Gilbert, E. (2014). VADER: A Parsimonious Rule-Based Model for Sentiment Analysis of Social Media Text. Proceedings of the Eighth International AAAI Conference on Weblogs and Social Media (ICWSM-14)."],
    ["[7]", "Guo, J., et al. (2025). Beyond the Hook: Predicting Billboard Hot 100 Chart Inclusion with Machine Learning from Streaming, Audio Signals, and Perceptual Features. arXiv preprint arXiv:2509.24856."],
    ["[8]", "Pachet, F., & Roy, P. (2008). Hit Song Science Is Not Yet a Science. Proceedings of the International Society for Music Information Retrieval Conference (ISMIR 2008)."],
  ].map((row, i) => para([
    txt(row[0] + " ", { bold: true }),
    txt(row[1]),
  ], { before: i === 0 ? 0 : 60, after: 60 })),
];

// ── Dummy section to close appendix figures block ──────────────────────────
const appendixFiguresTail = [
  img("fig4_odds_ratios.png", 4457700, 4750856),
  figCaption("Figure A1. Logistic regression odds ratios for all 15 features. Green = positive association with charting; red = negative association. Per 1 SD change."),
  img("fig1_class_balance.png", 3657600, 2561400),
  figCaption("Figure A2. Class distribution in the final dataset (n = 78,390 tracks). 2,157 charted (2.75%); 76,233 not charted (97.25%)."),
  img("fig8_longevity_distribution.png", 5943600, 3397500),
  figCaption("Figure A3. Distribution of weeks on chart for the 2,157 charted tracks. Right-skewed distribution motivates the log transformation in Log-OLS. Median = 8 weeks; mean = 17.6 weeks."),
  img("fig11_shap_instrumentalness.png", 5943600, 4120500),
  figCaption("Figure A4. SHAP dependence plot for instrumentalness. Each point is a test track. Negative SHAP values confirm that higher instrumentalness reduces the predicted probability of charting. Color indicates valence (green = happy, red = somber); no strong interaction is evident."),
];

// ── Build document ─────────────────────────────────────────────────────────
const doc = new Document({
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
          alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.",
          alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ]
  },
  styles: {
    default: { document: { run: { font: "Arial", size: 20 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: "1F3864" },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, font: "Arial", color: "2E75B6" },
        paragraph: { spacing: { before: 180, after: 80 }, outlineLevel: 1 } },
    ]
  },
  sections: [{
    properties: {
      page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } }
    },
    headers: {
      default: new Header({ children: [new Paragraph({
        children: [
          new TextRun({ text: "Predicting Billboard Hot 100 Chart Entry and Longevity", font: "Arial", size: 18, color: "595959" }),
          new TextRun({ text: "\tOIT367 | Stanford GSB | Winter 2026", font: "Arial", size: 18, color: "595959" }),
        ],
        tabStops: [{ type: TabStopType.RIGHT, position: TabStopPosition.MAX }],
        border: { bottom: { style: BorderStyle.SINGLE, size: 2, color: "CCCCCC", space: 4 } },
        spacing: { after: 0 },
      })] }),
    },
    footers: {
      default: new Footer({ children: [new Paragraph({
        children: [
          new TextRun({ text: "Wurm · Chen · Barli · Taruno", font: "Arial", size: 16, color: "595959" }),
          new TextRun({ text: "\t", font: "Arial", size: 16 }),
          new TextRun({ text: "Page ", font: "Arial", size: 16, color: "595959" }),
          new TextRun({ children: [PageNumber.CURRENT], font: "Arial", size: 16, color: "595959" }),
        ],
        tabStops: [{ type: TabStopType.RIGHT, position: TabStopPosition.MAX }],
        border: { top: { style: BorderStyle.SINGLE, size: 2, color: "CCCCCC", space: 4 } },
        spacing: { before: 0 },
      })] }),
    },
    children: [
      ...execSummary,
      ...intro,
      ...dataSection,
      ...methods,
      ...results,
      ...conclusions,
      ...appendix,
      ...appendixFiguresTail,
      ...appendixGlossary,
      ...appendixBiblio,
    ],
  }],
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync('/sessions/hopeful-optimistic-shannon/docx_build/oit367_final_report.docx', buffer);
  console.log('Report written: oit367_final_report.docx');
}).catch(err => { console.error(err); process.exit(1); });

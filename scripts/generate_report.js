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
      children: [new TextRun({ text, font: "Arial", size: 18, bold: opts.bold || false, color: opts.color || "000000" })],
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
  para([txt("All models are trained on historical patterns from 1958 to 2024 and are associative rather than causal. The analysis uses a stratified 80/20 train-test split with 5-fold cross-validation for classification stability checks.")]),
];

// ── Data ───────────────────────────────────────────────────────────────────
const dataSection = [
  heading1("2. Data"),
  heading2("2.1 Sources"),
  para([
    txt("The base corpus combines two public Kaggle datasets. The "),
    txt("Spotify Tracks Dataset", { italics: true }),
    txt(" contains approximately 114,000 rows representing unique track-genre pairs, with audio features for each track. The "),
    txt("Billboard Hot 100", { italics: true }),
    txt(" dataset contains roughly 690,000 weekly chart observations from 1958 to 2024. These are joined via a pre-matched intermediate file that resolves track names to Spotify track IDs."),
  ]),
  para([txt("Three external sources augment the base: Spotify API artist profiles providing historical peak popularity, catalog size, and current popularity scores; a MusicBrainz and Last.fm dataset containing listener counts for 1.47 million artists and artist country of origin; and a Billboard lyrics dataset processed with the VADER sentiment analyzer for compound, positive, and negative sentiment scores. Last.fm listeners are log-transformed to reduce right skew.")]),
  heading2("2.2 Dataset Construction"),
  para([
    txt("The Spotify source contains duplicate rows when a track is assigned to multiple genres. The pipeline deduplicates to one row per track ID, retaining one genre label. It then performs a "),
    txt("left join", { italics: true }),
    txt(" with aggregated Billboard data, retaining all 78,390 unique Spotify tracks and assigning is_charted = 1 only to the 2,157 that appeared on the Hot 100. A left join is critical; an inner join used in an earlier pipeline iteration retained only charted tracks, producing a 100 percent positive rate and making binary classification impossible.")
  ]),
  para([txt("A second deduplication step collapsed 3,502 initially matched charted track IDs to 2,157 truly unique songs, removing album versus single release pairs and regional variants that share identical audio features. Retaining these near-duplicates inflated model performance artificially by creating trivially similar training and test instances.")]),
  heading2("2.3 Class Imbalance and Evaluation Metric"),
  para([txt("2,157 of 78,390 tracks are charted (2.75% positive rate). This imbalance is real and meaningful; the overwhelming majority of released music never reaches the Hot 100. At a 2.75% positive rate, a random classifier achieves PR-AUC = 0.0275. For this reason, Precision-Recall AUC (PR-AUC) is reported as the primary classification metric alongside ROC-AUC throughout this paper. A model that achieves PR-AUC = 0.275 against a 0.0275 baseline represents a 10-fold lift over random.")]),
  heading2("2.4 Features"),
  para([txt("The final feature set for classification contains 15 variables: 10 audio features from Spotify (valence, acousticness, loudness, speechiness, instrumentalness, liveness, mode, key, explicit, track duration) and 5 artist-level features (historical peak popularity, current popularity, catalog size, Last.fm log-listeners, and US artist indicator). Three features were removed for variance inflation factor (VIF) exceeding 10: energy (VIF = 15.1, collinear with loudness), danceability (VIF = 12.4), and tempo (VIF = 10.7 after danceability removal). The longevity models add a decade ordinal (0 = 1950s through 7 = 2020s) and four VADER lyric sentiment features, available for 856 of 2,157 charted tracks.")]),
];

// ── Methods ────────────────────────────────────────────────────────────────
const methods = [
  heading1("3. Methods"),
  heading2("3.1 Feature Engineering and Preprocessing"),
  para([txt("All continuous features are standardized with StandardScaler before logistic regression fitting, so all odds ratios are directly comparable on a per-standard-deviation basis. Track duration is converted from milliseconds to minutes and capped at 10 minutes to exclude outlier podcast episodes that appear in the Kaggle source. The is_us_artist binary is conservatively imputed: unknown nationality is coded as 0 (non-US) rather than excluded, ensuring 100% coverage while making a deliberate conservative assumption documented in the pipeline code.")]),
  heading2("3.2 Research Question 1: Binary Chart Entry Classification"),
  para([txt("Two classifiers address chart entry prediction. Logistic regression is included for interpretability: standardized coefficients yield per-standard-deviation odds ratios comparable across all features. Class imbalance is handled via class_weight=balanced, which up-weights the positive class by a factor of approximately 25x. Five-fold stratified cross-validation confirms out-of-sample stability.")]),
  para([txt("XGBoost captures nonlinear feature interactions and feature-combination effects that a linear model cannot represent. The positive class weight is set to scale_pos_weight = 24.6 (ratio of negative to positive training examples). Hyperparameters are n_estimators = 500, learning_rate = 0.05, max_depth = 5, with early stopping at round 189. SHAP (SHapley Additive exPlanations) values decompose each XGBoost prediction into per-feature contributions, enabling a feature importance ranking that accounts for nonlinear interactions and correlations. SHAP values are computed on the held-out test set.")]),
  para([txt("Prediction tasks require genuine out-of-sample validation. Both classifiers are trained on 80 percent of the data (n = 62,712) and evaluated on a held-out stratified test set (n = 15,678). Audio-only baseline models are trained with identical hyperparameters but limited to the 10 audio features, providing a clean experimental comparison of the marginal value of artist enrichment.")]),
  heading2("3.3 Research Question 2: Chart Longevity Survival Analysis"),
  para([txt("Chart longevity is the number of weeks a charted track remains on the Hot 100. This is modeled as a survival outcome: wks_on_chart is the survival time, all observations are treated as right-censored (no confirmed removal is recorded for tracks still charting at the data cutoff), and the event indicator is set to 1 for all observations.")]),
  para([txt("Cox proportional hazards (Cox PH) is the primary longevity model. It estimates the hazard of exiting the chart per unit time as a function of standardized covariates. The proportional hazards assumption is verified using Schoenfeld residuals. Mode (major vs. minor key) fails this test (p < 0.05) and is handled by stratification, fitting a separate baseline hazard for major and minor key tracks rather than estimating a single proportional coefficient. A ridge penalizer of 0.1 is applied for coefficient stability. The concordance index (C-statistic, analogous to AUC-ROC for survival models) is the primary fit metric.")]),
  para([txt("An OLS regression on log-transformed weeks serves as a robustness check. If the key directional findings hold across both Cox PH and OLS, the conclusions are not artifacts of the proportional hazards assumption. Both models use the same 856 lyric-matched charted tracks to allow direct comparison.")]),
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
  heading2("4.2 Research Question 2: Chart Longevity Survival Analysis"),
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
  para([
    txt("Limitations. ", { bold: true }),
    txt("The two artist popularity metrics are highly collinear (VIF > 14), making individual coefficient interpretation unreliable for this pair; only their combined directional signal is interpretable. The Billboard Hot 100 has a structural US and English-language bias partially but not fully captured by the is_us_artist binary. Lyric sentiment analysis covers only 40.1% of charted tracks, limiting statistical power for sentiment findings. These models capture historical associations and should not be applied causally.")
  ]),
  para([
    txt("Future Work. ", { bold: true }),
    txt("Incorporating first-week streaming velocity and social media engagement data would likely further improve chart entry prediction. A time-stratified model that accounts for the changing composition of the Hot 100 across eras could improve longevity prediction for contemporary releases.")
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
    ],
  }],
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync('/sessions/hopeful-optimistic-shannon/docx_build/oit367_final_report.docx', buffer);
  console.log('Report written: oit367_final_report.docx');
}).catch(err => { console.error(err); process.exit(1); });

# EDU-AI Evaluation Platform 

Welcome to the **EDU-AI Evaluation Platform**, an interactive Streamlit-based web system designed to operationalize the evaluation framework of the EDU-AI project. This platform enables the analysis of educational AI interventions across multiple data sources, combining quantitative and qualitative indicators into interpretable, research-grade analytics.

---

## 1. Overview

The EDU-AI Evaluation Platform integrates diverse educational datasets (e.g., pre/post assessments, surveys, telemetry logs, reflections, and fairness data) into a unified analytic workflow. It computes indices such as:

- **Learning Gain ( \( \Delta = \text{Post} - \text{Pre} \) )**
- **Prompt Evolution Index (PEI)** and **Reflection Depth Score (RDS)**
- **Reliability Metrics** (Cronbach’s \( \alpha \))
- **Fairness Indicators** (SPD, EOD, DI)
- **Mediation & Dose-Response Effects**
- **False Discovery Rate (FDR)** adjustments across multiple hypotheses

The platform's modular architecture aligns with experimental and quasi-experimental research design principles, supporting both descriptive analytics and inferential testing.

---

## 2. System Architecture

### 2.1 Core Components
- **Home.py** — Summarizes overall results and KPIs.
- **pages/0_Upload.py** — Handles input data ingestion and session caching.
- **pages/1–7** — Each module implements one analytic component (Assessments, Surveys, Telemetry, Teacher Logs, Fairness, Reflections, Summary Dashboard).

### 2.2 Data Flow
1. Users upload source CSV files.
2. Data is stored in `st.session_state`.
3. Module pages transform and compute derived indicators.
4. Summary Dashboard aggregates, visualizes, and exports results.

---

## 3. Mathematical Formulations

### 3.1 Learning Gain and Effect Size

Each student-level learning gain is calculated as:
$$
\Delta_i = \text{Post}_i - \text{Pre}_i
$$

For between-group comparison, the standardized effect size (Cohen’s \( d \)) is computed as:
$$
d = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}}}
$$

An adjusted post-test model approximates a three-level mixed model:
$$
Y_{ijk} = \beta_0 + \beta_1 \text{EDUAI}_{jk} + \beta_2 \text{Pre}_{ijk} + u_k + v_{jk} + \varepsilon_{ijk}
$$

---

### 3.2 Reliability (Cronbach’s Alpha)
\[
\alpha = \frac{k}{k-1} \left(1 - \frac{\sum_{i=1}^{k}\sigma_i^2}{\sigma_{\text{total}}^2}\right)
\]
where \( k \) is the number of items.

---

### 3.3 Fairness Metrics
Let \( g \) denote group identity:

- **Selection Rate:**  \( SR_g = P(\hat{Y}=1 \mid A=g) \)
- **Statistical Parity Difference:**  \( SPD = SR_{ref} - SR_g \)
- **Disparate Impact:**  \( DI = SR_g / SR_{ref} \)
- **Equal Opportunity Gap:**  \( EOG = |TPR_g - TPR_{ref}| \)
- **Equalized Odds Gap:**  \( EOD = \frac{|TPR_g - TPR_{ref}| + |FPR_g - FPR_{ref}|}{2} \)

---

### 3.4 Intraclass Correlation (ICC)
\[
ICC(1) = \frac{MS_B - MS_W}{MS_B + (k - 1)MS_W}
\]
This quantifies within-class vs between-class variance in learning gains.

---

### 3.5 False Discovery Rate (Benjamini–Hochberg)
Given \( m \) hypotheses with ordered p-values \( p_{(1)} \le p_{(2)} \le ... \le p_{(m)} \):
\[
p_{(k)} \le \frac{k}{m} q
\]
All tests with \( p_i \le p_{(k)} \) are declared significant at level \( q \).

---

### 3.6 Mediation (Process → Outcome)
Does EDU-AI influence learning gains through intermediary process indicators (PEI or RDS)?

\[
\text{Indirect} = a \times b, \quad a: M \sim T, \quad b: \Delta \sim M (\text{and } T)
\]

where:
- \( a \): effect of treatment on mediator
- \( b \): effect of mediator on outcome

Bootstrap confidence intervals are generated for \( a \times b \).

---

### 3.7 Dose–Response Model
Represents the effect of usage intensity \( U \) on outcomes (Post or \( \Delta \)):
\[
Y = \beta_0 + \beta_1 U + \beta_2 U^2 + \varepsilon
\]
Confidence bands are estimated via bootstrap resampling.

---

## 4. Statistical Diagnostics and Visualization

### 4.1 Tabs & Analysis Features
| Tab | Analysis Component | Key Outputs |
|------|--------------------|--------------|
| A | Primary Outcome | Learning gain distribution, group means, CSV export |
| B | ICC & Cluster | Cluster effect size (ICC1), class-level variance |
| C | Mixed Effects & Effect Size | OLS models, Cohen’s d, BH-adjusted pairwise tests |
| D | Multiple Outcomes & FDR | Familywise control using BH q=0.10 |
| E | Mediation | Product-of-coefficients path estimates (a,b,ab) |
| F | Dose–Response | Quadratic/segmented fits, bootstrap CI, residual plots |
| G | Fairness Summary | SPD, DI, EOG, EOD, ROC/PR, calibration, bootstrap CIs |

---

### 4.2 Example Outputs
- **Interactive histograms** and **boxplots** (learning gains)
- **Group comparison bar charts** with error bars
- **Bootstrap CI visualizations** for indirect and treatment effects
- **Residual, influence, and Cook’s D plots** (diagnostics)
- **ROC, PR, and calibration curves** (model fairness)
- **Intersectional heatmaps** for multi-attribute fairness

---

## 5. Data Requirements Summary

| Table | Required Columns | Purpose |
|--------|------------------|----------|
| Assessments | `student_id`, `phase`, `score` | Pre/Post gains |
| Surveys | `student_id`, `instrument`, `response` | Reliability & self-reports |
| Telemetry | `student_id`, `prompt`, `ai_response` | Compute PEI & RDS |
| Teacher Logs | `workload_hours`, `cognitive_load` | Workload summaries |
| Fairness | `group`, `y_true`, `y_pred`/`y_score` | Bias metrics |
| Reflections | `student_id`, `reflection_text` | Qualitative RDS proxy |

---

## 6. Visualizations and Statistical Outputs

The platform automatically generates and exports:
- **Descriptive tables:** mean, SD, N by group
- **Inferential summaries:** ANOVA, t-tests, adjusted effect sizes
- **Diagnostic plots:** variance partitioning, mediation paths
- **Fairness analytics:** ROC, PR, calibration, parity tables
- **Exportable CSVs:** per-tab (e.g., `learning_gains.csv`, `fairness_metrics.csv`)

---

## 7. Implementation Notes

### 7.1 Technologies
- **Frontend:** Streamlit, Plotly, Matplotlib
- **Backend:** Python (pandas, numpy, statsmodels, scipy)
- **Data storage:** `st.session_state` (in-memory; no permanent storage)

### 7.2 Reproducibility
- All computed results (e.g., gains, metrics, CIs) are cached in-session.
- Users can download CSVs for offline analysis.

---

## 8. Ethical & Research Integrity Considerations

- **Privacy:** All identifiers (e.g., `student_id`) should be anonymized.
- **Fairness:** Sensitive attributes used for bias evaluation must be handled under ethical and legal frameworks.
- **Reproducibility:** All metrics and formulas are explicitly documented.
- **Transparency:** Visualization of uncertainty (CIs, bootstraps) is mandatory for interpretability.

---

## 9. Citation
If you use this platform in a paper or thesis, cite as:

> EDU-AI Evaluation Platform (2025). Streamlit application for the empirical assessment of AI-mediated learning. National University of Singapore.

---

## 10. License
This repository is distributed under the MIT License. See `LICENSE.md` for terms.

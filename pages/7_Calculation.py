# pages/8_Calculation.py
# ============================================================
# EDU-AI Evaluation Platform — Calculation Page (Step-by-Step)
# Implements advanced, reproducible analyses with formulas,
# outputs, and Plotly figures for each step (0–9).
# ============================================================

import math
import io
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Optional scientific packages
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    HAS_SM = True
except Exception:
    HAS_SM = False

try:
    from scipy import stats
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# Optional MICE (Iterative Imputer)
try:
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    HAS_IMPUTE = True
except Exception:
    HAS_IMPUTE = False

st.set_page_config(page_title="EDU-AI — Calculation Page", layout="wide")
st.title("EDU-AI Evaluation — Calculation Page (Step-by-Step)")
st.caption("This page performs the full analysis pipeline with academic formulas, outputs, and interactive figures.")

# ------------------------
# Utilities & Helpers
# ------------------------
def safe_rate(num, den):
    num = float(num)
    den = float(den)
    return num / den if den and np.isfinite(den) else np.nan

def bh_adjust(pvals, q=0.10):
    """Benjamini–Hochberg FDR adjust; returns adjusted p-values in original order."""
    p = np.array([np.nan if x is None else x for x in pvals], dtype=float)
    n = np.sum(np.isfinite(p))
    if n == 0:
        return [np.nan]*len(p)
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p)+1, dtype=float)
    # Handle NaNs: assign high rank but keep NaNs in place
    ranks[~np.isfinite(p)] = np.nan
    adj = np.full_like(p, np.nan, dtype=float)
    # Compute BH on finite
    finite_idx = np.isfinite(p)
    p_fin = p[finite_idx]
    r_fin = ranks[finite_idx]
    bh = (p_fin * len(p_fin)) / r_fin
    # Monotone non-increasing from tail
    bh_sorted = np.minimum.accumulate(bh[np.argsort(-p_fin)])[np.argsort(np.argsort(-p_fin))]
    # Now, we must map back to original finite order respecting sorting by p
    # Easier: redo standard BH via sorted order
    idx_sorted = np.argsort(p_fin)
    bh2 = np.empty_like(p_fin)
    prev = 1.0
    for i in range(len(p_fin)-1, -1, -1):
        rank = i+1
        val = p_fin[idx_sorted[i]] * len(p_fin) / rank
        prev = min(prev, val)
        bh2[idx_sorted[i]] = prev
    adj[finite_idx] = np.clip(bh2, 0, 1)
    return adj.tolist()


def df_download(df: pd.DataFrame, name: str, label: str):
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=name,
        mime="text/csv",
        key=f"dl_{name}"
    )

def ensure_str_series(s):
    """Ensure a 1-D string-like series (handles duplicated column names returning DataFrame)."""
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s.astype(str)

# ------------------------
# Data Presence Shortcut
# ------------------------
def has(key): 
    return key in st.session_state and st.session_state[key] is not None

# ============================================================
# 0) Data Preparation & Alignment (Prerequisite for All Steps)
# ============================================================
st.header("0) Data Preparation & Alignment (Prerequisite)")

st.markdown("### Data Integration Checklist")

st.markdown(r"""
**Goal.** Align primary keys, harmonize timestamps, and verify anonymization consistency across all tables.

**Canonical keys (recommended).** `student_id`, `class_id`, `site_id`, `term`, `condition`
- For factorial designs, encode `condition` as a 2×2 (e.g., **EDUAI × SCAFF**).

**Time fields.** `pretest_date`, `posttest_date`, `activity_timestamp`
- Normalize all timestamps to a common timezone (e.g., Asia/Singapore or UTC) and store consistently.

**Anonymization.** Verify that `uuid` / `pseudo_id` mappings are one-to-one and consistent across tables.
""")

st.markdown("**Intent-to-Treat (ITT) principle.** All randomized units are analyzed in their assigned group, regardless of compliance or missingness:")
st.latex(r"\hat{\tau}_{\text{ITT}} \;=\; \mathbb{E}[Y \mid Z=1] \;-\; \mathbb{E}[Y \mid Z=0]")
st.markdown("where $Z$ denotes assignment (not exposure).")

st.markdown(r"""
**Hierarchical nesting.** Students are nested within classes, and classes within sites; downstream analyses will include random components aligned to this structure.
""")
def has(key): 
    return key in st.session_state and st.session_state[key] is not None

# Collect sources from session (non-destructive copies, lowercased columns)
assess = st.session_state["assess_df"].copy() if has("assess_df") else None
surveys = st.session_state["surveys_df"].copy() if has("surveys_df") else None
tele = (st.session_state["tele_df"].copy() if has("tele_df") 
        else (st.session_state["telemetry_with_pei"].copy() if has("telemetry_with_pei") else None))
teacher = st.session_state["teacher_df"].copy() if has("teacher_df") else None
fair = st.session_state["fairness_df"].copy() if has("fairness_df") else None
refl = st.session_state["reflections_df"].copy() if has("reflections_df") else None

tables = {
    "Assessments": assess,
    "Surveys": surveys,
    "Telemetry": tele,
    "TeacherLogs": teacher,
    "Fairness": fair,
    "Reflections": refl
}

for name, df in tables.items():
    if df is not None:
        df.columns = [c.strip().lower() for c in df.columns]
        tables[name] = df

# --------------------------
# A) Session cache inventory
# --------------------------
with st.expander("A) Data sources & session cache status", expanded=True):
    src = []
    for k in ["assess_df","surveys_df","tele_df","teacher_df","fairness_df","reflections_df","learning_gains","telemetry_with_pei"]:
        src.append((k, has(k), 0 if not has(k) else len(st.session_state[k])))
    df_src = pd.DataFrame(src, columns=["object","available","n_rows"])
    fig_src = go.Figure(data=[go.Table(
        header=dict(values=list(df_src.columns), fill_color="#2c3e50", font=dict(color="white")),
        cells=dict(values=[df_src[c] for c in df_src.columns], align="left")
    )])
    fig_src.update_layout(title="Session Objects Inventory", template="plotly_white")
    st.plotly_chart(fig_src, use_container_width=True)

# --------------------------------------
# B) Key alignment & time fields summary
# --------------------------------------
with st.expander("B) Key alignment & basic cleaning", expanded=True):
    key_cols = ["student_id","class_id","site_id","term","condition"]
    time_cols = ["pretest_date","posttest_date","activity_timestamp","log_timestamp"]

    rows = []
    for name, df in tables.items():
        if df is None:
            rows.append((name, "—", "—", 0))
            continue
        present_keys = [c for c in key_cols if c in df.columns]
        present_time = [c for c in time_cols if c in df.columns]
        rows.append((
            name,
            ", ".join(present_keys) if present_keys else "—",
            ", ".join(present_time) if present_time else "—",
            len(df)
        ))
    df_keys = pd.DataFrame(rows, columns=["table","keys_present","time_fields","rows"])
    fig_keys = go.Figure(data=[go.Table(
        header=dict(values=list(df_keys.columns), fill_color="#2c3e50", font=dict(color="white")),
        cells=dict(values=[df_keys[c] for c in df_keys.columns], align="left")
    )])
    fig_keys.update_layout(title="Key & Time Fields by Table", template="plotly_white")
    st.plotly_chart(fig_keys, use_container_width=True)

    # Normalize time fields (UTC if possible)
    if tables["Telemetry"] is not None and "activity_timestamp" in tables["Telemetry"].columns:
        try:
            tables["Telemetry"]["_activity_ts"] = pd.to_datetime(tables["Telemetry"]["activity_timestamp"], errors="coerce", utc=True)
        except Exception:
            tables["Telemetry"]["_activity_ts"] = pd.to_datetime(tables["Telemetry"]["activity_timestamp"], errors="coerce")
    if tables["Assessments"] is not None:
        for c in ["pretest_date","posttest_date"]:
            if c in tables["Assessments"].columns:
                tables["Assessments"][f"_{c}"] = pd.to_datetime(tables["Assessments"][c], errors="coerce")

# =============================================================================
# C) Advanced diagnostics — 7–8 interactive figures/tables (Plotly)
# =============================================================================
st.subheader("Advanced Diagnostics: Alignment, Completeness, and Time Coverage")

# Helper for boolean presence matrix
def presence_matrix(tables: dict, columns: list) -> pd.DataFrame:
    out = []
    for name, df in tables.items():
        row = {"table": name}
        for col in columns:
            row[col] = 1 if (df is not None and col in df.columns) else 0
        out.append(row)
    return pd.DataFrame(out)

# -----------------------
# Chart 1: Key coverage heatmap
# -----------------------
cols_interest = ["student_id","class_id","site_id","term","condition"]
pm = presence_matrix(tables, cols_interest)
if not pm.empty:
    z = pm[cols_interest].to_numpy()
    fig_heat = go.Figure(data=go.Heatmap(
        z=z, x=cols_interest, y=pm["table"], colorscale="Blues", showscale=True
    ))
    fig_heat.update_layout(title="(1) Key Coverage Heatmap (Table × Key)", template="plotly_white",
                           xaxis_title="Key", yaxis_title="Table")
    st.plotly_chart(fig_heat, use_container_width=True)

# -----------------------
# Chart 2: Missingness bar (keys + time fields per table)
# -----------------------
miss_rows = []
for name, df in tables.items():
    if df is None:
        continue
    cols = [c for c in cols_interest + ["pretest_date","posttest_date","activity_timestamp","log_timestamp"] if c in df.columns]
    if not cols:
        continue
    miss = df[cols].isna().mean().reset_index()
    miss.columns = ["column","missing_rate"]
    miss["table"] = name
    miss_rows.append(miss)
if miss_rows:
    df_miss = pd.concat(miss_rows, ignore_index=True)
    fig_miss = px.bar(df_miss, x="column", y="missing_rate", color="table",
                      barmode="group", title="(2) Missingness Rate for Key/Time Fields by Table")
    fig_miss.update_layout(template="plotly_white", yaxis_tickformat=".0%")
    st.plotly_chart(fig_miss, use_container_width=True)

# -----------------------
# Chart 3: ECDF of telemetry timestamps (if present)
# -----------------------
if tables["Telemetry"] is not None and "_activity_ts" in tables["Telemetry"].columns:
    tdf = tables["Telemetry"].dropna(subset=["_activity_ts"]).copy()
    if not tdf.empty:
        # ECDF
        tdf = tdf.sort_values("_activity_ts")
        tdf["ecdf"] = np.linspace(0, 1, len(tdf))
        fig_ecdf = px.line(tdf, x="_activity_ts", y="ecdf",
                           title="(3) Telemetry Time Coverage — ECDF")
        fig_ecdf.update_layout(template="plotly_white", xaxis_title="Timestamp (UTC)", yaxis_title="Proportion ≤ t")
        st.plotly_chart(fig_ecdf, use_container_width=True)

# -----------------------
# Chart 4: Duplication rate by key-combo (per table)
# -----------------------
dup_stats = []
for name, df in tables.items():
    if df is None: 
        continue
    keys_here = [c for c in cols_interest if c in df.columns]
    if not keys_here:
        continue
    # simplest duplication: student_id-level
    if "student_id" in keys_here:
        dup_rate = 1.0 - float(df["student_id"].nunique())/max(1, len(df))
        dup_stats.append(dict(table=name, level="student_id", duplication_rate=dup_rate))
    # class-level (student_id within class if both exist)
    if {"student_id","class_id"}.issubset(df.columns):
        combo = df[["student_id","class_id"]].astype(str).agg("::".join, axis=1)
        dup_rate2 = 1.0 - float(combo.nunique())/max(1, len(df))
        dup_stats.append(dict(table=name, level="student_id×class_id", duplication_rate=dup_rate2))

if dup_stats:
    df_dup = pd.DataFrame(dup_stats)
    fig_dup = px.bar(df_dup, x="table", y="duplication_rate", color="level",
                     title="(4) Duplication Rate by Key Level", barmode="group")
    fig_dup.update_layout(template="plotly_white", yaxis_tickformat=".0%")
    st.plotly_chart(fig_dup, use_container_width=True)

# -----------------------
# Chart 5: Joinability matrix — student_id overlap counts
# -----------------------
# compute pairwise overlaps on student_id
overlaps = []
name_list = [n for n, df in tables.items() if df is not None and "student_id" in df.columns]
sid_sets = {n: set(ensure_str_series(tables[n]["student_id"]).str.strip().unique()) for n in name_list}
for i in range(len(name_list)):
    for j in range(i, len(name_list)):
        ni, nj = name_list[i], name_list[j]
        inter = len(sid_sets[ni].intersection(sid_sets[nj]))
        overlaps.append((ni, nj, inter))
df_ov = pd.DataFrame(overlaps, columns=["table_i","table_j","n_overlap"])
if not df_ov.empty:
    pivot_ov = df_ov.pivot(index="table_i", columns="table_j", values="n_overlap").fillna(0)
    # symmetrize
    pivot_ov = pivot_ov.reindex(index=name_list, columns=name_list, fill_value=0)
    z = pivot_ov.to_numpy()
    fig_ov = go.Figure(data=go.Heatmap(
        z=z, x=pivot_ov.columns, y=pivot_ov.index, colorscale="Viridis", showscale=True
    ))
    fig_ov.update_layout(title="(5) Joinability Heatmap (Overlap on student_id)", template="plotly_white")
    st.plotly_chart(fig_ov, use_container_width=True)

# -----------------------
# Chart 6: Anonymization consistency — uuid/pseudo_id collision & reuse
# -----------------------
anon_rows = []
for name, df in tables.items():
    if df is None: 
        continue
    for idcol in ["uuid","pseudo_id"]:
        if idcol in df.columns:
            nun = df[idcol].nunique(dropna=True)
            cnt = len(df)
            reuse_rate = 1.0 - float(nun)/max(1, cnt)
            anon_rows.append(dict(table=name, idcol=idcol, unique_ids=nun, rows=cnt, reuse_rate=reuse_rate))
if anon_rows:
    df_anon = pd.DataFrame(anon_rows)
    fig_anon = px.bar(df_anon, x="table", y="reuse_rate", color="idcol",
                      title="(6) Anonymization Reuse Rate by Table", barmode="group")
    fig_anon.update_layout(template="plotly_white", yaxis_tickformat=".0%")
    st.plotly_chart(fig_anon, use_container_width=True)
    fig_anon_tab = go.Figure(data=[go.Table(
        header=dict(values=list(df_anon.columns), fill_color="#2c3e50", font=dict(color="white")),
        cells=dict(values=[df_anon[c] for c in df_anon.columns], align="left")
    )])
    fig_anon_tab.update_layout(title="Anonymization Summary Table", template="plotly_white")
    st.plotly_chart(fig_anon_tab, use_container_width=True)

# -----------------------
# Chart 7: Time-window alignment — histogram/facet coverage
# -----------------------
time_cov = []
if tables["Assessments"] is not None:
    if "_pretest_date" in tables["Assessments"].columns:
        d = tables["Assessments"].dropna(subset=["_pretest_date"]).copy()
        d["which"] = "pretest_date"
        time_cov.append(d[["_pretest_date","which"]].rename(columns={"_pretest_date":"ts"}))
    if "_posttest_date" in tables["Assessments"].columns:
        d = tables["Assessments"].dropna(subset=["_posttest_date"]).copy()
        d["which"] = "posttest_date"
        time_cov.append(d[["_posttest_date","which"]].rename(columns={"_posttest_date":"ts"}))
if tables["Telemetry"] is not None and "_activity_ts" in tables["Telemetry"].columns:
    d = tables["Telemetry"].dropna(subset=["_activity_ts"]).copy()
    d["which"] = "activity_timestamp"
    time_cov.append(d[["_activity_ts","which"]].rename(columns={"_activity_ts":"ts"}))

if time_cov:
    tc = pd.concat(time_cov, ignore_index=True)
    tc["date"] = pd.to_datetime(tc["ts"]).dt.date
    df_cnt = tc.groupby(["which","date"]).size().reset_index(name="n")
    fig_time = px.line(df_cnt, x="date", y="n", color="which",
                       title="(7) Time Coverage by Source (Daily Counts)")
    fig_time.update_layout(template="plotly_white", xaxis_title="Date", yaxis_title="Rows per day")
    st.plotly_chart(fig_time, use_container_width=True)

# -----------------------
# Chart 8: Availability Sankey — table → has student_id / missing
# -----------------------
nodes, links = [], []
# nodes: each table + availability states
tbl_nodes = list(tables.keys())
avail_nodes = ["has_student_id", "missing_student_id"]
nodes = tbl_nodes + avail_nodes
node_index = {n:i for i,n in enumerate(nodes)}
for name, df in tables.items():
    if df is None:
        continue
    if "student_id" in df.columns:
        n_has = df["student_id"].notna().sum()
        n_miss = df["student_id"].isna().sum()
        links.append(dict(source=node_index[name], target=node_index["has_student_id"], value=int(n_has)))
        links.append(dict(source=node_index[name], target=node_index["missing_student_id"], value=int(n_miss)))
    else:
        links.append(dict(source=node_index[name], target=node_index["missing_student_id"], value=len(df) if df is not None else 0))

if links:
    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(label=nodes, pad=10, thickness=12),
        link=dict(source=[l["source"] for l in links],
                  target=[l["target"] for l in links],
                  value=[l["value"] for l in links])
    )])
    fig_sankey.update_layout(title="(8) Availability Flow: Table → student_id Presence", template="plotly_white")
    st.plotly_chart(fig_sankey, use_container_width=True)

# -----------------------
# Export: compact alignment report
# -----------------------
with st.expander("Download compact alignment report", expanded=False):
    rep_rows = []
    for name, df in tables.items():
        if df is None:
            rep_rows.append(dict(table=name, rows=0, has_student_id=False, has_time=False))
            continue
        has_sid = "student_id" in df.columns
        has_time_any = any(c in df.columns for c in ["pretest_date","posttest_date","activity_timestamp","log_timestamp",
                                                     "_pretest_date","_posttest_date","_activity_ts"])
        rep_rows.append(dict(table=name, rows=len(df), has_student_id=has_sid, has_time=has_time_any))
    df_rep = pd.DataFrame(rep_rows)
    st.dataframe(df_rep, use_container_width=True)
    st.download_button(
        "Download alignment_report.csv",
        data=df_rep.to_csv(index=False).encode("utf-8"),
        file_name="alignment_report.csv",
        mime="text/csv",
        key="dl_alignment_report"
    )

st.divider()

# ============================================================
# 1) Primary Model (Three-level LMM — approximated) & 2×2 Factorial
# ============================================================
st.header("1) Primary Model (Three-level LMM — approximated) & 2×2 Factorial")

st.markdown("### Statistical Modeling Plan")

st.markdown("""
**Objective.**  
Estimate the *Average Treatment Effect (ATE)* of the **EDU-AI** intervention and, in a 2×2 factorial design, test the interaction between *AI* and *Scaffolding*.  
The ATE captures the mean difference in post-test outcomes attributable to treatment assignment after controlling for baseline performance.  
In the factorial case, the *AI × Scaffolding* interaction examines whether adaptive AI feedback combined with instructor scaffolding produces synergistic learning gains beyond their individual effects.
""")

st.markdown("""
**Recommended Analytic Model.**  
Use a **linear mixed-effects model** with random intercepts for **site** and **class** to account for the nested data structure (students within classes, classes within sites).  
Apply **Restricted Maximum Likelihood (REML)** for unbiased estimation of variance components.  
Compute degrees of freedom using **Satterthwaite** or **Kenward–Roger** approximations, and use **cluster-robust standard errors (SE)** for inference at the class or site level.
""")

st.markdown("**Conceptual Outcome Model (post-test adjusted by pre-test):**")
st.latex(r"Y_{ijk} = \beta_0 + \beta_1 \mathrm{EDUAI}_{jk} + \beta_2 \mathrm{Pre}_{ijk} + u_k + v_{jk} + \varepsilon_{ijk}")

st.markdown(r"""
where  

- $Y_{ijk}$: post-test score for student $i$ in class $j$ within site $k$;  
- $\mathrm{EDUAI}_{jk}$: class-level treatment indicator;  
- $\mathrm{Pre}_{ijk}$: pre-test covariate (baseline performance);  
- $u_k \sim \mathcal{N}(0, \sigma^2_u)$: random site effect;  
- $v_{jk} \sim \mathcal{N}(0, \sigma^2_v)$: random class effect;  
- $\varepsilon_{ijk} \sim \mathcal{N}(0, \sigma^2_\varepsilon)$: individual-level residual error.
""")

st.markdown(r"""
**Interpretation.**  
The coefficient $\beta_1$ represents the **adjusted mean difference** in post-test outcomes between EDU-AI and control groups,  
after controlling for pre-test performance and accounting for variation across sites and classes.  
Including random intercepts at both the site and class levels corrects for within-group dependency  
and prevents underestimation of standard errors due to nested data structures.
""")
st.markdown("**Standardized Effect Size (approximation):**")
st.latex(r"d_{\text{adj}} \approx \frac{\widehat{\beta}_1}{\hat{\sigma}_\mathrm{resid}}")

st.markdown("""
where \$\\widehat{\\beta}_1\$ is the estimated treatment coefficient and \$\\hat{\\sigma}_\\mathrm{resid}\$ is the residual standard deviation from the fitted model.  
This provides a standardized mean difference comparable to Cohen’s *d*, allowing for effect-size interpretation across studies.
""")

st.markdown("""
**Note.**  
If the multilevel model cannot be estimated due to small cluster sizes or convergence issues, use **ordinary least squares (OLS)** with **cluster-robust SEs** (clustered by class or site) as an approximate but interpretable alternative.
""")

# --------------------------
# Build modeling dataset
# --------------------------
if assess is None:
    st.info("Please upload *Assessments* on the **Upload** page first.")
else:
    af = assess.copy()
    if {"student_id","phase","score"}.issubset(af.columns):
        # Aggregate item/row-level scores to one Pre and one Post per student
        phase_scores = (
            af.groupby(["student_id","phase"], dropna=False)["score"]
              .mean().reset_index()
        )
        pre = phase_scores[phase_scores["phase"].str.contains("pre", case=False, na=False)][["student_id","score"]]
        post = phase_scores[phase_scores["phase"].str.contains("post", case=False, na=False)][["student_id","score"]]
        pre = pre.rename(columns={"score":"pre"})
        post = post.rename(columns={"score":"post"})
        merged = pd.merge(pre, post, on="student_id", how="inner")

        # Bring design keys if present (one row per student)
        for c in ["class_id","site_id","condition","ai","scaffolding"]:
            if c in af.columns:
                merged = merged.merge(
                    af[["student_id", c]].drop_duplicates("student_id"),
                    on="student_id", how="left"
                )

        # Optional student-level covariates from surveys
        if surveys is not None:
            covars = [c for c in [
                "gpa","ai_familiarity","self_efficacy_pre","mslq_motivation_pre",
                "mslq_strategy_pre"
            ] if c in surveys.columns]
            if covars:
                cov_map = (surveys.groupby("student_id")[covars]
                                  .mean(numeric_only=True).reset_index())
                merged = merged.merge(cov_map, on="student_id", how="left")

        st.subheader("Modeling Data — Preview")
        st.dataframe(merged.head(20), use_container_width=True)

        # --------------------------
        # Controls & Model Formula
        # --------------------------
        st.subheader("Model Specification")
        cov_candidates = [c for c in ["gpa","ai_familiarity","self_efficacy_pre","mslq_motivation_pre","mslq_strategy_pre"] if c in merged.columns]
        use_covs = st.multiselect("Select additional covariates (optional):", options=cov_candidates, default=cov_candidates[:2])

        # treatment options
        treat_opts = ["<none>"]
        if "condition" in merged.columns: treat_opts.append("condition")     # categorical condition (e.g., 4-arm or 2×2 coding)
        if "ai" in merged.columns:        treat_opts.append("ai")            # binary 0/1
        if "scaffolding" in merged.columns: treat_opts.append("scaffolding") # binary 0/1

        treat_col = st.selectbox("Select treatment variable:", options=treat_opts, index=(1 if "condition" in treat_opts else 0))
        include_interaction = False
        if {"ai","scaffolding"}.issubset(merged.columns):
            include_interaction = st.checkbox("Include AI × Scaffolding interaction (2×2)", value=False)

        # cluster column for robust SE
        cluster_col = st.selectbox(
            "Cluster-robust SE by:",
            options=["<none>"] + [c for c in ["class_id","site_id"] if c in merged.columns],
            index=(1 if "class_id" in merged.columns else (2 if "site_id" in merged.columns else 0))
        )
        cluster_col = None if cluster_col == "<none>" else cluster_col

        # build formula
        rhs_terms = ["pre"] + use_covs
        if treat_col != "<none>":
            if treat_col == "condition":
                rhs_terms = [f"C({treat_col})"] + rhs_terms
            else:
                rhs_terms = [treat_col] + rhs_terms
        if include_interaction:
            rhs_terms = [t for t in rhs_terms if t not in ["ai","scaffolding"]]  # will add explicitly
            rhs_full = "pre " + (" + " + " + ".join(use_covs) if use_covs else "")
            fml = f"post ~ ai * scaffolding + {rhs_full}"
        else:
            fml = "post ~ " + " + ".join(rhs_terms)

        st.caption(f"**Fitted formula:** `{fml}`")

        # --------------------------
        # Fit model (statsmodels if available)
        # --------------------------
        HAS_SM = 'smf' in globals()
        model, model_hc3 = None, None
        if HAS_SM:
            dfm = merged.dropna(subset=["post","pre"]).copy()
            try:
                if cluster_col:
                    model = smf.ols(fml, data=dfm).fit(cov_type="cluster", cov_kwds={"groups": dfm[cluster_col]})
                    model_hc3 = smf.ols(fml, data=dfm).fit(cov_type="HC3")
                    st.info(f"Fitted OLS with cluster-robust SE by **{cluster_col}** (and HC3 for comparison).")
                else:
                    model = smf.ols(fml, data=dfm).fit(cov_type="HC3")
                    st.info("Fitted OLS with **HC3** robust SE (no clustering column selected).")

                # Coefficient table (cluster robust as primary)
                coefs = (model.summary2().tables[1]
                         .rename(columns={"Coef.":"coef","Std.Err.":"se","P>|t|":"p","[0.025":"ci_lo","0.975]":"ci_hi"}))
                coefs = coefs.reset_index().rename(columns={"index":"term"})
                coefs["term"] = coefs["term"].astype(str)

                # Effect-size proxy for the main treatment parameter(s)
                resid_sd = float(np.sqrt(np.nanvar(model.resid, ddof=1)))
                def effect_rows_for_treatment(df_coef, resid_sd):
                    rows = []
                    for t in df_coef["term"]:
                        if t.startswith("C(condition)[T.") or t in ["ai","scaffolding","ai:scaffolding"]:
                            beta = float(df_coef.loc[df_coef["term"]==t, "coef"].values[0])
                            d_adj = beta / resid_sd if resid_sd > 0 else np.nan
                            rows.append(dict(term=t, beta=beta, d_adj=d_adj))
                        # For the 2×2 case add the interaction explicitly
                        if include_interaction and t == "ai:scaffolding":
                            beta = float(df_coef.loc[df_coef["term"]==t, "coef"].values[0])
                            d_adj = beta / resid_sd if resid_sd > 0 else np.nan
                            rows.append(dict(term=t, beta=beta, d_adj=d_adj))
                    return pd.DataFrame(rows)
                eff_tbl = effect_rows_for_treatment(coefs, resid_sd)

                # --------------------------
                # Advanced visuals (7–8)
                # --------------------------

                st.subheader("Model Outputs & Advanced Diagnostics")

                # 1) Forest (coefficient) plot with 95% CI
                fig_coef = go.Figure()
                plot_df = coefs.copy()
                plot_df = plot_df[plot_df["term"]!="Intercept"]
                fig_coef.add_trace(go.Scatter(
                    x=plot_df["coef"], y=plot_df["term"],
                    mode="markers",
                    error_x=dict(type="data", symmetric=False,
                                 array=(plot_df["ci_hi"]-plot_df["coef"]),
                                 arrayminus=(plot_df["coef"]-plot_df["ci_lo"])),
                    name="Coef (95% CI)"
                ))
                fig_coef.update_layout(template="plotly_white",
                                       title="(1) Coefficient Forest Plot (robust SE)",
                                       xaxis_title="Estimate", yaxis_title="")
                st.plotly_chart(fig_coef, use_container_width=True)

                # 2) Coefficient table (Plotly)
                fig_tbl = go.Figure(data=[go.Table(
                    header=dict(values=list(coefs.columns), fill_color="#2c3e50", font=dict(color="white")),
                    cells=dict(values=[coefs[c] for c in coefs.columns], align="left")
                )])
                fig_tbl.update_layout(template="plotly_white", title="(2) Robust Coefficient Table")
                st.plotly_chart(fig_tbl, use_container_width=True)

                # 3) Partial effect curve for Pre (marginal lines by treatment/group)
                #    We grid pre over its range and hold other covariates at mean.
                grid = pd.DataFrame({"pre": np.linspace(dfm["pre"].min(), dfm["pre"].max(), 80)})
                for cv in use_covs:
                    grid[cv] = float(pd.to_numeric(dfm[cv], errors="coerce").mean())
                if treat_col == "condition":
                    levels = dfm["condition"].dropna().unique().tolist()
                    out_frames = []
                    for lv in levels:
                        gg = grid.copy()
                        gg["condition"] = lv
                        gg["yhat"] = model.predict(gg)
                        gg["group"] = f"cond={lv}"
                        out_frames.append(gg)
                    pe = pd.concat(out_frames)
                elif include_interaction and {"ai","scaffolding"}.issubset(dfm.columns):
                    combos = [(0,0),(1,0),(0,1),(1,1)]
                    out_frames = []
                    for a,s in combos:
                        gg = grid.copy()
                        gg["ai"] = a; gg["scaffolding"] = s
                        gg["yhat"] = model.predict(gg)
                        gg["group"] = f"AI={a}, SCAFF={s}"
                        out_frames.append(gg)
                    pe = pd.concat(out_frames)
                elif treat_col in dfm.columns and treat_col != "<none>":
                    # binary or numeric
                    vals = sorted(pd.Series(dfm[treat_col]).dropna().unique().tolist())
                    vals = vals[:4]  # limit legend size
                    out_frames = []
                    for v in vals:
                        gg = grid.copy()
                        gg[treat_col] = v
                        gg["yhat"] = model.predict(gg)
                        gg["group"] = f"{treat_col}={v}"
                        out_frames.append(gg)
                    pe = pd.concat(out_frames)
                else:
                    pe = grid.copy()
                    pe["yhat"] = model.predict(grid)
                    pe["group"] = "marginal"

                fig_pe = px.line(pe, x="pre", y="yhat", color="group",
                                 title="(3) Partial Effect of Pre on Post (by treatment)")
                fig_pe.update_layout(template="plotly_white", xaxis_title="Pre", yaxis_title="Predicted Post")
                st.plotly_chart(fig_pe, use_container_width=True)

                # 4) Residuals vs Fitted
                rvf = pd.DataFrame({"fitted": model.fittedvalues, "resid": model.resid})
                fig_rvf = px.scatter(rvf, x="fitted", y="resid", trendline="ols",
                                     title="(4) Residuals vs Fitted (robust OLS trend)")
                fig_rvf.update_layout(template="plotly_white", xaxis_title="Fitted", yaxis_title="Residuals")
                st.plotly_chart(fig_rvf, use_container_width=True)

                # 5) Normal Q–Q plot of residuals
                res = pd.Series(model.resid).dropna().to_numpy()
                if len(res) > 5:
                    res_std = (res - res.mean()) / (res.std() if res.std() else 1)
                    q = np.linspace(0.01, 0.99, len(res_std))
                    norm_q = stats.norm.ppf(q) if 'stats' in globals() else np.quantile(res_std, q)
                    df_qq = pd.DataFrame({"theoretical": norm_q, "sample": np.sort(res_std)})
                    fig_qq = px.scatter(df_qq, x="theoretical", y="sample", title="(5) Normal Q–Q of Residuals")
                    # 45-degree line
                    lo, hi = df_qq["theoretical"].min(), df_qq["theoretical"].max()
                    fig_qq.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="45° line"))
                    fig_qq.update_layout(template="plotly_white", xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles")
                    st.plotly_chart(fig_qq, use_container_width=True)

                # 6) Influence (Cook's distance) — top 20
                infl = None
                try:
                    infl = model.get_influence()
                    cooks = infl.cooks_distance[0]
                    topk = min(20, len(cooks))
                    idx = np.argsort(cooks)[-topk:]
                    df_cook = pd.DataFrame({"row": idx, "cooksD": cooks[idx]})
                    fig_cook = px.bar(df_cook, x="row", y="cooksD", title="(6) Top Cook's Distance (Influential Points)")
                    fig_cook.update_layout(template="plotly_white")
                    st.plotly_chart(fig_cook, use_container_width=True)
                except Exception:
                    pass

                # 7) Outcome distribution by treatment (violin + box)
                if treat_col != "<none>":
                    plt_df = dfm.copy()
                    if treat_col == "condition":
                        fig_vio = px.violin(plt_df, x="condition", y="post", box=True, points="all",
                                            title="(7) Post Score by Condition (Violin + Box)")
                    else:
                        fig_vio = px.violin(plt_df, x=treat_col, y="post", box=True, points="all",
                                            title=f"(7) Post Score by {treat_col} (Violin + Box)")
                    fig_vio.update_layout(template="plotly_white", yaxis_title="Post")
                    st.plotly_chart(fig_vio, use_container_width=True)

                # 8) ECDF of Post by treatment
                if treat_col != "<none>":
                    ec = dfm.copy()
                    grpvar = "condition" if treat_col=="condition" else treat_col
                    curves = []
                    for g, sub in ec.groupby(grpvar):
                        y = pd.to_numeric(sub["post"], errors="coerce").dropna().sort_values().to_numpy()
                        if len(y) == 0: 
                            continue
                        F = np.linspace(0, 1, len(y))
                        curves.append(pd.DataFrame({grpvar:[g]*len(y), "post":y, "ecdf":F}))
                    if curves:
                        df_ec = pd.concat(curves)
                        fig_ecdf = px.line(df_ec, x="post", y="ecdf", color=grpvar, title="(8) ECDF of Post by Treatment")
                        fig_ecdf.update_layout(template="plotly_white", yaxis_title="Proportion ≤ y")
                        st.plotly_chart(fig_ecdf, use_container_width=True)

                # Effect-size table (beta & d_adj)
                if not eff_tbl.empty:
                    eff_fig = go.Figure(data=[go.Table(
                        header=dict(values=list(eff_tbl.columns), fill_color="#2c3e50", font=dict(color="white")),
                        cells=dict(values=[eff_tbl[c] for c in eff_tbl.columns], align="left")
                    )])
                    eff_fig.update_layout(template="plotly_white", title="Treatment Effect Proxies (β & d_adj)")
                    st.plotly_chart(eff_fig, use_container_width=True)

                # Downloadables
                st.subheader("Downloads")
                st.download_button("Download robust coefficients (CSV)",
                                   data=coefs.to_csv(index=False).encode("utf-8"),
                                   file_name="model_coefficients_robust.csv",
                                   mime="text/csv")
                if not eff_tbl.empty:
                    st.download_button("Download effect-size proxies (CSV)",
                                       data=eff_tbl.to_csv(index=False).encode("utf-8"),
                                       file_name="effect_sizes_proxy.csv",
                                       mime="text/csv")

                # Optional: show HC3 vs Cluster-robust side-by-side (if cluster selected)
                if model_hc3 is not None:
                    st.subheader("Robustness Check: HC3 vs Cluster-robust SE")
                    t1 = (model_hc3.summary2().tables[1]
                          .rename(columns={"Coef.":"coef","Std.Err.":"se","P>|t|":"p","[0.025":"ci_lo","0.975]":"ci_hi"})
                          .reset_index().rename(columns={"index":"term"}))
                    t1["estimator"] = "HC3"
                    t2 = (model.summary2().tables[1]
                          .rename(columns={"Coef.":"coef","Std.Err.":"se","P>|t|":"p","[0.025":"ci_lo","0.975]":"ci_hi"})
                          .reset_index().rename(columns={"index":"term"}))
                    t2["estimator"] = f"cluster({cluster_col})"
                    comb = pd.concat([t1,t2], ignore_index=True)
                    fig_rb = go.Figure(data=[go.Table(
                        header=dict(values=list(comb.columns), fill_color="#2c3e50", font=dict(color="white")),
                        cells=dict(values=[comb[c] for c in comb.columns], align="left")
                    )])
                    fig_rb.update_layout(template="plotly_white", title="HC3 vs Cluster-robust Coefficients")
                    st.plotly_chart(fig_rb, use_container_width=True)

            except Exception as e:
                st.warning(f"Model fitting failed: {e}")
        else:
            st.info("`statsmodels` is not available. Please install it to run the robust OLS approximation.")
    else:
        st.warning("`Assessments` must contain columns: student_id, phase, score (to aggregate to Pre/Post).")

st.divider()

# ============================================================
# 2) Missing Data Handling (MICE — Multiple Imputation)
# ============================================================
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.header("2) Missing Data Handling (MICE — Multiple Imputation)")


def _has(key: str) -> bool:
    return key in st.session_state and st.session_state[key] is not None

def _df_download(df: pd.DataFrame, fname: str, label: str, key: str):
    st.download_button(label, data=df.to_csv(index=False).encode("utf-8"),
                       file_name=fname, mime="text/csv", key=key)

def ensure_student_id(df: pd.DataFrame, table_name: str = "Telemetry") -> pd.DataFrame:
    """
    Ensure df has a 'student_id' column by auto-detecting a likely ID column,
    offering a UI override, and renaming the chosen column to 'student_id'.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    d = df.copy()
    d.columns = [c.strip().lower() for c in d.columns]
    if "student_id" in d.columns:
        d["student_id"] = d["student_id"].astype(str).str.strip()
        return d

    candidates = [
        "student_id","user_id","learner_id","sid","student","id",
        "uuid","pseudo_id","account_id","actor_id","person_id"
    ]
    auto = next((c for c in candidates if c in d.columns), None)
    options = [auto] if auto else []
    options += [c for c in d.columns if c not in options]
    if options:
        picked = st.selectbox(
            f"Select identifier column for {table_name} → treat as student_id",
            options=options, index=0
        )
    else:
        st.error(f"{table_name}: No columns available to map to 'student_id'.")
        return d

    if picked and picked != "student_id":
        d = d.rename(columns={picked: "student_id"})
    d["student_id"] = d["student_id"].astype(str).str.strip()
    return d

# ---------- Narrative & formulas ----------
st.markdown("### Multiple Imputation and Rubin’s Pooling")

st.markdown(r"""
**Goal.**  
The goal of multiple imputation is to handle missing data in a statistically principled manner while preserving the structure and variability of the original dataset.  
We apply **chained-equations multiple imputation** (typically around 20 imputations, $M \approx 20$) to generate several complete datasets, each containing plausible values for the missing entries based on predictive models.  

To maintain data validity, imputations preserve the **hierarchical structure** (e.g., students nested within classes) by including relevant group-level predictors in the imputation process.  
Each imputed dataset is then analyzed **independently** using the same model specification.  

Finally, the resulting parameter estimates and variances are combined using **Rubin’s rules**, ensuring that both the **within-imputation** and **between-imputation** sources of uncertainty are properly incorporated into the final inference.
""")

st.markdown("""
**Rubin’s Pooling Framework.**  
Let $ Q_m $ be the estimate of a parameter (e.g., a regression coefficient) from imputed dataset $ m $, and $ U_m $ its associated within-imputation variance.  
Across all $ M $ imputations, we compute the pooled quantities as follows:
""")

st.latex(r"\bar{Q} = \frac{1}{M}\sum_{m=1}^{M} Q_m,\quad \bar{U} = \frac{1}{M}\sum_{m=1}^{M} U_m,\quad B = \frac{1}{M-1}\sum_{m=1}^{M} (Q_m - \bar{Q})^2")

st.markdown(r"""
Here:  

- $\bar{Q}$: **pooled point estimate**, representing the average parameter estimate across all imputations;  
- $\bar{U}$: **average within-imputation variance**, reflecting sampling uncertainty within each complete dataset;  
- $B$: **between-imputation variance**, indicating how much parameter estimates differ across imputations due to uncertainty in the missing-data mechanism.  

Intuitively, $\bar{U}$ captures the variability arising from estimation within each dataset,  
whereas $B$ quantifies the additional uncertainty introduced by imputation—that is, by not knowing the true values of the missing data.
""")

st.markdown("""
**Total Variance and Standard Error.**  
The total uncertainty of the pooled estimate combines both sources of variance, with an adjustment for the finite number of imputations:
""")

st.latex(r"T = \bar{U} + \left(1+\frac{1}{M}\right) B,\quad \text{SE}(\bar{Q}) = \sqrt{T}")

st.markdown("""
- $ T $ is the **total variance**, capturing both within- and between-imputation variation.  
- The adjustment factor $ (1 + 1/M) $ slightly inflates the between-imputation variance $ B $ to account for the limited number of imputations.  
- The **standard error** of the pooled estimate is simply $ \sqrt{T} $, which can be used for hypothesis testing and constructing confidence intervals.

Practically, if the imputations are consistent and $ B $ is small, the uncertainty added by imputation is minimal.  
If $ B $ is large, it indicates that missing data have a meaningful effect on the stability of estimates.
""")

st.markdown("""
**Small-Sample Degrees of Freedom (Barnard–Rubin Approximation).**  
When the number of imputations $ M $ is small or the sample size is limited, degrees of freedom for $ t $-based inference can be estimated using the **Barnard–Rubin** adjustment:
""")

st.latex(r"\nu \approx \left( \frac{M-1}{\lambda^2} \right),\quad \lambda = \frac{(1+1/M)B}{T}")

st.markdown("""
Here:  
- $ \lambda $ is the **fraction of missing information**, quantifying how much uncertainty comes from imputation rather than sampling.  
- $ \nu $ represents the **effective degrees of freedom** for the $ t $-distribution used to compute confidence intervals.  
This correction avoids overconfidence in the results when the number of imputations is small, ensuring the confidence intervals remain conservative and well-calibrated.
""")

st.markdown("""
**95% Confidence Interval.**
""")

st.latex(r"\bar{Q} \pm t_{0.975,\nu}\,\sqrt{T}")

st.markdown("""
The final confidence interval combines both estimation and imputation uncertainty using the total variance $ T $ and adjusted degrees of freedom $ \nu $.  
This ensures that the reported uncertainty accurately reflects both the variability due to missing data and the inherent sampling error.

In short, Rubin’s framework provides a rigorous method for combining information across multiple imputations,  
yielding statistically valid point estimates, standard errors, and confidence intervals that properly account for missingness uncertainty.
""")

# ---------- Availability of IterativeImputer ----------
try:
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    HAS_IMPUTE = True
except Exception:
    HAS_IMPUTE = False

with st.expander("Run MICE (demo: Assessments + Telemetry + Surveys)", expanded=True):
    if not HAS_IMPUTE:
        st.warning("`sklearn` IterativeImputer not available. Please install `scikit-learn>=0.24` to enable MICE.")
        st.stop()

    # --------------------------
    # A) Build a joint analysis frame
    # --------------------------
    assess = st.session_state["assess_df"].copy() if _has("assess_df") else None
    surveys = st.session_state["surveys_df"].copy() if _has("surveys_df") else None
    tele    = (st.session_state["tele_df"].copy() if _has("tele_df")
               else (st.session_state["telemetry_with_pei"].copy() if _has("telemetry_with_pei") else None))

    for df in [assess, surveys, tele]:
        if df is not None:
            df.columns = [c.strip().lower() for c in df.columns]

    base = None
    if assess is not None and {"student_id","phase","score"}.issubset(assess.columns):
        ps = (assess.groupby(["student_id","phase"], dropna=False)["score"]
                    .mean().reset_index())
        pre  = ps[ps["phase"].str.contains("pre", case=False, na=False)][["student_id","score"]].rename(columns={"score":"pre"})
        post = ps[ps["phase"].str.contains("post", case=False, na=False)][["student_id","score"]].rename(columns={"score":"post"})
        base = pd.merge(pre, post, on="student_id", how="outer")
    else:
        st.info("Assessments must contain `student_id, phase, score` to demonstrate MICE.")
        st.stop()

    # Optional covariates from surveys
    if surveys is not None:
        cand_s = [c for c in [
            "self_efficacy_pre","self_efficacy_post",
            "mslq_motivation_pre","mslq_motivation_post",
            "gpa","ai_familiarity"
        ] if c in surveys.columns]
        if cand_s:
            sv = surveys.groupby("student_id")[cand_s].mean(numeric_only=True).reset_index()
            base = base.merge(sv, on="student_id", how="left")

    # Optional telemetry usage signals (ensure student_id first)
    tv = None
    if tele is not None:
        tele = ensure_student_id(tele, table_name="Telemetry")
        cand_t = [c for c in ["usage_minutes","num_turns","hint_requests","dwell_time",
                              "abandon_flag","prompt_evolution_index"] if c in tele.columns]
        if "student_id" not in tele.columns:
            st.error("Telemetry still has no 'student_id' after mapping. Cannot aggregate usage by student.")
        else:
            if cand_t:
                tele[cand_t] = tele[cand_t].apply(pd.to_numeric, errors="coerce")
                tv = (tele.groupby("student_id")[cand_t]
                           .mean(numeric_only=True)
                           .reset_index())
                base = base.merge(tv, on="student_id", how="left")

    st.write("Input matrix for imputation — preview:")
    st.dataframe(base.head(15), use_container_width=True)

    # --------------------------
    # B) Missingness diagnostics (advanced)
    # --------------------------
    st.subheader("Diagnostics before Imputation")

    num_cols = [c for c in base.columns if c != "student_id"]
    X = base[num_cols].copy()

    # (1) Missingness rate bar
    miss_rate = X.isna().mean().sort_values(ascending=False).reset_index()
    miss_rate.columns = ["variable","missing_rate"]
    fig_miss = px.bar(miss_rate, x="variable", y="missing_rate", title="(1) Missingness Rate by Variable")
    fig_miss.update_layout(template="plotly_white", yaxis_tickformat=".0%")
    st.plotly_chart(fig_miss, use_container_width=True)

    # (2) Missingness heatmap (row-sampled for speed)
    samp = X.sample(min(400, len(X)), random_state=42) if len(X) > 400 else X.copy()
    try:
        fig_heat = px.imshow(samp.isna().astype(int),
                             labels=dict(color="Missing"),
                             title="(2) Missingness Pattern (sampled rows)",
                             aspect="auto")
    except Exception:
        # Fallback for older Plotly versions
        fig_heat = go.Figure(data=go.Heatmap(z=samp.isna().astype(int).to_numpy(),
                                             x=samp.columns, y=list(range(len(samp))),
                                             coloraxis="coloraxis"))
        fig_heat.update_layout(coloraxis_colorscale="Blues",
                               title="(2) Missingness Pattern (sampled rows)")
    fig_heat.update_layout(template="plotly_white")
    st.plotly_chart(fig_heat, use_container_width=True)

    # (3) Pairwise missingness co-occurrence
    miss_co = pd.DataFrame(
        np.dot(X.isna().astype(int).T, X.isna().astype(int)),
        index=num_cols, columns=num_cols
    )
    fig_co = px.imshow(miss_co, title="(3) Pairwise Missingness Co-occurrence (counts)")
    fig_co.update_layout(template="plotly_white")
    st.plotly_chart(fig_co, use_container_width=True)

    # --------------------------
    # C) Multiple Imputation (M datasets)
    # --------------------------
    st.subheader("Multiple Imputation (MICE)")

    M = st.number_input("Number of imputations (M)", 5, 50, 20, 1, key="mice_M")
    max_iter = st.number_input("Max iterations per imputation", 5, 50, 20, 1, key="mice_iter")

    imputed_sets = []
    seeds = np.random.SeedSequence(42).spawn(int(M))
    for m, ss in enumerate(seeds, start=1):
        rng = np.random.default_rng(ss.entropy)
        imp = IterativeImputer(random_state=int(ss.entropy), max_iter=int(max_iter), sample_posterior=True)
        Xi = imp.fit_transform(X)  # shape: (n, p)
        Xi = pd.DataFrame(Xi, columns=num_cols, index=X.index)
        ds = base[["student_id"]].copy()
        ds[num_cols] = Xi
        ds["_m"] = m
        imputed_sets.append(ds)

    imp_long = pd.concat(imputed_sets, ignore_index=True)

    st.write("Imputed datasets (long format) — preview:")
    st.dataframe(imp_long.head(15), use_container_width=True)
    _df_download(imp_long, "imputed_long.csv", "Download all imputations (long)", key="dl_imp_long")

    # --------------------------
    # D) Post-imputation diagnostics (7–8 visuals/tables)
    # --------------------------
    st.subheader("Post-imputation Diagnostics")

    # (4) Distribution overlay: pre/post by imputation (violin+box)
    melted = imp_long.melt(id_vars=["student_id","_m"], value_vars=[c for c in ["pre","post"] if c in num_cols],
                           var_name="metric", value_name="value")
    if not melted.empty:
        fig_vio = px.violin(melted, x="metric", y="value", color="_m", points=False, box=True,
                            title="(4) Post-Imputation Distributions by Imputation (Violin+Box)")
        fig_vio.update_layout(template="plotly_white", legend_title="Imputation m")
        st.plotly_chart(fig_vio, use_container_width=True)

    # (5) ECDF overlay of a key variable (choose)
    var_for_ecdf = st.selectbox("Variable for ECDF overlay", options=num_cols, index=0, key="mice_ecdf_var")
    ecdfs = []
    for m, grp in imp_long.groupby("_m"):
        vals = pd.to_numeric(grp[var_for_ecdf], errors="coerce").dropna().sort_values().to_numpy()
        if len(vals) == 0:
            continue
        F = np.linspace(0, 1, len(vals))
        ecdfs.append(pd.DataFrame({"_m": m, "x": vals, "F": F}))
    if ecdfs:
        ecdf_df = pd.concat(ecdfs, ignore_index=True)
        fig_ecdf = px.line(ecdf_df, x="x", y="F", color="_m", title=f"(5) ECDF of {var_for_ecdf} by Imputation")
        fig_ecdf.update_layout(template="plotly_white", xaxis_title=var_for_ecdf, yaxis_title="Proportion ≤ x")
        st.plotly_chart(fig_ecdf, use_container_width=True)

# (6) Correlation heatmap averaged over imputations
    cors = []
    for m, grp in imp_long.groupby("_m"):
        # Coerce each column to numeric safely, leave non-numeric as NaN
        gnum = grp[num_cols].apply(pd.to_numeric, errors="coerce")
        # If there are fewer than 2 valid numeric columns, skip
        valid_cols = [c for c in gnum.columns if gnum[c].notna().sum() > 1]
        if len(valid_cols) < 2:
            continue
        cors.append(gnum[valid_cols].corr())

    if cors:
        # Align all correlation matrices (in case columns differ across imputations)
        # and take the mean over common columns
        all_cols = sorted(set().union(*[c.columns for c in cors]))
        aligned = []
        for C in cors:
            aligned.append(C.reindex(index=all_cols, columns=all_cols))
        cor_mean = sum(A.fillna(0) for A in aligned) / len(aligned)

        fig_cor = px.imshow(
            cor_mean,
            title="(6) Mean Correlation Matrix across Imputations",
            aspect="auto",
            color_continuous_scale="RdBu",
            zmin=-1, zmax=1
        )
        fig_cor.update_layout(template="plotly_white")
        st.plotly_chart(fig_cor, use_container_width=True)
    else:
        st.info("Not enough numeric overlap to compute a correlation heatmap across imputations.")



    # (7) Variation across imputations (per-variable SD of mean)
    var_stats = []
    for c in num_cols:
        means = [pd.to_numeric(g[c], errors="coerce").mean() for _, g in imp_long.groupby("_m")]
        var_stats.append(dict(variable=c, mean=np.nanmean(means), sd=np.nanstd(means, ddof=1)))
    df_var = pd.DataFrame(var_stats)
    fig_var = px.bar(df_var.sort_values("sd", ascending=False), x="variable", y="sd",
                     title="(7) Between-Imputation Variation of Variable Means (SD over m)")
    fig_var.update_layout(template="plotly_white", yaxis_title="SD of mean across imputations")
    st.plotly_chart(fig_var, use_container_width=True)

    # (8) Imputation index vs. summary (trace-like) — convergence proxy
    if "pre" in num_cols:
        trace_df = pd.DataFrame({
            "_m": sorted(imp_long["_m"].unique()),
            "mean_pre": [pd.to_numeric(g["pre"], errors="coerce").mean() for _, g in imp_long.groupby("_m")]
        })
        fig_trace = px.line(trace_df, x="_m", y="mean_pre", markers=True,
                            title="(8) Convergence Proxy: Mean(Pre) across Imputations")
        fig_trace.update_layout(template="plotly_white", xaxis_title="Imputation m", yaxis_title="Mean(Pre)")
        st.plotly_chart(fig_trace, use_container_width=True)

    # --------------------------
    # E) Example model on each imputed dataset & Rubin pooling
    # --------------------------
    st.subheader("Rubin Pooling: Example OLS (post ~ pre)")
    # Minimal example: OLS beta_pre & its variance on each imputed set
    results = []
    for m, grp in imp_long.groupby("_m"):
        d = grp.dropna(subset=["pre","post"]).copy() if {"pre","post"}.issubset(grp.columns) else pd.DataFrame()
        if len(d) < 5:
            continue
        # OLS with intercept: post = b0 + b1*pre + e
        X_ = np.column_stack([np.ones(len(d)), pd.to_numeric(d["pre"], errors="coerce").to_numpy()])
        y_ = pd.to_numeric(d["post"], errors="coerce").to_numpy()
        ok = np.isfinite(X_).all(axis=1) & np.isfinite(y_)
        X_, y_ = X_[ok], y_[ok]
        if len(y_) < 5:
            continue
        beta_hat, residuals, rank, s = np.linalg.lstsq(X_, y_, rcond=None)
        # variance of beta: sigma^2 * (X'X)^-1
        if residuals.size == 0:
            resid = y_ - X_ @ beta_hat
            sigma2 = (resid @ resid) / max(1, (len(y_) - X_.shape[1]))
        else:
            sigma2 = residuals[0] / max(1, (len(y_) - X_.shape[1]))
        XtX_inv = np.linalg.inv(X_.T @ X_)
        var_beta = sigma2 * XtX_inv
        b1 = float(beta_hat[1])
        v1 = float(var_beta[1,1])
        results.append(dict(m=int(m), beta_pre=b1, var_beta_pre=v1, n=len(y_)))

    if not results:
        st.info("Not enough overlap to fit the demo model on imputed sets.")
    else:
        res_df = pd.DataFrame(results).sort_values("m")
        Qs = res_df["beta_pre"].to_numpy()
        Us = res_df["var_beta_pre"].to_numpy()
        M_eff = len(Qs)

        Qbar = np.mean(Qs)
        Ubar = np.mean(Us)
        B = np.var(Qs, ddof=1) if M_eff > 1 else 0.0
        T = Ubar + (1 + 1/M_eff) * B
        SE = float(np.sqrt(T)) if T >= 0 else np.nan
        lam = ((1 + 1/M_eff) * B) / T if T > 0 else np.nan
        df = ((M_eff - 1) / (lam**2)) if (M_eff > 1 and np.isfinite(lam) and lam > 0) else np.nan

        try:
            from scipy import stats
            tcrit = stats.t.ppf(0.975, df) if np.isfinite(df) and df > 1 else 1.96
        except Exception:
            tcrit = 1.96
        lo, hi = Qbar - tcrit * SE, Qbar + tcrit * SE

        # Table: per-imputation & pooled
        pooled = pd.DataFrame([{
            "pooled_beta_pre": Qbar, "SE": SE, "df": df, "CI_lo": lo, "CI_hi": hi,
            "M": M_eff, "Ubar": Ubar, "B": B, "T": T, "lambda": lam
        }])

        st.write("Per-imputation coefficients (β_pre) and pooled result:")
        col1, col2 = st.columns(2)
        with col1:
            fig_imp = go.Figure(data=[go.Table(
                header=dict(values=list(res_df.columns), fill_color="#2c3e50", font=dict(color="white")),
                cells=dict(values=[res_df[c] for c in res_df.columns], align="left")
            )])
            fig_imp.update_layout(template="plotly_white", title="Per-imputation OLS results")
            st.plotly_chart(fig_imp, use_container_width=True)
        with col2:
            fig_pool = go.Figure(data=[go.Table(
                header=dict(values=list(pooled.columns), fill_color="#2c3e50", font=dict(color="white")),
                cells=dict(values=[pooled[c] for c in pooled.columns], align="left")
            )])
            fig_pool.update_layout(template="plotly_white", title="Rubin-pooled estimate of β_pre")
            st.plotly_chart(fig_pool, use_container_width=True)

        # (9) Forest of β_pre across imputations with pooled CI
        fig_forest = go.Figure()
        fig_forest.add_trace(go.Scatter(x=res_df["beta_pre"], y=res_df["m"], mode="markers",
                                        name="β_pre (per imputation)"))
        fig_forest.add_trace(go.Scatter(x=[lo, hi], y=[-0.5, -0.5], mode="lines",
                                        name="Pooled 95% CI", line=dict(width=4)))
        fig_forest.add_trace(go.Scatter(x=[Qbar, Qbar], y=[-1, res_df["m"].max()+1],
                                        mode="lines", name="Pooled β_pre", line=dict(dash="dash")))
        fig_forest.update_layout(template="plotly_white", title="(9) Forest: β_pre by Imputation & Pooled CI",
                                 xaxis_title="β_pre", yaxis_title="Imputation m")
        st.plotly_chart(fig_forest, use_container_width=True)

        # Downloads
        _df_download(res_df, "per_imputation_ols.csv", "Download per-imputation OLS results (CSV)", key="dl_pi_ols")
        _df_download(pooled, "rubin_pooled_beta_pre.csv", "Download pooled estimate (CSV)", key="dl_pooled")

st.divider()

# ============================================================
# 3) Multiple Outcomes Error-Rate Control & Effect Sizes (BH-FDR)
# ============================================================
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Optional backends
try:
    from scipy import stats
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    HAS_SM = True
except Exception:
    HAS_SM = False

# ---------- Small helpers ----------
def has(key: str) -> bool:
    return key in st.session_state and st.session_state[key] is not None


def df_download(df: pd.DataFrame, fname: str, label: str, key: str = None):
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=fname,
        mime="text/csv",
        key=key or f"dl_{fname}"
    )

def safe_rate(num: float, den: float) -> float:
    num = float(num)
    den = float(den)
    if den <= 0 or not np.isfinite(den):
        return np.nan
    return num / den

def bh_adjust(pvals, q=0.10):
    """
    Return BH critical values and adjusted decisions (q given only for display).
    Output is the vector of BH-adjusted p-values (Benjamini–Hochberg).
    """
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranks = np.arange(1, m + 1)
    p_sorted = p[order]
    # BH adjusted p-values (monotone non-decreasing when sorted)
    adj = np.minimum.accumulate((m / ranks) * p_sorted[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty_like(adj)
    out[order] = adj  # place back to original order
    return out

# ============================================================
# 3) 多重结局的错误率控制与效应量报告（BH-FDR）
# ============================================================
st.header("3) Multiple Outcomes with BH-FDR & Effect Size Reporting")

st.markdown("""
**Goal.**  
When multiple outcomes (e.g., **O1–O5**) are tested, the probability of obtaining false positives increases due to multiple comparisons.  
To maintain inferential validity while retaining statistical power, we control the **False Discovery Rate (FDR)** using the **Benjamini–Hochberg (BH)** procedure.  
If dependence among tests is substantial, the **Benjamini–Yekutieli (BY)** correction can be used for a more conservative adjustment.

In addition to significance testing, we report **standardized effect sizes** and **95% confidence intervals** to convey practical importance, not just statistical significance.
""")

st.markdown("""
**Benjamini–Hochberg (BH) Procedure (Informal Rule).**  
Given a family of $ m $ hypothesis tests with corresponding $ p $-values:
""")

st.latex(r"p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}")

st.markdown("""
1. **Sort** the $ p $-values from smallest to largest.  
2. Choose a desired **FDR level** $ q $ (e.g., 0.05).  
3. Identify the largest $ k $ satisfying:
""")

st.latex(r"p_{(k)} \le \frac{k}{m}q")

st.markdown("""
4. **Declare significant** all hypotheses with $ p_{(i)} \le p_{(k)} $.  

This ensures that, on average, no more than a proportion $ q $ of the rejected hypotheses are false discoveries.  
Unlike the Bonferroni correction (which controls the family-wise error rate and is highly conservative), BH-FDR maintains **greater power** by allowing a small, controlled proportion of false rejections.
""")

st.markdown("""
**When to Use BY Correction.**  
The **Benjamini–Yekutieli (BY)** variant adjusts for arbitrary dependence among tests:
""")

st.latex(r"p_{(k)} \le \frac{k}{m\,c(m)}q,\quad c(m)=\sum_{i=1}^{m}\frac{1}{i}")

st.markdown("""
This makes the procedure more conservative but robust under correlated outcomes,  
such as when O1–O5 are derived from overlapping constructs or measurements (e.g., reading comprehension, vocabulary, and writing quality).
""")

st.markdown("""
**Effect Size Reporting.**  
Beyond statistical significance, each outcome is accompanied by a **standardized effect size**—such as Cohen’s $ d $, Hedges’ $ g $, or standardized regression coefficients—together with their confidence intervals:
""")

st.latex(r"d = \frac{\bar{X}_1 - \bar{X}_0}{s_\text{pooled}},\quad 95\%~\text{CI}:~ d \pm 1.96\,\text{SE}(d)")

st.markdown("""
Effect sizes allow interpretation of the *magnitude* and *practical relevance* of EDU-AI’s impact, providing context beyond binary significance decisions.  
They also support meta-analytic integration and cumulative evidence synthesis across multiple studies.
""")

st.markdown("""
**Summary.**  
The BH-FDR framework offers a balanced trade-off between Type I error control and power when testing multiple outcomes.  
Reporting standardized effect sizes alongside adjusted $ p $-values ensures both statistical rigor and substantive interpretability.
""")
with st.expander("Assemble tests & apply FDR (demo)", expanded=True):
    tests = []

    # ---- (1) Learning gain: group ANOVA
    if has("learning_gains"):
        lg = st.session_state["learning_gains"].copy()
        lg.columns = [c.lower() for c in lg.columns]
        if HAS_SCIPY and "group" in lg.columns and lg["group"].nunique() >= 2:
            arrays = [
                pd.to_numeric(lg.loc[lg["group"] == g, "learning_gain"], errors="coerce").dropna().values
                for g in sorted(lg["group"].dropna().unique())
            ]
            if all(len(a) > 1 for a in arrays):
                F, p = stats.f_oneway(*arrays)
                tests.append(dict(outcome="O1: Learning Gain", test="ANOVA gain~group", pval=p))

    # ---- (2) PEI: group ANOVA
    if has("telemetry_with_pei"):
        te = st.session_state["telemetry_with_pei"].copy()
        te.columns = [c.lower() for c in te.columns]
        if HAS_SCIPY and "group" in te.columns and "prompt_evolution_index" in te.columns and te["group"].nunique() >= 2:
            arrays = [
                pd.to_numeric(te.loc[te["group"] == g, "prompt_evolution_index"], errors="coerce").dropna().values
                for g in sorted(te["group"].dropna().unique())
            ]
            if all(len(a) > 1 for a in arrays):
                F, p = stats.f_oneway(*arrays)
                tests.append(dict(outcome="O2: PEI", test="ANOVA PEI~group", pval=p))

    # ---- (3) Fairness: simple TPR gap z-test vs. largest-N reference
    if has("fairness_df"):
        fd = st.session_state["fairness_df"].copy()
        fd.columns = [c.strip().lower() for c in fd.columns]
        if {"group", "y_true"}.issubset(fd.columns) and (("y_pred" in fd.columns) or ("y_score" in fd.columns)):
            if "y_pred" in fd.columns:
                yhat = "y_pred"
            else:
                fd["_y_pred_thr"] = (pd.to_numeric(fd["y_score"], errors="coerce") >= 0.5).astype(int)
                yhat = "_y_pred_thr"

            rows = []
            for g, sub in fd.dropna(subset=["group"]).groupby("group"):
                y = pd.to_numeric(sub["y_true"], errors="coerce").astype(int)
                yp = pd.to_numeric(sub[yhat], errors="coerce").astype(int)
                P = int((y == 1).sum())
                TPR = safe_rate(int(((yp == 1) & (y == 1)).sum()), P)
                rows.append((g, len(sub), P, TPR))

            diag = pd.DataFrame(rows, columns=["group", "n", "pos", "tpr"]).sort_values("n", ascending=False)
            if len(diag) >= 2 and HAS_SCIPY:
                ref = diag.iloc[0]
                for _, r in diag.iloc[1:].iterrows():
                    if ref["pos"] > 0 and r["pos"] > 0 and np.isfinite(ref["tpr"]) and np.isfinite(r["tpr"]):
                        p1, p2 = ref["tpr"], r["tpr"]
                        n1, n2 = ref["pos"], r["pos"]
                        p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
                        se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
                        if se > 0:
                            z = (p1 - p2) / se
                            pval = 2 * (1 - stats.norm.cdf(abs(z)))
                            tests.append(dict(outcome="O-Fairness: TPR gap", test=f"TPR diff vs {ref['group']}", pval=pval))

    if not tests:
        st.info("No eligible tests found to build a family. Please run earlier modules first.")
    else:
        D = pd.DataFrame(tests)
        D["pval_adj_BH_q=.10"] = bh_adjust(D["pval"].tolist(), q=0.10)
        st.dataframe(D, use_container_width=True)

        figtab = go.Figure(data=[go.Table(
            header=dict(values=list(D.columns), fill_color="#2c3e50", font=dict(color="white")),
            cells=dict(values=[D[c] for c in D.columns], align="left")
        )])
        figtab.update_layout(template="plotly_white", title="BH-FDR across outcomes (q=0.10)")
        st.plotly_chart(figtab, use_container_width=True)

        df_download(D, "fdr_family_tests.csv", "Download FDR family results (CSV)", key="dl_fdr_family")

st.divider()


# ============================================================
# 4) Mechanism Test A: Mediation (PEI / RDS → Learning)
# ============================================================
st.header("4) Mechanism A — Mediation (PEI / RDS → Learning Gain)")

st.markdown("""
**Goal.**  
To investigate *how* EDU-AI improves learning, we test whether its effects operate **through intermediate process indicators**, such as:
- **PEI (Process Engagement Index):** captures behavioral engagement and interaction depth.  
- **RDS (Reflection Depth Score):** captures metacognitive quality in reflections.

In mediation analysis, we ask whether EDU-AI influences learning **indirectly**, by first improving these process-level mediators, which in turn enhance learning gains (post–pre test differences).
""")

st.markdown("""
**Conceptual Framework (Product-of-Coefficients Approximation).**  
Within-student mediation can be expressed as the product of two effects:
""")

st.latex(r"\text{Indirect} = a \times b,\quad a:\; M \sim T,\quad b:\; \Delta \sim M \;(\text{and } T)")

st.markdown("""
where:  
- $T$: Treatment indicator (EDU-AI vs. Control)  
- $M$: Mediator (e.g., PEI or RDS)  
- $\Delta$: Learning gain (post-test − pre-test)  
- $a$: Path from treatment to mediator  
- $b$: Path from mediator to outcome, controlling for treatment  
- $a \times b$: Estimated **indirect (mediated) effect**

The **indirect effect** quantifies how much of EDU-AI’s total impact on learning operates *through* changes in the mediator.  
A significant $a \times b$ implies that EDU-AI enhances learning by improving underlying processes like engagement or reflection.
""")

st.markdown("""
**Estimation Approach.**  
For exploratory analysis, we estimate each path using **Ordinary Least Squares (OLS)** regression:
""")

st.markdown("""
- **Path a:** regress mediator on treatment  
- **Path b:** regress learning gain on both mediator and treatment  
""")

st.markdown("""
Robust standard errors are used if available to account for heteroskedasticity.  
For confirmatory or preregistered analyses, a **Multilevel Structural Equation Model (MSEM)** is preferred, as it:
- Corrects for hierarchical data (students nested in classes/sites)  
- Decomposes mediation into within- and between-class effects  
- Provides unbiased standard errors under complex dependency structures
""")

st.markdown("""
**Interpretation.**  
- If $a$ is significant but $b$ is not, EDU-AI affects the mediator, but the mediator does not explain learning.  
- If both $a$ and $b$ are significant, there is evidence for **partial mediation** (EDU-AI improves learning partly via the mediator).  
- If only $b$ is significant after controlling for $T$, mediation may occur through unobserved variables correlated with $M$.  

The **direct effect** of EDU-AI on learning, controlling for mediators, represents the portion of the total effect not explained by the mediators.
""")

st.markdown("""
**Summary.**  
This mechanism analysis helps identify *why* EDU-AI works — distinguishing direct instructional benefits from indirect effects mediated by behavioral and reflective engagement.  
Such insight supports the design of more adaptive feedback loops and targeted scaffolding in future iterations of EDU-AI.
""")

with st.expander("Run OLS-based mediation (approx) with bootstrap CI", expanded=True):
    # Pull data
    if not (has("learning_gains") and (has("telemetry_with_pei") or has("reflections_df"))):
        st.info("Need Learning Gains and Telemetry (with PEI) or Reflections (with RDS).")
    else:
        lg = st.session_state["learning_gains"].copy()
        lg.columns = [c.lower() for c in lg.columns]

        # Prefer telemetry_with_pei as mediator source; fallback to reflections
        if has("telemetry_with_pei"):
            medsrc = st.session_state["telemetry_with_pei"].copy()
        else:
            medsrc = st.session_state["reflections_df"].copy()
        medsrc.columns = [c.lower() for c in medsrc.columns]

        # Sanity check for student_id
        if "student_id" not in lg.columns or "student_id" not in medsrc.columns:
            st.warning("Mediation requires `student_id` in both tables.")
        else:
            # Candidate mediators
            candidate_meds = [c for c in ["prompt_evolution_index", "rds_proxy"] if c in medsrc.columns]
            if not candidate_meds:
                st.info("No mediator found. Expect `prompt_evolution_index` or `rds_proxy` in mediator table.")
            else:
                # Prepare joined frame (one row per student)
                keep_cols = ["student_id"] + candidate_meds
                for c in ["group", "condition"]:
                    if c in medsrc.columns:
                        keep_cols.append(c)
                joined = lg.merge(
                    medsrc[keep_cols].drop_duplicates(subset=["student_id"]),
                    on="student_id",
                    how="inner"
                )

                st.write("Joined data (preview):")
                st.dataframe(joined.head(20), use_container_width=True)

                med = st.selectbox("Choose mediator", options=candidate_meds, index=0, key="med_sel_4")
                yopt = [c for c in ["learning_gain", "post"] if c in joined.columns]
                if not yopt:
                    st.warning("Outcome not found (need `learning_gain` or `post`).")
                else:
                    yvar = st.selectbox("Choose outcome", options=yopt, index=0, key="y_sel_4")
                    treat = "condition" if "condition" in joined.columns else None

                    if not HAS_SM:
                        st.info("`statsmodels` not available. Install it to run regression-based mediation.")
                    else:
                        # a-path: M ~ treatment (if available)
                        a_est = np.nan
                        if treat:
                            a_mod = smf.ols(f"{med} ~ C({treat})", data=joined).fit(cov_type="HC3")
                            st.markdown("**a-path model**")
                            st.code(a_mod.summary().as_text(), language="text")
                            # Take first non-intercept term (reference coding)
                            a_terms = [c for c in a_mod.params.index if c != "Intercept"]
                            a_est = float(a_mod.params[a_terms[0]]) if a_terms else np.nan
                        else:
                            st.info("No treatment variable detected; proceeding with b-path only.")
                            a_mod = None

                        # b-path: Δ or post ~ M (+ treatment)
                        if treat:
                            b_mod = smf.ols(f"{yvar} ~ {med} + C({treat})", data=joined).fit(cov_type="HC3")
                        else:
                            b_mod = smf.ols(f"{yvar} ~ {med}", data=joined).fit(cov_type="HC3")
                        st.markdown("**b-path model**")
                        st.code(b_mod.summary().as_text(), language="text")
                        b_est = float(b_mod.params.get(med, np.nan))

                        # Indirect effect
                        indirect = a_est * b_est if np.isfinite(a_est) and np.isfinite(b_est) else np.nan
                        st.metric("Indirect effect (a×b)", "—" if not np.isfinite(indirect) else f"{indirect:.3f}")

                        # Bootstrap CI for a×b
                        rng = np.random.default_rng(7)
                        B = st.number_input("Bootstrap draws (for a×b CI)", 200, 5000, 800, 100, key="boot_mediation")
                        idx = np.arange(len(joined))
                        boots = []
                        for _ in range(int(B)):
                            sel = rng.choice(idx, size=len(idx), replace=True)
                            dfb = joined.iloc[sel]
                            try:
                                if treat:
                                    am = smf.ols(f"{med} ~ C({treat})", data=dfb).fit()
                                    bm = smf.ols(f"{yvar} ~ {med} + C({treat})", data=dfb).fit()
                                    a_terms = [c for c in am.params.index if c != "Intercept"]
                                    a_b = float(am.params[a_terms[0]]) if a_terms else 0.0
                                    b_b = float(bm.params.get(med, 0.0))
                                else:
                                    # Without treatment, a is not identifiable in the same sense; use corr-proxy a=1.0 for shape
                                    am = None
                                    bm = smf.ols(f"{yvar} ~ {med}", data=dfb).fit()
                                    a_b = 1.0
                                    b_b = float(bm.params.get(med, 0.0))
                                boots.append(a_b * b_b)
                            except Exception:
                                continue
                        if boots:
                            lo, hi = np.percentile(boots, [2.5, 97.5])
                            st.caption(f"Bootstrap 95% CI for a×b: [{lo:.3f}, {hi:.3f}]")

                        # Scatter with optional trendline
                        fig_sc = px.scatter(
                            joined, x=med, y=yvar,
                            color=("group" if "group" in joined.columns else None),
                            trendline=("ols" if HAS_SM else None),
                            title=f"Mediator vs Outcome: {med} → {yvar}"
                        )
                        fig_sc.update_layout(template="plotly_white")
                        st.plotly_chart(fig_sc, use_container_width=True)

                        # Export joined data
                        df_download(joined, "mediation_joined.csv", "Download joined data (CSV)", key="dl_med_joined")

st.divider()
# ============================================================
# 5) Mechanism B — Dose–Response (Nonlinear Usage Intensity)
# ============================================================
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.header("5) Mechanism B — Dose–Response (Nonlinear Usage Intensity)")

# ---------- Helpers ----------

def _has(key: str) -> bool:
    return key in st.session_state and st.session_state[key] is not None

def ensure_student_id(df: pd.DataFrame, table_name: str = "Telemetry") -> pd.DataFrame:
    """Map a chosen identifier to 'student_id' for joins and groupbys."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    d = df.copy()
    d.columns = [c.strip().lower() for c in d.columns]
    if "student_id" in d.columns:
        d["student_id"] = d["student_id"].astype(str).str.strip()
        return d
    candidates = [
        "student_id","user_id","learner_id","sid","student","id",
        "uuid","pseudo_id","account_id","actor_id","person_id"
    ]
    auto = next((c for c in candidates if c in d.columns), None)
    opts = [auto] if auto else []
    opts += [c for c in d.columns if c not in opts]
    if not opts:
        st.error(f"{table_name}: no column can be mapped to 'student_id'.")
        return d
    picked = st.selectbox(
        f"Select identifier column for {table_name} → treat as student_id",
        options=opts, index=0, key=f"{table_name}_id_pick"
    )
    if picked and picked != "student_id":
        d = d.rename(columns={picked: "student_id"})
    d["student_id"] = d["student_id"].astype(str).str.strip()
    return d

def _download_df(df: pd.DataFrame, name: str, label: str, key: str):
    st.download_button(label, data=df.to_csv(index=False).encode("utf-8"),
                       file_name=name, mime="text/csv", key=key)

st.header("5) Mechanism B — Nonlinear Usage and the Zone of Productive Engagement")

st.markdown("""
**Objective.**  
Examine whether the relationship between **usage intensity** (e.g., time on EDU-AI, number of interactions, or feedback sessions) and **learning outcomes** follows a **nonlinear** pattern.  
The goal is to identify a *zone of productive engagement*—a range of moderate usage associated with the greatest learning gains—beyond which additional usage may yield diminishing or negative returns.

We model this using a **quadratic approximation**, and compute **bootstrap confidence intervals** for the fitted curve.  
In a full preregistered analysis, a **Generalized Additive Mixed Model (GAMM)** can be applied to flexibly capture more complex nonlinearities while preserving hierarchical structure.
""")

st.markdown("""
**Quadratic Approximation (Centered Usage).**  
To reduce multicollinearity and aid interpretation, usage $ U $ is standardized by centering and scaling:
""")

st.latex(r"U_c = \frac{U - \bar U}{s_U}")

st.markdown("""
The fitted model then takes the form:
""")

st.latex(r"Y = \beta_0 + \beta_1 U_c + \beta_2 U_c^2 + \varepsilon")

st.markdown(r"""
where:  

- $Y$: learning outcome (e.g., post-test score or learning gain);  
- $U_c$: centered usage (standardized measure of engagement intensity);  
- $\beta_1$: linear effect — captures whether increased usage tends to improve performance overall;  
- $\beta_2$: quadratic curvature — indicates whether the effect of usage levels off or becomes negative at higher intensities;  
- $\varepsilon$: residual error term representing unexplained variance.  

A **significant negative** $\beta_2$ suggests **diminishing returns**: learning improves with moderate engagement but may plateau or decline with excessive use,  
potentially reflecting **cognitive overload** or **inefficient effort allocation**.
""")

st.markdown(r"""
**Optimal Usage (Argmax of the Quadratic Function).**  
When $\beta_2 < 0$, the peak (maximum predicted outcome) occurs at:
""")

st.latex(r"U_c^* = -\frac{\beta_1}{2\beta_2}, \quad U^* = \bar U + s_U \cdot U_c^*")

st.markdown("""
Here:  
- $ U_c^* $: standardized location of optimal usage,  
- $ U^* $: optimal usage value in original units (e.g., minutes, logins, or interactions).  

This value $ U^* $ defines the **zone of productive engagement** — where learning efficiency is maximized.  
Empirically, the region near $ U^* $ is visualized using **bootstrapped confidence bands** to illustrate uncertainty around the estimated peak.
""")

st.markdown(r"""
**Interpretation.**  

- If $\beta_1 > 0$ and $\beta_2 < 0$: learning improves with greater usage up to an optimal point, after which returns diminish or even decline.  
- If both $\beta_1$ and $\beta_2$ are close to zero: usage intensity exerts minimal or no measurable influence on learning outcomes.  
- If $\beta_1 < 0$ but $\beta_2 > 0$: there may be an initial adjustment cost (e.g., a learning curve), followed by improvements at higher levels of engagement.  

These patterns illuminate **optimal engagement thresholds**—helping educators and learners strike a balance between sufficient interaction for learning benefits and overuse that may lead to fatigue or reduced efficiency.
""")

# ---------- Pull data ----------
with st.expander("Fit quadratic curve with bootstrap band", expanded=True):
    # Assessments (learning gains)
    if not _has("learning_gains"):
        st.info("Learning Gains are required. Please compute on the '1) Assessments' page.")
        st.stop()

    lg = st.session_state["learning_gains"].copy()
    lg.columns = [c.strip().lower() for c in lg.columns]

    # Telemetry source (prefer tele_df, fallback to telemetry_with_pei)
    if _has("tele_df"):
        tel = st.session_state["tele_df"].copy()
    elif _has("telemetry_with_pei"):
        tel = st.session_state["telemetry_with_pei"].copy()
    else:
        tel = None

    if tel is None:
        st.info("Telemetry is required for dose–response.")
        st.stop()

    tel.columns = [c.strip().lower() for c in tel.columns]
    tel = ensure_student_id(tel, table_name="Telemetry")

    if "student_id" not in lg.columns or "student_id" not in tel.columns:
        st.warning("Both tables must include `student_id`.")
        st.stop()

    # Choose or construct a usage proxy
    usage_candidates = ["usage_minutes","turns","sessions","events",
                        "interaction_count","num_turns","dwell_time","prompt_evolution_index"]
    present = [c for c in usage_candidates if c in tel.columns]

    st.markdown("**Usage signal selection**")
    if present:
        usage_pick = st.selectbox("Pick a telemetry column as usage (or choose 'row-count proxy')",
                                  options=["<row-count proxy>"] + present, index=0)
    else:
        usage_pick = "<row-count proxy>"

    if usage_pick == "<row-count proxy>":
        usage = tel.groupby("student_id").size().rename("usage_proxy").reset_index()
    else:
        tel[usage_pick] = pd.to_numeric(tel[usage_pick], errors="coerce")
        usage = (tel.groupby("student_id")[usage_pick]
                    .sum(min_count=1)
                    .rename("usage_proxy")
                    .reset_index())

    # Merge outcome (post or learning_gain)
    dfu = lg.merge(usage, on="student_id", how="inner")
    dfu = dfu.dropna(subset=["usage_proxy"]).copy()

    # Allow outcome selection
    y_options = [c for c in ["learning_gain","post"] if c in dfu.columns]
    if not y_options:
        st.info("No available outcome (need 'learning_gain' or 'post').")
        st.stop()

    yvar = st.selectbox("Outcome", y_options, index=0)
    # Optional stratifier
    group_col = st.selectbox("Optional group stratifier (faceting)", 
                             options=["<none>"] + [c for c in ["group","course_id","class_id","site_id"] if c in dfu.columns],
                             index=0)

    # Clean X, Y
    dfu["usage_proxy"] = pd.to_numeric(dfu["usage_proxy"], errors="coerce")
    dfu[yvar] = pd.to_numeric(dfu[yvar], errors="coerce")
    dfu = dfu[np.isfinite(dfu["usage_proxy"]) & np.isfinite(dfu[yvar])]
    if len(dfu) < 10:
        st.info("Too few rows after cleaning to fit a curve.")
        st.stop()

    # Centered usage
    U = dfu["usage_proxy"].to_numpy()
    Y = dfu[yvar].to_numpy()
    U_mean = float(np.mean(U))
    U_std  = float(np.std(U, ddof=0)) if np.std(U, ddof=0) > 0 else 1.0
    Uc = (U - U_mean) / U_std
    A = np.column_stack([np.ones_like(Uc), Uc, Uc**2])

    # Estimate coefficients via OLS (closed-form)
    beta, *_ = np.linalg.lstsq(A, Y, rcond=None)
    yhat = A @ beta
    b0, b1, b2 = map(float, beta)

    # Bootstrap band
    B = st.number_input("Bootstrap draws for CI band", 200, 5000, 800, 100, key="dose_boot")
    rng = np.random.default_rng(123)
    idx = np.arange(len(Uc))
    boots = []
    for _ in range(int(B)):
        sel = rng.choice(idx, size=len(idx), replace=True)
        A_b = A[sel]; Y_b = Y[sel]
        try:
            beta_b, *_ = np.linalg.lstsq(A_b, Y_b, rcond=None)
            boots.append((A @ beta_b))  # predict on original X-grid for band
        except Exception:
            continue
    boots = np.vstack(boots) if len(boots) else np.empty((0, len(U)))

    lo = np.percentile(boots, 2.5, axis=0) if boots.size else np.full_like(yhat, np.nan)
    hi = np.percentile(boots, 97.5, axis=0) if boots.size else np.full_like(yhat, np.nan)

    # Optimal usage (if concave)
    opt_Uc = -b1/(2*b2) if (np.isfinite(b1) and np.isfinite(b2) and b2 < 0) else np.nan
    opt_U  = U_mean + U_std * opt_Uc if np.isfinite(opt_Uc) else np.nan

    # ---------------------------
    # A) Main scatter + quadratic + band
    # ---------------------------
    df_plot = pd.DataFrame({
        "usage": U,
        yvar: Y,
        "yhat": yhat,
        "lo": lo,
        "hi": hi,
    })
    if group_col != "<none>":
        df_plot[group_col] = dfu[group_col].values

    fig = go.Figure()
    if group_col == "<none>":
        fig.add_trace(go.Scatter(x=df_plot["usage"], y=df_plot[yvar],
                                 mode="markers", name="Observed", opacity=0.6))
    else:
        for g, sub in df_plot.groupby(group_col):
            fig.add_trace(go.Scatter(x=sub["usage"], y=sub[yvar], mode="markers",
                                     name=f"Observed — {g}", opacity=0.6))
    fig.add_trace(go.Scatter(x=df_plot["usage"], y=df_plot["yhat"],
                             mode="lines", name="Quadratic fit"))
    fig.add_trace(go.Scatter(
        x=df_plot["usage"].tolist() + df_plot["usage"].tolist()[::-1],
        y=df_plot["hi"].tolist() + df_plot["lo"].tolist()[::-1],
        fill="toself", line=dict(width=0), name="95% CI", hoverinfo="skip", opacity=0.18
    ))
    if np.isfinite(opt_U):
        fig.add_trace(go.Scatter(x=[opt_U, opt_U],
                                 y=[np.nanmin([df_plot[yvar].min(), df_plot["yhat"].min()]),
                                    np.nanmax([df_plot[yvar].max(), df_plot["yhat"].max()])],
                                 mode="lines", name="Optimal U*", line=dict(dash="dash")))
    fig.update_layout(template="plotly_white",
                      title=f"Dose–Response: Usage → {yvar}",
                      xaxis_title="Usage (proxy)", yaxis_title=yvar)
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # B) Coefficient table + optimal usage
    # ---------------------------
    coef_tbl = pd.DataFrame({
        "term": ["Intercept (β0)", "Linear (β1)", "Quadratic (β2)", "U_mean", "U_std", "U* (raw)"],
        "value": [b0, b1, b2, U_mean, U_std, opt_U]
    })
    st.write("Quadratic model coefficients and derived optimal usage:")
    st.dataframe(coef_tbl, use_container_width=True)

    # ---------------------------
    # C) Decile table (usage bins) with outcome stats
    # ---------------------------
    dfu_local = dfu.copy()
    dfu_local["usage_decile"] = pd.qcut(dfu_local["usage_proxy"].rank(method="first"),
                                        q=10, labels=[f"D{i}" for i in range(1, 11)])
    dec = (dfu_local.groupby("usage_decile")[yvar]
           .agg(["count","mean","std","median","min","max"])
           .reset_index()
           .sort_values("usage_decile"))
    st.write("Outcome summary by usage deciles:")
    st.dataframe(dec, use_container_width=True)

    # ---------------------------
    # D) Hexbin-like density (via 2D histogram heatmap)
    # ---------------------------
    # Use coarse bins for stability
    fig_hex = go.Figure(data=go.Histogram2d(
        x=dfu["usage_proxy"], y=dfu[yvar],
        nbinsx=30, nbinsy=30, coloraxis="coloraxis"))
    fig_hex.update_layout(template="plotly_white", coloraxis_colorscale="Viridis",
                          title="2D Density of Usage vs Outcome (Histogram2d)",
                          xaxis_title="Usage (proxy)", yaxis_title=yvar)
    st.plotly_chart(fig_hex, use_container_width=True)

    # ---------------------------
    # E) Group-specific curves (if group_col provided)
    # ---------------------------
    if group_col != "<none>":
        st.markdown("**Group-specific quadratic fits** (separate curves):")
        fig_g = go.Figure()
        for g, sub in dfu.merge(df_plot[["usage","yhat"]], left_on="usage_proxy", right_on="usage", how="left").groupby(group_col):
            U_g = sub["usage_proxy"].to_numpy()
            Y_g = sub[yvar].to_numpy()
            if len(U_g) < 8:
                continue
            Uc_g = (U_g - np.mean(U_g)) / (np.std(U_g) if np.std(U_g) > 0 else 1.0)
            A_g = np.column_stack([np.ones_like(Uc_g), Uc_g, Uc_g**2])
            beta_g, *_ = np.linalg.lstsq(A_g, Y_g, rcond=None)
            yhat_g = A_g @ beta_g
            fig_g.add_trace(go.Scatter(x=U_g, y=Y_g, mode="markers",
                                       name=f"{g} — obs", opacity=0.45))
            fig_g.add_trace(go.Scatter(x=U_g, y=yhat_g, mode="lines",
                                       name=f"{g} — fit"))
        fig_g.update_layout(template="plotly_white",
                            title=f"Group-specific Dose–Response: Usage → {yvar}",
                            xaxis_title="Usage (proxy)", yaxis_title=yvar)
        st.plotly_chart(fig_g, use_container_width=True)

    # ---------------------------
    # F) Quartile violin plots (usage stratification)
    # ---------------------------
    dfq = dfu.copy()
    dfq["usage_quartile"] = pd.qcut(dfq["usage_proxy"].rank(method="first"), 4, labels=["Q1","Q2","Q3","Q4"])
    fig_v = px.violin(dfq, x="usage_quartile", y=yvar, box=True, points="all",
                      title=f"{yvar} by Usage Quartiles")
    fig_v.update_layout(template="plotly_white")
    st.plotly_chart(fig_v, use_container_width=True)

    # ---------------------------
    # G) Partial-residual style plot (r ~ usage)
    # ---------------------------
    # Fit simple linear model Y ~ usage (uncentered for residual plot) for visualization
    X_lin = np.column_stack([np.ones_like(U), U])
    beta_lin, *_ = np.linalg.lstsq(X_lin, Y, rcond=None)
    resid = Y - X_lin @ beta_lin
    fig_pr = px.scatter(x=U, y=resid, trendline="lowess",
                        labels={"x":"Usage (proxy)", "y":"Partial residual (Y − Ŷ_linear)"},
                        title="Partial-residual style view (lowess on residuals vs usage)")
    fig_pr.update_layout(template="plotly_white")
    st.plotly_chart(fig_pr, use_container_width=True)

    # ---------------------------
    # H) Export analysis data
    # ---------------------------
    out = df_plot.copy()
    out["usage_proxy"] = out["usage"]
    out["outcome"] = out[yvar]
    out = out.drop(columns=[yvar])
    _download_df(out, "dose_response_curve.csv", "Download fitted curve & band (CSV)", key="dl_dose_curve")
    _download_df(dec, "dose_usage_deciles.csv", "Download usage-decile summary (CSV)", key="dl_dose_deciles")

st.divider()
# ============================================================
# 6) Heterogeneous Treatment Effects (HTE)
# ============================================================
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Optional backends
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    HAS_SM = True
except Exception:
    HAS_SM = False

try:
    from scipy import stats
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# ---------------- Helpers ----------------
def _has(k: str) -> bool:
    return k in st.session_state and st.session_state[k] is not None



def _download_df(df: pd.DataFrame, name: str, label: str, key: str):
    st.download_button(label, data=df.to_csv(index=False).encode("utf-8"),
                       file_name=name, mime="text/csv", key=key)

def cohen_d(a: pd.Series, b: pd.Series) -> float:
    xa = pd.to_numeric(a, errors="coerce").dropna().to_numpy()
    xb = pd.to_numeric(b, errors="coerce").dropna().to_numpy()
    if len(xa) < 2 or len(xb) < 2:
        return np.nan
    va = np.var(xa, ddof=1)
    vb = np.var(xb, ddof=1)
    sp = np.sqrt(((len(xa)-1)*va + (len(xb)-1)*vb) / (len(xa)+len(xb)-2))
    if sp <= 0 or not np.isfinite(sp):
        return np.nan
    return (np.mean(xa) - np.mean(xb)) / sp

# ---------------- UI & Theory ----------------
st.header("6) Heterogeneous Treatment Effects (HTE)")

st.markdown("""
**Goal.**  
Assess whether the EDU-AI effect varies by **moderators** (e.g., baseline ability, motivation, prior achievement, demographics).  
Report both:  
1) **Confirmatory interaction tests** in a regression with cluster-robust SEs, and  
2) **Exploratory CATE profiles** (e.g., conditional average treatment effects) to visualize patterns across the moderator distribution.
""")

st.markdown("""
**Confirmatory Model (Interaction OLS with Cluster-Robust SE).**  
We adjust for baseline and include a treatment–moderator interaction:
""")

st.latex(r"\text{Post} \;=\; \beta_0 \;+\; \beta_1\,\text{Treat} \;+\; \beta_2\,M \;+\; \beta_3\,(\text{Treat}\times M) \;+\; \beta_4\,\text{Pre} \;+\; \varepsilon")

st.markdown(r"""
where:  

- $\text{Post}$: post-test outcome;  
- $\text{Pre}$: baseline or pre-test covariate;  
- $\text{Treat}$: treatment indicator (EDU-AI = 1, Control = 0);  
- $M$: moderator variable (e.g., baseline ability, motivation, or other learner characteristics);  
- Cluster-robust SEs adjust for class and site clustering to avoid anti-conservative inference.
""")

st.markdown("""
**Marginal (Simple-Slope) Effect of Treatment at Moderator Value $M=m$.**  
This is the estimated EDU-AI effect **conditional on** the moderator:
""")

st.latex(r"\frac{\partial\,\widehat{\text{Post}}}{\partial\,\text{Treat}}\Big|_{M=m} \;=\; \beta_1 \;+\; \beta_3\,m")

st.markdown(r"""
**Interpretation:**  

- $\beta_1$: treatment effect when $M = 0$ (thus, center or standardize $M$ for meaningful interpretation);  
- $\beta_3$: captures **how the treatment effect changes** with each unit increase in $M$;  
- A significant $\beta_3$ indicates **heterogeneity** — suggesting that EDU-AI benefits differ across subgroups.
""")
st.markdown(r"""
**Visualization & Probing.**  
We recommend:  

- **Simple-slope plots** of $\beta_1 + \beta_3 m$ across a range of $m$ values (e.g., mean ± 2 SD), with 95% confidence bands to illustrate uncertainty.  
- **Subgroup profiles** (e.g., terciles of $M$: low / medium / high) showing adjusted post-test means and corresponding treatment effects.  
- *(Optional)* **Johnson–Neyman intervals** to identify ranges of $M$ where the treatment effect differs significantly from zero.
""")

st.markdown(r"""
**Modeling Choices & Robustness.**  

- **Centering or standardizing** $M$ enhances interpretability of $\beta_1$, ensuring it represents the treatment effect at the average moderator level.  
- Use **cluster-robust SEs** (clustered by class or site). For preregistered hierarchical analyses, fit a **mixed-effects model** with random intercepts (and optionally random slopes for $\text{Treat}$).  
- Examine **linearity** of the moderator effect; if residual diagnostics suggest curvature, consider spline transformations or higher-order interaction terms for $M$.
""")

st.markdown("""
**Exploratory CATE (Profile-Level Insights).**  
Beyond a single moderator, explore **conditional average treatment effects** using flexible learners (e.g., uplift trees, causal forests, T-learners).  
These tools can reveal **multivariate** heterogeneity patterns.  
Present results as partial-dependence style plots or binned CATE summaries, but treat as **exploratory** and validate with out-of-sample checks to avoid over-interpretation.
""")



with st.expander("Interaction model & layered profiles", expanded=True):
    # ---------- Build analysis frame ----------
    if not _has("assess_df"):
        st.info("Assessments are required (with columns student_id, phase, score).")
        st.stop()

    assess = st.session_state["assess_df"].copy()
    assess.columns = [c.strip().lower() for c in assess.columns]

    if not {"student_id","phase","score"}.issubset(assess.columns):
        st.warning("Assessments must contain: student_id, phase, score.")
        st.stop()

    # Aggregate to one pre & post per student
    phase_scores = assess.groupby(["student_id","phase"], dropna=False)["score"].mean().reset_index()
    pre  = phase_scores[phase_scores["phase"].str.contains("pre", case=False, na=False)][["student_id","score"]].rename(columns={"score":"pre"})
    post = phase_scores[phase_scores["phase"].str.contains("post", case=False, na=False)][["student_id","score"]].rename(columns={"score":"post"})
    merged = pd.merge(pre, post, on="student_id", how="inner")

    # Attach cohort keys if available
    for key in ["class_id","site_id","condition","group","course_id"]:
        if key in assess.columns:
            merged = merged.merge(assess[["student_id", key]].drop_duplicates("student_id"), on="student_id", how="left")

    # Bring moderators from surveys if present
    if _has("surveys_df"):
        surveys = st.session_state["surveys_df"].copy()
        surveys.columns = [c.strip().lower() for c in surveys.columns]
        mvars = [c for c in ["ai_familiarity","motivation_pre","self_efficacy_pre","gpa"] if c in surveys.columns]
        if mvars:
            mod_map = surveys.groupby("student_id")[mvars].mean(numeric_only=True).reset_index()
            merged = merged.merge(mod_map, on="student_id", how="left")

    # Basic preview
    st.write("Merged data for HTE (head):")
    st.dataframe(merged.head(20), use_container_width=True)

    # Moderator selection
    mod_candidates = [c for c in ["ai_familiarity","motivation_pre","self_efficacy_pre","gpa","pre"] if c in merged.columns]
    if not mod_candidates:
        st.info("No moderator candidates found. Consider adding ai_familiarity / motivation_pre / self_efficacy_pre / gpa.")
        st.stop()

    moderator = st.selectbox("Choose moderator (M)", options=mod_candidates, index=0, key="hte_mod_pick")

    # Treatment variable (categorical condition expected)
    treat = "condition" if "condition" in merged.columns else None
    # Cluster key (for cluster-robust SE)
    cluster_key = None
    for c in ["class_id","site_id"]:
        if c in merged.columns and merged[c].nunique() > 1:
            cluster_key = c
            break

    # ---------- (1) Fit confirmatory interaction model ----------
    if HAS_SM:
        # Clean numeric columns
        merged["pre"] = pd.to_numeric(merged["pre"], errors="coerce")
        merged["post"] = pd.to_numeric(merged["post"], errors="coerce")
        merged[moderator] = pd.to_numeric(merged[moderator], errors="coerce")
        model_df = merged.dropna(subset=["post","pre",moderator]).copy()

        if treat:
            fml = f"post ~ C({treat})*{moderator} + pre"
        else:
            st.info("No treatment variable detected; showing descriptive HTE only (no interaction model).")
            fml = None

        if fml:
            try:
                if cluster_key:
                    m = smf.ols(fml, data=model_df).fit(cov_type="cluster", cov_kwds={"groups": model_df[cluster_key]})
                    st.caption(f"Cluster-robust SE by `{cluster_key}`")
                else:
                    m = smf.ols(fml, data=model_df).fit(cov_type="HC3")
                    st.caption("Robust SE: HC3")

                st.code(m.summary().as_text(), language="text")

                # Coefficient table
                coef_tbl = pd.DataFrame({
                    "term": m.params.index,
                    "coef": m.params.values,
                    "se": m.bse.values,
                    "t": m.tvalues.values,
                    "p": m.pvalues.values
                })
                fig_coef = go.Figure(data=[go.Table(
                    header=dict(values=list(coef_tbl.columns), fill_color="#2c3e50", font=dict(color="white")),
                    cells=dict(values=[coef_tbl[c] for c in coef_tbl.columns], align="left")
                )])
                fig_coef.update_layout(template="plotly_white", title="Interaction model — coefficients")
                st.plotly_chart(fig_coef, use_container_width=True)

            except Exception as e:
                st.warning(f"Interaction model failed: {e}")
                m = None
        else:
            m = None
    else:
        st.info("`statsmodels` not available — skipping confirmatory interaction model.")
        m = None

    # ---------- (2) Marginal effect curve of treatment across moderator ----------
    if m is not None and treat:
        # Identify a baseline level & the interaction term name
        # We will compute delta = E[post|Treat=t1, M] − E[post|Treat=t0, M]
        # Use the first non-reference level vs reference
        levels = sorted(model_df[treat].dropna().unique().tolist())
        if len(levels) >= 2:
            base, alt = levels[0], levels[1]
            Mgrid = np.linspace(np.nanpercentile(model_df[moderator], 5),
                                np.nanpercentile(model_df[moderator], 95), 50)
            # Build design matrices by using patsy manually via formula evaluation
            # Simpler: predict by creating two copies with Treat=base/alt
            preds = []
            for mval in Mgrid:
                # Use mean(pre) for conditioning
                row_base = {moderator: mval, "pre": float(np.nanmean(model_df["pre"])), treat: base}
                row_alt  = {moderator: mval, "pre": float(np.nanmean(model_df["pre"])), treat: alt}
                y0 = float(m.predict(pd.DataFrame([row_base]))[0])
                y1 = float(m.predict(pd.DataFrame([row_alt]))[0])
                preds.append(dict(M=mval, delta=y1 - y0))
            df_me = pd.DataFrame(preds)
            fig_me = px.line(df_me, x="M", y="delta",
                             title=f"Marginal effect of Treatment ({alt}−{base}) across moderator {moderator}")
            fig_me.update_layout(template="plotly_white", xaxis_title=moderator, yaxis_title="Δ Predicted Post")
            st.plotly_chart(fig_me, use_container_width=True)

    # ---------- (3) Subgroup forest by moderator quartiles (Cohen's d of gain) ----------
    lg_ready = merged.copy()
    lg_ready["learning_gain"] = pd.to_numeric(lg_ready["post"], errors="coerce") - pd.to_numeric(lg_ready["pre"], errors="coerce")
    lg_ready = lg_ready.dropna(subset=["learning_gain", moderator])

    if treat:
        lg_ready["mod_quartile"] = pd.qcut(pd.to_numeric(lg_ready[moderator], errors="coerce"),
                                           4, labels=["Q1","Q2","Q3","Q4"], duplicates="drop")
        rows = []
        for q, sub in lg_ready.groupby("mod_quartile"):
            if sub[treat].nunique() >= 2:
                # pick two groups (first two levels for display)
                levels = sorted(sub[treat].dropna().unique().tolist())
                g0, g1 = levels[0], levels[1]
                d = cohen_d(sub.loc[sub[treat]==g1, "learning_gain"],
                            sub.loc[sub[treat]==g0, "learning_gain"])
                rows.append(dict(moderator_quartile=str(q), treat_ref=g0, treat_alt=g1, cohens_d=d,
                                 n_ref=int((sub[treat]==g0).sum()), n_alt=int((sub[treat]==g1).sum())))
        forest = pd.DataFrame(rows)
        if not forest.empty:
            st.write("Cohen’s d of learning gain by moderator quartile (alt vs ref):")
            st.dataframe(forest, use_container_width=True)
            fig_for = px.scatter(forest, x="cohens_d", y="moderator_quartile",
                                 size=(forest["n_ref"] + forest["n_alt"]).clip(lower=1),
                                 title="Forest-style view: subgroup Cohen’s d",
                                 labels={"cohens_d":"Cohen's d", "moderator_quartile":"Moderator quartile"})
            fig_for.update_layout(template="plotly_white")
            st.plotly_chart(fig_for, use_container_width=True)

    # ---------- (4) Heatmap of mean post by (moderator decile × pre quartile) and (treatment) ----------
    # Build grids
    grid_df = merged.dropna(subset=["post", "pre", moderator]).copy()
    grid_df["mod_decile"] = pd.qcut(pd.to_numeric(grid_df[moderator], errors="coerce").rank(method="first"),
                                    10, labels=[f"D{i}" for i in range(1, 11)], duplicates="drop")
    grid_df["pre_quartile"] = pd.qcut(pd.to_numeric(grid_df["pre"], errors="coerce"),
                                      4, labels=["Q1","Q2","Q3","Q4"], duplicates="drop")
    if treat and grid_df[treat].nunique() >= 2:
        # Show surface for alt − base
        base, alt = sorted(grid_df[treat].dropna().unique().tolist())[:2]
        surf = (grid_df.groupby(["mod_decile","pre_quartile", treat])["post"]
                      .mean().reset_index().pivot_table(index=["mod_decile","pre_quartile"],
                                                        columns=treat, values="post"))
        if base in surf.columns and alt in surf.columns:
            diff = (surf[alt] - surf[base]).reset_index().rename(columns={0: "diff"})
            fig_hm = px.imshow(
                diff.pivot(index="mod_decile", columns="pre_quartile", values=0),
                title=f"Mean(Post) difference ( {alt} − {base} ) across moderator deciles × pre quartiles",
                aspect="auto", color_continuous_scale="RdBu_r", origin="lower"
            )
            fig_hm.update_layout(template="plotly_white")
            st.plotly_chart(fig_hm, use_container_width=True)

    # ---------- (5) Violin + strip: Post by treatment across moderator quartiles ----------
    if treat:
        vdf = merged.dropna(subset=["post", moderator]).copy()
        vdf["mod_quartile"] = pd.qcut(pd.to_numeric(vdf[moderator], errors="coerce"),
                                      4, labels=["Q1","Q2","Q3","Q4"], duplicates="drop")
        fig_v = px.violin(vdf, x="mod_quartile", y="post", color=treat, box=True, points="all",
                          title=f"Post by {treat} across moderator quartiles ({moderator})")
        fig_v.update_layout(template="plotly_white")
        st.plotly_chart(fig_v, use_container_width=True)

    # ---------- (6) Quantile contrasts (exploratory): median diff across moderator bins ----------
    if treat:
        qdf = merged.dropna(subset=["post", moderator]).copy()
        qdf["mod_bin"] = pd.qcut(pd.to_numeric(qdf[moderator], errors="coerce"),
                                 6, labels=[f"B{i}" for i in range(1, 7)], duplicates="drop")
        rows = []
        levels = sorted(qdf[treat].dropna().unique().tolist())
        if len(levels) >= 2:
            g0, g1 = levels[0], levels[1]
            for b, sub in qdf.groupby("mod_bin"):
                med0 = pd.to_numeric(sub.loc[sub[treat]==g0, "post"], errors="coerce").median()
                med1 = pd.to_numeric(sub.loc[sub[treat]==g1, "post"], errors="coerce").median()
                rows.append(dict(mod_bin=str(b), median_diff=med1 - med0))
            qtab = pd.DataFrame(rows)
            fig_q = px.bar(qtab, x="mod_bin", y="median_diff",
                           title=f"Median(Post) difference ({g1} − {g0}) by moderator bins")
            fig_q.update_layout(template="plotly_white")
            st.plotly_chart(fig_q, use_container_width=True)

    # ---------- (7) Partial dependence-style curve: average over covariates ----------
    # For simplicity: Lowess on scatter of post vs moderator, colored by treatment
    if treat:
        scat = merged.dropna(subset=["post", moderator]).copy()
        fig_pd = px.scatter(scat, x=moderator, y="post", color=treat, trendline="lowess",
                            title="Partial dependence-style: Post vs Moderator (LOWESS by treatment)")
        fig_pd.update_layout(template="plotly_white")
        st.plotly_chart(fig_pd, use_container_width=True)

    # ---------- (8) Table: subgroup Ns & means by moderator quartile × treatment ----------
    if treat:
        tab = (merged.dropna(subset=["post", moderator])
                     .assign(mod_quartile=lambda d: pd.qcut(pd.to_numeric(d[moderator], errors="coerce"),
                                                           4, labels=["Q1","Q2","Q3","Q4"], duplicates="drop"))
                     .groupby(["mod_quartile", treat])["post"]
                     .agg(["count","mean","std","median","min","max"])
                     .reset_index())
        st.write("Subgroup summary table:")
        st.dataframe(tab, use_container_width=True)

        _download_df(tab, "hte_subgroup_summary.csv", "Download subgroup table (CSV)", key="dl_hte_tab")

st.divider()

# ============================================================
# 7) Statistical Diagnostics & Robustness Checks
# ============================================================


# Optional deps
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.diagnostic import het_breuschpagan, het_white
    HAS_SM = True
except Exception:
    HAS_SM = False

try:
    from scipy import stats
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# ---------------- Helpers ----------------
def _has(k: str) -> bool:
    return k in st.session_state and st.session_state[k] is not None


def _download_df(df: pd.DataFrame, name: str, label: str, key: str):
    st.download_button(label, data=df.to_csv(index=False).encode("utf-8"),
                       file_name=name, mime="text/csv", key=key)

def _safe_num(s):
    return pd.to_numeric(s, errors="coerce")

def _quick_icc_oneway(values: pd.Series, clusters: pd.Series) -> float:
    """One-way ICC(1) proxy = (MSB - MSW) / (MSB + (k-1)MSW)."""
    try:
        df = pd.DataFrame({"y": _safe_num(values), "g": clusters})
        df = df.dropna()
        if df["g"].nunique() < 2:
            return np.nan
        groups = [v.dropna().values for _, v in df.groupby("g")["y"]]
        n_i = np.array([len(g) for g in groups], dtype=float)
        if np.any(n_i <= 1):
            return np.nan
        k_bar = n_i.mean()
        overall_mean = df["y"].mean()
        ssb = np.sum(n_i * (np.array([g.mean() for g in groups]) - overall_mean) ** 2)
        ssw = np.sum([np.sum((g - g.mean()) ** 2) for g in groups])
        dfb = df["g"].nunique() - 1
        dfw = len(df) - df["g"].nunique()
        if dfb <= 0 or dfw <= 0:
            return np.nan
        msb = ssb / dfb
        msw = ssw / dfw
        return float((msb - msw) / (msb + (k_bar - 1) * msw)) if (msb + (k_bar - 1) * msw) != 0 else np.nan
    except Exception:
        return np.nan
st.header("7) Statistical Diagnostics & Robustness Checks")

st.markdown("""
**Goals.**  
Evaluate model assumptions and result stability via: residual shape, heteroscedasticity, leverage/influence, ITT vs. per-protocol (PP) contrasts (proxy), bootstrap robustness, and ICC sensitivity (for multilevel data).
""")

st.markdown("""
**Autofit (if no cached model).**  
If no fitted model is available, we fit a minimal confirmatory model and refresh all plots/tables from that fit:
""")
st.latex(r"\text{post} = \beta_0 + \beta_1\,\text{pre} + \varepsilon")
st.markdown("or, when a treatment indicator exists:")
st.code("post ~ pre + C(condition)", language="r")

# ---------------- Residuals ----------------
st.markdown("### A) Residual Shape & Normality")
st.markdown("""
We assess whether residuals are symmetric and light-tailed enough for t-based inference (important for small samples).
""")
st.latex(r"e_i = y_i - \hat{y}_i,\qquad \tilde{e}_i = \frac{e_i}{\hat{\sigma}_\varepsilon}")
st.markdown("""
- Inspect **residual vs. fitted** plots for curvature or patterns.  
- **QQ-plots** check approximate normality of residuals.  
- Mild deviations are usually acceptable with robust SE; severe skew/heavy tails suggest bootstrap CIs.
""")

# ---------------- Heteroscedasticity ----------------
st.markdown("### B) Heteroscedasticity (Non-constant Variance)")
st.markdown("We test and guard against heteroscedastic errors.")
st.latex(r"\mathbb{V}(e_i \mid X) \neq \sigma^2 \quad \Rightarrow \quad \text{use robust SE}")
st.markdown("""
- **Tests**: Breusch–Pagan / White.  
- **Mitigation**: report **heteroscedasticity-robust SE** (HC1/HC3) and compare with classical SE.
""")

# ---------------- Leverage & Influence ----------------
st.markdown("### C) Leverage & Influence")
st.markdown("Identify points that unduly affect coefficients.")
st.latex(r"H = X (X'X)^{-1} X',\quad h_{ii} = \text{diag}(H)")
st.latex(r"D_i \approx \frac{e_i^2}{p\,\hat{\sigma}_\varepsilon^2}\cdot \frac{h_{ii}}{(1-h_{ii})^2}\quad\text{(Cook's D, OLS)}")
st.markdown("""
**Rules of thumb**: high leverage if $h_{ii} > 2p/n$; influential if Cook’s $D_i > 4/n$.  
Action: inspect, verify data quality, and report **with/without** influential points.
""")

# ---------------- ITT vs PP ----------------
st.markdown("### D) ITT vs. Per-Protocol (Proxy) Contrasts")
st.markdown("""
**Intent-to-Treat (ITT)** uses randomized assignment; **Per-Protocol (PP)** restricts to compliers/adequate exposure (proxy).  
Comparing both gauges sensitivity to adherence.
""")
st.latex(r"\hat{\tau}_{\text{ITT}} = \mathbb{E}[Y \mid Z=1] - \mathbb{E}[Y \mid Z=0]")
st.markdown("""
- **ITT** preserves randomization; preferred for confirmatory claims.  
- **PP (proxy)**: restrict to $U \ge u_0$ (e.g., minimum usage) → **exploratory** only (selection bias risk).  
We display both; interpret PP cautiously.
""")

# ---------------- Bootstrap ----------------
st.markdown("### E) Bootstrap Robustness")
st.markdown("""
Use resampling to stabilize inference under mild assumption violations.
""")
st.latex(r"\hat{\theta}^{*(b)},\; b=1,\dots,B \quad\Rightarrow\quad \text{CI}_{\alpha} = [\hat{\theta}^{*(\alpha/2)},\; \hat{\theta}^{*(1-\alpha/2)}]")
st.markdown("""
- Default $B$ (e.g., 1,000) with percentile or BCa intervals.  
- Compare bootstrap CIs to model-based CIs; large gaps suggest model misspecification.
""")

# ---------------- ICC Sensitivity ----------------
st.markdown("### F) ICC Sensitivity (Multilevel Data)")
st.markdown("""
Quantify outcome clustering and its impact on SEs/effect estimates.
""")
st.latex(r"\text{ICC} = \frac{\sigma^2_{\text{class}}}{\sigma^2_{\text{class}} + \sigma^2_{\varepsilon}}")
st.markdown("""
- Higher ICC ⇒ more within-class dependence, larger SEs if ignored.  
- We vary assumed ICC or refit a mixed model to assess sensitivity of treatment effects and CIs.
""")

# ---------------- Summary ----------------
st.markdown("""
**Summary.**  
We (1) check residual shape and heteroscedasticity; (2) flag high-influence points;  
(3) contrast ITT with a PP proxy; (4) add bootstrap CIs; and (5) probe ICC sensitivity.  
Concordant conclusions across these checks increase confidence in the reported findings.
""")
# ------------------------------------------------------------
# A) Prepare minimal modeling frame
# ------------------------------------------------------------
assess = st.session_state.get("assess_df", None)
model_obj = None  # will hold statsmodels results if available

if assess is None:
    st.info("Upload Assessments first to run diagnostics.")
    st.stop()

af = assess.copy()
af.columns = [c.strip().lower() for c in af.columns]
if not {"student_id", "phase", "score"}.issubset(af.columns):
    st.warning("Assessments must include: student_id, phase, score.")
    st.stop()

# Aggregate to pre/post per student
phase_scores = af.groupby(["student_id", "phase"], dropna=False)["score"].mean().reset_index()
pre  = phase_scores[phase_scores["phase"].str.contains("pre", case=False, na=False)][["student_id","score"]].rename(columns={"score":"pre"})
post = phase_scores[phase_scores["phase"].str.contains("post", case=False, na=False)][["student_id","score"]].rename(columns={"score":"post"})
df = pd.merge(pre, post, on="student_id", how="inner")

# Attach potential covariates (optional)
for c in ["condition", "class_id", "site_id", "group"]:
    if c in af.columns:
        df = df.merge(af[["student_id", c]].drop_duplicates("student_id"), on="student_id", how="left")

df["pre"] = _safe_num(df["pre"])
df["post"] = _safe_num(df["post"])
dfm = df.dropna(subset=["pre", "post"]).copy()

st.subheader("Data used for diagnostics (preview)")
st.dataframe(dfm.head(15), use_container_width=True)

# ------------------------------------------------------------
# B) Fit OLS model (with or without condition)
# ------------------------------------------------------------
st.subheader("Model Fit (for diagnostics)")
if HAS_SM and len(dfm) >= 10:
    if "condition" in dfm.columns and dfm["condition"].nunique() >= 2:
        fml = "post ~ pre + C(condition)"
    else:
        fml = "post ~ pre"
    # Cluster-robust if cluster key available
    cluster_key = None
    for c in ["class_id", "site_id"]:
        if c in dfm.columns and dfm[c].nunique() > 1:
            cluster_key = c
            break
    try:
        if cluster_key:
            model_obj = smf.ols(fml, data=dfm).fit(cov_type="cluster", cov_kwds={"groups": dfm[cluster_key]})
            st.caption(f"Cluster-robust SE by `{cluster_key}`")
        else:
            model_obj = smf.ols(fml, data=dfm).fit(cov_type="HC3")
            st.caption("Robust SE: HC3")
        st.code(model_obj.summary().as_text(), language="text")
    except Exception as e:
        st.warning(f"Model fit failed: {e}")
else:
    st.info("`statsmodels` missing or too few rows; falling back to NumPy OLS.")
    # Fallback: OLS post ~ pre (no robust SE)
    X = np.column_stack([np.ones(len(dfm)), dfm["pre"].to_numpy()])
    y = dfm["post"].to_numpy()
    try:
        beta, resid_sum, rank, s = np.linalg.lstsq(X, y, rcond=None)
        dfm["_yhat"] = X @ beta
        dfm["_resid"] = y - dfm["_yhat"]
        st.write("Fallback coefficients (β0, β1):", [float(beta[0]), float(beta[1])])
    except Exception as e:
        st.error(f"Fallback OLS failed: {e}")

# If we have a model_obj, compute fitted & residuals
if model_obj is not None:
    dfm["_yhat"] = model_obj.fittedvalues
    dfm["_resid"] = model_obj.resid.values
    # Influence measures
    infl = model_obj.get_influence()
    dfm["_std_resid"] = infl.resid_studentized_internal
    dfm["_leverage"] = infl.hat_matrix_diag
    # Cook's distance
    cooks_d = infl.cooks_distance[0]
    dfm["_cooks_d"] = cooks_d

# ------------------------------------------------------------
# C) Residual diagnostics (6+ visuals)
# ------------------------------------------------------------
st.subheader("Residual Diagnostics")

if "_resid" in dfm.columns:
    # 1) Residual vs Fitted
    fig_rvf = px.scatter(dfm, x="_yhat", y="_resid",
                         title="(1) Residuals vs Fitted",
                         labels={"_yhat": "Fitted (Ŷ)", "_resid": "Residual"})
    fig_rvf.add_hline(y=0, line_dash="dot")
    fig_rvf.update_layout(template="plotly_white")
    st.plotly_chart(fig_rvf, use_container_width=True)

    # 2) Scale-Location: |std resid| vs Fitted
    if "_std_resid" in dfm.columns:
        fig_sl = px.scatter(dfm, x="_yhat", y=np.abs(dfm["_std_resid"]),
                            title="(2) Scale–Location: |Std Residual| vs Fitted",
                            labels={"x": "Fitted (Ŷ)", "y": "|Std Residual|"})
        fig_sl.update_layout(template="plotly_white")
        st.plotly_chart(fig_sl, use_container_width=True)

    # 3) Residual histogram + KDE
    fig_hist = px.histogram(dfm, x="_resid", nbins=30, marginal="box",
                            title="(3) Residual Distribution")
    fig_hist.update_layout(template="plotly_white")
    st.plotly_chart(fig_hist, use_container_width=True)

    # 4) Residual QQ (approx) — plot residual quantiles vs Normal
    r = _safe_num(dfm["_resid"]).dropna().to_numpy()
    if len(r) > 5:
        r_sorted = np.sort(r)
        ppos = (np.arange(1, len(r_sorted) + 1) - 0.5) / len(r_sorted)
        if HAS_SCIPY:
            z = stats.norm.ppf(ppos)
        else:
            # Approx inverse CDF via np.erfinv
            z = np.sqrt(2) * stats.erfinv(2*ppos-1) if hasattr(stats, "erfinv") else r_sorted * 0
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(x=z, y=r_sorted, mode="markers", name="Residual quantiles"))
        # Add y=x line scaled
        slope = np.std(r_sorted) / (np.std(z) if np.std(z) > 0 else 1)
        fig_qq.add_trace(go.Scatter(x=z, y=z*slope, mode="lines", name="Reference (scaled)",
                                    line=dict(dash="dash")))
        fig_qq.update_layout(template="plotly_white", title="(4) QQ Plot (Residuals vs Normal)",
                             xaxis_title="Theoretical Quantiles", yaxis_title="Residual Quantiles")
        st.plotly_chart(fig_qq, use_container_width=True)

    # 5) Leverage vs Std Residual (influence map)
    if "_leverage" in dfm.columns and "_std_resid" in dfm.columns:
        fig_inf = px.scatter(dfm, x="_leverage", y=np.abs(dfm["_std_resid"]),
                             size=("_cooks_d" if "_cooks_d" in dfm.columns else None),
                             title="(5) Influence: |Std Residual| vs Leverage",
                             labels={"_leverage": "Leverage (hat)", "y": "|Std Residual|"})
        fig_inf.update_layout(template="plotly_white")
        st.plotly_chart(fig_inf, use_container_width=True)

    # 6) Residuals by group (if available)
    gcol = next((c for c in ["group", "class_id", "site_id", "condition"] if c in dfm.columns), None)
    if gcol:
        fig_grp = px.box(dfm, x=gcol, y="_resid", points="all",
                         title=f"(6) Residuals by {gcol}")
        fig_grp.update_layout(template="plotly_white")
        st.plotly_chart(fig_grp, use_container_width=True)

    # 7) Heteroscedasticity tests (if statsmodels)
    st.subheader("Heteroscedasticity Tests")
    if HAS_SM and "_yhat" in dfm.columns:
        try:
            X = sm.add_constant(_safe_num(dfm["_yhat"]))
            white = het_white(_safe_num(dfm["_resid"]), X)
            bp = het_breuschpagan(_safe_num(dfm["_resid"]), X)
            wt = pd.DataFrame({"stat":[white[0]], "pval":[white[1]], "fstat":[white[2]], "fpval":[white[3]]})
            bt = pd.DataFrame({"Lagrange Multiplier":[bp[0]], "p-value":[bp[1]], "F-stat":[bp[2]], "F p-value":[bp[3]]})
            st.write("White test:")
            st.dataframe(wt, use_container_width=True)
            st.write("Breusch–Pagan test:")
            st.dataframe(bt, use_container_width=True)
        except Exception as e:
            st.info(f"Heteroscedasticity tests unavailable: {e}")

    # 8) Bootstrap check: mean residual ≈ 0
    st.subheader("Bootstrap Robustness: Mean Residual ~ 0")
    if HAS_SCIPY:
        B = st.number_input("Bootstrap draws", 200, 5000, 1000, 100, key="boot_resid")
        rng = np.random.default_rng(202)
        r = _safe_num(dfm["_resid"]).dropna().to_numpy()
        if len(r) >= 5:
            means = []
            idx = np.arange(len(r))
            for _ in range(int(B)):
                sel = rng.choice(idx, size=len(idx), replace=True)
                means.append(float(np.mean(r[sel])))
            lo, hi = np.percentile(means, [2.5, 97.5])
            fig_bs = px.histogram(means, nbins=40, title="Bootstrap distribution of mean residual")
            fig_bs.add_vline(x=lo, line_dash="dash", annotation_text=f"2.5%={lo:.3f}")
            fig_bs.add_vline(x=hi, line_dash="dash", annotation_text=f"97.5%={hi:.3f}")
            fig_bs.add_vline(x=0.0, line_dash="dot", annotation_text="0")
            fig_bs.update_layout(template="plotly_white")
            st.plotly_chart(fig_bs, use_container_width=True)
            st.caption(f"95% bootstrap CI for mean residual: [{lo:.3f}, {hi:.3f}]")
    else:
        st.info("Install SciPy to run bootstrap CI.")

else:
    st.info("Model residuals not available.")

# ------------------------------------------------------------
# D) ITT vs PP Proxy (completion-based)
# ------------------------------------------------------------
st.subheader("ITT vs Per-Protocol (PP) — Proxy Comparison")
st.markdown("**Note:** As a proxy, we define PP as students with complete pre & post and (optionally) telemetry completion flag.")

# ITT: everyone in df (post model frame)
itt_n = len(df)
pp_mask = np.isfinite(df["pre"]) & np.isfinite(df["post"])
# If a completion flag exists in telemetry or assessments, incorporate
pp_flag_cols = [c for c in ["completed_core_assessments", "completion_rate", "pct_tasks_completed"] if c in af.columns]
if pp_flag_cols:
    comp_map = af.groupby("student_id")[pp_flag_cols].max(numeric_only=True).reset_index()
    df_pp = df.merge(comp_map, on="student_id", how="left")
    comp_ok = (df_pp[pp_flag_cols].fillna(0).max(axis=1) >= 1).to_numpy()
    pp_mask = pp_mask & comp_ok
else:
    df_pp = df.copy()

pp_n = int(pp_mask.sum())
st.write(f"ITT N = {itt_n}, PP N = {pp_n}")

fig_pp = px.histogram(df.loc[pp_mask], x="post", nbins=30, title="PP: Post distribution")
fig_pp.update_layout(template="plotly_white")
st.plotly_chart(fig_pp, use_container_width=True)

fig_itt = px.histogram(df, x="post", nbins=30, title="ITT: Post distribution")
fig_itt.update_layout(template="plotly_white")
st.plotly_chart(fig_itt, use_container_width=True)

# ------------------------------------------------------------
# E) ICC Sensitivity (cluster structure proxy)
# ------------------------------------------------------------
st.subheader("ICC Sensitivity (One-way ICC proxy on Learning Gain)")
df_icc = df.copy()
df_icc["learning_gain"] = _safe_num(df_icc["post"]) - _safe_num(df_icc["pre"])
cluster_opts = [c for c in ["class_id","site_id","group"] if c in df_icc.columns]
if not cluster_opts:
    st.info("No clustering columns available for ICC (need class_id/site_id/group).")
else:
    cl = st.selectbox("Cluster column for ICC", options=cluster_opts, index=0)
    icc_val = _quick_icc_oneway(df_icc["learning_gain"], df_icc[cl])
    st.metric(f"ICC(1) on Learning Gain (cluster={cl})", "—" if not np.isfinite(icc_val) else f"{icc_val:.3f}")
    # Show per-cluster means and Ns
    summ = (df_icc.groupby(cl)["learning_gain"]
            .agg(["count","mean","std","median","min","max"])
            .reset_index()
            .sort_values("count", ascending=False))
    st.dataframe(summ, use_container_width=True)
    fig_icc = px.bar(summ, x=cl, y="mean", error_y="std",
                     title="Cluster means (Learning Gain) with SD",
                     labels={"mean":"Mean Gain"})
    fig_icc.update_layout(template="plotly_white", xaxis_tickangle=30)
    st.plotly_chart(fig_icc, use_container_width=True)
    _download_df(summ, "icc_cluster_summary.csv", "Download cluster summary (CSV)", key="dl_icc_sum")

st.divider()
# ============================================================
# 8) Outcome-Family Mapping across Tables (for FDR & Reporting)
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ---------- Safe helpers (idempotent) ----------
def _has(key: str) -> bool:
    return key in st.session_state and st.session_state[key] is not None

def _df_download(df: pd.DataFrame, fname: str, label: str, key: str):
    st.download_button(label, data=df.to_csv(index=False).encode("utf-8"),
                       file_name=fname, mime="text/csv", key=key)

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = [c.strip().lower() for c in d.columns]
    return d

st.header("8) Outcome-Family Mapping across Tables (for FDR & Reporting)")
st.markdown("""
We link each **outcome family** (O1–O5, plus exploratory mechanisms and fairness slices) to **source tables** and **candidate fields** that can feed
the statistical analyses and the FDR families. Use this overview to validate schema readiness before modeling.
""")

# ---------- Pull from session (normalize if present) ----------
assess  = _normalize_cols(st.session_state["assess_df"])        if _has("assess_df")           else None
surveys = _normalize_cols(st.session_state["surveys_df"])       if _has("surveys_df")          else None
tele    = _normalize_cols(st.session_state["tele_df"])          if _has("tele_df")             else (_normalize_cols(st.session_state["telemetry_with_pei"]) if _has("telemetry_with_pei") else None)
teacher = _normalize_cols(st.session_state["teacher_df"])       if _has("teacher_df")          else None
fair    = _normalize_cols(st.session_state["fairness_df"])      if _has("fairness_df")         else None
refl    = _normalize_cols(st.session_state["reflections_df"])   if _has("reflections_df")      else None
lg_tbl  = _normalize_cols(st.session_state["learning_gains"])   if _has("learning_gains")      else None

# ---------- Define families & expected fields ----------
families_spec = [
    dict(
        family="O1: Cognitive Learning (Primary)",
        source="Assessments",
        want_any=[
            # Either direct post score or pre/post to create Δ
            "post", "post_total_score", "post_theta", "learning_gain", "pre", "pre_total_score", "pre_theta",
            # Row-level raw if aggregation is needed
            "student_id", "phase", "score"
        ]
    ),
    dict(
        family="O2: Metacognition / SRL (Secondary)",
        source="Surveys",
        want_any=[
            "mslq_", "self_efficacy", "calibration_accuracy", "srl_", "motivation_", "ai_familiarity"
        ]
    ),
    dict(
        family="O3: Transfer (Secondary)",
        source="Assessments",
        want_any=[
            "near_transfer_score", "far_transfer_score", "transfer_"
        ]
    ),
    dict(
        family="O4: Efficiency (Secondary)",
        source="Telemetry",
        want_any=[
            "time_to_mastery", "attempts_to_mastery", "abandon_rate", "usage_minutes", "num_turns", "sessions", "events"
        ]
    ),
    dict(
        family="O5: Teacher Orchestration Load (Secondary)",
        source="Teacher Logs",
        want_any=[
            "teacher_minutes_feedback", "proactive_interventions", "reactive_interventions", "scaffold_events",
            "perceived_cognitive_load", "workload_hours"
        ]
    ),
    dict(
        family="Exploratory: Process Mechanisms — PEI",
        source="Telemetry",
        want_any=[
            "prompt_evolution_index", "lexical_specificity", "constraint_density", "strategy_token_rate"
        ]
    ),
    dict(
        family="Exploratory: Process Mechanisms — RDS",
        source="Reflections",
        want_any=[
            "rds_proxy", "rds_score", "reflection_text", "ai_response"
        ]
    ),
    dict(
        family="Slices: Fairness (for HTE/Parity)",
        source="Fairness",
        want_any=[
            "group", "gender", "ses_bucket", "language_group", "program", "y_true", "y_pred", "y_score", "ai_accuracy"
        ]
    ),
]

source_to_df = {
    "Assessments": assess,
    "Surveys": surveys,
    "Telemetry": tele,
    "Teacher Logs": teacher,
    "Fairness": fair,
    "Reflections": refl
}

# ---------- Build availability table ----------
rows = []
for spec in families_spec:
    fam   = spec["family"]
    src   = spec["source"]
    wants = spec["want_any"]
    df    = source_to_df.get(src, None)

    if df is None or df.empty:
        present = []
        coverage = 0.0
        status = "missing source"
    else:
        cols = df.columns.tolist()
        # match by prefix support
        present = []
        for w in wants:
            if w.endswith("_"):  # prefix pattern
                hit = any(c.startswith(w) for c in cols)
            else:
                hit = (w in cols)
            if hit:
                present.append(w)
        coverage = len(present) / max(1, len(wants))
        status = "ok" if coverage > 0 else "no matching fields"

    rows.append(dict(family=fam, source=src, fields_present=", ".join(present) if present else "—",
                     n_present=len(present), n_wanted=len(wants),
                     coverage=coverage, status=status))

map_df = pd.DataFrame(rows).sort_values(["source","family"]).reset_index(drop=True)

st.subheader("A) Family-to-Source Mapping Table (with coverage)")
st.dataframe(map_df, use_container_width=True)
_df_download = _df_download  # alias for linter
_df_download(map_df, "outcome_family_mapping.csv", "Download mapping CSV", key="dl_map_csv")

# ---------- B) Coverage bar and heatmaps ----------
st.subheader("B) Coverage Overview")

# (1) Coverage bar by family
fig_cov = px.bar(map_df, x="family", y="coverage", color="status",
                 title="Coverage by Outcome Family (fraction of wanted fields present)",
                 text=map_df["n_present"].astype(str) + "/" + map_df["n_wanted"].astype(str))
fig_cov.update_layout(template="plotly_white", xaxis_tickangle=30, yaxis_tickformat=".0%")
st.plotly_chart(fig_cov, use_container_width=True)

# (2) Binary heatmap of field availability by family (explode rows)
exploded = []
for _, r in map_df.iterrows():
    fam, src = r["family"], r["source"]
    df = source_to_df.get(src, None)
    wants = next(s["want_any"] for s in families_spec if s["family"] == fam)
    for w in wants:
        available = False
        if df is not None and not df.empty:
            if w.endswith("_"):
                available = any(c.startswith(w) for c in df.columns)
            else:
                available = (w in df.columns)
        exploded.append(dict(family=fam, wanted=w, available=int(available)))
heat = pd.DataFrame(exploded)

if not heat.empty:
    pivot = heat.pivot_table(index="family", columns="wanted", values="available", fill_value=0)
    fig_hm = px.imshow(pivot, title="Binary Availability Heatmap (1=present, 0=absent)",
                       aspect="auto", color_continuous_scale=[[0,"#f8f9fa"],[1,"#2c3e50"]])
    fig_hm.update_layout(template="plotly_white")
    st.plotly_chart(fig_hm, use_container_width=True)

# ---------- C) Treemap / Sunburst for Families → Sources → Fields ----------
st.subheader("C) Families → Sources → Fields (Treemap/Sunburst)")

def _present_fields_for(fam: str, src: str):
    df = source_to_df.get(src, None)
    wants = next(s["want_any"] for s in families_spec if s["family"] == fam)
    if df is None or df.empty:
        return []
    cols = df.columns.tolist()
    out = []
    for w in wants:
        if w.endswith("_"):
            hits = [c for c in cols if c.startswith(w)]
            out.extend(hits if hits else [])
        elif w in cols:
            out.append(w)
    return sorted(list(set(out)))

treemap_rows = []
for spec in families_spec:
    fam, src = spec["family"], spec["source"]
    pres = _present_fields_for(fam, src)
    if pres:
        for f in pres:
            treemap_rows.append(dict(path=f"{fam}/{src}/{f}", value=1))
    else:
        treemap_rows.append(dict(path=f"{fam}/{src}/(none)", value=1))

if treemap_rows:
    treemap_df = pd.DataFrame(treemap_rows)
    # Treemap
    fig_tree = px.treemap(treemap_df, path=[treemap_df["path"].str.split("/").str[0],
                                            treemap_df["path"].str.split("/").str[1],
                                            treemap_df["path"].str.split("/").str[2]],
                          values="value", title="Treemap: Families → Sources → Fields")
    fig_tree.update_layout(template="plotly_white")
    st.plotly_chart(fig_tree, use_container_width=True)

    # Sunburst
    fig_sun = px.sunburst(treemap_df, path=[treemap_df["path"].str.split("/").str[0],
                                            treemap_df["path"].str.split("/").str[1],
                                            treemap_df["path"].str.split("/").str[2]],
                          values="value", title="Sunburst: Families → Sources → Fields")
    fig_sun.update_layout(template="plotly_white")
    st.plotly_chart(fig_sun, use_container_width=True)

# ---------- D) Family readiness scorecard ----------
st.subheader("D) Family Readiness Scorecard")

scorecard = map_df[["family","source","n_present","n_wanted","coverage","status"]].copy()
scorecard["readiness"] = pd.cut(
    scorecard["coverage"],
    bins=[-0.01, 0.2, 0.5, 0.8, 1.0],
    labels=["low", "partial", "good", "excellent"]
)
fig_tab = go.Figure(data=[go.Table(
    header=dict(values=list(scorecard.columns), fill_color="#2c3e50", font=dict(color="white")),
    cells=dict(values=[scorecard[c] for c in scorecard.columns], align="left")
)])
fig_tab.update_layout(template="plotly_white", title="Outcome Family Readiness")
st.plotly_chart(fig_tab, use_container_width=True)

# ---------- E) Cross-family coverage summary ----------
st.subheader("E) Cross-family Coverage Summary")

cov_summary = (map_df
               .groupby("source", as_index=False)
               .agg(mean_coverage=("coverage","mean"),
                    families=("family","nunique"),
                    families_ok=("status", lambda s: int((s=="ok").sum()))))
cov_summary["pct_ok"] = cov_summary["families_ok"] / cov_summary["families"]
fig_covsrc = px.bar(cov_summary, x="source", y="pct_ok", text=(cov_summary["families_ok"].astype(str)+"/"+cov_summary["families"].astype(str)),
                    title="Percent of Families with OK Coverage by Source")
fig_covsrc.update_layout(template="plotly_white", yaxis_tickformat=".0%")
st.plotly_chart(fig_covsrc, use_container_width=True)
st.dataframe(cov_summary, use_container_width=True)

# ---------- F) Missing-field to-do list (actionable) ----------
st.subheader("F) Missing-field To-Do List")

todo_rows = []
for spec in families_spec:
    fam, src, wants = spec["family"], spec["source"], spec["want_any"]
    df = source_to_df.get(src, None)
    present = []
    if df is not None and not df.empty:
        cols = df.columns.tolist()
        for w in wants:
            if w.endswith("_"):
                if any(c.startswith(w) for c in cols):
                    present.append(w)
            elif w in cols:
                present.append(w)
    missing = [w for w in wants if w not in present]
    if missing:
        todo_rows.append(dict(family=fam, source=src, missing_fields=", ".join(missing), n_missing=len(missing)))

todo = pd.DataFrame(todo_rows).sort_values(["source","n_missing"], ascending=[True, False])
if todo.empty:
    st.success("All families have at least one matching field in their sources — good to proceed.")
else:
    st.warning("Some families are missing desired fields. See the checklist below.")
    st.dataframe(todo, use_container_width=True)
    _df_download(todo, "outcome_family_missing_fields.csv", "Download to-do checklist (CSV)", key="dl_todo_csv")

st.divider()

# ============================================================
# 9) Fairness (Parity) Analysis Bridge
# ============================================================
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------- Safe helpers (idempotent, won't overwrite if already defined) ----------
if "safe_rate" not in globals():
    def safe_rate(num, den):
        num = float(num)
        den = float(den)
        return float(num/den) if den and np.isfinite(den) else np.nan

if "_norm_cols" not in globals():
    def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d.columns = [c.strip().lower() for c in d.columns]
        return d

if "_df_download" not in globals():
    def _df_download(df: pd.DataFrame, fname: str, label: str, key: str):
        st.download_button(label, data=df.to_csv(index=False).encode("utf-8"),
                           file_name=fname, mime="text/csv", key=key)

st.header("9) Fairness (Parity) Analysis Bridge")
st.markdown("""
This section summarizes **group-level parity** and prepares outputs that align with your statistical-conclusion-validity plan.
We report classification parity metrics when **`y_true`** and **`y_pred` or `y_score`** are available, and accuracy summaries when only aggregate accuracy is provided.
""")

# ---------- Load/normalize ----------
fair = st.session_state.get("fairness_df", None)
if fair is not None:
    fair = _norm_cols(fair)

with st.expander("Minimal Parity Summary (requires `group` and labels/scores)", expanded=True):
    if fair is None or "group" not in fair.columns:
        st.info("A `fairness_df` with a `group` column is required.")
    else:
        # Decide a prediction source
        yhat_col = None
        if "y_pred" in fair.columns:
            yhat_col = "y_pred"
        elif "y_score" in fair.columns:
            # On-the-fly threshold for demo; users can tune threshold below
            fair["_y_pred_thr"] = (pd.to_numeric(fair["y_score"], errors="coerce") >= 0.5).astype(int)
            yhat_col = "_y_pred_thr"

        # =========================
        # A) Accuracy-only summary
        # =========================
        if "ai_accuracy" in fair.columns:
            st.subheader("A) AI Accuracy by Group (available even without labels)")
            acc = fair.groupby("group", dropna=False)["ai_accuracy"].mean().reset_index()
            st.dataframe(acc, use_container_width=True)

            fig_acc = px.bar(acc, x="group", y="ai_accuracy", title="Mean AI Accuracy by Group")
            fig_acc.update_layout(template="plotly_white", yaxis_title="Accuracy")
            st.plotly_chart(fig_acc, use_container_width=True)

            gap = float(acc["ai_accuracy"].max() - acc["ai_accuracy"].min())
            st.metric("Accuracy Gap (max − min)", f"{gap:.3f}")

        # ======================================
        # B) Full parity metrics if labels exist
        # ======================================
        if ("y_true" in fair.columns) and (yhat_col is not None):
            st.subheader("B) Classification Parity Metrics")
            thr_col = None
            if "y_score" in fair.columns:
                # Allow user to tune threshold and recompute prediction
                thr = st.slider("Decision threshold on y_score (for parity metrics)",
                                min_value=0.0, max_value=1.0, value=0.50, step=0.01, key="fair_thr")
                thr_col = "_y_pred_thr_user"
                fair[thr_col] = (pd.to_numeric(fair["y_score"], errors="coerce") >= thr).astype(int)
                yhat_use = thr_col
            else:
                yhat_use = yhat_col

            rows = []
            for g, sub in fair.dropna(subset=["group"]).groupby("group"):
                y  = pd.to_numeric(sub["y_true"], errors="coerce").astype(int)
                yp = pd.to_numeric(sub[yhat_use], errors="coerce").astype(int)

                TP = int(((yp == 1) & (y == 1)).sum())
                TN = int(((yp == 0) & (y == 0)).sum())
                FP = int(((yp == 1) & (y == 0)).sum())
                FN = int(((yp == 0) & (y == 1)).sum())

                n  = len(sub)
                P  = TP + FN
                Nn = TN + FP

                SR  = safe_rate((yp == 1).sum(), n)         # Selection rate
                ACC = safe_rate((yp == y).sum(), n)         # Accuracy
                TPR = safe_rate(TP, P)                      # True positive rate (Recall+)
                FPR = safe_rate(FP, Nn)                     # False positive rate
                PPV = safe_rate(TP, TP + FP)                # Precision
                FNR = safe_rate(FN, P)                      # Miss rate
                TNR = safe_rate(TN, Nn)                     # Specificity

                rows.append(dict(group=g, N=n, TP=TP, TN=TN, FP=FP, FN=FN,
                                  SR=SR, ACC=ACC, TPR=TPR, FPR=FPR, TNR=TNR, FNR=FNR, PPV=PPV))

            gm = pd.DataFrame(rows).sort_values("N", ascending=False)
            st.caption("Reference group is the **largest-N** group for gap calculations.")
            st.dataframe(gm, use_container_width=True)

            if not gm.empty:
                ref = gm.iloc[0]
                gm["SPD"] = ref["SR"] - gm["SR"]                                           # Statistical Parity Difference
                gm["DI"]  = gm["SR"] / ref["SR"] if ref["SR"] and np.isfinite(ref["SR"]) else np.nan  # Disparate Impact
                gm["EOG"] = (gm["TPR"] - ref["TPR"]).abs()                                 # Equal Opportunity Gap
                gm["EOD"] = ((gm["TPR"] - ref["TPR"]).abs() + (gm["FPR"] - ref["FPR"]).abs())/2      # Equalized Odds Gap

                st.subheader("Parity Gap Table (vs largest-N group)")
                st.dataframe(gm, use_container_width=True)
                _df_download(gm, "fairness_parity_metrics.csv", "Download parity metrics (CSV)", key="dl_fair_csv")

                # C1) Equalized Odds Gap bar
                fig_eod = px.bar(gm, x="group", y="EOD", title="Equalized Odds Gap by Group")
                fig_eod.update_layout(template="plotly_white")
                st.plotly_chart(fig_eod, use_container_width=True)

                # C2) Statistical Parity Difference
                fig_spd = px.bar(gm, x="group", y="SPD", title="Statistical Parity Difference (Ref − Group)")
                fig_spd.update_layout(template="plotly_white", yaxis_title="SPD (ref SR − group SR)")
                st.plotly_chart(fig_spd, use_container_width=True)

                # C3) Disparate Impact (SR_g / SR_ref)
                fig_di = px.bar(gm, x="group", y="DI", title="Disparate Impact (Selection Rate Ratio)")
                fig_di.add_hline(y=0.8, line_dash="dot", annotation_text="80% rule")
                fig_di.update_layout(template="plotly_white", yaxis_title="SR_g / SR_ref")
                st.plotly_chart(fig_di, use_container_width=True)

                # C4) ROC by group (if y_score exists)
                if "y_score" in fair.columns:
                    st.subheader("ROC by Group")
                    # Compute ROC points per group
                    roc_data = []
                    # thresholds grid
                    thr_grid = np.linspace(0.0, 1.0, 51)
                    for g, sub in fair.dropna(subset=["group"]).groupby("group"):
                        y  = pd.to_numeric(sub["y_true"], errors="coerce").astype(int).to_numpy()
                        s  = pd.to_numeric(sub["y_score"], errors="coerce").to_numpy()
                        ok = np.isfinite(y) & np.isfinite(s)
                        y, s = y[ok], s[ok]
                        if y.size < 5 or len(np.unique(y)) < 2:
                            continue
                        P  = (y == 1).sum()
                        Nn = (y == 0).sum()
                        for t in thr_grid:
                            yp = (s >= t).astype(int)
                            TP = ((yp == 1) & (y == 1)).sum()
                            FP = ((yp == 1) & (y == 0)).sum()
                            TPR = safe_rate(TP, P)
                            FPR = safe_rate(FP, Nn)
                            roc_data.append(dict(group=g, thr=t, TPR=TPR, FPR=FPR))
                    roc_df = pd.DataFrame(roc_data)
                    if not roc_df.empty:
                        fig_roc = px.line(roc_df, x="FPR", y="TPR", color="group",
                                          title="ROC Curves by Group (threshold sweep)")
                        fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                                          line=dict(dash="dot"), name="random")
                        fig_roc.update_layout(template="plotly_white", xaxis_title="FPR", yaxis_title="TPR")
                        st.plotly_chart(fig_roc, use_container_width=True)
                    else:
                        st.info("Insufficient variance in labels/scores to draw ROC.")

                # C5) Precision–Recall by group (if y_score exists)
                if "y_score" in fair.columns:
                    st.subheader("Precision–Recall by Group")
                    pr_data = []
                    thr_grid = np.linspace(0.0, 1.0, 51)
                    for g, sub in fair.dropna(subset=["group"]).groupby("group"):
                        y  = pd.to_numeric(sub["y_true"], errors="coerce").astype(int).to_numpy()
                        s  = pd.to_numeric(sub["y_score"], errors="coerce").to_numpy()
                        ok = np.isfinite(y) & np.isfinite(s)
                        y, s = y[ok], s[ok]
                        if y.size < 5 or len(np.unique(y)) < 2:
                            continue
                        for t in thr_grid:
                            yp = (s >= t).astype(int)
                            TP = ((yp == 1) & (y == 1)).sum()
                            FP = ((yp == 1) & (y == 0)).sum()
                            FN = ((yp == 0) & (y == 1)).sum()
                            prec = safe_rate(TP, TP + FP)
                            rec  = safe_rate(TP, TP + FN)
                            pr_data.append(dict(group=g, thr=t, Precision=prec, Recall=rec))
                    pr_df = pd.DataFrame(pr_data)
                    if not pr_df.empty:
                        fig_pr = px.line(pr_df, x="Recall", y="Precision", color="group",
                                         title="Precision–Recall Curves by Group")
                        fig_pr.update_layout(template="plotly_white")
                        st.plotly_chart(fig_pr, use_container_width=True)

                # C6) Parity frontier: plot TPR vs FPR with lines to ref
                st.subheader("Parity Frontier (TPR/FPR vs Reference)")
                ref_name = str(ref["group"])
                fig_front = go.Figure()
                fig_front.add_trace(go.Scatter(
                    x=[ref["FPR"]], y=[ref["TPR"]], mode="markers+text",
                    text=[f"ref ({ref_name})"], name="Reference", textposition="top center"
                ))
                for _, r in gm.iterrows():
                    fig_front.add_trace(go.Scatter(
                        x=[r["FPR"]], y=[r["TPR"]], mode="markers+text",
                        text=[str(r["group"])], name=str(r["group"]), textposition="bottom center"
                    ))
                    fig_front.add_trace(go.Scatter(
                        x=[ref["FPR"], r["FPR"]], y=[ref["TPR"], r["TPR"]],
                        mode="lines", showlegend=False, line=dict(dash="dot")
                    ))
                fig_front.update_layout(template="plotly_white", title="Parity Frontier: TPR/FPR vs Reference",
                                        xaxis_title="FPR", yaxis_title="TPR")
                st.plotly_chart(fig_front, use_container_width=True)

                # C7) Radar (Spider) chart of metrics by group
                st.subheader("Radar of Metrics by Group")
                radar_metrics = ["ACC", "TPR", "TNR", "PPV", "SR"]
                rad = gm[["group"] + radar_metrics].copy()
                rad = rad.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                cats = radar_metrics + [radar_metrics[0]]
                fig_rad = go.Figure()
                for _, r in rad.iterrows():
                    vals = [float(r[m]) for m in radar_metrics] + [float(r[radar_metrics[0]])]
                    fig_rad.add_trace(go.Scatterpolar(r=vals, theta=cats, fill="toself", name=str(r["group"])))
                fig_rad.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                                      template="plotly_white",
                                      title="Radar of Group Metrics (scaled to [0,1])")
                st.plotly_chart(fig_rad, use_container_width=True)

        elif "ai_accuracy" not in fair.columns:
            st.info("Provide `y_true` with (`y_pred` or `y_score`) for parity metrics, or supply `ai_accuracy` for accuracy-only summaries.")

st.divider()
st.success("Fairness analysis section complete. Use the downloads above to carry metrics into your reporting pipeline.")
# pages/4_Teacher_Logs.py — Advanced
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from math import log
from datetime import timedelta

st.title("4) Teacher Logs — Orchestration (Advanced)")

# ------------------------------------------------------------------
# Load
# ------------------------------------------------------------------
if "teacher_df" not in st.session_state:
    st.info("Please upload teacher logs on the **0. Upload** page.")
    st.stop()

raw = st.session_state["teacher_df"].copy()
st.write("Preview:", raw.head())
raw.columns = [c.strip().lower() for c in raw.columns]

# Soft schema (all optional except at least one metric column)
has_ts  = "timestamp" in raw.columns
has_id  = "instructor_id" in raw.columns
has_act = "activity" in raw.columns
has_int = "interruptions" in raw.columns
has_w   = "workload_hours" in raw.columns
has_cl  = "perceived_cognitive_load" in raw.columns
has_cs  = "class_size" in raw.columns
has_ai  = "ai_usage_pct" in raw.columns
has_ev  = "event_type" in raw.columns
has_sid = "session_id" in raw.columns

df = raw.copy()

# ------------------------------------------------------------------
# Preprocess
# ------------------------------------------------------------------
if has_ts:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"] = df["timestamp"].dt.date
    df["week"] = df["timestamp"].dt.isocalendar().week.astype(int)
else:
    df["date"] = pd.NaT
    df["week"] = np.nan

# numeric coercions
for col in ["workload_hours","perceived_cognitive_load","interruptions","class_size","ai_usage_pct"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# fill activity if absent
if not has_act:
    df["activity"] = "unspecified"

# ------------------------------------------------------------------
# Filters
# ------------------------------------------------------------------
with st.expander("Filters"):
    c1, c2, c3 = st.columns(3)
    if has_id:
        insts = sorted(df["instructor_id"].dropna().astype(str).unique())
        sel_inst = c1.multiselect("Instructor(s)", insts, default=insts if insts else None, key="t_insts")
    else:
        sel_inst = []
    if has_act:
        acts = sorted(df["activity"].dropna().astype(str).unique())
        sel_act = c2.multiselect("Activity type(s)", acts, default=acts if acts else None, key="t_acts")
    else:
        sel_act = []

    if has_ts and df["timestamp"].notna().any():
        tmin, tmax = df["timestamp"].min(), df["timestamp"].max()
        tr = c3.slider("Time window", min_value=tmin.to_pydatetime(), max_value=tmax.to_pydatetime(),
                       value=(tmin.to_pydatetime(), tmax.to_pydatetime()), key="t_time")
    else:
        tr = None

mask = pd.Series(True, index=df.index)
if has_id and sel_inst:
    mask &= df["instructor_id"].astype(str).isin(sel_inst)
if has_act and sel_act:
    mask &= df["activity"].astype(str).isin(sel_act)
if tr and has_ts:
    mask &= df["timestamp"].between(tr[0], tr[1])

df = df[mask].copy()
st.success(f"Filtered rows: {len(df)}")

# ------------------------------------------------------------------
# Academic metrics
# ------------------------------------------------------------------

st.markdown("### Academic Metrics & Formulas")

st.markdown(r"""
This section formalizes the key **cognitive and behavioral performance metrics** used to quantify 
task orchestration, workload distribution, and mental fatigue.  
Let the total logged time be $T$, the number of task transitions be $N_{trans}$, and interruptions be $N_{int}$.  
Each metric provides insight into distinct dimensions of human–AI interaction and self-regulated learning performance.
""")

# --- Task Switching Rate (TSR) ---
st.markdown("**Task-Switching Rate (TSR) — indicator of cognitive fragmentation:**")
st.latex(r"\text{TSR} = \dfrac{\#\,\text{activity transitions}}{\text{total logged hours}}")
st.markdown(r"""
**Interpretation:** TSR quantifies the *frequency of context switches per unit time*.  
A higher TSR implies greater cognitive fragmentation or multitasking, often associated with increased mental load  
and reduced sustained attention (Monsell, 2003).  
Monitoring TSR enables the evaluation of task continuity and attentional control across work sessions.
""")

# --- Interrupt Rate (IR) ---
st.markdown("**Interrupt Rate (IR) — frequency of external or self-initiated interruptions:**")
st.latex(r"\text{IR} = \dfrac{\text{interruptions}}{\text{total logged hours}}")
st.markdown(r"""
**Interpretation:** IR measures how frequently a user’s workflow is disrupted within a given time frame.  
Interruptions may originate from external alerts, peer messages, or internal task switching.  
A high IR generally correlates with increased cognitive load and lower flow-state engagement (Mark et al., 2008).  
Balancing IR is critical for optimizing productivity and minimizing task-switching costs.
""")

# --- Shannon Entropy (H) ---
st.markdown("**Shannon Entropy (H) — diversity of activity distribution:**")
st.latex(r"H = -\sum_i p_i \log p_i")
st.markdown(r"""
**Interpretation:**$H$ measures the *diversity and unpredictability* of a participant’s activity mix,  
where $p_i$ denotes the relative proportion of time allocated to activity $i$.  
Higher entropy signifies a more balanced and heterogeneous engagement pattern,  
while lower entropy indicates task specialization or potential monotony.  
Entropy-based indicators are often used in behavioral analytics to characterize adaptive task orchestration.
""")

# --- Orchestration Load Index (OLI) ---
st.markdown("**Orchestration Load Index (OLI) — composite index of cognitive intensity:**")
st.latex(r"\text{OLI} = z(\text{Workload}) + z(\text{Cognitive Load}) + z(\text{TSR}) + z(\text{IR})")
st.markdown(r"""
**Interpretation:** OLI integrates multiple *z-scored cognitive indicators* into a single standardized measure.  
It reflects the overall mental and behavioral load imposed on an individual, combining subjective workload,  
perceived cognitive strain, task-switching frequency, and interruption density.  
A higher OLI indicates increased multitasking pressure and cognitive effort across modalities,  
serving as a unified measure for dynamic workload assessment in human–AI collaboration environments.
""")

# --- EWMA for Cognitive Fatigue ---
st.markdown("**Exponentially Weighted Moving Average (EWMA) — fatigue trend estimation:**")
st.latex(r"\text{EWMA}_t = \lambda \, x_t + (1 - \lambda)\,\text{EWMA}_{t-1}")
st.markdown(r"""
**Interpretation:** EWMA applies a smoothing factor $\lambda \in [0,1]$ to track evolving trends  
in cognitive load or performance indicators.  
It assigns higher weight to recent observations while preserving historical memory,  
making it effective for detecting gradual fatigue accumulation or learning adaptation over time.  
A higher $\lambda$ increases sensitivity to short-term fluctuations, whereas smaller values emphasize long-term stability.
""")

# --- Summary ---
st.markdown(r"""
Together, these metrics enable a **multi-dimensional characterization** of learner or worker behavior,  
capturing temporal, cognitive, and contextual dimensions of engagement.  
By analyzing TSR, IR, H, OLI, and EWMA collectively, researchers can evaluate how individuals balance workload,  
respond to interruptions, and maintain sustained attention under varying cognitive demands.
""")

# Helper: daily hours fallback (if workload_hours missing, infer by session timing if possible)
def infer_hours_day(g):
    """Prefer workload_hours sum; else approximate by time span (>=15min) if timestamps exist."""
    if has_w and g["workload_hours"].notna().any():
        return g["workload_hours"].sum()
    if has_ts and g["timestamp"].notna().any():
        span = (g["timestamp"].max() - g["timestamp"].min()) / timedelta(hours=1)
        return max(float(span), 0.25)  # floor at 15 minutes
    return np.nan

# Task-switch transitions per day
def transitions_per_day(g):
    if not has_act:
        return 0
    seq = g.sort_values("timestamp")["activity"].astype(str).tolist() if has_ts else g["activity"].astype(str).tolist()
    return sum(a != b for a, b in zip(seq, seq[1:]))

# Entropy of activity mix per day
def activity_entropy(g):
    if not has_act or g["activity"].isna().all():
        return np.nan
    p = g["activity"].value_counts(normalize=True)
    return float(-np.sum([pi * log(pi + 1e-12) for pi in p.values]))

# Build daily aggregates per instructor (if present) else global per day
grp_keys = []
if has_id: grp_keys.append("instructor_id")
if has_ts: grp_keys.append("date")
if not grp_keys:  # fallback single bucket
    df["__dummy_date__"] = "all"
    grp_keys = ["__dummy_date__"]

agg = df.groupby(grp_keys, dropna=False).apply(lambda g: pd.Series({
    "hours": infer_hours_day(g),
    "mean_cog_load": g["perceived_cognitive_load"].mean() if has_cl else np.nan,
    "interruptions": g["interruptions"].sum() if has_int else np.nan,
    "switches": transitions_per_day(g),
    "entropy": activity_entropy(g),
    "n_rows": len(g),
})).reset_index()

# Rates
agg["tsr"] = agg["switches"] / agg["hours"] if "hours" in agg.columns else np.nan
agg["ir"]  = agg["interruptions"] / agg["hours"] if "hours" in agg.columns else np.nan

# Z-scores (robust: if stdev=0 or NaN, keep NaN)
def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    mu, sd = s.mean(), s.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series([np.nan]*len(s), index=s.index)
    return (s - mu) / sd

agg["z_workload"] = zscore(agg["hours"])
agg["z_cog"]      = zscore(agg["mean_cog_load"])
agg["z_tsr"]      = zscore(agg["tsr"])
agg["z_ir"]       = zscore(agg["ir"])
agg["OLI"]        = agg[["z_workload","z_cog","z_tsr","z_ir"]].sum(axis=1)

st.subheader("Daily/Per-Instructor Aggregates")
st.dataframe(agg.head(20), use_container_width=True)

# ------------------------------------------------------------------
# Advanced visuals
# ------------------------------------------------------------------
st.subheader("A. Distributions & Relationships (Plotly)")

# 1) Histograms + marginal box
if has_w:
    fig_w = px.histogram(df, x="workload_hours", nbins=30, marginal="box",
                         title="Workload Hours Distribution")
    fig_w.update_layout(template="plotly_white")
    st.plotly_chart(fig_w, use_container_width=True)

if has_cl:
    fig_cl = px.histogram(df, x="perceived_cognitive_load", nbins=30, marginal="violin",
                          title="Perceived Cognitive Load Distribution")
    fig_cl.update_layout(template="plotly_white")
    st.plotly_chart(fig_cl, use_container_width=True)

# 2) Violin/ECDF of OLI
if agg["OLI"].notna().any():
    fig_oli = px.violin(agg, y="OLI", box=True, points="all", title="Orchestration Load Index (OLI)")
    fig_oli.update_layout(template="plotly_white")
    st.plotly_chart(fig_oli, use_container_width=True)

    fig_ecdf = px.ecdf(agg.dropna(subset=["OLI"]), x="OLI", title="ECDF — OLI")
    fig_ecdf.update_layout(template="plotly_white")
    st.plotly_chart(fig_ecdf, use_container_width=True)

# 3) Scatter relationships
cols_xy = []
if has_w: cols_xy.append("hours")
if has_cl: cols_xy.append("mean_cog_load")
if "tsr" in agg.columns: cols_xy.append("tsr")
if "ir"  in agg.columns: cols_xy.append("ir")
if "entropy" in agg.columns: cols_xy.append("entropy")

if len(cols_xy) >= 2:
    color_col = "instructor_id" if has_id and "instructor_id" in agg.columns else None
    fig_sc = px.scatter_matrix(agg, dimensions=cols_xy, color=color_col, title="Relationships Between Orchestration Metrics")
    fig_sc.update_layout(template="plotly_white", height=600)
    st.plotly_chart(fig_sc, use_container_width=True)

# ------------------------------------------------------------------
# B. Time Trends & EWMA
# ------------------------------------------------------------------
st.subheader("B. Time Trends & EWMA")

if has_ts:
    # Weekly means
    by = ["week"]
    if has_id:
        by = ["instructor_id","week"]
    wmean = df.groupby(by)["perceived_cognitive_load"].mean().reset_index() if has_cl else pd.DataFrame()

    if not wmean.empty:
        title = "Weekly Cognitive Load"
        fig_t = px.line(wmean, x="week", y="perceived_cognitive_load",
                        color="instructor_id" if has_id else None, markers=True, title=title)
        fig_t.update_layout(template="plotly_white")
        st.plotly_chart(fig_t, use_container_width=True)

        # EWMA per instructor
        lam = st.slider("EWMA λ (smoothing)", 0.05, 0.5, 0.2, step=0.05, key="ewma_lambda")
        def ewma(x, alpha):
            return x.ewm(alpha=alpha, adjust=False).mean()

        ew = []
        if has_id:
            for iid, g in wmean.sort_values("week").groupby("instructor_id"):
                g = g.copy()
                g["EWMA"] = ewma(g["perceived_cognitive_load"], lam)
                ew.append(g)
            ew = pd.concat(ew, ignore_index=True)
        else:
            g = wmean.sort_values("week").copy()
            g["EWMA"] = ewma(g["perceived_cognitive_load"], lam)
            ew = g

        if not ew.empty:
            fig_ew = px.line(ew, x="week", y="EWMA", color="instructor_id" if has_id else None,
                             markers=True, title="EWMA of Cognitive Load")
            fig_ew.update_layout(template="plotly_white")
            st.plotly_chart(fig_ew, use_container_width=True)

# ------------------------------------------------------------------
# C. Activity Mix, Heatmaps & Treemap
# ------------------------------------------------------------------
st.subheader("C. Orchestration Mix")

if has_act:
    # Activity frequency
    act_cnt = df["activity"].value_counts().reset_index()
    act_cnt.columns = ["activity","count"]
    fig_bar = px.bar(act_cnt, x="activity", y="count", title="Activity Frequency")
    fig_bar.update_layout(template="plotly_white", xaxis_tickangle=30)
    st.plotly_chart(fig_bar, use_container_width=True)

    # Treemap by activity (weighted by hours if available)
    if has_w and df["workload_hours"].notna().any():
        w_by_act = df.groupby("activity")["workload_hours"].sum().reset_index()
        fig_tree = px.treemap(w_by_act, path=["activity"], values="workload_hours", title="Workload Hours by Activity (Treemap)")
        st.plotly_chart(fig_tree, use_container_width=True)

    # Heatmap hour-of-day vs weekday (if timestamp present)
    if has_ts:
        tmp = df.dropna(subset=["timestamp"]).copy()
        tmp["hour"] = tmp["timestamp"].dt.hour
        tmp["weekday"] = tmp["timestamp"].dt.weekday  # 0=Mon
        piv = tmp.pivot_table(index="weekday", columns="hour", values="activity", aggfunc="count", fill_value=0)
        hm = go.Figure(data=go.Heatmap(z=piv.values, x=piv.columns, y=piv.index, colorscale="YlGnBu"))
        hm.update_layout(title="Activity Heatmap (weekday × hour)", xaxis_title="Hour", yaxis_title="Weekday", template="plotly_white")
        st.plotly_chart(hm, use_container_width=True)

# ------------------------------------------------------------------
# D. Event Flow (optional)
# ------------------------------------------------------------------
st.subheader("D. Event Flow (optional)")
if has_ev and has_sid:
    # Markov transitions across event_type within session
    s = df[["session_id","event_type"]].dropna().astype(str)
    trans = {}
    for sid, g in s.groupby("session_id"):
        seq = g["event_type"].tolist()
        for a, b in zip(seq, seq[1:]):
            trans[(a,b)] = trans.get((a,b), 0) + 1
    if trans:
        states = sorted({x for pair in trans.keys() for x in pair})
        src, tgt, val = [], [], []
        for (a,b), v in trans.items():
            src.append(states.index(a)); tgt.append(states.index(b)); val.append(v)
        figS = go.Figure(data=[go.Sankey(
            node=dict(label=states, pad=15, thickness=15),
            link=dict(source=src, target=tgt, value=val)
        )])
        figS.update_layout(title="Event Flow (Sankey)", template="plotly_white")
        st.plotly_chart(figS, use_container_width=True)
else:
    st.info("Event flow requires both 'event_type' and 'session_id' columns.")

# ------------------------------------------------------------------
# E. Correlation Matrix & Table
# ------------------------------------------------------------------
st.subheader("E. Correlations & Tables")

num_cols = []
if has_w: num_cols.append("workload_hours")
if has_cl: num_cols.append("perceived_cognitive_load")
for k in ["interruptions","class_size","ai_usage_pct"]:
    if k in df.columns: num_cols.append(k)
for k in ["tsr","ir","entropy","OLI"]:
    if k in agg.columns: pass

# Merge day/instructor aggregates back (suffix _agg)
to_merge = agg.copy()
if has_id:  # merge on instructor + date if possible
    on = ["instructor_id"]
    if "date" in df.columns and "date" in agg.columns:
        on += ["date"]
else:
    on = None

if on:
    merged_view = df.merge(to_merge, on=on, how="left", suffixes=("","_agg"))
else:
    merged_view = df.copy()
    for col in ["hours","mean_cog_load","interruptions","switches","entropy","tsr","ir","OLI","n_rows"]:
        merged_view[f"{col}_agg"] = agg[col].iloc[0] if col in agg.columns and len(agg)>0 else np.nan

# Correlation heatmap (numeric only)
corr_candidates = []
corr_candidates += [c for c in ["workload_hours","perceived_cognitive_load","interruptions","class_size","ai_usage_pct"] if c in merged_view.columns]
corr_candidates += [c for c in ["hours_agg","mean_cog_load_agg","tsr_agg","ir_agg","entropy_agg","OLI_agg"] if c in merged_view.columns]
corr_df = merged_view[corr_candidates].select_dtypes(include=[np.number]).copy()

if not corr_df.empty and corr_df.shape[1] >= 2:
    corr = corr_df.corr()
    heat = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale="RdBu", zmin=-1, zmax=1))
    heat.update_layout(title="Correlation Matrix", template="plotly_white")
    st.plotly_chart(heat, use_container_width=True)

# Pretty datatable of aggregates
table_cols = ["instructor_id","date","hours","mean_cog_load","tsr","ir","entropy","OLI","n_rows"]
table_cols = [c for c in table_cols if c in agg.columns or c in ["instructor_id","date"]]
tshow = agg[table_cols].head(50) if not agg.empty else pd.DataFrame()
if not tshow.empty:
    tfig = go.Figure(data=[go.Table(
        header=dict(values=list(tshow.columns), fill_color="#2c3e50", font=dict(color="white")),
        cells=dict(values=[tshow[c] for c in tshow.columns], align="left")
    )])
    tfig.update_layout(title="Aggregate Snapshot", template="plotly_white")
    st.plotly_chart(tfig, use_container_width=True)

# ------------------------------------------------------------------
# Exports
# ------------------------------------------------------------------
st.subheader("Exports")
def dl(_df, name, label):
    st.download_button(label=label, data=_df.to_csv(index=False).encode("utf-8"),
                       file_name=name, mime="text/csv")

dl(agg, "teacher_aggregates.csv", "Download Aggregates CSV")
dl(merged_view.head(1000), "teacher_logs_enriched_sample.csv", "Download Enriched Sample CSV")

# Persist for Home page (optional KPIs)
st.session_state["teacher_df"] = df
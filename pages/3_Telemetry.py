# pages/3_Telemetry.py  — Advanced
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import scipy.optimize as opt

st.title("3) Telemetry — Prompts / AI Replies / Events (Advanced)")

# -----------------------------------------------------------------------------
# Load & normalize
# -----------------------------------------------------------------------------
if "tele_df" not in st.session_state:
    st.info("Please upload telemetry on the **0. Upload** page.")
    st.stop()

df = st.session_state["tele_df"].copy()
st.write("Preview:", df.head())
df.columns = [c.strip().lower() for c in df.columns]

# -----------------------------------------------------------------------------
# Academic metrics (PEI / RDS) — definitions & computation
# -----------------------------------------------------------------------------
st.markdown(r"""
**Academic formulas**  
**Prompt Evolution Index (PEI)**  
$$
\text{PEI} = 0.3 \cdot \text{LexSpec} + 0.35 \cdot \min\left(\frac{\#\text{strategy verbs}}{6}, 1\right)
+ 0.35 \cdot \min\left(\frac{\#\text{constraint cues}}{6}, 1\right)
$$
where LexSpec is the ratio of unique alphabetic tokens over total alphabetic tokens.  

**Reflection Depth Score (RDS proxy)** (0–4): counts reasoning cues (because, therefore, hence, justify, so that, however, evidence) and adds +1 if text length > 40 tokens, bucketed into 0–4.
""")

def lexical_specificity(text: str) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0
    toks = [t for t in text.split() if t.isalpha()]
    if not toks:
        return 0.0
    unique_ratio = len(set(toks)) / len(toks)
    return round(0.2 + 0.75 * unique_ratio, 3)

STRATEGY = {"plan","debug","optimize","compare","analyze","verify","refactor","test","justify","clarify","connect"}
CONSTRAINT = {"must","include","exactly","at least","no more than","use","ensure","without","limit","constrain"}

def compute_pei(prompt: str) -> float:
    if not isinstance(prompt, str):
        prompt = ""
    sv = sum(1 for w in prompt.lower().split() if w in STRATEGY)
    cd = sum(prompt.lower().count(c) for c in CONSTRAINT)
    return round(0.3*lexical_specificity(prompt) + 0.35*(min(sv,6)/6) + 0.35*(min(cd,6)/6), 3)

def rds_proxy(text: str) -> int:
    if not isinstance(text, str):
        text = ""
    cues = ["because","therefore","hence","justify","so that","however","evidence"]
    score = sum(c in text.lower() for c in cues)
    if len(text.split()) > 40:
        score += 1
    return int(min(4, max(0, score // 2)))

# Compute PEI / RDS if prompt column exists
if "prompt" not in df.columns:
    st.warning("Telemetry must include a 'prompt' column.")
    st.stop()

df["prompt_evolution_index"] = df["prompt"].astype(str).apply(compute_pei)
if "reflection_text" in df.columns:
    df["rds_proxy"] = df["reflection_text"].astype(str).apply(rds_proxy)
elif "ai_response" in df.columns:
    df["rds_proxy"] = df["ai_response"].astype(str).apply(rds_proxy)
else:
    df["rds_proxy"] = 0

# -----------------------------------------------------------------------------
# Feature engineering (optional columns guarded)
# -----------------------------------------------------------------------------
df["prompt_len"] = df["prompt"].astype(str).apply(lambda s: len(s.split()))
if "ai_response" in df.columns:
    df["ai_len"] = df["ai_response"].astype(str).apply(lambda s: len(s.split()))

# Timestamp handling
has_ts = "timestamp" in df.columns
if has_ts:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# If latency columns exist, use them; else try to derive (best-effort)
if "latency_ms" in df.columns:
    df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")
elif has_ts and {"session_id","event_type"}.issubset(df.columns):
    # derive latency between successive events inside a session
    dlat = []
    for sid, g in df.sort_values(["session_id","timestamp"]).groupby("session_id"):
        prev_t = None
        for _, r in g.iterrows():
            t = r["timestamp"]
            if pd.notnull(t) and pd.notnull(prev_t):
                dlat.append((_, (t - prev_t).total_seconds()*1000.0))
            else:
                dlat.append((_, np.nan))
            prev_t = t
    dd = pd.DataFrame(dlat, columns=["_idx","latency_ms"]).set_index("_idx")
    df = df.join(dd, how="left")
else:
    df["latency_ms"] = np.nan

# Session-local turn index (within each session)
if "session_id" in df.columns:
    df["_turn_idx"] = df.sort_values(["session_id", "timestamp"]).groupby("session_id").cumcount() + 1
else:
    df["_turn_idx"] = np.arange(1, len(df)+1)

# -----------------------------------------------------------------------------
# Filters
# -----------------------------------------------------------------------------
with st.expander("Filters"):
    cols = st.columns(4)
    # group/course/topic filters if present
    groups = sorted(df["group"].dropna().unique()) if "group" in df.columns else []
    sessions = sorted(df["session_id"].dropna().unique()) if "session_id" in df.columns else []
    evtypes = sorted(df["event_type"].dropna().unique()) if "event_type" in df.columns else []

    sel_groups = cols[0].multiselect("Group", groups, default=groups if groups else None) if groups else []
    sel_sessions = cols[1].multiselect("Session", sessions, default=sessions[:25] if len(sessions)>25 else sessions) if sessions else []
    sel_events = cols[2].multiselect("Event types", evtypes, default=evtypes if evtypes else None) if evtypes else []
    pei_min, pei_max = cols[3].slider("PEI range", 0.0, 1.0, (0.0, 1.0), step=0.05)

    if has_ts:
        tmin = pd.to_datetime(df["timestamp"]).min()
        tmax = pd.to_datetime(df["timestamp"]).max()
        tr = st.slider("Time window", min_value=tmin.to_pydatetime() if pd.notnull(tmin) else None,
                       max_value=tmax.to_pydatetime() if pd.notnull(tmax) else None,
                       value=(tmin.to_pydatetime(), tmax.to_pydatetime()) if pd.notnull(tmin) and pd.notnull(tmax) else None) if pd.notnull(tmin) and pd.notnull(tmax) else None
    else:
        tr = None

mask = (df["prompt_evolution_index"].between(pei_min, pei_max))
if groups and sel_groups:
    mask &= df["group"].isin(sel_groups)
if "session_id" in df.columns and sel_sessions:
    mask &= df["session_id"].isin(sel_sessions)
if "event_type" in df.columns and sel_events:
    mask &= df["event_type"].isin(sel_events)
if has_ts and tr:
    mask &= df["timestamp"].between(tr[0], tr[1])
df_f = df[mask].copy()

st.success(f"Computed PEI & RDS proxy for N={len(df_f)} rows (after filters).")

cols = ["prompt","prompt_evolution_index","rds_proxy","prompt_len","ai_len","latency_ms","_turn_idx"]
cols = [c for c in cols if c in df_f.columns]  # keep only existing ones
st.dataframe(df_f[cols].head(25))

# -----------------------------------------------------------------------------
# Advanced visual analytics (Plotly)
# -----------------------------------------------------------------------------
st.subheader("A. Distributions & Relationships")

# 1) PEI distribution with marginal box & rug
fig1 = px.histogram(df_f, x="prompt_evolution_index", nbins=30, marginal="box",
                    title="PEI Distribution", opacity=0.8)
fig1.update_layout(template="plotly_white", xaxis_title="PEI", yaxis_title="Count")
st.plotly_chart(fig1, use_container_width=True)

# 2) RDS distribution + violin
fig2 = px.violin(df_f, y="rds_proxy", box=True, points="all", title="RDS Proxy Distribution (0–4)")
fig2.update_layout(template="plotly_white", yaxis_title="RDS Proxy")
st.plotly_chart(fig2, use_container_width=True)

# 3) Scatter: PEI vs prompt length, colored by group (if exists)
color_col = "group" if "group" in df_f.columns else None
fig3 = px.scatter(df_f, x="prompt_len", y="prompt_evolution_index",
                  color=color_col, trendline="ols" if len(df_f)>5 else None,
                  title="PEI vs Prompt Length")
fig3.update_layout(template="plotly_white", xaxis_title="Prompt length (tokens)", yaxis_title="PEI")
st.plotly_chart(fig3, use_container_width=True)

# 4) Latency vs PEI (if latency present)
if df_f["latency_ms"].notna().any():
    fig4 = px.scatter(df_f, x="latency_ms", y="prompt_evolution_index", color=color_col,
                      title="Latency (ms) vs PEI", trendline="lowess" if len(df_f)>20 else None)
    fig4.update_layout(template="plotly_white", xaxis_title="Latency (ms)", yaxis_title="PEI")
    st.plotly_chart(fig4, use_container_width=True)

# -----------------------------------------------------------------------------
# B. Time & session analytics
# -----------------------------------------------------------------------------
st.subheader("B. Time & Session Analytics")

# 1) PEI over turn index (learning curve view)
agg_turn = df_f.groupby("_turn_idx")["prompt_evolution_index"].mean().reset_index()
if len(agg_turn) >= 3:
    # Simple exponential learning curve: y = L - (L - y0)*exp(-k*t)
    def exp_curve(t, L, y0, k):
        return L - (L - y0)*np.exp(-k*t)
    try:
        popt, pcov = opt.curve_fit(exp_curve, agg_turn["_turn_idx"], agg_turn["prompt_evolution_index"],
                                   p0=(0.9, 0.3, 0.1), maxfev=5000)
        Lhat, y0hat, khat = popt
    except Exception:
        Lhat, y0hat, khat = np.nan, np.nan, np.nan

    figlc = go.Figure()
    figlc.add_trace(go.Scatter(x=agg_turn["_turn_idx"], y=agg_turn["prompt_evolution_index"],
                               mode="markers+lines", name="Mean PEI"))
    if np.isfinite(Lhat):
        xs = np.linspace(agg_turn["_turn_idx"].min(), agg_turn["_turn_idx"].max(), 200)
        figlc.add_trace(go.Scatter(x=xs, y=exp_curve(xs, Lhat, y0hat, khat),
                                   mode="lines", name=f"Fit: L={Lhat:.2f}, y0={y0hat:.2f}, k={khat:.2f}"))
    figlc.update_layout(title="Learning Curve: PEI over Turn Index", template="plotly_white",
                        xaxis_title="Turn index (within session)", yaxis_title="Mean PEI")
    st.plotly_chart(figlc, use_container_width=True)

# 2) Timeline (Gantt) per session if timestamps exist
if has_ts and "session_id" in df_f.columns:
    # Build durations between events for a rough timeline
    gg = df_f.sort_values(["session_id","timestamp"]).copy()
    gg["t_end"] = gg.groupby("session_id")["timestamp"].shift(-1)
    gg["dur"] = (gg["t_end"] - gg["timestamp"]).dt.total_seconds()
    gg["dur"] = gg["dur"].fillna(gg["dur"].median())  # fill last event duration
    tl = gg.dropna(subset=["timestamp"]).copy()
    if not tl.empty:
        figtl = px.timeline(tl, x_start="timestamp", x_end="t_end", y="session_id",
                            color="event_type" if "event_type" in tl.columns else None,
                            hover_data=["prompt_evolution_index","rds_proxy","latency_ms"])
        figtl.update_layout(title="Session Timelines (approx.)", template="plotly_white", showlegend=True)
        st.plotly_chart(figtl, use_container_width=True)

# -----------------------------------------------------------------------------
# C. Process mining (Markov transitions, Sankey)
# -----------------------------------------------------------------------------
st.subheader("C. Process Mining")

if {"event_type","session_id"}.issubset(df_f.columns):
    s = df_f[["session_id","event_type"]].dropna().copy()
    s["event_type"] = s["event_type"].astype(str)
    # Build first-order transitions
    trans = {}
    for sid, g in s.groupby("session_id"):
        seq = g["event_type"].tolist()
        for a, b in zip(seq, seq[1:]):
            trans[(a,b)] = trans.get((a,b), 0) + 1
    if trans:
        # Heatmap (transition counts)
        states = sorted(set([a for a,b in trans.keys()]) | set([b for a,b in trans.keys()]))
        idx = {stt:i for i,stt in enumerate(states)}
        mat = np.zeros((len(states), len(states)), dtype=int)
        for (a,b), v in trans.items():
            mat[idx[a], idx[b]] = v

        figH = go.Figure(data=go.Heatmap(z=mat, x=states, y=states, colorscale="Blues"))
        figH.update_layout(title="Transition Matrix (counts)", xaxis_title="To", yaxis_title="From", template="plotly_white")
        st.plotly_chart(figH, use_container_width=True)

        # Sankey
        src, tgt, val = [], [], []
        for (a,b), v in trans.items():
            src.append(states.index(a)); tgt.append(states.index(b)); val.append(v)
        figS = go.Figure(data=[go.Sankey(
            node=dict(label=states, pad=15, thickness=15),
            link=dict(source=src, target=tgt, value=val)
        )])
        figS.update_layout(title_text="Event Flow (Sankey)", template="plotly_white")
        st.plotly_chart(figS, use_container_width=True)
else:
    st.info("Process mining requires 'event_type' and 'session_id' columns.")

# -----------------------------------------------------------------------------
# D. Group comparisons & ECDFs
# -----------------------------------------------------------------------------
st.subheader("D. Group Comparisons")

if "group" in df_f.columns:
    grp = df_f.groupby("group")[["prompt_evolution_index","rds_proxy","prompt_len","ai_len","latency_ms"]].agg(["mean","std","count"])
    grp.columns = ["_".join(c).strip() for c in grp.columns.to_flat_index()]
    st.dataframe(grp.reset_index(), use_container_width=True)

    figG = px.box(df_f, x="group", y="prompt_evolution_index", points="all", title="PEI by Group")
    figG.update_layout(template="plotly_white")
    st.plotly_chart(figG, use_container_width=True)

# ECDF Prevalence
melt = df_f[["prompt_evolution_index","rds_proxy"]].rename(columns={"prompt_evolution_index":"PEI","rds_proxy":"RDS"})
melt = melt.melt(value_name="value", var_name="metric")
figE = px.ecdf(melt, x="value", color="metric", title="ECDF — PEI vs RDS")
figE.update_layout(template="plotly_white", xaxis_title="Value")
st.plotly_chart(figE, use_container_width=True)

# -----------------------------------------------------------------------------
# E. Data tables (Plotly Table) & exports
# -----------------------------------------------------------------------------
st.subheader("E. Tables & Export")

# Table: top 50 rows with key metrics
cols_show = [c for c in ["timestamp","session_id","event_type","prompt","ai_response","prompt_evolution_index","rds_proxy","prompt_len","ai_len","latency_ms","_turn_idx","group"] if c in df_f.columns]
tab_df = df_f[cols_show].head(50).copy()
if not tab_df.empty:
    table = go.Figure(data=[go.Table(
        header=dict(values=list(tab_df.columns), fill_color="#34495e", font=dict(color="white")),
        cells=dict(values=[tab_df[c] for c in tab_df.columns], align="left")
    )])
    table.update_layout(title="Sample Telemetry Rows", template="plotly_white")
    st.plotly_chart(table, use_container_width=True)
else:
    st.info("No rows to show for the current filters.")

def dl(_df, name, label):
    st.download_button(label=label,
                       data=_df.to_csv(index=False).encode("utf-8"),
                       file_name=name, mime="text/csv")

dl(df_f, "telemetry_with_pei_rds.csv", "Download Telemetry + PEI/RDS (filtered)")
if "group" in df_f.columns:
    dl(grp.reset_index(), "telemetry_group_summary.csv", "Download Group Summary")

# -----------------------------------------------------------------------------
# Save for other pages
# -----------------------------------------------------------------------------
st.session_state["telemetry_with_pei"] = df_f[["prompt_evolution_index","rds_proxy"] + ([ "group"] if "group" in df_f.columns else [])]
# --- Save for other pages (keep text columns!) ---
keep_cols = [c for c in [
    "student_id","group","week","session_id","event_type","timestamp",
    "prompt","ai_response",
    "prompt_evolution_index","rds_proxy","prompt_len","ai_len","latency_ms","_turn_idx"
] if c in df_f.columns]

# --- Save for other pages (keep IDs + mediator columns) ---
keep_cols = [c for c in [
    "student_id", "group", "week",
    "prompt", "ai_response",
    "prompt_evolution_index", "rds_proxy",
    # any usage columns if you have them:
    "usage_minutes", "turns", "sessions", "events", "interaction_count"
] if c in df.columns]  # df is your telemetry dataframe after cleaning

st.session_state["telemetry_with_pei"] = df_f[keep_cols].copy()

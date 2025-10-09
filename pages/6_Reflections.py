# pages/6_Reflections.py — Advanced RDS & Reflection Analytics
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.title("6) Reflections — RDS & Text Analytics (Advanced)")

# --------------------------------------------------------------------------------------
# 0) Data source selection & preview
# --------------------------------------------------------------------------------------
if "reflections_df" not in st.session_state and "telemetry_with_pei" not in st.session_state:
    st.info("Please upload reflections (or rely on telemetry’s ai_response) on the **0. Upload** page.")
    st.stop()

source_opt = st.radio(
    "Choose reflections source",
    options=[
        "reflections_df (preferred if uploaded)",
        "telemetry_with_pei (use ai_response)"
    ],
    index=0 if "reflections_df" in st.session_state else 1,
    help="If both are present, you can choose."
)

if source_opt.startswith("reflections_df") and "reflections_df" in st.session_state:
    df = st.session_state["reflections_df"].copy()
    text_col_guess = "reflection_text" if "reflection_text" in map(str.lower, df.columns) else None
    fallback_col = "ai_response" if "ai_response" in map(str.lower, df.columns) else None
    source_label = "reflection_text"
else:
    df = st.session_state["telemetry_with_pei"].copy()
    text_col_guess = "ai_response"
    fallback_col = None
    source_label = "ai_response"

st.write("Preview:", df.head())
df.columns = [c.strip().lower() for c in df.columns]
text_col = text_col_guess if text_col_guess in df.columns else (fallback_col if fallback_col in df.columns else None)
if text_col is None:
    st.warning("Provide a text column: 'reflection_text' or 'ai_response'.")
    st.stop()

# --------------------------------------------------------------------------------------
# 1) Academic formula & helper functions
# --------------------------------------------------------------------------------------
st.markdown(r"""
### Academic Formula
**Reflection Depth Score (RDS proxy)** (0–4) measures the presence of reasoning & metacognitive cues:
\[
\text{RDS} = \Big\lfloor \frac{\#\{\text{cues}\}}{2} \Big\rfloor + \mathbb{1}\{\text{tokens} > 40\},
\;\; \text{capped to } [0,4]
\]
where **cues** ∈ {because, therefore, hence, justify, *so that*, however, evidence}.

We also compute supporting indicators:
- **Length** (tokens, sentences), **Type–Token Ratio** (TTR) = \(|V|/N\)  
- **Cohesion proxy** = proportion of discourse markers (e.g., however, therefore, because, first, next, finally, in conclusion, for example)  
- **Readability proxies**: Flesch Reading Ease (heuristic syllable estimate), normalized to \([0, 1]\) for comparability.
""")

CUES = ["because","therefore","hence","justify","so that","however","evidence"]
DISCOURSE = [
    "however","therefore","because","thus","hence","moreover","furthermore",
    "first","second","next","finally","in conclusion","for example","for instance","on the other hand"
]

def tokenize(s: str):
    if not isinstance(s, str): 
        return []
    return re.findall(r"[A-Za-z']+", s.lower())

def sentence_split(s: str):
    if not isinstance(s, str): return []
    return re.split(r"[.!?]+", s)

def estimate_syllables(word: str):
    # naive heuristic
    w = word.lower()
    if not w: return 0
    vowels = "aeiouy"
    count = 0
    prev_v = False
    for ch in w:
        is_v = ch in vowels
        if is_v and not prev_v:
            count += 1
        prev_v = is_v
    if w.endswith("e") and count > 1:
        count -= 1
    return max(1, count)

def flesch_reading_ease(text: str):
    toks = tokenize(text)
    sents = [s for s in sentence_split(text) if s.strip()]
    if len(toks) == 0 or len(sents) == 0:
        return np.nan
    words = len(toks)
    sentences = max(1, len(sents))
    syllables = sum(estimate_syllables(w) for w in toks)
    # Flesch Reading Ease (English): 206.835 - 1.015*(words/sentences) - 84.6*(syllables/words)
    fre = 206.835 - 1.015*(words/sentences) - 84.6*(syllables/words)
    return fre

def normalize_0_1(x):
    x = pd.to_numeric(x, errors="coerce")
    mn, mx = x.min(), x.max()
    if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
        return pd.Series([np.nan]*len(x), index=x.index)
    return (x - mn) / (mx - mn)

def rds_proxy(text: str) -> int:
    if not isinstance(text, str):
        text = ""
    cues = sum(c in text.lower() for c in CUES)
    tokens = len(tokenize(text))
    score = (cues // 2) + (1 if tokens > 40 else 0)
    return int(min(4, max(0, score)))

def cohesion_proxy(text: str) -> float:
    if not isinstance(text, str): return 0.0
    toks = tokenize(text)
    N = len(toks)
    if N == 0: return 0.0
    hits = sum(1 for cue in DISCOURSE if cue in " ".join(toks))
    return hits / max(1, len(DISCOURSE))

# --------------------------------------------------------------------------------------
# 2) Compute features
# --------------------------------------------------------------------------------------
df = df.copy()
df["rds_proxy"] = df[text_col].astype(str).apply(rds_proxy)
df["tokens"] = df[text_col].astype(str).apply(lambda s: len(tokenize(s)))
df["sentences"] = df[text_col].astype(str).apply(lambda s: len([x for x in sentence_split(s) if x.strip()]))
df["ttr"] = df[text_col].astype(str).apply(lambda s: (len(set(tokenize(s))) / max(1, len(tokenize(s)))))
df["cohesion"] = df[text_col].astype(str).apply(cohesion_proxy)
df["flesch"] = df[text_col].astype(str).apply(flesch_reading_ease)
df["flesch_norm"] = normalize_0_1(df["flesch"])

# --------------------------------------------------------------------------------------
# 3) Filters
# --------------------------------------------------------------------------------------
with st.expander("Filters"):
    cols = st.columns(3)
    groups = sorted(df["group"].dropna().unique()) if "group" in df.columns else []
    sel_groups = cols[0].multiselect("Group", groups, default=groups if groups else None)
    if "week" in df.columns and pd.api.types.is_numeric_dtype(df["week"]):
        wmin, wmax = int(df["week"].min()), int(df["week"].max())
        sel_weeks = cols[1].slider("Week range", wmin, wmax, (wmin, wmax), step=1)
    else:
        sel_weeks = None
    rmin, rmax = cols[2].slider("RDS range", 0, 4, (0, 4), step=1)
    tmin, tmax = st.slider("Token length range", int(df["tokens"].min()), int(df["tokens"].max()),
                           (int(df["tokens"].min()), int(df["tokens"].max())), step=5)

mask = pd.Series(True, index=df.index)
if groups and sel_groups:
    mask &= df["group"].isin(sel_groups)
if sel_weeks:
    mask &= df["week"].between(sel_weeks[0], sel_weeks[1])
mask &= df["rds_proxy"].between(rmin, rmax)
mask &= df["tokens"].between(tmin, tmax)
df_f = df[mask].copy()

st.success(f"Rows after filters: {len(df_f)}")
st.dataframe(df_f[[text_col, "rds_proxy", "tokens", "ttr", "cohesion", "flesch", "flesch_norm"]].head(20),
             use_container_width=True)

# --------------------------------------------------------------------------------------
# 4) Advanced visualizations (Plotly)
# --------------------------------------------------------------------------------------
st.subheader("A) Distributions & Relationships")

# Histogram + marginal for RDS
fig1 = px.histogram(df_f, x="rds_proxy", nbins=5, category_orders={"rds_proxy":[0,1,2,3,4]},
                    title="RDS Proxy Distribution", marginal="box")
fig1.update_layout(template="plotly_white", xaxis_tickmode="linear")
st.plotly_chart(fig1, use_container_width=True)

# Violin by group (if available)
if "group" in df_f.columns and df_f["group"].notna().any():
    fig2 = px.violin(df_f, x="group", y="rds_proxy", box=True, points="all",
                     title="RDS by Group")
    fig2.update_layout(template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

# ECDF of key metrics
melt = df_f[["rds_proxy","tokens","ttr","cohesion","flesch_norm"]].rename(columns={
    "rds_proxy":"RDS","tokens":"Tokens","ttr":"TTR","cohesion":"Cohesion","flesch_norm":"Flesch(norm)"
}).melt(var_name="metric", value_name="value")
fig3 = px.ecdf(melt.dropna(), x="value", color="metric", title="ECDF — Reflection Metrics")
fig3.update_layout(template="plotly_white")
st.plotly_chart(fig3, use_container_width=True)

# Scatter relationships
fig4 = px.scatter(df_f, x="tokens", y="rds_proxy", color="group" if "group" in df_f.columns else None,
                  trendline="ols" if len(df_f) > 20 else None,
                  title="RDS vs Token Length")
fig4.update_layout(template="plotly_white")
st.plotly_chart(fig4, use_container_width=True)

fig5 = px.scatter(df_f, x="flesch_norm", y="rds_proxy",
                  color="group" if "group" in df_f.columns else None,
                  title="RDS vs Readability (Flesch, normalized)")
fig5.update_layout(template="plotly_white", xaxis_title="Flesch (0–1 scaled)")
st.plotly_chart(fig5, use_container_width=True)

# Heatmap: TTR vs Cohesion binned, colored by mean RDS
bins_ttr = pd.cut(df_f["ttr"], bins=np.linspace(0, 1, 11), include_lowest=True)
bins_coh = pd.cut(df_f["cohesion"], bins=np.linspace(0, 1, 11), include_lowest=True)
grid = df_f.assign(ttr_bin=bins_ttr, coh_bin=bins_coh).groupby(["ttr_bin","coh_bin"])["rds_proxy"].mean().reset_index()
if not grid.empty:
    grid["ttr_bin"] = grid["ttr_bin"].astype(str)
    grid["coh_bin"] = grid["coh_bin"].astype(str)
    heat = go.Figure(data=go.Heatmap(
        z=grid["rds_proxy"], x=grid["ttr_bin"], y=grid["coh_bin"], colorscale="YlGnBu", colorbar_title="Mean RDS"
    ))
    heat.update_layout(title="Mean RDS across TTR × Cohesion bins", template="plotly_white",
                       xaxis_title="TTR bin", yaxis_title="Cohesion bin")
    st.plotly_chart(heat, use_container_width=True)

# --------------------------------------------------------------------------------------
# 5) Weekly trends (if week exists)
# --------------------------------------------------------------------------------------
if "week" in df_f.columns and pd.api.types.is_numeric_dtype(df_f["week"]):
    st.subheader("B) Weekly Trend")
    wtrend = df_f.groupby("week")[["rds_proxy","tokens","ttr","cohesion","flesch_norm"]].mean().reset_index()
    figw = px.line(wtrend, x="week", y=["rds_proxy","tokens","ttr","cohesion","flesch_norm"],
                   markers=True, title="Weekly Mean of Reflection Metrics")
    figw.update_layout(template="plotly_white")
    st.plotly_chart(figw, use_container_width=True)

# --------------------------------------------------------------------------------------
# 6) Link to outcomes (if learning gains available)
# --------------------------------------------------------------------------------------
st.subheader("C) Reflection Quality ↔ Learning Outcomes (optional)")
if "learning_gains" in st.session_state and "student_id" in df_f.columns:
    lg = st.session_state["learning_gains"].copy()
    lg.columns = [c.lower() for c in lg.columns]
    if {"student_id","learning_gain"}.issubset(lg.columns):
        joined = df_f.merge(lg[["student_id","learning_gain"]], on="student_id", how="inner")
        if not joined.empty:
            st.info(f"Joined rows with learning gains: {len(joined)}")
            figjx = px.scatter(joined, x="rds_proxy", y="learning_gain",
                               color="group" if "group" in joined.columns else None,
                               trendline="ols" if len(joined) > 20 else None,
                               title="RDS vs Learning Gain")
            figjx.update_layout(template="plotly_white")
            st.plotly_chart(figjx, use_container_width=True)

            # bucket by RDS
            bucket = joined.groupby("rds_proxy")["learning_gain"].agg(["mean","std","count"]).reset_index()
            tbl = go.Figure(data=[go.Table(
                header=dict(values=list(bucket.columns), fill_color="#2c3e50", font=dict(color="white")),
                cells=dict(values=[bucket[c] for c in bucket.columns], align="left")
            )])
            tbl.update_layout(title="Learning Gain by RDS bucket", template="plotly_white")
            st.plotly_chart(tbl, use_container_width=True)
        else:
            st.info("No overlap between reflections and learning_gains on student_id.")
else:
    st.caption("Tip: Upload assessments first so this section can relate reflection quality to learning outcomes.")

# --------------------------------------------------------------------------------------
# 7) Tables & exports
# --------------------------------------------------------------------------------------
st.subheader("D) Tables & Export")

# Top rows DataTable
show_cols = [c for c in [text_col, "rds_proxy","tokens","sentences","ttr","cohesion","flesch","flesch_norm","student_id","group","week"] if c in df_f.columns]
sample = df_f[show_cols].head(100)
if not sample.empty:
    tfig = go.Figure(data=[go.Table(
        header=dict(values=list(sample.columns), fill_color="#34495e", font=dict(color="white")),
        cells=dict(values=[sample[c] for c in sample.columns], align="left")
    )])
    tfig.update_layout(title="Sample Reflections with Metrics", template="plotly_white")
    st.plotly_chart(tfig, use_container_width=True)

# Download buttons
def dl(_df, name, label):
    st.download_button(label=label, data=_df.to_csv(index=False).encode("utf-8"),
                       file_name=name, mime="text/csv")

dl(df_f[[c for c in df_f.columns if c != ""]], "reflections_enriched.csv", "Download Enriched Reflections CSV")
if "learning_gains" in st.session_state and "student_id" in df_f.columns:
    if 'joined' in locals() and not joined.empty:
        dl(joined, "reflections_with_learning_gains.csv", "Download Reflections + Learning Gains CSV")
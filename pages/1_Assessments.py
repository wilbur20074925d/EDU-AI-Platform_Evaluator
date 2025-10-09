import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import scipy.optimize as opt
import plotly.express as px
import plotly.graph_objects as go

st.title("1) Assessments — Pre/Post & Learning Gains (Advanced)")

# ---------- Load ----------
if "assess_df" not in st.session_state:
    st.info("Please upload assessments on the **0. Upload** page.")
    st.stop()

raw = st.session_state["assess_df"].copy()
st.write("Preview of uploaded data:", raw.head())

# ---------- Normalize ----------
raw.columns = [c.strip().lower() for c in raw.columns]
needed = {"student_id", "phase", "score"}
if not needed.issubset(raw.columns):
    st.error("Columns required: student_id, phase, score (and optional group)")
    st.stop()

has_item_bank = {"course_id","course_name","instructor_or_pi","question_id","topic","answer"}.issubset(raw.columns)

work = raw.copy()
work["phase"] = work["phase"].astype(str).str.lower().str.strip()
work["score"] = pd.to_numeric(work["score"], errors="coerce")

# ---------- Filters (applied BEFORE aggregation) ----------
with st.expander("Filters"):
    colf1, colf2, colf3 = st.columns(3)
    groups = sorted(work["group"].dropna().unique()) if "group" in work.columns else []
    courses = sorted(work["course_id"].dropna().unique()) if "course_id" in work.columns else []
    topics = sorted(work["topic"].dropna().unique()) if "topic" in work.columns else []

    sel_groups = colf1.multiselect("Group filter", groups, default=groups if groups else None) if groups else []
    sel_courses = colf2.multiselect("Course filter", courses, default=courses if courses else None) if courses else []
    sel_topics = colf3.multiselect("Topic filter", topics, default=topics if topics else None) if topics else []

    smin, smax = st.slider("Score range filter", 0.0, 100.0, (0.0, 100.0), step=1.0)
    trim_outliers = st.checkbox("Trim outliers by IQR (phase-level before aggregation)", value=False)

# Apply filters
def iqr_trim(df, value_col):
    q1, q3 = df[value_col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return df[(df[value_col] >= lo) & (df[value_col] <= hi)]

mask = (work["score"].between(smin, smax))
if groups and sel_groups:
    mask &= work["group"].isin(sel_groups)
if courses and sel_courses:
    mask &= work["course_id"].isin(sel_courses)
if topics and sel_topics:
    mask &= work["topic"].isin(sel_topics)

work = work[mask].copy()
if trim_outliers:
    # Trim within each (student, phase) bucket to be safe
    work = work.groupby(["student_id","phase"], group_keys=False).apply(lambda d: iqr_trim(d, "score"))

# ---------- Aggregation ----------
st.subheader("Aggregation Settings")
agg_method = st.selectbox("When multiple rows per (student_id, phase) exist, aggregate by:",
                          ["mean", "median", "max", "min"], index=0)
agg_map = {"mean":"mean","median":"median","max":"max","min":"min"}[agg_method]

phase_scores = (work.groupby(["student_id","phase"], dropna=False)["score"]
                    .agg(agg_map)
                    .reset_index()
                    .rename(columns={"score":"phase_score"}))

pre = phase_scores[phase_scores["phase"].str.contains("pre", na=False)] \
        .rename(columns={"phase_score":"pre"})[["student_id","pre"]]
post = phase_scores[phase_scores["phase"].str.contains("post", na=False)] \
        .rename(columns={"phase_score":"post"})[["student_id","post"]]

merged = pd.merge(pre, post, on="student_id", how="inner")
merged["learning_gain"] = merged["post"] - merged["pre"]

# Attach group (one group per student)
if "group" in work.columns:
    group_map = (work[["student_id","group"]]
                 .dropna()
                 .drop_duplicates(subset=["student_id"], keep="last"))
    merged = merged.merge(group_map, on="student_id", how="left")

# Optional: attach dominant course/topic for breakdowns
with st.expander("Optional: attach dominant course/topic (for breakdowns)"):
    attach_breakdowns = st.checkbox("Attach dominant course/topic", value=False)
    if attach_breakdowns and has_item_bank:
        def _mode(s): 
            m = s.mode()
            return m.iloc[0] if not m.empty else np.nan
        merged = (merged
                  .merge(work.groupby("student_id")["course_id"].apply(_mode).reset_index(), on="student_id", how="left")
                  .merge(work.groupby("student_id")["topic"].apply(_mode).reset_index(), on="student_id", how="left"))

st.success(f"Computed learning gains for N={len(merged)} students.")
st.dataframe(merged.head(25), use_container_width=True)

# ---------- Stats ----------
st.subheader("Primary Statistics")

def cohens_d_paired(diff):
    diff = pd.Series(diff).dropna()
    if diff.empty: return np.nan
    return diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan

def bootstrap_ci_mean(x, B=2000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    x = pd.Series(x).dropna().values
    if len(x) == 0: return (np.nan, np.nan)
    boot = []
    n = len(x)
    for _ in range(B):
        sample = rng.choice(x, size=n, replace=True)
        boot.append(np.mean(sample))
    lo = np.percentile(boot, 100*alpha/2)
    hi = np.percentile(boot, 100*(1-alpha/2))
    return lo, hi

diff = merged["learning_gain"]
d = cohens_d_paired(diff)
ci_lo, ci_hi = bootstrap_ci_mean(diff, B=2000, alpha=0.05)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Mean Gain (Post−Pre)", f"{diff.mean():.2f}")
kpi2.metric("Std. Dev (Gain)", f"{diff.std(ddof=1):.2f}")
kpi3.metric("95% CI (Mean Gain)", f"[{ci_lo:.2f}, {ci_hi:.2f}]")
kpi4.metric("Cohen's d (paired)", f"{d:.3f}")

# ---------- Model fit: Post = a + b * Pre ----------
st.subheader("Regression-style Fit (Post as function of Pre)")
def linear_fn(x, a, b):  # a + b*pre
    return a + b * x

try:
    popt, pcov = opt.curve_fit(linear_fn, merged["pre"].values, merged["post"].values)
    a_hat, b_hat = popt
    st.write(f"Fitted model: **post ≈ {a_hat:.3f} + {b_hat:.3f} × pre**")
except Exception as e:
    a_hat, b_hat = np.nan, np.nan
    st.info(f"Model fit skipped: {e}")

# ---------- Visuals ----------
st.subheader("Visualizations (Interactive)")

# 1) Plotly Histogram of Learning Gains with marginal
fig_hist = px.histogram(merged, x="learning_gain", nbins=30, marginal="box",
                        title="Learning Gain Distribution (Plotly)")
st.plotly_chart(fig_hist, use_container_width=True)

# 2) ECDF of Pre vs Post
fig_ecdf = px.ecdf(merged.melt(id_vars=["student_id"], value_vars=["pre","post"],
                               var_name="phase", value_name="score"),
                   x="score", color="phase", title="ECDF — Pre vs Post")
st.plotly_chart(fig_ecdf, use_container_width=True)

# 3) Scatter Pre vs Post with y=x and fitted line
scatter = go.Figure()
scatter.add_trace(go.Scatter(
    x=merged["pre"], y=merged["post"], mode="markers",
    name="Students", opacity=0.6
))
# identity line
xymin = float(np.nanmin([merged["pre"].min(), merged["post"].min()]))
xymax = float(np.nanmax([merged["pre"].max(), merged["post"].max()]))
scatter.add_trace(go.Scatter(x=[xymin, xymax], y=[xymin, xymax],
                             mode="lines", name="y = x", line=dict(dash="dash")))
# fitted line
if np.isfinite(a_hat) and np.isfinite(b_hat):
    xs = np.linspace(xymin, xymax, 100)
    ys = a_hat + b_hat*xs
    scatter.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Fit: post = a + b*pre"))
scatter.update_layout(title="Pre vs Post (with Identity & Fitted Line)",
                      xaxis_title="Pre", yaxis_title="Post")
st.plotly_chart(scatter, use_container_width=True)

# 4) Violin/Box by Group (if available)
if "group" in merged.columns:
    fig_v = px.violin(merged, x="group", y="learning_gain", box=True, points="all",
                      title="Learning Gain by Group (Violin + Points)")
    st.plotly_chart(fig_v, use_container_width=True)

# ---------- Classic Matplotlib (kept) ----------
# ---------- Advanced Interactive Distribution Plots ----------


st.subheader("Interactive Score Distributions (Plotly)")

col1, col2, col3 = st.columns(3)

# Overlaid Pre vs Post Distribution
with col1:
    df_long = merged.melt(id_vars=["student_id"], value_vars=["pre", "post"],
                          var_name="phase", value_name="score")
    fig1 = px.histogram(df_long, x="score", color="phase",
                        nbins=25, barmode="overlay", opacity=0.65,
                        marginal="box",
                        title="Pre vs Post Distribution (Overlay)",
                        color_discrete_sequence=["#4c78a8", "#f58518"])
    fig1.update_layout(
        xaxis_title="Score",
        yaxis_title="Count",
        legend_title="Phase",
        template="plotly_white"
    )
    st.plotly_chart(fig1, use_container_width=True)

# Learning Gain Distribution + KDE + ECDF
with col2:
    fig2 = go.Figure()
    # Histogram
    fig2.add_trace(go.Histogram(
        x=merged["learning_gain"],
        nbinsx=25,
        name="Learning Gain",
        marker=dict(color="#2ca02c", line=dict(width=1, color="white")),
        opacity=0.7,
        histnorm="probability density"
    ))
    # Add smoothed KDE using numpy hist with gaussian filter
    x = np.linspace(merged["learning_gain"].min(), merged["learning_gain"].max(), 200)
    hist, bins = np.histogram(merged["learning_gain"], bins=30, density=True)
    centers = (bins[:-1] + bins[1:]) / 2
    from scipy.ndimage import gaussian_filter1d
    smooth = gaussian_filter1d(hist, sigma=2)
    fig2.add_trace(go.Scatter(
        x=centers, y=smooth, mode="lines",
        name="Smoothed KDE", line=dict(color="red", width=2)
    ))
    fig2.update_layout(
        title="Learning Gain Distribution (KDE + Histogram)",
        xaxis_title="Learning Gain",
        yaxis_title="Density",
        template="plotly_white"
    )
    st.plotly_chart(fig2, use_container_width=True)

# 3D Surface of Joint Pre/Post Distribution
with col3:
    # Generate 2D histogram density
    H, xedges, yedges = np.histogram2d(
        merged["pre"], merged["post"], bins=20, density=True
    )
    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])
    fig3 = go.Figure(
        data=[go.Surface(
            z=H.T,
            x=xcenters,
            y=ycenters,
            colorscale="Viridis",
            contours={"z": {"show": True, "usecolormap": True, "highlightcolor": "limegreen", "project_z": True}}
        )]
    )
    fig3.update_layout(
        title="3D Surface — Pre vs Post Joint Distribution",
        scene=dict(
            xaxis_title="Pre Score",
            yaxis_title="Post Score",
            zaxis_title="Density"
        ),
        template="plotly_white",
        height=500
    )
    st.plotly_chart(fig3, use_container_width=True)

# Combined Facet View: Phase Comparison
st.subheader("Facet View: Phase Comparison")
fig4 = px.histogram(df_long, x="score", facet_col="phase", color="phase",
                    nbins=25, opacity=0.7,
                    color_discrete_sequence=["#4c78a8", "#f58518"],
                    title="Facet Comparison: Pre vs Post")
fig4.update_layout(template="plotly_white", showlegend=False)
st.plotly_chart(fig4, use_container_width=True)
# ---------- Summaries ----------
st.subheader("Summaries & Breakdowns")

def bar_from_summary(df, x, y, title, ylabel):
    if df.empty:
        return
    bar = px.bar(df, x=x, y=y, title=title)
    st.plotly_chart(bar, use_container_width=True)

# Group summary
if "group" in merged.columns:
    grp = (merged.groupby("group")["learning_gain"]
           .agg(["mean","count","std"])
           .reset_index()
           .rename(columns={"mean":"mean_gain"}))
    st.write("By Group:", grp)
    bar_from_summary(grp, "group", "mean_gain", "Mean Learning Gain by Group", "Mean Gain")

# Course summary (if attached)
if "course_id" in merged.columns:
    crs = (merged.groupby("course_id")["learning_gain"]
           .agg(["mean","count","std"]).reset_index()
           .rename(columns={"mean":"mean_gain"}))
    st.write("By Course:", crs)
    bar_from_summary(crs, "course_id", "mean_gain", "Mean Learning Gain by Course", "Mean Gain")

# Topic summary (if attached)
if "topic" in merged.columns:
    tpc = (merged.groupby("topic")["learning_gain"]
           .agg(["mean","count","std"]).reset_index()
           .rename(columns={"mean":"mean_gain"}))
    st.write("By Topic:", tpc)
    # show top 12 by count for readability
    topN = tpc.sort_values("count", ascending=False).head(12)
    bar_from_summary(topN, "topic", "mean_gain", "Mean Learning Gain by Topic (Top 12)", "Mean Gain")

# ---------- Exports ----------
st.subheader("Exports")
def dl(df, name, label):
    st.download_button(label=label,
                       data=df.to_csv(index=False).encode("utf-8"),
                       file_name=name, mime="text/csv")

dl(merged, "learning_gains.csv", "Download Learning Gains CSV")
if "group" in merged.columns:
    dl(grp, "summary_by_group.csv", "Download Group Summary CSV")
if "course_id" in merged.columns:
    dl(crs, "summary_by_course.csv", "Download Course Summary CSV")
if "topic" in merged.columns:
    dl(tpc, "summary_by_topic.csv", "Download Topic Summary CSV")

# ---------- Save for other pages ----------
st.session_state["learning_gains"] = merged
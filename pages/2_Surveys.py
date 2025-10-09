import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.title("2) Surveys — MSLQ / Self-Efficacy / NASA-TLX (Advanced)")

# --------------------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------------------
if "surveys_df" not in st.session_state:
    st.info("Please upload surveys on the **0. Upload** page.")
    st.stop()

df = st.session_state["surveys_df"].copy()
st.write("Preview:", df.head())

# Normalize
df.columns = [c.strip().lower() for c in df.columns]
if not {"instrument", "response"}.issubset(df.columns):
    st.warning("Columns required: instrument, response (+ optional week, item_code, student_id, group, subscale)")
    st.stop()

# Coerce response numeric & clip to [1, 7] if Likert
df["response"] = pd.to_numeric(df["response"], errors="coerce")
if df["response"].max() <= 9:  # assume Likert-like
    df["response"] = df["response"].clip(1, 7)

# Optional columns
has_week  = "week" in df.columns
has_id    = "student_id" in df.columns
has_group = "group" in df.columns
has_item  = "item_code" in df.columns
has_sub   = "subscale" in df.columns

# --------------------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------------------
with st.expander("Filters"):
    colf1, colf2, colf3 = st.columns(3)
    instruments = sorted(df["instrument"].dropna().unique())
    sel_inst = colf1.multiselect("Instrument", instruments, default=instruments)
    sel_group = colf2.multiselect("Group", sorted(df["group"].dropna().unique()) if has_group else [], 
                                  default=(sorted(df["group"].dropna().unique()) if has_group else []))
    if has_week:
        wmin, wmax = int(df["week"].min()), int(df["week"].max())
        sel_weeks = colf3.slider("Weeks", wmin, wmax, (wmin, wmax), step=1)
    else:
        sel_weeks = None

    rmin, rmax = st.slider("Response range", float(df["response"].min()), float(df["response"].max()),
                           (float(df["response"].min()), float(df["response"].max())), step=0.5)

mask = df["instrument"].isin(sel_inst)
if has_group and len(sel_group):
    mask &= df["group"].isin(sel_group)
if has_week and sel_weeks:
    mask &= df["week"].between(sel_weeks[0], sel_weeks[1])
mask &= df["response"].between(rmin, rmax)
df = df[mask].copy()

# --------------------------------------------------------------------------------------
# Academic utilities
# --------------------------------------------------------------------------------------
def normalize_0_1(x, min_v=1.0, max_v=7.0):
    x = pd.to_numeric(x, errors="coerce")
    return (x - min_v) / (max_v - min_v)

def cronbach_alpha(items_wide: pd.DataFrame):
    """items_wide: columns are items of the same scale (numeric)."""
    X = items_wide.dropna(axis=0, how="any")
    k = X.shape[1]
    if k < 2 or X.empty: 
        return np.nan
    variances = X.var(axis=0, ddof=1)
    total_var = X.sum(axis=1).var(ddof=1)
    if total_var <= 0:
        return np.nan
    return float(k / (k - 1) * (1 - variances.sum() / total_var))

# NASA-TLX: weighted workload (Hart & Staveland). If no weights provided, use unweighted mean.
# Expected subscales (common): MD, PD, TD, P (performance, reverse), E (effort), FR (frustration)
# If df has 'subscale' column, we’ll use that. Otherwise try to infer from item_code.
SUB_MAP_DEFAULT = {
    "mental": ["md","mental"],
    "physical": ["pd","physical"],
    "temporal": ["td","temporal"],
    "performance": ["p","perf","performance"],
    "effort": ["e","effort"],
    "frustration": ["fr","frus","frustration"]
}
def infer_subscale_from_item(item: str):
    s = str(item).lower()
    for key, keys in SUB_MAP_DEFAULT.items():
        if any(k in s for k in keys):
            return key
    return None

def nasatlx_score(rows: pd.DataFrame, weights: dict|None=None):
    """
    rows: columns -> ['subscale','response'], 7-point typical.
    If weights given (e.g., {'mental':3, 'physical':1, ...} sums to 15),
    use weighted average; else unweighted mean of subscales.
    Performance is reverse-scored in some variants; toggle as needed.
    """
    sub_means = rows.groupby("subscale")["response"].mean()
    if "performance" in sub_means.index:
        # Reverse performance: high performance = low workload
        # Map 1..7 to 7..1
        sub_means.loc["performance"] = 8 - sub_means.loc["performance"]
    if weights:
        wsum = sum(weights.get(k, 0) for k in sub_means.index)
        if wsum > 0:
            return float(sum(sub_means.get(k, np.nan) * weights.get(k, 0) for k in sub_means.index) / wsum)
    return float(sub_means.mean())

# --------------------------------------------------------------------------------------
# Subscale mapping for MSLQ & Self-Efficacy (optional)
# You can upload a mapping JSON: {"MSLQ":{"INTR":["Q1","Q5"],"SRL":["Q2","Q6"], ...}, "SelfEfficacy":{"SEF":["Q1","Q3",...]}}
# --------------------------------------------------------------------------------------
with st.expander("Optional: Upload Subscale Mapping (JSON)"):
    umap = st.file_uploader("JSON mapping (instrument → subscale → list[item_code])", type=["json"], key="submap")
    user_map = None
    if umap is not None:
        try:
            user_map = json.load(umap)
            st.success("Subscale mapping loaded.")
        except Exception as e:
            st.error(f"Failed to parse JSON: {e}")

# --------------------------------------------------------------------------------------
# Instrument-level summaries (+ reliability where applicable)
# --------------------------------------------------------------------------------------
st.subheader("A. Instrument summaries")

summary = df.groupby("instrument")["response"].agg(["mean","std","count"]).reset_index()
summary["mean_norm_0_1"] = normalize_0_1(summary["mean"])
st.plotly_chart(px.bar(summary, x="instrument", y="mean", title="Mean Response by Instrument"), use_container_width=True)

# Plotly datatable
table_fig = go.Figure(data=[go.Table(
    header=dict(values=list(summary.columns), fill_color="#2c3e50", font_color="white", align="left"),
    cells=dict(values=[summary[c] for c in summary.columns], align="left")
)])
table_fig.update_layout(title="Instrument Summary Table")
st.plotly_chart(table_fig, use_container_width=True)

# ================== Reliability (Cronbach's α) with diagnostics ==================
st.subheader("B. Reliability (Cronbach’s α) — with diagnostics")

def cronbach_alpha(items_wide: pd.DataFrame):
    X = items_wide.dropna(axis=0, how="any")
    k = X.shape[1]
    if k < 2 or X.empty:
        return np.nan
    variances = X.var(axis=0, ddof=1)
    total_var = X.sum(axis=1).var(ddof=1)
    if total_var <= 0:
        return np.nan
    return float(k/(k-1) * (1 - variances.sum()/total_var))

if "item_code" in df.columns and "student_id" in df.columns:
    for inst in sorted(df["instrument"].unique()):
        sub_df = df[df["instrument"] == inst].copy()
        n_items = sub_df["item_code"].nunique()
        n_students = sub_df["student_id"].nunique()
        st.write(f"**{inst}** — items: {n_items}, students: {n_students}, responses: {len(sub_df)}")

        if n_items < 2:
            st.info(f"{inst} — α not computable: need ≥ 2 distinct item_code.")
            continue

        wide = sub_df.pivot_table(index="student_id", columns="item_code",
                                  values="response", aggfunc="mean")
        # require ≥ 2 columns after pivot and ≥ 2 students with non-NA
        if wide.shape[1] < 2 or wide.dropna(how='any').shape[0] < 2:
            st.info(f"{inst} — α not computable: too few items or too few complete rows after pivot.")
            continue

        alpha = cronbach_alpha(wide)
        if np.isnan(alpha):
            # Try to detect reasons
            col_var = wide.var(axis=0, ddof=1).sum()
            row_var = wide.sum(axis=1).var(ddof=1)
            st.info(f"{inst} — α not computable: variance too low (items_var_sum={col_var:.4f}, total_score_var={row_var:.4f}).")
        else:
            st.success(f"**{inst}** — Cronbach’s α = {alpha:.3f}")
else:
    st.info("Reliability skipped: need 'item_code' and 'student_id' columns.")

# ================== NASA-TLX Workload (robust) ==================
st.subheader("C. NASA-TLX Workload (robust)")

def infer_subscale_from_item(item: str):
    s = str(item).lower()
    if any(k in s for k in ["md","mental"]): return "mental"
    if any(k in s for k in ["pd","physical"]): return "physical"
    if any(k in s for k in ["td","temporal","time"]): return "temporal"
    if any(k in s for k in ["perf","performance","p "]): return "performance"
    if any(k in s for k in ["effort","e "]): return "effort"
    if any(k in s for k in ["fr","frus","frustration"]): return "frustration"
    return None

def nasatlx_score(rows: pd.DataFrame, weights=None):
    if rows.empty or "subscale" not in rows.columns:
        return np.nan
    sub_means = rows.groupby("subscale")["response"].mean()
    if "performance" in sub_means.index:
        sub_means.loc["performance"] = 8 - sub_means.loc["performance"]  # reverse P
    if weights and sum(weights.values()) > 0:
        use = {k: weights.get(k, 0) for k in sub_means.index}
        den = sum(use.values())
        if den > 0:
            return float(sum(sub_means[k]*use[k] for k in sub_means.index)/den)
    return float(sub_means.mean())

has_tlx = any(df["instrument"].str.lower() == "nasa_tlx")
if not has_tlx:
    st.info("No NASA-TLX rows found in 'instrument'.")
else:
    tlx = df[df["instrument"].str.lower() == "nasa_tlx"].copy()
    # get/derive subscales
    if "subscale" not in tlx.columns:
        if "item_code" in tlx.columns:
            tlx["subscale"] = tlx["item_code"].apply(infer_subscale_from_item)
        else:
            tlx["subscale"] = np.nan
    tlx = tlx[tlx["subscale"].notna()].copy()

    if tlx.empty:
        st.warning("NASA-TLX present but no subscales identified. "
                   "Add a 'subscale' column or encode item_code as MD/PD/TD/P/E/FR.")
    else:
        # weights (optional)
        wcols = st.columns(6)
        weights = dict(
            mental=wcols[0].number_input("Mental", 0, 15, 0, key="tlx_mental"),
            physical=wcols[1].number_input("Physical", 0, 15, 0, key="tlx_physical"),
            temporal=wcols[2].number_input("Temporal", 0, 15, 0, key="tlx_temporal"),
            performance=wcols[3].number_input("Performance", 0, 15, 0, key="tlx_performance"),
            effort=wcols[4].number_input("Effort", 0, 15, 0, key="tlx_effort"),
            frustration=wcols[5].number_input("Frustration", 0, 15, 0, key="tlx_frustration"),
        )
        if sum(weights.values()) == 0:
            weights = None

        # compute per-student
        wdf = []
        if "student_id" in tlx.columns:
            for sid, g in tlx.groupby("student_id"):
                score = nasatlx_score(g[["subscale","response"]], weights=weights)
                if not np.isnan(score):
                    wdf.append({"student_id": sid, "nasatlx": score})
            wdf = pd.DataFrame(wdf)

        if not wdf.empty:
            st.dataframe(wdf.head(10), use_container_width=True)
            st.plotly_chart(
                px.histogram(wdf, x="nasatlx", nbins=25, title="NASA-TLX Workload Distribution"),
                use_container_width=True
            )
            # radar: mean subscale
            sub_means = tlx.groupby("subscale")["response"].mean().reindex(
                ["mental","physical","temporal","performance","effort","frustration"]
            )
            sub_means = sub_means.dropna()
            if not sub_means.empty:
                radar = go.Figure()
                radar.add_trace(go.Scatterpolar(r=sub_means.values, theta=sub_means.index, fill="toself"))
                radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[1,7])),
                                    template="plotly_white", title="NASA-TLX Subscale Means")
                st.plotly_chart(radar, use_container_width=True)
        else:
            st.info("No per-student NASA-TLX scores computed (check 'student_id' and subscales).")
# --------------------------------------------------------------------------------------
# C. NASA-TLX workload (robust & guarded)
# --------------------------------------------------------------------------------------
st.subheader("C. NASA-TLX Workload")

def _safe_hist(df, x, **kwargs):
    """Plotly histogram with safety checks."""
    if isinstance(df, pd.DataFrame) and (x in df.columns) and not df.empty:
        st.plotly_chart(px.histogram(df, x=x, **kwargs), use_container_width=True)
    else:
        st.info("No NASA-TLX scores to plot yet.")

def infer_subscale_from_item(item: str):
    s = str(item).lower()
    # heuristic mapping; tweak as needed
    if any(k in s for k in ["md", "mental"]): return "mental"
    if any(k in s for k in ["pd", "physical"]): return "physical"
    if any(k in s for k in ["td", "temporal", "time"]): return "temporal"
    if any(k in s for k in ["perf", "performance"]): return "performance"
    if any(k in s for k in ["effort", "eff"]): return "effort"
    if any(k in s for k in ["fr", "frus", "frustration"]): return "frustration"
    return None

def nasatlx_score(rows: pd.DataFrame, weights: dict | None = None):
    """
    rows must contain: ['subscale','response'].
    Performance is reverse-scored (7->1). If weights is None, use unweighted mean.
    """
    if rows.empty or "subscale" not in rows.columns:
        return np.nan
    sub_means = rows.groupby("subscale")["response"].mean()

    # Reverse 'performance' so higher performance -> lower workload
    if "performance" in sub_means.index:
        sub_means.loc["performance"] = 8 - sub_means.loc["performance"]  # 1..7 -> 7..1

    if weights:
        # only use weights for available subscales
        w = {k: weights.get(k, 0) for k in sub_means.index}
        wsum = sum(w.values())
        if wsum > 0:
            return float(sum(sub_means[k] * w[k] for k in sub_means.index) / wsum)

    return float(sub_means.mean())

# Proceed only if NASA-TLX rows exist
has_tlx = any(df["instrument"].str.lower() == "nasa_tlx")
if not has_tlx:
    st.info("No NASA-TLX rows detected in 'instrument'. (Your current columns: student_id, instrument, week, item_code, response, reflection_text, ai_response)")
else:
    tlx = df[df["instrument"].str.lower() == "nasa_tlx"].copy()

    # Ensure we have subscales: use provided 'subscale' or infer from 'item_code'
    if "subscale" not in tlx.columns:
        if "item_code" in tlx.columns:
            tlx["subscale"] = tlx["item_code"].apply(infer_subscale_from_item)
        else:
            tlx["subscale"] = np.nan

    # Keep only rows where we could assign a subscale
    tlx = tlx[tlx["subscale"].notna()].copy()

    # If still empty, bail out gracefully
    if tlx.empty:
        st.info("NASA-TLX present but no subscales could be identified. "
                "Add a 'subscale' column or make item codes identifiable (e.g., MD, PD, TD, P, E, FR).")
    else:
        # Optional weights input
        st.markdown("**Weights (optional)** — leave as zeros to use unweighted mean of subscales.")
        wcols = st.columns(6)
        weights = dict(
            mental=wcols[0].number_input("Mental", 0, 15, 0),
            physical=wcols[1].number_input("Physical", 0, 15, 0),
            temporal=wcols[2].number_input("Temporal", 0, 15, 0),
            performance=wcols[3].number_input("Performance", 0, 15, 0),
            effort=wcols[4].number_input("Effort", 0, 15, 0),
            frustration=wcols[5].number_input("Frustration", 0, 15, 0),
        )
        if sum(weights.values()) == 0:
            weights = None  # unweighted average

        # Compute per-student NASA-TLX
        wdf = []
        if "student_id" in tlx.columns:
            for sid, g in tlx.groupby("student_id"):
                score = nasatlx_score(g[["subscale","response"]], weights=weights)
                if not np.isnan(score):
                    wdf.append({"student_id": sid, "nasatlx": score})
            wdf = pd.DataFrame(wdf)

        if isinstance(wdf, pd.DataFrame) and not wdf.empty:
            st.write("NASA-TLX per student (first 10):")
            st.dataframe(wdf.head(10), use_container_width=True)

            # Histogram (guarded)
            _safe_hist(wdf, "nasatlx", nbins=25, title="NASA-TLX Workload Distribution")

            # Radar of subscales (mean)
            sub_means = (tlx.groupby("subscale")["response"].mean()
                           .reindex(["mental","physical","temporal","performance","effort","frustration"]))
            sub_means = sub_means.dropna()
            if not sub_means.empty:
                radar = go.Figure()
                radar.add_trace(go.Scatterpolar(
                    r=sub_means.values, theta=sub_means.index, fill="toself", name="Mean"
                ))
                radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[1,7])),
                    template="plotly_white", title="NASA-TLX Subscale Means"
                )
                st.plotly_chart(radar, use_container_width=True)
        else:
            st.info("No per-student NASA-TLX scores were computed (check 'student_id' presence and subscale mapping).")
# --------------------------------------------------------------------------------------
# D. MSLQ / Self-Efficacy: Subscale scoring
# --------------------------------------------------------------------------------------
st.subheader("D. Subscale scoring for MSLQ / Self-Efficacy")

def compute_subscales(_df, instrument_name, mapping: dict):
    """mapping: {subscale_name: [item_codes]}"""
    dd = _df[_df["instrument"].str.lower()==instrument_name.lower()].copy()
    if dd.empty or not has_item:
        return pd.DataFrame()
    # pivot to wide per student over items listed in mapping
    wide = dd.pivot_table(index="student_id", columns="item_code", values="response", aggfunc="mean")
    out = {"student_id": wide.index}
    for sub, items in mapping.items():
        cols = [c for c in items if c in wide.columns]
        if len(cols) == 0: 
            continue
        out[sub] = wide[cols].mean(axis=1)
    return pd.DataFrame(out)

# Default tiny example mapping (replace with your JSON for research use)
default_mslq_map = {
    "IntrinsicMot": ["Q1","Q5","Q9"],
    "SelfRegulation": ["Q2","Q6","Q10"],
    "TaskValue": ["Q3","Q7","Q11"],
    "TestAnxiety": ["Q4","Q8","Q12"],
}
default_sef_map = {"SelfEfficacy":["Q1","Q3","Q5","Q7"]}

mapping = user_map if user_map else {"MSLQ": default_mslq_map, "SelfEfficacy": default_sef_map}

for inst_name, mp in mapping.items():
    if inst_name.lower() in df["instrument"].str.lower().unique():
        res = compute_subscales(df, inst_name, mp)
        if not res.empty:
            st.write(f"**{inst_name} Subscales (per student)**")
            st.dataframe(res.head(10), use_container_width=True)

            # Reliability per subscale if >=2 items
            if has_item and has_id:
                inst_df = df[df["instrument"].str.lower()==inst_name.lower()]
                wide_all = inst_df.pivot_table(index="student_id", columns="item_code", values="response", aggfunc="mean")
                st.write(f"Reliability (α) by subscale for **{inst_name}**")
                alphas = []
                for sub, items in mp.items():
                    cols = [c for c in items if c in wide_all.columns]
                    if len(cols) >= 2:
                        alpha = cronbach_alpha(wide_all[cols])
                        alphas.append((sub, alpha))
                if alphas:
                    a_df = pd.DataFrame(alphas, columns=["subscale","alpha"])
                    st.plotly_chart(px.bar(a_df, x="subscale", y="alpha", title=f"{inst_name} — Cronbach’s α by Subscale", range_y=[0,1]),
                                    use_container_width=True)

            # Distribution and box/violin
            melted = res.melt(id_vars=["student_id"], var_name="subscale", value_name="score")
            st.plotly_chart(px.violin(melted, x="subscale", y="score", box=True, points="all",
                                      title=f"{inst_name} Subscale Distributions"),
                            use_container_width=True)

            # Group comparisons if available
            if has_group:
                # attach group to each student
                sid_group = df[["student_id","group"]].drop_duplicates()
                mg = melted.merge(sid_group, on="student_id", how="left")
                st.plotly_chart(px.box(mg, x="group", y="score", color="subscale",
                                       title=f"{inst_name} Subscales by Group"),
                                use_container_width=True)

# --------------------------------------------------------------------------------------
# E. Weekly trends (Plotly)
# --------------------------------------------------------------------------------------
st.subheader("E. Weekly Trends (mean response)")
if has_week:
    trend = df.groupby(["instrument","week"])["response"].mean().reset_index()
    fig_tr = px.line(trend, x="week", y="response", color="instrument",
                     markers=True, title="Weekly Trend by Instrument")
    fig_tr.update_layout(template="plotly_white")
    st.plotly_chart(fig_tr, use_container_width=True)

# --------------------------------------------------------------------------------------
# F. Exports
# --------------------------------------------------------------------------------------
st.subheader("Exports")
def dl(_df, name, label):
    st.download_button(label=label,
                       data=_df.to_csv(index=False).encode("utf-8"),
                       file_name=name, mime="text/csv")

dl(summary, "survey_instrument_summary.csv", "Download Instrument Summary CSV")
if has_week:
    dl(trend, "survey_weekly_trend.csv", "Download Weekly Trend CSV")
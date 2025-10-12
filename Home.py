# Home.py — EDU-AI Evaluation Platform — Home (Overview)
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="EDU-AI Evaluation Platform — Home", layout="wide")
st.title("EDU-AI Evaluation Platform — Home")
st.caption("This page summarizes final outputs after you upload on the Upload page and review each module page.")

def has(key: str) -> bool:
    return key in st.session_state and st.session_state[key] is not None

# ---------- helpers ----------
def kpi(label, val, precision=3, help_text=None):
    if val is None or (isinstance(val, float) and (not np.isfinite(val))):
        st.metric(label, "—", help=help_text)
    else:
        st.metric(label, f"{val:.{precision}f}", help=help_text)

def _safe_rate(num, den):
    return float(num/den) if den and den > 0 else np.nan

def fairness_quick(df: pd.DataFrame):
    """Return quick fairness tables if possible; otherwise graceful fallbacks."""
    out = {}
    if df is None or df.empty: return out
    d = df.copy()
    d.columns = [c.lower() for c in d.columns]
    has_group = "group" in d.columns
    if not has_group:
        return out

    # Basic accuracy-by-group (if provided)
    if "ai_accuracy" in d.columns:
        acc = d.groupby("group")["ai_accuracy"].mean().reset_index()
        out["acc"] = acc
        out["acc_gap"] = float(acc["ai_accuracy"].max() - acc["ai_accuracy"].min())
    else:
        out["acc"] = None
        out["acc_gap"] = None

    # Parity metrics if we have y_true + (y_pred or y_score)
    yhat = None
    if "y_pred" in d.columns: yhat = "y_pred"
    elif "y_score" in d.columns:
        d["_y_pred_thr"] = (pd.to_numeric(d["y_score"], errors="coerce") >= 0.5).astype(int)
        yhat = "_y_pred_thr"

    if "y_true" in d.columns and yhat is not None:
        rows = []
        for g, sub in d.dropna(subset=["group"]).groupby("group"):
            y  = pd.to_numeric(sub["y_true"], errors="coerce").astype(int)
            yp = pd.to_numeric(sub[yhat],   errors="coerce").astype(int)
            TP = int(((yp==1)&(y==1)).sum())
            TN = int(((yp==0)&(y==0)).sum())
            FP = int(((yp==1)&(y==0)).sum())
            FN = int(((yp==0)&(y==1)).sum())
            n  = int(len(sub)); P, Nn = TP+FN, TN+FP
            SR  = _safe_rate(TP+FP, n)
            ACC = _safe_rate(TP+TN, n)
            TPR = _safe_rate(TP, P)
            FPR = _safe_rate(FP, Nn)
            rows.append(dict(group=g, N=n, SR=SR, ACC=ACC, TPR=TPR, FPR=FPR))
        gm = pd.DataFrame(rows).sort_values("N", ascending=False)
        if not gm.empty:
            ref = gm.iloc[0]
            gm["SPD"] = ref["SR"] - gm["SR"]
            gm["DI"]  = gm["SR"] / ref["SR"] if ref["SR"] and np.isfinite(ref["SR"]) else np.nan
            gm["EOG"] = (gm["TPR"] - ref["TPR"]).abs()
            gm["EOD"] = ((gm["TPR"] - ref["TPR"]).abs() + (gm["FPR"] - ref["FPR"]).abs()) / 2
            out["gm"] = gm
            out["ref_group"] = ref["group"]
    return out

# ---------- KPI row ----------
k1, k2, k3, k4 = st.columns(4)

with k1:
    if has("learning_gains"):
        lg = st.session_state["learning_gains"]
        kpi("Students with Gain", float(len(lg)), precision=0)
    else:
        kpi("Students with Gain", None)

with k2:
    if has("learning_gains"):
        kpi("Mean Learning Gain", st.session_state["learning_gains"]["learning_gain"].mean(), precision=2)
    else:
        kpi("Mean Learning Gain", None)

with k3:
    if has("telemetry_with_pei") and "prompt_evolution_index" in st.session_state["telemetry_with_pei"].columns:
        kpi("Mean PEI", st.session_state["telemetry_with_pei"]["prompt_evolution_index"].astype(float).mean(), precision=3)
    else:
        kpi("Mean PEI", None)

with k4:
    if has("fairness_df"):
        fq = fairness_quick(st.session_state["fairness_df"])
        kpi("Accuracy Gap (max−min)", fq.get("acc_gap"), precision=3)
    else:
        kpi("Accuracy Gap (max−min)", None)

st.divider()

# ---------- A) Learning Gains ----------
st.subheader("A) Learning Gains")
st.markdown(r"""
**Formula (per student):**  
$$
\Delta = \text{Post} - \text{Pre}
$$
**Interpretation:** Positive values indicate improvement from pre to post.
""")
if has("learning_gains"):
    lg = st.session_state["learning_gains"].copy()
    lg.columns = [c.lower() for c in lg.columns]
    st.dataframe(lg.head(25), use_container_width=True)

    if "learning_gain" in lg.columns:
        fig = px.histogram(lg, x="learning_gain", nbins=30, marginal="box",
                           title="Distribution of Learning Gains (Δ = Post − Pre)")
        fig.update_layout(template="plotly_white", xaxis_title="Learning Gain (Δ)")
        st.plotly_chart(fig, use_container_width=True)

    # Optional group/course/topic bars
    for dim in ["group", "course_id", "topic"]:
        if dim in lg.columns:
            agg = (lg.groupby(dim)["learning_gain"]
                     .agg(mean="mean", std="std", count="count").reset_index())
            st.write(f"Summary by **{dim}**")
            st.dataframe(agg, use_container_width=True)
            figb = px.bar(agg, x=dim, y="mean", error_y="std",
                          title=f"Mean Learning Gain by {dim}")
            figb.update_layout(template="plotly_white", xaxis_tickangle=30, yaxis_title="Mean Δ")
            st.plotly_chart(figb, use_container_width=True)

    st.download_button("Download Learning Gains CSV",
                       data=lg.to_csv(index=False).encode("utf-8"),
                       file_name="learning_gains.csv", mime="text/csv")
else:
    st.info("Go to **0. Upload** to upload assessments and compute learning gains on the Assessments page.")

# ---------- B) PEI & RDS ----------
st.subheader("B) Prompt Evolution (PEI) & Reflection Depth (RDS proxy)")
st.markdown(r"""
**PEI (Prompt Evolution Index) — conceptual formula:**  
$$
\text{PEI} = 0.3\cdot\text{LexSpec} + 0.35\cdot\min\!\left(\frac{\#\text{strategy verbs}}{6},1\right)
+ 0.35\cdot\min\!\left(\frac{\#\text{constraint cues}}{6},1\right)
$$

**RDS proxy (0–4):** counts discourse/justification cues and long-text bonus.  
""")
if has("telemetry_with_pei"):
    tele = st.session_state["telemetry_with_pei"].copy()
    tele.columns = [c.lower() for c in tele.columns]
    cols_show = [c for c in ["timestamp","session_id","prompt","ai_response","prompt_evolution_index","rds_proxy","group"] if c in tele.columns]
    if cols_show:
        st.dataframe(tele[cols_show].head(25), use_container_width=True)
    else:
        st.dataframe(tele.head(25), use_container_width=True)

    if "prompt_evolution_index" in tele.columns:
        figp = px.histogram(tele, x="prompt_evolution_index", nbins=30, marginal="box",
                            title="PEI Distribution")
        figp.update_layout(template="plotly_white", xaxis_title="PEI")
        st.plotly_chart(figp, use_container_width=True)

    if "rds_proxy" in tele.columns:
        figr = px.histogram(tele, x="rds_proxy", nbins=5,
                            category_orders={"rds_proxy":[0,1,2,3,4]},
                            title="RDS Proxy Distribution")
        figr.update_layout(template="plotly_white", xaxis_title="RDS Proxy")
        st.plotly_chart(figr, use_container_width=True)

    st.download_button("Download Telemetry (with PEI/RDS)",
                       data=tele.to_csv(index=False).encode("utf-8"),
                       file_name="telemetry_with_pei.csv", mime="text/csv")
else:
    st.info("Upload Telemetry to compute PEI/RDS (see Upload page).")

# ---------- C) Surveys ----------
st.subheader("C) Surveys Snapshot")
st.markdown(r"""
We summarize instruments (e.g., **MSLQ**, **Self-Efficacy**, **NASA-TLX**).  
**Display:** mean ± SD and sample size per instrument.
""")
if has("surveys_df"):
    sv = st.session_state["surveys_df"].copy()
    sv.columns = [c.lower() for c in sv.columns]
    if {"instrument","response"}.issubset(sv.columns):
        agg = sv.groupby("instrument")["response"].agg(["mean","std","count"]).reset_index()
        st.dataframe(agg, use_container_width=True)
        fig = px.bar(agg, x="instrument", y="mean", error_y="std",
                     title="Mean Response by Instrument")
        fig.update_layout(template="plotly_white", yaxis_title="Mean Response")
        st.plotly_chart(fig, use_container_width=True)

        st.download_button("Download Surveys CSV",
                           data=sv.to_csv(index=False).encode("utf-8"),
                           file_name="surveys.csv", mime="text/csv")
    else:
        st.warning("Surveys missing required columns: instrument, response.")
else:
    st.info("Upload surveys (MSLQ / Self-Efficacy / NASA-TLX).")

# ---------- D) Fairness ----------
st.subheader("D) Fairness / Bias")
st.markdown(r"""
**Key formulas:**  
- **Selection Rate** \(SR_g = P(\hat{Y}=1 \mid A=g)\)  
- **Statistical Parity Difference** \(SPD = SR_{ref} - SR_g\)  
- **Disparate Impact** \(DI = SR_g / SR_{ref}\) (80% rule if \(DI \ge 0.8\))  
- **Equal Opportunity Gap** \(EOG = |TPR_g - TPR_{ref}|\)  
- **Equalized Odds Gap** \(EOD = \frac{|TPR_g - TPR_{ref}| + |FPR_g - FPR_{ref}|}{2}\)  
Reference group: **largest-N** group for stability.
""")
if has("fairness_df"):
    fair = st.session_state["fairness_df"].copy()
    st.dataframe(fair.head(25), use_container_width=True)

    fq = fairness_quick(fair)
    # Accuracy by group (if given)
    if fq.get("acc") is not None:
        acc = fq["acc"]
        fig = px.bar(acc, x="group", y="ai_accuracy", title="Mean AI Accuracy by Group")
        fig.update_layout(template="plotly_white", yaxis_title="AI Accuracy")
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Accuracy Gap (max−min)", f"{fq['acc_gap']:.3f}" if fq["acc_gap"] is not None else "—")

    # Parity metrics if computable
    if fq.get("gm") is not None:
        st.caption(f"Reference group (largest N): **{fq['ref_group']}**")
        gm = fq["gm"]
        # Plotly Table
        table = go.Figure(data=[go.Table(
            header=dict(values=list(gm.columns), fill_color="#2c3e50", font=dict(color="white")),
            cells=dict(values=[gm[c] for c in gm.columns], align="left")
        )])
        table.update_layout(title="Per-Group Fairness Metrics", template="plotly_white")
        st.plotly_chart(table, use_container_width=True)

        for metric, title in [("SPD","Statistical Parity Difference"),
                              ("DI","Disparate Impact"),
                              ("EOD","Equalized Odds Gap")]:
            if metric in gm.columns:
                figm = px.bar(gm, x="group", y=metric, title=title)
                figm.update_layout(template="plotly_white")
                st.plotly_chart(figm, use_container_width=True)

        st.download_button("Download Fairness Metrics CSV",
                           data=gm.to_csv(index=False).encode("utf-8"),
                           file_name="fairness_metrics.csv", mime="text/csv")
    elif fq.get("acc") is None:
        st.info("Provide (`group` + `y_true` + (`y_pred` or `y_score`)) for parity metrics, or (`group` + `ai_accuracy`) for a basic summary.")
else:
    st.info("Upload fairness data to see group accuracy and parity gaps.")

# ---------- E) Teacher Orchestration ----------
st.subheader("E) Teacher Orchestration")
st.markdown("We summarize teacher workload/cognitive load if available.")
if has("teacher_df"):
    teach = st.session_state["teacher_df"].copy()
    teach.columns = [c.lower() for c in teach.columns]
    cols = [c for c in ["workload_hours","perceived_cognitive_load","class_size","ai_usage_pct"] if c in teach.columns]
    if cols:
        st.dataframe(teach[cols].describe(), use_container_width=True)
        for c in cols:
            fig = px.histogram(teach, x=c, nbins=30, marginal="box", title=f"Distribution: {c}")
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        st.download_button("Download Teacher Logs CSV",
                           data=teach.to_csv(index=False).encode("utf-8"),
                           file_name="teacher_logs.csv", mime="text/csv")
    else:
        st.info("Teacher logs present but missing typical columns (e.g., workload_hours, perceived_cognitive_load).")
else:
    st.info("Upload Teacher Logs to view orchestration summaries.")

st.divider()
st.caption("Tip: See **7) Summary Dashboard** to compare groups across Learning Gain, PEI, and Fairness with advanced charts.")

# =========================
# Supplementary Academic Analyses (add-on block for Home.py)
# =========================


st.markdown("---")
st.header("Supplementary Academic Analyses")

def _is_num(s):
    return pd.api.types.is_numeric_dtype(s)

# ---------- 1) Learning Gains: ECDF, QQ-style check, and correlation matrix ----------
st.subheader("1) Learning Gains — ECDF · QQ-style check · Correlations")
if "learning_gains" in st.session_state and st.session_state["learning_gains"] is not None:
    lgx = st.session_state["learning_gains"].copy()
    lgx.columns = [c.lower() for c in lgx.columns]

    if "learning_gain" in lgx.columns:
        # ECDF
        fig_ecdf = px.ecdf(lgx.dropna(subset=["learning_gain"]), x="learning_gain",
                           title="ECDF of Learning Gain (Δ)")
        fig_ecdf.update_layout(template="plotly_white", xaxis_title="Learning Gain (Δ)")
        st.plotly_chart(fig_ecdf, use_container_width=True)

        # QQ-style plot against Normal(μ,σ)
        x = pd.to_numeric(lgx["learning_gain"], errors="coerce").dropna().values
        if x.size >= 10:
            q_emp = np.quantile(x, np.linspace(0.01, 0.99, 99))
            mu, sd = np.mean(x), np.std(x, ddof=1) if np.std(x, ddof=1) > 0 else 1.0
            q_the = np.quantile(np.random.default_rng(0).normal(mu, sd, 100000), np.linspace(0.01, 0.99, 99))
            qq = pd.DataFrame(dict(theoretical=q_the, empirical=q_emp))
            figqq = px.scatter(qq, x="theoretical", y="empirical", title="QQ-style Plot vs Normal(μ,σ)")
            figqq.add_shape(type="line", x0=qq["theoretical"].min(), y0=qq["theoretical"].min(),
                            x1=qq["theoretical"].max(), y1=qq["theoretical"].max(),
                            line=dict(dash="dash"))
            figqq.update_layout(template="plotly_white", xaxis_title="Theoretical Quantiles", yaxis_title="Empirical Quantiles")
            st.plotly_chart(figqq, use_container_width=True)

        # Correlation matrix for numeric columns (pre/post/gain/etc.)
        numcols = [c for c in ["pre","post","learning_gain"] if c in lgx.columns]
        if len(numcols) >= 2:
            corr = lgx[numcols].corr()
            heat = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale="RdBu", zmid=0))
            heat.update_layout(title="Correlation Matrix (Assessments)", template="plotly_white")
            st.plotly_chart(heat, use_container_width=True)

        # Group violin (if group exists)
        if "group" in lgx.columns:
            figv = px.violin(lgx, x="group", y="learning_gain", box=True, points="all",
                             title="Learning Gain by Group (Violin + Box)")
            figv.update_layout(template="plotly_white", xaxis_tickangle=30)
            st.plotly_chart(figv, use_container_width=True)
else:
    st.caption("Upload assessments to enable these learning gain analyses.")

# ---------- 2) Telemetry: PEI×RDS joint density, length effects, and ECDF ----------
st.subheader("2) Telemetry — Joint Density & Distributions")
if "telemetry_with_pei" in st.session_state and st.session_state["telemetry_with_pei"] is not None:
    telx = st.session_state["telemetry_with_pei"].copy()
    telx.columns = [c.lower() for c in telx.columns]

    # Try to create basic length fields if missing
    if "prompt_len" not in telx.columns and "prompt" in telx.columns:
        telx["prompt_len"] = telx["prompt"].astype(str).str.split().apply(len)
    if "ai_len" not in telx.columns and "ai_response" in telx.columns:
        telx["ai_len"] = telx["ai_response"].astype(str).str.split().apply(len)

    # Joint density of PEI vs RDS (if both exist)
    if "prompt_evolution_index" in telx.columns and "rds_proxy" in telx.columns:
        figden = px.density_heatmap(
            telx.dropna(subset=["prompt_evolution_index","rds_proxy"]),
            x="prompt_evolution_index", y="rds_proxy", nbinsx=30, nbinsy=5,
            title="Joint Density: PEI × RDS"
        )
        figden.update_layout(template="plotly_white")
        st.plotly_chart(figden, use_container_width=True)

    # Token length vs PEI with trendline
    if "prompt_len" in telx.columns and "prompt_evolution_index" in telx.columns:
        fig_sc = px.scatter(
            telx, x="prompt_len", y="prompt_evolution_index",
            color="group" if "group" in telx.columns else None,
            trendline="ols" if len(telx) > 30 else None,
            title="PEI vs Prompt Length"
        )
        fig_sc.update_layout(template="plotly_white")
        st.plotly_chart(fig_sc, use_container_width=True)

    # ECDFs for PEI and RDS
    for metric in ["prompt_evolution_index","rds_proxy"]:
        if metric in telx.columns:
            figc = px.ecdf(telx.dropna(subset=[metric]), x=metric, title=f"ECDF — {metric}")
            figc.update_layout(template="plotly_white")
            st.plotly_chart(figc, use_container_width=True)
else:
    st.caption("Upload telemetry to enable these PEI/RDS analyses.")

# ---------- 3) Surveys: Violins & ECDF by Instrument ----------
st.subheader("3) Surveys — Distributions by Instrument")
if "surveys_df" in st.session_state and st.session_state["surveys_df"] is not None:
    svx = st.session_state["surveys_df"].copy()
    svx.columns = [c.lower() for c in svx.columns]
    if {"instrument","response"}.issubset(svx.columns):
        figv = px.violin(svx, x="instrument", y="response", box=True, points="all",
                         title="Survey Responses by Instrument (Violin + Box)")
        figv.update_layout(template="plotly_white", xaxis_tickangle=30)
        st.plotly_chart(figv, use_container_width=True)

        figc = px.ecdf(svx.dropna(subset=["response"]), x="response", color="instrument",
                       title="ECDF — Survey Responses by Instrument")
        figc.update_layout(template="plotly_white")
        st.plotly_chart(figc, use_container_width=True)
else:
    st.caption("Upload surveys to enable these survey distribution analyses.")

# ---------- 4) Fairness: Calibration, PR curves, and parity pass/fail table ----------
st.subheader("4) Fairness — Calibration · PR Curves · Parity Table")
if "fairness_df" in st.session_state and st.session_state["fairness_df"] is not None:
    fx = st.session_state["fairness_df"].copy()
    fx.columns = [c.lower() for c in fx.columns]

    # Calibration-in-the-large and reliability curve (per group, if y_score available)
    if {"group","y_true","y_score"}.issubset(fx.columns):
        # Bin scores and compute observed rate per bin
        fx["_score"] = pd.to_numeric(fx["y_score"], errors="coerce")
        fx = fx.dropna(subset=["_score","y_true"])
        fx["_bin"] = pd.qcut(fx["_score"], q=10, duplicates="drop")
        cal = (fx.groupby(["group","_bin"])
                 .apply(lambda d: pd.Series({
                     "mean_score": d["_score"].mean(),
                     "obs_rate":  pd.to_numeric(d["y_true"], errors="coerce").mean(),
                     "n": len(d)
                 }))
                 .reset_index())
        if not cal.empty:
            figcal = px.line(cal.sort_values("mean_score"), x="mean_score", y="obs_rate",
                             color="group", markers=True, title="Reliability Curve (Observed vs Predicted)")
            figcal.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
            figcal.update_layout(template="plotly_white", xaxis_title="Mean Predicted (bin)", yaxis_title="Observed Positive Rate")
            st.plotly_chart(figcal, use_container_width=True)

    # Precision-Recall by group (if y_score & y_true)
    if {"group","y_true","y_score"}.issubset(fx.columns):
        prs = []
        for g, sub in fx.groupby("group"):
            s = pd.to_numeric(sub["y_score"], errors="coerce").dropna()
            y = pd.to_numeric(sub.loc[s.index, "y_true"], errors="coerce")
            # thresholds
            thr = np.unique(s.values)
            if thr.size < 2: 
                continue
            for t in np.r_[0.0, thr, 1.0]:
                yp = (s >= t).astype(int)
                TP = int(((yp==1)&(y==1)).sum()); FP = int(((yp==1)&(y==0)).sum())
                FN = int(((yp==0)&(y==1)).sum())
                prec = TP / (TP + FP) if (TP+FP) > 0 else np.nan
                rec  = TP / (TP + FN) if (TP+FN) > 0 else np.nan
                prs.append((g, rec, prec))
        if prs:
            prdf = pd.DataFrame(prs, columns=["group","recall","precision"]).dropna()
            figpr = px.line(prdf, x="recall", y="precision", color="group", title="Precision–Recall by Group")
            figpr.update_layout(template="plotly_white", xaxis_title="Recall", yaxis_title="Precision")
            st.plotly_chart(figpr, use_container_width=True)

    # Parity pass/fail table: DI >= 0.8 rule (needs SR by group)
    # Try to derive SR via y_true/y_pred or direct ai_accuracy as fallback
    def _sr_from_df(f):
        if "y_pred" in f.columns:
            return f.groupby("group")["y_pred"].mean().rename("SR").reset_index()
        elif "y_score" in f.columns:
            tmp = f.copy()
            tmp["_y_pred_thr"] = (pd.to_numeric(tmp["y_score"], errors="coerce") >= 0.5).astype(int)
            return tmp.groupby("group")["_y_pred_thr"].mean().rename("SR").reset_index()
        return None

    sr = _sr_from_df(fx)
    if sr is not None and not sr.empty:
        ref_sr = sr.loc[sr["SR"].idxmax(), "SR"]  # use group with highest SR as reference
        di_tbl = sr.copy()
        di_tbl["DI"] = di_tbl["SR"] / ref_sr if ref_sr and np.isfinite(ref_sr) else np.nan
        di_tbl["80% rule pass"] = di_tbl["DI"].apply(lambda z: "PASS" if (isinstance(z,(int,float)) and z >= 0.8) else "FAIL")
        table = go.Figure(data=[go.Table(
            header=dict(values=list(di_tbl.columns), fill_color="#2c3e50", font=dict(color="white")),
            cells=dict(values=[di_tbl[c] for c in di_tbl.columns], align="left")
        )])
        table.update_layout(title="Disparate Impact (DI) — 80% Rule (reference = highest SR)", template="plotly_white")
        st.plotly_chart(table, use_container_width=True)

else:
    st.caption("Upload fairness data to enable calibration, PR curves, and DI table.")

# ---------- 5) Teacher: Time-of-day / Weekday heatmap ----------
st.subheader("5) Teacher — Time-of-Day × Weekday Heatmap")
if "teacher_df" in st.session_state and st.session_state["teacher_df"] is not None:
    tdx = st.session_state["teacher_df"].copy()
    tdx.columns = [c.lower() for c in tdx.columns]
    if "timestamp" in tdx.columns:
        tdx["timestamp"] = pd.to_datetime(tdx["timestamp"], errors="coerce")
        tdx = tdx.dropna(subset=["timestamp"])
        if not tdx.empty:
            tdx["hour"] = tdx["timestamp"].dt.hour
            tdx["weekday"] = tdx["timestamp"].dt.weekday  # 0=Mon
            piv = tdx.pivot_table(index="weekday", columns="hour", values=tdx.columns[0], aggfunc="count", fill_value=0)
            heat = go.Figure(data=go.Heatmap(z=piv.values, x=piv.columns, y=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][:piv.shape[0]],
                                             colorscale="YlGnBu"))
            heat.update_layout(title="Teacher Activity Heatmap (Weekday × Hour)", template="plotly_white",
                               xaxis_title="Hour", yaxis_title="Weekday")
            st.plotly_chart(heat, use_container_width=True)
else:
    st.caption("Upload teacher logs with timestamps to enable the orchestration heatmap.")



# ================================
# Advanced Add-Ons (Sections 6–10)
# ================================
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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


# -----------------------------------------
# 6) Learning Gains — Quantile Bands & Heterogeneity
# -----------------------------------------
st.subheader("6) Learning Gains — Quantile Bands & Heterogeneity")
if "learning_gains" in st.session_state and st.session_state["learning_gains"] is not None:
    lg6 = st.session_state["learning_gains"].copy()
    lg6.columns = [c.lower() for c in lg6.columns]
    # A) Quantile bands of Δ by Pre (nonparametric via binning)
    if {"pre","learning_gain"}.issubset(lg6.columns):
        dfq = lg6.dropna(subset=["pre","learning_gain"]).copy()
        if not dfq.empty:
            dfq["_bin"] = pd.qcut(dfq["pre"], q=10, duplicates="drop")
            band = (dfq.groupby("_bin")["learning_gain"]
                        .agg(n="size", q10=lambda s: s.quantile(0.10),
                             q25=lambda s: s.quantile(0.25),
                             q50="median",
                             q75=lambda s: s.quantile(0.75),
                             q90=lambda s: s.quantile(0.90))
                        .reset_index())
            # represent bins by midpoint
            band["pre_mid"] = band["_bin"].apply(lambda iv: (iv.left + iv.right)/2)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=band["pre_mid"], y=band["q50"], mode="lines+markers", name="Median (Q50)"))
            fig.add_trace(go.Scatter(x=band["pre_mid"], y=band["q75"], mode="lines", name="Q75"))
            fig.add_trace(go.Scatter(x=band["pre_mid"], y=band["q25"], mode="lines", name="Q25", fill='tonexty'))
            fig.add_trace(go.Scatter(x=band["pre_mid"], y=band["q90"], mode="lines", name="Q90"))
            fig.add_trace(go.Scatter(x=band["pre_mid"], y=band["q10"], mode="lines", name="Q10", fill='tonexty', opacity=0.3))
            fig.update_layout(template="plotly_white", title="Learning Gain vs Pre — Quantile Bands",
                              xaxis_title="Pre", yaxis_title="Learning Gain (Δ)")
            st.plotly_chart(fig, use_container_width=True)

    # B) Heterogeneity by subgroup (interaction-style view)
    if {"group","learning_gain"}.issubset(lg6.columns):
        agg = (lg6.groupby("group")["learning_gain"]
                  .agg(mean="mean", std="std", count="count").reset_index())
        agg["se"] = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
        agg["lo"] = agg["mean"] - 1.96*agg["se"]
        agg["hi"] = agg["mean"] + 1.96*agg["se"]
        figf = go.Figure()
        figf.add_trace(go.Scatter(x=agg["group"], y=agg["mean"], mode="markers", name="Mean Δ",
                                  error_y=dict(type="data", array=agg["hi"]-agg["mean"],
                                               arrayminus=agg["mean"]-agg["lo"])))
        figf.update_layout(template="plotly_white", title="Learning Gain by Group (±95% CI)",
                           xaxis_title="Group", yaxis_title="Mean Δ")
        st.plotly_chart(figf, use_container_width=True)

        # Table
        tbl = go.Figure(data=[go.Table(
            header=dict(values=list(agg.columns), fill_color="#2c3e50", font=dict(color="white")),
            cells=dict(values=[agg[c] for c in agg.columns], align="left")
        )])
        tbl.update_layout(title="Learning Gain by Group — Summary", template="plotly_white")
        st.plotly_chart(tbl, use_container_width=True)
else:
    st.caption("Upload assessments to enable Section 6.")

# -----------------------------------------
# 7) Telemetry — Cue Usage, Top Tokens & Bigram Table
# -----------------------------------------
st.subheader("7) Telemetry — Discourse Cue Usage & Top Tokens")
if "telemetry_with_pei" in st.session_state and st.session_state["telemetry_with_pei"] is not None:
    tel7 = st.session_state["telemetry_with_pei"].copy()
    tel7.columns = [c.lower() for c in tel7.columns]
    # Prepare text col
    text_col = "ai_response" if "ai_response" in tel7.columns else ("prompt" if "prompt" in tel7.columns else None)
    if text_col:
        toks = tel7[text_col].astype(str).str.lower()
        # A) Discourse cues frequency by group
        CUES = ["because","therefore","however","hence","justify","so that","evidence","for example"]
        rows = []
        if "group" in tel7.columns:
            for g, sub in tel7.groupby("group"):
                s = sub[text_col].astype(str).str.lower().str.cat(sep=" ")
                for c in CUES:
                    rows.append(dict(group=g, cue=c, freq=s.count(c)))
        else:
            s = toks.str.cat(sep=" ")
            for c in CUES:
                rows.append(dict(group="ALL", cue=c, freq=s.count(c)))
        cue_df = pd.DataFrame(rows)
        if not cue_df.empty:
            figc = px.bar(cue_df, x="cue", y="freq", color="group", barmode="group",
                          title="Discourse Cue Frequency")
            figc.update_layout(template="plotly_white")
            st.plotly_chart(figc, use_container_width=True)

        # B) Top unigrams table (simple stoplist)
        STOP = set("the a an and or of to in is are for with on by as at from that this it be was were".split())
        words = (toks.str.replace(r"[^a-z\s]", " ", regex=True)
                      .str.split()
                      .explode()
                      .dropna())
        words = words[~words.isin(STOP)]
        top_uni = words.value_counts().head(30).reset_index()
        top_uni.columns = ["token","count"]
        table_uni = go.Figure(data=[go.Table(
            header=dict(values=list(top_uni.columns), fill_color="#34495e", font=dict(color="white")),
            cells=dict(values=[top_uni[c] for c in top_uni.columns], align="left")
        )])
        table_uni.update_layout(title="Top Tokens (Unigrams)", template="plotly_white")
        st.plotly_chart(table_uni, use_container_width=True)

        # C) Bigram table (very lightweight)
        tokens = (toks.str.replace(r"[^a-z\s]", " ", regex=True)
                       .str.split())
        def bigrams(lst):
            return [" ".join(pair) for pair in zip(lst, lst[1:])] if isinstance(lst, list) else []
        bigs = tokens.apply(bigrams).explode().dropna()
        bigs = bigs[~bigs.str.contains(r"\b("+"|".join(STOP)+r")\b", regex=True, na=False)]
        top_bi = bigs.value_counts().head(30).reset_index()
        top_bi.columns = ["bigram","count"]
        table_bi = go.Figure(data=[go.Table(
            header=dict(values=list(top_bi.columns), fill_color="#34495e", font=dict(color="white")),
            cells=dict(values=[top_bi[c] for c in top_bi.columns], align="left")
        )])
        table_bi.update_layout(title="Top Bigrams", template="plotly_white")
        st.plotly_chart(table_bi, use_container_width=True)
else:
    st.caption("Upload telemetry to enable Section 7.")

# -----------------------------------------
# 8) Surveys — Item Diagnostics (Item–Total r, α-if-deleted) & PCA Scree
# -----------------------------------------
st.subheader("8) Surveys — Item Diagnostics & Scree")
if "surveys_df" in st.session_state and st.session_state["surveys_df"] is not None:
    sv8 = st.session_state["surveys_df"].copy()
    sv8.columns = [c.lower() for c in sv8.columns]
    # Require instrument + item_code + response
    if {"instrument","item_code","response","student_id"}.issubset(sv8.columns):
        inst = st.selectbox("Choose instrument for item diagnostics", sorted(sv8["instrument"].unique()))
        sub = sv8[sv8["instrument"]==inst].dropna(subset=["item_code","response","student_id"]).copy()
        if not sub.empty:
            # Pivot: rows=student, cols=item, values=response
            wide = sub.pivot_table(index="student_id", columns="item_code", values="response", aggfunc="mean")
            wide = wide.dropna(axis=1, how="all").dropna(axis=0, how="any")
            if wide.shape[1] >= 3 and wide.shape[0] >= 20:
                # Item-total correlations & alpha-if-deleted
                items = wide.columns.tolist()
                it_rows = []
                total = wide.sum(axis=1)
                k = len(items)
                var_total = total.var(ddof=1) if k>1 else np.nan
                for it in items:
                    rest = (total - wide[it])
                    r_it = np.corrcoef(wide[it], rest)[0,1] if rest.std(ddof=1)>0 and wide[it].std(ddof=1)>0 else np.nan
                    # Cronbach alpha if item deleted
                    submat = wide.drop(columns=[it])
                    k2 = submat.shape[1]
                    var_sum = submat.sum(axis=1).var(ddof=1) if k2>1 else np.nan
                    var_items = submat.var(ddof=1, axis=0).sum() if k2>1 else np.nan
                    alpha_del = (k2/(k2-1))*(1 - var_items/var_sum) if (k2>1 and var_sum>0) else np.nan
                    it_rows.append(dict(item=it, item_total_r=r_it, alpha_if_deleted=alpha_del))
                diag = pd.DataFrame(it_rows)

                t1 = go.Figure(data=[go.Table(
                    header=dict(values=list(diag.columns), fill_color="#2c3e50", font=dict(color="white")),
                    cells=dict(values=[diag[c] for c in diag.columns], align="left")
                )])
                t1.update_layout(title=f"{inst}: Item Diagnostics (item-total r, α-if-deleted)", template="plotly_white")
                st.plotly_chart(t1, use_container_width=True)

                # PCA Scree
                X = (wide - wide.mean())/wide.std(ddof=1)
                X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="any")
                if X.shape[1] >= 3:
                    cov = np.cov(X.values, rowvar=False)
                    evals, _ = np.linalg.eig(cov)
                    evals = np.sort(np.real(evals))[::-1]
                    scree = pd.DataFrame({"component": np.arange(1, len(evals)+1), "eigenvalue": evals})
                    fig_scree = px.line(scree, x="component", y="eigenvalue", markers=True,
                                        title=f"{inst}: PCA Scree Plot")
                    fig_scree.update_layout(template="plotly_white")
                    st.plotly_chart(fig_scree, use_container_width=True)
            else:
                st.info("Need ≥20 students and ≥3 items with complete rows for diagnostics.")
    else:
        st.info("Surveys need columns: instrument, item_code, response, student_id.")
else:
    st.caption("Upload surveys to enable Section 8.")

# -----------------------------------------
# 9) Fairness — Threshold Sweep, ROC/AUC by Group, Confusion Table
# -----------------------------------------
st.subheader("9) Fairness — Threshold Sweep & ROC/AUC by Group")
if "fairness_df" in st.session_state and st.session_state["fairness_df"] is not None:
    fx9 = st.session_state["fairness_df"].copy()
    fx9.columns = [c.lower() for c in fx9.columns]
    if {"group","y_true"}.issubset(fx9.columns) and "y_score" in fx9.columns:
        fx9["_y_score"] = pd.to_numeric(fx9["y_score"], errors="coerce")
        fx9 = fx9.dropna(subset=["_y_score","y_true","group"])
        if not fx9.empty:
            # A) Threshold sweep of SR/TPR/FPR gaps vs threshold
            thr_grid = np.linspace(0.05, 0.95, 19)
            rows = []
            for t in thr_grid:
                fx9["_yp"] = (fx9["_y_score"] >= t).astype(int)
                # per group rates
                grp = []
                for g, sub in fx9.groupby("group"):
                    y = sub["y_true"].astype(int).values
                    yp = sub["_yp"].astype(int).values
                    TP = ((yp==1)&(y==1)).sum()
                    TN = ((yp==0)&(y==0)).sum()
                    FP = ((yp==1)&(y==0)).sum()
                    FN = ((yp==0)&(y==1)).sum()
                    P, Nn, n = TP+FN, TN+FP, len(sub)
                    grp.append(dict(g=g, SR=(TP+FP)/n if n else np.nan,
                                    TPR=TP/P if P else np.nan,
                                    FPR=FP/Nn if Nn else np.nan))
                gdf = pd.DataFrame(grp)
                if gdf.empty: 
                    continue
                # reference = largest SR group at this threshold
                gref = gdf.iloc[gdf["SR"].idxmax()]
                rows.append(dict(thr=t,
                                 SPD=(gref["SR"] - gdf["SR"]).abs().max(),
                                 EOG=(gdf["TPR"] - gref["TPR"]).abs().max(),
                                 EOD=(((gdf["TPR"] - gref["TPR"]).abs() + (gdf["FPR"] - gref["FPR"]).abs())/2).max()))
            sweep = pd.DataFrame(rows)
            if not sweep.empty:
                figs = px.line(sweep.melt("thr", var_name="metric", value_name="gap"),
                               x="thr", y="gap", color="metric",
                               title="Threshold Sweep — Max Gap across Groups")
                figs.update_layout(template="plotly_white", xaxis_title="Threshold", yaxis_title="Max Gap")
                st.plotly_chart(figs, use_container_width=True)

            # B) ROC & AUC by group
            rocs = []
            aucs = []
            for g, sub in fx9.groupby("group"):
                y = sub["y_true"].astype(int).values
                s = sub["_y_score"].values
                # compute ROC stepwise
                thr = np.unique(s)
                if thr.size < 2:
                    continue
                pts = []
                for t in np.r_[0.0, thr, 1.0]:
                    yp = (s >= t).astype(int)
                    TP = ((yp==1)&(y==1)).sum()
                    TN = ((yp==0) & (y==0)).sum() 
                    FP = ((yp==1) & (y==0)).sum()
                    FN = ((yp==0) & (y==1)).sum()
                    P, Nn = TP+FN, TN+FP
                    TPR = TP/P if P else np.nan
                    FPR = FP/Nn if Nn else np.nan
                    pts.append((FPR, TPR))
                roc = pd.DataFrame(pts, columns=["FPR","TPR"]).dropna()
                if not roc.empty:
                    roc["group"] = g
                    rocs.append(roc)
                    # trapezoidal AUC
                    auc = np.trapz(y=np.sort(roc["TPR"].values), x=np.sort(roc["FPR"].values))
                    aucs.append(dict(group=g, AUC=auc))
            if rocs:
                roc_all = pd.concat(rocs, ignore_index=True)
                figroc = px.line(roc_all, x="FPR", y="TPR", color="group", title="ROC by Group")
                figroc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
                figroc.update_layout(template="plotly_white")
                st.plotly_chart(figroc, use_container_width=True)
            if aucs:
                auc_df = pd.DataFrame(aucs)
                tauc = go.Figure(data=[go.Table(
                    header=dict(values=list(auc_df.columns), fill_color="#2c3e50", font=dict(color="white")),
                    cells=dict(values=[auc_df[c] for c in auc_df.columns], align="left")
                )])
                tauc.update_layout(title="AUC by Group (trapezoid approx.)", template="plotly_white")
                st.plotly_chart(tauc, use_container_width=True)

            # C) Confusion matrix table per group at default thr=0.5
            fx9["_yp"] = (fx9["_y_score"] >= 0.5).astype(int)
            rows = []
            for g, sub in fx9.groupby("group"):
                y = sub["y_true"].astype(int).values
                yp = sub["_yp"].astype(int).values
                TP = ((yp==1)&(y==1)).sum(); TN = ((yp==0)&(y==0)).sum()
                FP = ((yp==1)&(y==0)).sum(); FN = ((yp==0)&(y==1)).sum()
                rows.append(dict(group=g, TP=TP, FP=FP, FN=FN, TN=TN))
            cm = pd.DataFrame(rows)
            if not cm.empty:
                tcm = go.Figure(data=[go.Table(
                    header=dict(values=list(cm.columns), fill_color="#2c3e50", font=dict(color="white")),
                    cells=dict(values=[cm[c] for c in cm.columns], align="left")
                )])
                tcm.update_layout(title="Confusion Counts by Group (thr=0.5)", template="plotly_white")
                st.plotly_chart(tcm, use_container_width=True)
    else:
        st.info("Provide group + y_true + y_score for Section 9.")
else:
    st.caption("Upload fairness data to enable Section 9.")

# -----------------------------------------
# 10) Cross-Module — Group KPI Matrix & Correlations
# -----------------------------------------
st.subheader("10) Cross-Module — Group KPI Matrix & Correlations")
# Build a per-group KPI table across modules when possible
pieces = []

# Learning Gain by group
if "learning_gains" in st.session_state and st.session_state["learning_gains"] is not None:
    lg10 = st.session_state["learning_gains"].copy()
    lg10.columns = [c.lower() for c in lg10.columns]
    if {"group","learning_gain"}.issubset(lg10.columns):
        g_lg = lg10.groupby("group")["learning_gain"].mean().rename("mean_gain").reset_index()
        pieces.append(g_lg)

# PEI by group
if "telemetry_with_pei" in st.session_state and st.session_state["telemetry_with_pei"] is not None:
    t10 = st.session_state["telemetry_with_pei"].copy()
    t10.columns = [c.lower() for c in t10.columns]
    if {"group","prompt_evolution_index"}.issubset(t10.columns):
        g_pei = t10.groupby("group")["prompt_evolution_index"].mean().rename("mean_pei").reset_index()
        pieces.append(g_pei)

# Fairness: DI/TPR/FPR if available
if "fairness_df" in st.session_state and st.session_state["fairness_df"] is not None:
    f10 = st.session_state["fairness_df"].copy()
    f10.columns = [c.lower() for c in f10.columns]
    yhat = "y_pred" if "y_pred" in f10.columns else None
    if not yhat and "y_score" in f10.columns:
        f10["_y_pred_thr"] = (pd.to_numeric(f10["y_score"], errors="coerce") >= 0.5).astype(int)
        yhat = "_y_pred_thr"
    if {"group","y_true"}.issubset(f10.columns) and yhat:
        rows = []
        for g, sub in f10.dropna(subset=["group"]).groupby("group"):
            y = pd.to_numeric(sub["y_true"], errors="coerce").astype(int)
            yp = pd.to_numeric(sub[yhat], errors="coerce").astype(int)
            TP = int(((yp==1)&(y==1)).sum()); TN = int(((yp==0)&(y==0)).sum())
            FP = int(((yp==1)&(y==0)).sum()); FN = int(((yp==0)&(y==1)).sum())
            n  = len(sub); P, Nn = TP+FN, TN+FP
            SR  = (TP+FP)/n if n else np.nan
            TPR = TP/P if P else np.nan
            FPR = FP/Nn if Nn else np.nan
            rows.append(dict(group=g, SR=SR, TPR=TPR, FPR=FPR))
        g_fair = pd.DataFrame(rows)
        # DI vs ref (largest SR)
        if not g_fair.empty:
            ref_sr = g_fair.loc[g_fair["SR"].idxmax(), "SR"]
            g_fair["DI"] = g_fair["SR"]/ref_sr if ref_sr and np.isfinite(ref_sr) else np.nan
            pieces.append(g_fair)

# Merge all per-group pieces
if pieces:
    from functools import reduce
    gmat = reduce(lambda l, r: pd.merge(l, r, on="group", how="outer"), pieces)
    st.write("Per-Group KPI Matrix (merged across modules):")
    st.dataframe(gmat, use_container_width=True)

    # Correlation heatmap among numeric KPIs
    num_cols = [c for c in gmat.columns if c != "group" and pd.api.types.is_numeric_dtype(gmat[c])]
    if len(num_cols) >= 2:
        corr = gmat[num_cols].corr()
        heat = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index,
                                         colorscale="RdBu", zmid=0))
        heat.update_layout(title="Correlation Matrix — Group KPIs", template="plotly_white")
        st.plotly_chart(heat, use_container_width=True)

    # Export
    st.download_button("Download Group KPI Matrix CSV",
                       data=gmat.to_csv(index=False).encode("utf-8"),
                       file_name="group_kpi_matrix.csv", mime="text/csv")
else:
    st.info("Not enough overlapping group metrics to build a cross-module KPI matrix yet.")    
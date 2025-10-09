# pages/5_Fairness.py — Advanced Bias Auditing
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.title("5) Fairness — Bias Auditing (Advanced)")

# --------------------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------------------
if "fairness_df" not in st.session_state:
    st.info("Please upload fairness data on the **0. Upload** page.")
    st.stop()

raw = st.session_state["fairness_df"].copy()
st.write("Preview:", raw.head())
raw.columns = [c.strip().lower() for c in raw.columns]

# soft schema (all optional)
has_group = "group" in raw.columns
has_ytrue = "y_true" in raw.columns
has_ypred = "y_pred" in raw.columns
has_score = "y_score" in raw.columns  # probability / score
has_acc   = "ai_accuracy" in raw.columns
has_tprgap = "tpr_gap" in raw.columns
has_fprgap = "fpr_gap" in raw.columns
has_eodiff = "equalized_odds_diff" in raw.columns

df = raw.copy()

# --------------------------------------------------------------------------------------
# Academic formulas (display)
# --------------------------------------------------------------------------------------
st.markdown(r"""
### Fairness Metrics (Formulas)
Let \(A\) be the sensitive attribute (group), \( \hat{Y} \) the prediction at a threshold, \(Y\) the ground truth.

- **Selection Rate (SR)**: \(\; SR_g = P(\hat{Y}=1 \mid A=g)\)
- **Statistical Parity Difference (SPD)**: \(\; SPD = SR_{ref} - SR_{g}\)
- **Disparate Impact (DI)**: \(\; DI = SR_g / SR_{ref}\)  (80% rule if \(DI \ge 0.8\))
- **True Positive Rate (TPR / Recall)**: \(\; TPR_g = P(\hat{Y}=1 \mid Y=1, A=g)\)
- **False Positive Rate (FPR)**: \(\; FPR_g = P(\hat{Y}=1 \mid Y=0, A=g)\)
- **Equal Opportunity Gap (EOG)**: \(\; |TPR_g - TPR_{ref}|\)
- **Equalized Odds Gap (EOD)**: \(\; \frac{|TPR_g - TPR_{ref}| + |FPR_g - FPR_{ref}|}{2}\)
- **Calibration-in-the-large (CIL)**: \(\; \text{mean}(Y) - \text{mean}( \hat{p})\)
- **Brier Score (per group)**: \(\; \frac{1}{n}\sum ( \hat{p} - Y)^2\)
""")

# --------------------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------------------
with st.expander("Filters"):
    if has_group:
        groups = sorted(df["group"].dropna().astype(str).unique())
        sel_groups = st.multiselect("Groups", groups, default=groups, key="fair_groups")
    else:
        sel_groups = []
    if has_score:
        smin, smax = float(df["y_score"].min()), float(df["y_score"].max())
        score_range = st.slider("Score range filter (y_score)", smin, smax, (smin, smax), step=(smax-smin)/20 if smax>smin else 0.05, key="fair_score_rng")
    else:
        score_range = None

mask = pd.Series(True, index=df.index)
if has_group and sel_groups:
    mask &= df["group"].astype(str).isin(sel_groups)
if has_score and score_range:
    mask &= df["y_score"].between(score_range[0], score_range[1])
df = df[mask].copy()

# --------------------------------------------------------------------------------------
# Thresholding (if y_score available)
# --------------------------------------------------------------------------------------
if has_score:
    st.subheader("A) Thresholding")
    thr = st.slider("Decision threshold for y_score → y_pred", 0.0, 1.0, 0.5, 0.01, key="fair_thr")
    df["_y_pred_thr"] = (df["y_score"] >= thr).astype(int)
    if not has_ypred:
        ypred_col = "_y_pred_thr"
    else:
        # let user choose to use provided y_pred or thresholded y_score
        ypred_col = st.radio("Prediction column to use", ["y_pred (provided)", "thresholded y_score"],
                             index=1, key="fair_pred_choice")
        ypred_col = "y_pred" if "y_pred" in ypred_col else "_y_pred_thr"
else:
    thr = None
    ypred_col = "y_pred" if has_ypred else None

# --------------------------------------------------------------------------------------
# Utility: per-group confusion components & metrics
# --------------------------------------------------------------------------------------
def confusion_by_group(x, y, a):
    """Return per-group TP, FP, TN, FN, counts."""
    out = []
    for g, sub in zip(a.unique(), [x[a==gg] for gg in a.unique()]):  # we’ll reindex correctly
        pass
    # Proper implementation
    res = []
    for g in pd.Series(a).dropna().unique():
        mask_g = (a == g)
        yy = pd.Series(y)[mask_g]
        xx = pd.Series(x)[mask_g]
        tp = int(((xx==1) & (yy==1)).sum())
        tn = int(((xx==0) & (yy==0)).sum())
        fp = int(((xx==1) & (yy==0)).sum())
        fn = int(((xx==0) & (yy==1)).sum())
        n  = int(mask_g.sum())
        res.append(dict(group=g, TP=tp, FP=fp, TN=tn, FN=fn, N=n))
    return pd.DataFrame(res)

def safe_rate(num, den):
    return float(num/den) if den and den>0 else np.nan

def group_metrics(conf, y=None, p=None):
    """conf: DataFrame with TP/FP/TN/FN; y (true) and p (score) for calibration metrics (same length as df; we recompute per group)."""
    rows = []
    for _, r in conf.iterrows():
        g = r["group"]
        TP, FP, TN, FN = r["TP"], r["FP"], r["TN"], r["FN"]
        P  = TP + FN
        Nn = TN + FP
        TPR = safe_rate(TP, P)
        FPR = safe_rate(FP, Nn)
        FNR = safe_rate(FN, P)
        TNR = safe_rate(TN, Nn)
        PPV = safe_rate(TP, TP+FP)
        NPV = safe_rate(TN, TN+FN)
        ACC = safe_rate(TP+TN, TP+TN+FP+FN)
        SR  = safe_rate(TP+FP, TP+FP+TN+FN)

        cil = np.nan
        brier = np.nan
        if (y is not None) and (p is not None):
            yy = y[df["group"] == g]
            pp = p[df["group"] == g]
            if len(yy) > 0:
                cil = float(np.nanmean(yy) - np.nanmean(pp))
                brier = float(np.nanmean((pp - yy)**2))

        rows.append(dict(group=g, SR=SR, ACC=ACC, TPR=TPR, FPR=FPR, FNR=FNR, TNR=TNR, PPV=PPV, NPV=NPV,
                         CIL=cil, Brier=brier, N=r["N"]))
    gm = pd.DataFrame(rows)
    return gm.sort_values("group")

def bootstrap_ci(metric_values, B=1000, alpha=0.05, seed=42):
    vals = pd.Series(metric_values).dropna().values
    if len(vals)==0: return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = []
    n = len(vals)
    for _ in range(B):
        boots.append(np.mean(rng.choice(vals, size=n, replace=True)))
    lo = float(np.percentile(boots, 100*alpha/2))
    hi = float(np.percentile(boots, 100*(1-alpha/2)))
    return (lo, hi)

# --------------------------------------------------------------------------------------
# Compute metrics (if we have the needed columns)
# --------------------------------------------------------------------------------------
st.subheader("B) Per-Group Metrics")

metrics_table = pd.DataFrame()
gap_table = pd.DataFrame()
diag_conf = None

if has_group and has_ytrue and (ypred_col is not None):
    conf = confusion_by_group(df[ypred_col], df["y_true"], df["group"])
    diag_conf = conf.copy()
    gm = group_metrics(conf, y=df["y_true"] if has_ytrue else None, p=df["y_score"] if has_score else None)

    # choose reference group (largest N)
    ref_group = gm.sort_values("N", ascending=False)["group"].iloc[0]
    st.write(f"Reference group (largest N): **{ref_group}**")

    ref = gm[gm["group"]==ref_group].iloc[0]
    gm["SPD"] = ref["SR"] - gm["SR"]
    gm["DI"]  = gm["SR"] / ref["SR"] if ref["SR"] and ref["SR"]>0 else np.nan
    gm["EOG"] = (gm["TPR"] - ref["TPR"]).abs()
    gm["EOD"] = ((gm["TPR"] - ref["TPR"]).abs() + (gm["FPR"] - ref["FPR"]).abs()) / 2

    metrics_table = gm.copy()

    # Plotly table
    tbl = go.Figure(data=[go.Table(
        header=dict(values=list(gm.columns), fill_color="#2c3e50", font=dict(color="white")),
        cells=dict(values=[gm[c] for c in gm.columns], align="left")
    )])
    tbl.update_layout(title="Per-Group Metrics", template="plotly_white")
    st.plotly_chart(tbl, use_container_width=True)

    # Bar charts
    for metric, title in [("SR","Selection Rate (SR)"),
                          ("ACC","Accuracy"),
                          ("TPR","True Positive Rate (TPR)"),
                          ("FPR","False Positive Rate (FPR)"),
                          ("EOD","Equalized Odds Gap (EOD)"),
                          ("SPD","Statistical Parity Difference (SPD)"),
                          ("DI","Disparate Impact (DI)"),
                          ("Brier","Brier Score (Calibration)"),
                          ("CIL","Calibration-in-the-large")]:
        if metric in gm.columns:
            fig = px.bar(gm, x="group", y=metric, title=title)
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

    # Gaps heatmap vs reference
    gap_cols = ["SPD","EOG","EOD","DI"]
    gt = gm[["group"] + gap_cols].set_index("group")
    heat = go.Figure(data=go.Heatmap(z=gt.values,
                                     x=gt.columns, y=gt.index,
                                     colorscale="RdBu", zmin=-np.nanmax(abs(gt.values)), zmax=np.nanmax(abs(gt.values))))
    heat.update_layout(title="Gap Heatmap vs Reference", template="plotly_white")
    st.plotly_chart(heat, use_container_width=True)

elif has_group and has_acc:
    # Fallback: summarize given accuracies by group
    acc = df.groupby("group")["ai_accuracy"].mean().reset_index()
    st.subheader("Per-Group AI Accuracy (given)")
    st.dataframe(acc, use_container_width=True)
    fig = px.bar(acc, x="group", y="ai_accuracy", title="Mean AI Accuracy by Group")
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # compute accuracy range
    st.info(f"Accuracy range across groups: {float(acc['ai_accuracy'].max() - acc['ai_accuracy'].min()):.4f}")
else:
    st.warning("Provide either: (group + y_true + y_pred / y_score) for full fairness audit, or (group + ai_accuracy) for a basic summary.")
# --------------------------------------------------------------------------------------
# C) ROC by Group (if y_score available) — robust with diagnostics
# --------------------------------------------------------------------------------------
st.subheader("C) ROC by Group (if y_score available)")

# Data readiness summary
readiness = {
    "has_group": has_group,
    "has_ytrue": has_ytrue,
    "has_score": has_score,
    "rows_after_filters": int(len(df))
}
st.caption(f"Data readiness: {readiness}")

if has_group and has_ytrue and has_score and len(df) > 0:
    # Coerce numeric and drop rows without y_true or y_score
    df["_y_score_num"] = pd.to_numeric(df["y_score"], errors="coerce")
    roc_input = df.dropna(subset=["_y_score_num", "y_true", "group"]).copy()
    if roc_input.empty:
        st.warning("No rows with non-null y_score and y_true remain after filters.")
    else:
        # per-group diagnostics: counts and unique score cardinality
        diag = (roc_input
                .groupby("group")
                .agg(n=("y_true","size"),
                     n_pos=("y_true", lambda s: int((s==1).sum())),
                     n_neg=("y_true", lambda s: int((s==0).sum())),
                     n_score_nonnull=("_y_score_num","count"),
                     score_cardinality=("_y_score_num", lambda s: s.nunique()))
                .reset_index())
        st.write("ROC Diagnostics (per group):")
        st.dataframe(diag, use_container_width=True)

        # Build ROC per group only if we have at least 2 unique scores
        rocs = []
        for g in diag["group"]:
            sub = roc_input[roc_input["group"] == g]
            if sub["_y_score_num"].nunique() < 2:
                # Can't build an ROC curve with <2 distinct thresholds
                continue
            ts = np.unique(sub["_y_score_num"].values)
            pts = []
            for t in np.r_[0.0, ts, 1.0]:
                yp = (sub["_y_score_num"] >= t).astype(int)
                y  = sub["y_true"].astype(int)
                TP = int(((yp==1)&(y==1)).sum()); FP = int(((yp==1)&(y==0)).sum())
                TN = int(((yp==0)&(y==0)).sum()); FN = int(((yp==0)&(y==1)).sum())
                P, Nn = TP+FN, TN+FP
                TPR = safe_rate(TP, P); FPR = safe_rate(FP, Nn)
                pts.append((FPR, TPR, t))
            roc = pd.DataFrame(pts, columns=["FPR","TPR","thr"])
            roc["group"] = g
            rocs.append(roc)

        if not rocs:
            st.warning("No group has ≥2 distinct y_score values after filters; ROC curves cannot be plotted.")
        else:
            roc_all = pd.concat(rocs, ignore_index=True)
            figroc = px.line(roc_all, x="FPR", y="TPR", color="group",
                             hover_data=["thr"], title="ROC by Group")
            figroc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
            figroc.update_layout(template="plotly_white")
            st.plotly_chart(figroc, use_container_width=True)
else:
    st.info("To show ROC, ensure columns present and non-empty after filters: group, y_true, y_score.")


# --------------------------------------------------------------------------------------
# D) Bootstrap 95% CI for a Metric — robust with diagnostics
# --------------------------------------------------------------------------------------
st.subheader("D) Bootstrap 95% CI for a Metric")

# Use thresholded y_score if available, else y_pred
_pred_source_msg = ""
_use_col = None
if has_group and has_ytrue:
    if has_score:
        # Create thresholded column if not already present
        if "_y_pred_thr" not in df.columns:
            thr = st.session_state.get("fair_thr", 0.5)  # reuse earlier slider if set
            df["_y_pred_thr"] = (pd.to_numeric(df["y_score"], errors="coerce") >= thr).astype(int)
        _use_col = "_y_pred_thr"
        _pred_source_msg = f"Using thresholded y_score at thr={st.session_state.get('fair_thr', 0.5)}."
    elif has_ypred:
        _use_col = "y_pred"
        _pred_source_msg = "Using provided y_pred."
else:
    _use_col = None

st.caption(f"Prediction source: {_pred_source_msg if _use_col else 'Not available'}")

if has_group and has_ytrue and (_use_col is not None) and len(df) > 0:
    # Coerce types and drop NaNs needed for calc
    bdf = df.dropna(subset=["y_true", _use_col, "group"]).copy()
    bdf["y_true"] = bdf["y_true"].astype(int)
    bdf[_use_col] = bdf[_use_col].astype(int)

    if bdf.empty:
        st.warning("No rows with non-null group, y_true, and predictions after filters.")
    else:
        target_metric = st.selectbox("Metric for bootstrap CI", ["SR","TPR","FPR","ACC"], index=0, key="fair_boot_metric")

        def metric_vec(sub):
            y = sub["y_true"].values
            yp = sub[_use_col].values
            if target_metric == "SR":
                return (yp==1).astype(int)
            if target_metric == "ACC":
                return (yp==y).astype(int)
            if target_metric == "TPR":
                m = y==1
                return (yp[m]==1).astype(int) if m.any() else np.array([])
            if target_metric == "FPR":
                m = y==0
                return (yp[m]==1).astype(int) if m.any() else np.array([])
            return np.array([])

        rows = []
        for g, sub in bdf.groupby("group"):
            v = metric_vec(sub)
            if len(v) == 0:
                rows.append(dict(group=g, mean=np.nan, lo=np.nan, hi=np.nan, n=0))
            else:
                mu = float(np.mean(v))
                lo, hi = bootstrap_ci(v, B=1000, alpha=0.05)
                rows.append(dict(group=g, mean=mu, lo=lo, hi=hi, n=len(v)))
        ci_df = pd.DataFrame(rows).sort_values("group")
        st.dataframe(ci_df, use_container_width=True)

        # Only plot when there is at least one valid mean
        good = ci_df.dropna(subset=["mean","lo","hi"])
        if good.empty:
            st.warning("No valid groups for CI plotting (possibly due to no positive/negative cases).")
        else:
            figci = go.Figure()
            figci.add_trace(go.Scatter(
                x=good["group"], y=good["mean"], mode="markers",
                error_y=dict(type="data", array=good["hi"]-good["mean"], arrayminus=good["mean"]-good["lo"]),
                name=f"{target_metric} ±95% CI"
            ))
            figci.update_layout(title=f"{target_metric} with 95% CI (bootstrap)",
                                template="plotly_white",
                                xaxis_title="Group", yaxis_title=target_metric)
            st.plotly_chart(figci, use_container_width=True)
else:
    st.info("To compute bootstrap CIs, ensure: group + y_true + (y_score or y_pred) and non-empty data after filters.")
       
# --------------------------------------------------------------------------------------
# Legacy inputs (if present)
# --------------------------------------------------------------------------------------
if has_tprgap or has_fprgap or has_eodiff:
    st.subheader("E) Legacy Fairness Columns (provided)")
    cols = [c for c in ["tpr_gap","fpr_gap","equalized_odds_diff"] if c in df.columns]
    st.dataframe(df[cols].describe(), use_container_width=True)

# --------------------------------------------------------------------------------------
# Diagnostics & Exports
# --------------------------------------------------------------------------------------
st.subheader("F) Diagnostics & Exports")

if diag_conf is not None:
    st.write("Per-Group Confusion Counts:")
    st.dataframe(diag_conf, use_container_width=True)

def dl(_df, name, label):
    st.download_button(label=label,
                       data=_df.to_csv(index=False).encode("utf-8"),
                       file_name=name, mime="text/csv")

if not df.empty:
    dl(df, "fairness_input_filtered.csv", "Download Filtered Input CSV")
if not metrics_table.empty:
    dl(metrics_table, "fairness_metrics_by_group.csv", "Download Metrics by Group CSV")
if diag_conf is not None and not diag_conf.empty:
    dl(diag_conf, "fairness_confusion_by_group.csv", "Download Confusion by Group CSV")
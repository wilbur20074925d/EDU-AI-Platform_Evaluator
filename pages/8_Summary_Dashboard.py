# pages/7_Summary_Dashboard.py
# Summary Dashboard — Part 05 Statistical Analysis Plan (step by step)
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Optional scientific tools (used if available; page still runs without them)
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import multipletests
    HAS_SM = True
except Exception:
    HAS_SM = False

try:
    from scipy import stats
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

st.set_page_config(page_title="7) Summary Dashboard — Part 05", layout="wide")
st.title("7) Summary Dashboard — Group Comparisons (Part 05)")
st.caption("This page implements the pre-registered analysis plan in steps with formulas, diagnostics, and interactive charts.")

def has(key: str) -> bool:
    return key in st.session_state and st.session_state[key] is not None

# -----------------------------
# Utilities (pure-Python safe)
# -----------------------------
def safe_rate(num, den):
    return float(num/den) if den and np.isfinite(den) and den > 0 else np.nan

def cohen_d(x, y):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().values
    y = pd.to_numeric(pd.Series(y), errors="coerce").dropna().values
    if len(x) < 2 or len(y) < 2: return np.nan
    nx, ny = len(x), len(y)
    pooled = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / (nx+ny-2))
    return (np.mean(x) - np.mean(y)) / pooled if pooled and np.isfinite(pooled) else np.nan

def bh_adjust(pvals, q=0.10):
    p = np.array([np.nan if v is None else v for v in pvals], dtype=float)
    mask = np.isfinite(p)
    padj = np.full_like(p, np.nan)
    if mask.sum() == 0: return padj.tolist()
    if HAS_SM:
        padj_sub = multipletests(p[mask], alpha=q, method="fdr_bh")[1]
        padj[mask] = padj_sub
    else:
        # Manual BH (Benjamini–Hochberg)
        idx = np.argsort(p[mask])
        p_sorted = p[mask][idx]
        m = len(p_sorted)
        adj = np.minimum.accumulate((p_sorted * m / (np.arange(m)+1))[::-1])[::-1]
        # ensure monotone and ≤1
        adj = np.minimum(adj, 1.0)
        padj_sub = np.empty_like(adj); padj_sub[idx] = adj
        padj[mask] = padj_sub
    return padj.tolist()

def quick_icc_oneway(y, groups):
    """
    ICC(1) one-way random effects (rough proxy)
    Assumes 'groups' categorical (e.g., class/instructor). Returns NaN if not feasible.
    """
    df = pd.DataFrame(dict(y=y, g=groups)).dropna()
    if df["g"].nunique() < 2: return np.nan
    # Between/within MS (ANOVA)
    grand = df["y"].mean()
    n_i = df.groupby("g")["y"].count()
    k = n_i.mean()  # avg cluster size
    if not np.isfinite(k) or k < 2: return np.nan
    mean_i = df.groupby("g")["y"].mean()
    ssb = (n_i * (mean_i - grand)**2).sum()
    ssw = ((df["y"] - df.groupby("g")["y"].transform("mean"))**2).sum()
    dfb = df["g"].nunique() - 1
    dfw = len(df) - df["g"].nunique()
    msb = ssb / dfb if dfb > 0 else np.nan
    msw = ssw / dfw if dfw > 0 else np.nan
    if not (np.isfinite(msb) and np.isfinite(msw)): return np.nan
    icc = (msb - msw) / (msb + (k-1)*msw)
    return float(icc)

# -----------------------------
# Step navigation
# -----------------------------
tabs = st.tabs([
    "A) Primary Outcome — Learning Gains",
    "B) ICC & Cluster Structure",
    "C) Mixed-Effects (approx) & Effect Size",
    "D) Multiple Outcomes & FDR",
    "E) Mediation (PEI/RDS → Gains)",
    "F) Dose–Response (Usage → Post/∆)",
    "G) Fairness Summary (Gaps & Tables)"
])

# ============================================================
# A) Primary Outcome — Learning Gains (∆ = Post − Pre)
# ============================================================
with tabs[0]:
    st.subheader("A) Primary Outcome — Learning Gains")
    st.markdown(r"""
**Definition (per student)**  
$$
\Delta = \text{Post} - \text{Pre}
$$
This is the primary outcome (O1) in the analysis plan. Comparable **Pre/Post** forms are assumed or calibrated for fairness of ∆.  
""")
    if not has("learning_gains"):
        st.info("Upload/compute assessments first (see '1) Assessments' page).")
    else:
        lg = st.session_state["learning_gains"].copy()
        lg.columns = [c.lower() for c in lg.columns]
        show_cols = [c for c in ["student_id","group","pre","post","learning_gain","course_id","topic"] if c in lg.columns]
        st.write("Preview of learning gains:")
        st.dataframe(lg[show_cols].head(30), use_container_width=True)

        # Summaries
        st.markdown("**Distribution & Summary**")
        fig = px.histogram(lg, x="learning_gain", nbins=30, marginal="box",
                           title="Distribution of Learning Gains (Δ = Post − Pre)")
        fig.update_layout(template="plotly_white", xaxis_title="Learning Gain (Δ)")
        st.plotly_chart(fig, use_container_width=True)

        # Group mean bars (if group exists)
        if "group" in lg.columns:
            agg = lg.groupby("group")["learning_gain"].agg(["mean","std","count"]).reset_index()
            st.write("Group summary (Learning Gain):")
            st.dataframe(agg, use_container_width=True)
            figb = px.bar(agg, x="group", y="mean", error_y="std",
                          title="Mean Learning Gain by Group")
            figb.update_layout(template="plotly_white", xaxis_tickangle=30, yaxis_title="Mean Δ")
            st.plotly_chart(figb, use_container_width=True)

        # Export
        st.download_button("Download Learning Gains CSV",
                           data=lg.to_csv(index=False).encode("utf-8"),
                           file_name="learning_gains.csv", mime="text/csv")


        # === Advanced add-ons under "A) Primary Outcome — Learning Gains" ===
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

        st.markdown("### Advanced Analyses for Learning Gains")

        lg_adv = lg.copy()  # from the section above
        # Ensure numeric
        for c in ["pre","post","learning_gain"]:
            if c in lg_adv.columns:
                lg_adv[c] = pd.to_numeric(lg_adv[c], errors="coerce")

        # 1) ECDF + Summary Table -------------------------------------------------------
        if "learning_gain" in lg_adv.columns:
            st.markdown("**1) ECDF & Summary Table**")
            fig_ecdf = px.ecdf(lg_adv.dropna(subset=["learning_gain"]), x="learning_gain",
                            title="ECDF of Learning Gain (Δ)")
            fig_ecdf.update_layout(template="plotly_white", xaxis_title="Learning Gain (Δ)")
            st.plotly_chart(fig_ecdf, use_container_width=True)

            desc = lg_adv[["learning_gain"]].describe(percentiles=[.1,.25,.5,.75,.9]).reset_index()
            t1 = go.Figure(data=[go.Table(
                header=dict(values=list(desc.columns), fill_color="#2c3e50", font=dict(color="white")),
                cells=dict(values=[desc[c] for c in desc.columns], align="left")
            )])
            t1.update_layout(title="Learning Gain — Descriptives", template="plotly_white")
            st.plotly_chart(t1, use_container_width=True)

        # 2) Bland–Altman agreement (Post vs Pre) --------------------------------------
        if {"pre","post"}.issubset(lg_adv.columns):
            st.markdown(r"""**2) Bland–Altman Plot (Post vs Pre)**  
        Assesses agreement between Pre and Post:  
        $$
        \text{Mean}=\frac{\text{Pre}+\text{Post}}{2},\quad \Delta=\text{Post}-\text{Pre}
        $$
        """)
            ba = lg_adv.dropna(subset=["pre","post"]).copy()
            if not ba.empty:
                ba["mean"] = (ba["pre"] + ba["post"]) / 2.0
                ba["diff"] = ba["post"] - ba["pre"]
                m = ba["diff"].mean(); s = ba["diff"].std(ddof=1)
                loA, hiA = m - 1.96*s, m + 1.96*s
                fig_ba = px.scatter(ba, x="mean", y="diff", title="Bland–Altman Plot (Post−Pre vs Mean)")
                fig_ba.add_hline(y=m, line_dash="dash", annotation_text=f"Mean Δ={m:.2f}")
                fig_ba.add_hline(y=loA, line_dash="dot", annotation_text=f"LoA={loA:.2f}")
                fig_ba.add_hline(y=hiA, line_dash="dot", annotation_text=f"HiA={hiA:.2f}")
                fig_ba.update_layout(template="plotly_white", xaxis_title="Mean of (Pre, Post)", yaxis_title="Post−Pre (Δ)")
                st.plotly_chart(fig_ba, use_container_width=True)

        # 3) Quantile Bands: Δ vs Pre ---------------------------------------------------
        if {"pre","learning_gain"}.issubset(lg_adv.columns):
            st.markdown("**3) Quantile Bands of Δ vs Pre**")
            dfq = lg_adv.dropna(subset=["pre","learning_gain"]).copy()
            if not dfq.empty:
                dfq["_bin"] = pd.qcut(dfq["pre"], q=10, duplicates="drop")
                band = (dfq.groupby("_bin")["learning_gain"]
                            .agg(n="size", q10=lambda s: s.quantile(0.10),
                                q25=lambda s: s.quantile(0.25),
                                q50="median",
                                q75=lambda s: s.quantile(0.75),
                                q90=lambda s: s.quantile(0.90))
                            .reset_index())
                band["pre_mid"] = band["_bin"].apply(lambda iv: (iv.left + iv.right)/2)
                fig_band = go.Figure()
                fig_band.add_trace(go.Scatter(x=band["pre_mid"], y=band["q50"], mode="lines+markers", name="Median (Q50)"))
                fig_band.add_trace(go.Scatter(x=band["pre_mid"], y=band["q75"], mode="lines", name="Q75"))
                fig_band.add_trace(go.Scatter(x=band["pre_mid"], y=band["q25"], mode="lines", name="Q25", fill='tonexty'))
                fig_band.add_trace(go.Scatter(x=band["pre_mid"], y=band["q90"], mode="lines", name="Q90"))
                fig_band.add_trace(go.Scatter(x=band["pre_mid"], y=band["q10"], mode="lines", name="Q10", fill='tonexty', opacity=0.3))
                fig_band.update_layout(template="plotly_white", title="Learning Gain vs Pre — Quantile Bands",
                                    xaxis_title="Pre", yaxis_title="Learning Gain (Δ)")
                st.plotly_chart(fig_band, use_container_width=True)

        # 4) Forest plot by Group (mean Δ ±95% CI) -------------------------------------
        if {"group","learning_gain"}.issubset(lg_adv.columns):
            st.markdown("**4) Forest Plot — Mean Δ by Group (±95% CI)**")
            g = (lg_adv.groupby("group")["learning_gain"]
                    .agg(mean="mean", std="std", count="count").reset_index())
            g["se"] = g["std"] / np.sqrt(g["count"].clip(lower=1))
            g["lo"] = g["mean"] - 1.96*g["se"]
            g["hi"] = g["mean"] + 1.96*g["se"]
            fig_for = go.Figure()
            fig_for.add_trace(go.Scatter(
                x=g["mean"], y=g["group"], mode="markers", error_x=dict(type="data", array=g["hi"]-g["mean"],
                                                                        arrayminus=g["mean"]-g["lo"]),
                name="Mean Δ ±95% CI"
            ))
            fig_for.update_layout(template="plotly_white", title="Forest: Learning Gain by Group",
                                xaxis_title="Mean Δ", yaxis_title="Group")
            st.plotly_chart(fig_for, use_container_width=True)

            t2 = go.Figure(data=[go.Table(
                header=dict(values=list(g.columns), fill_color="#2c3e50", font=dict(color="white")),
                cells=dict(values=[g[c] for c in g.columns], align="left")
            )])
            t2.update_layout(title="Group Summary (Δ)", template="plotly_white")
            st.plotly_chart(t2, use_container_width=True)

        # 5) ANCOVA: Post ~ Group + Pre (OLS) ------------------------------------------
        if HAS_SM and {"post","pre"}.issubset(lg_adv.columns):
            st.markdown(r"""**5) ANCOVA (OLS)**  
        $$
        \text{Post} \sim C(\text{group}) + \text{Pre}
        $$
        Adjusts Post for Pre and estimates group contrasts on adjusted outcome.
        """)
            dfm = lg_adv.dropna(subset=["post","pre"]).copy()
            if "group" in dfm.columns and dfm["group"].nunique() >= 2:
                try:
                    model = smf.ols("post ~ C(group) + pre", data=dfm).fit(cov_type="HC3")
                    st.code(model.summary().as_text(), language="text")
                except Exception as e:
                    st.warning(f"ANCOVA failed: {e}")

        # 6) Outlier table (IQR rule on Δ) ---------------------------------------------
        if "learning_gain" in lg_adv.columns:
            st.markdown("**6) Outlier Table (IQR rule on Δ)**")
            x = pd.to_numeric(lg_adv["learning_gain"], errors="coerce").dropna()
            if len(x) >= 5:
                q1, q3 = x.quantile(0.25), x.quantile(0.75)
                iqr = q3 - q1
                lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
                out = lg_adv[(lg_adv["learning_gain"] < lo) | (lg_adv["learning_gain"] > hi)]
                if not out.empty:
                    cols = [c for c in ["student_id","group","pre","post","learning_gain","course_id","topic"] if c in out.columns]
                    t_out = go.Figure(data=[go.Table(
                        header=dict(values=cols, fill_color="#2c3e50", font=dict(color="white")),
                        cells=dict(values=[out[c] for c in cols], align="left")
                    )])
                    t_out.update_layout(title=f"Outliers on Δ (IQR rule): {len(out)} records", template="plotly_white")
                    st.plotly_chart(t_out, use_container_width=True)
                else:
                    st.caption("No IQR outliers detected for Δ.")

        # 7) Bootstrap 95% CI of Mean Δ (overall & by group) ---------------------------
        if "learning_gain" in lg_adv.columns:
            st.markdown("**7) Bootstrap 95% CI of Mean Δ (overall & by group)**")
            def boot_ci(arr, B=2000, seed=42):
                rng = np.random.default_rng(seed)
                arr = pd.to_numeric(pd.Series(arr), errors="coerce").dropna().values
                if len(arr) < 5: return np.nan, np.nan, np.nan
                boots = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(B)]
                return float(np.mean(boots)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

            mu, lo, hi = boot_ci(lg_adv["learning_gain"])
            if np.isfinite(mu):
                st.caption(f"Overall mean Δ bootstrap 95% CI: mean={mu:.3f}, CI=({lo:.3f}, {hi:.3f})")
            else:
                st.caption("Not enough data for bootstrap.")
            if "group" in lg_adv.columns:
                rows = []
                for gname, sub in lg_adv.groupby("group"):
                    m, l, h = boot_ci(sub["learning_gain"])
                    rows.append(dict(group=gname, mean=m, lo=l, hi=h))
                btab = pd.DataFrame(rows)
                if not btab.empty and btab["mean"].notna().any():
                    fig_ci = go.Figure()
                    fig_ci.add_trace(go.Scatter(
                        x=btab["mean"], y=btab["group"], mode="markers",
                        error_x=dict(type="data", array=btab["hi"]-btab["mean"], arrayminus=btab["mean"]-btab["lo"]),
                        name="Bootstrap 95% CI"
                    ))
                    fig_ci.update_layout(template="plotly_white", title="Bootstrap 95% CI of Mean Δ by Group",
                                        xaxis_title="Mean Δ", yaxis_title="Group")
                    st.plotly_chart(fig_ci, use_container_width=True)

        # 8) Pre–Post Hexbin & Gain-colored Scatter ------------------------------------
        if {"pre","post"}.issubset(lg_adv.columns):
            st.markdown("**8) Pre–Post Hexbin & Gain-Colored Scatter**")
            dfpp = lg_adv.dropna(subset=["pre","post"]).copy()
            if not dfpp.empty:
                dfpp["gain"] = dfpp["post"] - dfpp["pre"]
                fig_hex = px.density_heatmap(dfpp, x="pre", y="post", nbinsx=30, nbinsy=30,
                                            title="Pre vs Post — Density Heatmap")
                fig_hex.add_shape(type="line", x0=dfpp["pre"].min(), y0=dfpp["pre"].min(),
                                x1=dfpp["post"].max(), y1=dfpp["post"].max(), line=dict(dash="dash"))
                fig_hex.update_layout(template="plotly_white")
                st.plotly_chart(fig_hex, use_container_width=True)

                fig_sc = px.scatter(dfpp, x="pre", y="post", color="gain",
                                    color_continuous_scale="RdBu", title="Pre vs Post — Colored by Gain (Δ)")
                fig_sc.add_shape(type="line", x0=dfpp["pre"].min(), y0=dfpp["pre"].min(),
                                x1=dfpp["post"].max(), y1=dfpp["post"].max(), line=dict(dash="dash"))
                fig_sc.update_layout(template="plotly_white")
                st.plotly_chart(fig_sc, use_container_width=True)

        # 9) Small multiples: Gain by Course/Topic (if available) ----------------------
        if "learning_gain" in lg_adv.columns and ("course_id" in lg_adv.columns or "topic" in lg_adv.columns):
            st.markdown("**9) Small Multiples — Gain by Course/Topic**")
            facet_col = "course_id" if "course_id" in lg_adv.columns else "topic"
            dfsm = lg_adv.dropna(subset=["learning_gain"]).copy()
            fig_fac = px.box(dfsm, x="group" if "group" in dfsm.columns else None,
                            y="learning_gain", facet_col=facet_col, facet_col_wrap=4,
                            points="all", title=f"Learning Gain by {facet_col} (faceted)")
            fig_fac.update_layout(template="plotly_white")
            st.plotly_chart(fig_fac, use_container_width=True)

        # 10) Effect sizes vs reference group (Hedges g) --------------------------------
        if {"group","learning_gain"}.issubset(lg_adv.columns) and lg_adv["group"].nunique() >= 2:
            st.markdown(r"""**10) Effect Sizes vs Reference Group (Hedges' g)**  
        $$
        g = J \cdot \frac{\bar{x}_1 - \bar{x}_0}{s_p},\quad
        J = 1 - \frac{3}{4(n_1+n_0) - 9}
        $$
        """)
            def hedges_g(x, y):
                x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().values
                y = pd.to_numeric(pd.Series(y), errors="coerce").dropna().values
                nx, ny = len(x), len(y)
                if nx < 2 or ny < 2: return np.nan
                sp = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / (nx+ny-2))
                d = (np.mean(x) - np.mean(y)) / sp if sp and np.isfinite(sp) else np.nan
                J = 1 - 3/(4*(nx+ny)-9) if (nx+ny) > 2 else 1.0
                return J*d

            ref = lg_adv["group"].value_counts().idxmax()  # largest-N as reference
            rows = []
            for gname, sub in lg_adv.groupby("group"):
                if gname == ref: continue
                gval = hedges_g(lg_adv.loc[lg_adv["group"]==gname, "learning_gain"],
                                lg_adv.loc[lg_adv["group"]==ref,    "learning_gain"])
                rows.append(dict(group=gname, ref_group=ref, hedges_g=gval))
            es = pd.DataFrame(rows)
            if not es.empty:
                fig_es = px.bar(es, x="group", y="hedges_g",
                                title=f"Hedges' g vs Reference Group ({ref})")
                fig_es.update_layout(template="plotly_white", yaxis_title="Hedges' g")
                st.plotly_chart(fig_es, use_container_width=True)

                t_es = go.Figure(data=[go.Table(
                    header=dict(values=list(es.columns), fill_color="#2c3e50", font=dict(color="white")),
                    cells=dict(values=[es[c] for c in es.columns], align="left")
                )])
                t_es.update_layout(title="Effect Sizes vs Reference Group", template="plotly_white")
                st.plotly_chart(t_es, use_container_width=True)
        
# =================================================================
# B) ICC & Cluster Structure — variance partitioning (proxy)
# =================================================================
with tabs[1]:
    st.subheader("B) ICC & Cluster Structure (Proxy)")
    st.markdown(r"""
**Intraclass Correlation (ICC)** quantifies clustering (e.g., class/site).  
One-way random effects (proxy)  
$$
ICC(1) = \frac{MS_B - MS_W}{MS_B + (k-1)MS_W}
$$
where \(MS_B\) is between-group mean square, \(MS_W\) is within-group mean square, and \(k\) is average cluster size.  
""")
    if not has("learning_gains"):
        st.info("Learning gains required.")
    else:
        lg = st.session_state["learning_gains"].copy()
        lg.columns = [c.lower() for c in lg.columns]
        choices = [c for c in ["class_id","section_id","instructor","course_id","group"] if c in lg.columns]
        if not choices:
            st.info("No clustering columns found (e.g., class_id/section_id/instructor/course_id/group).")
        else:
            cl = st.selectbox("Choose clustering column for ICC", choices, index=0)
            icc = quick_icc_oneway(lg["learning_gain"], lg[cl])
            st.metric(f"ICC(1) on Learning Gain (cluster={cl})", "—" if not np.isfinite(icc) else f"{icc:.3f}")
            st.caption("Higher ICC implies stronger clustering; designs should model this in inference.")
        # ================================
        # Advanced Add-Ons for Section B) ICC & Cluster Structure
        # Paste this block **right after** your current ICC section
        # Requires: numpy, pandas, plotly, streamlit, and the helper quick_icc_oneway()
        # ================================
        import numpy as np
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go

        st.markdown("### Advanced ICC & Cluster Diagnostics")

        # Guard: we reuse `lg` (learning_gains dataframe) and selected cluster label `cl`
        # from the section immediately above. If you placed this elsewhere, recreate them.
        if "lg" in locals() and "cl" in locals() and (cl in lg.columns) and ("learning_gain" in lg.columns):
            # Clean & prep
            icc_df = lg[[cl, "learning_gain"]].dropna().copy()
            icc_df["learning_gain"] = pd.to_numeric(icc_df["learning_gain"], errors="coerce")
            icc_df = icc_df.dropna(subset=["learning_gain"])
            if icc_df.empty:
                st.info("No rows available after cleaning for ICC diagnostics.")
            else:
                # ---------- A) ANOVA table & variance components ----------
                st.subheader("B.1) One-way ANOVA Decomposition & Variance Components")
                grand = icc_df["learning_gain"].mean()
                grp_stats = icc_df.groupby(cl)["learning_gain"].agg(n="count", mean="mean", var="var").reset_index()
                # Between & within SS
                ssb = float(((grp_stats["n"] * (grp_stats["mean"] - grand) ** 2).sum()))
                # Within: sum over groups of (n_i-1)*var_i
                ssw = float(((grp_stats["n"] - 1) * grp_stats["var"]).sum())
                G = int(grp_stats.shape[0])               # number of clusters
                N = int(icc_df.shape[0])                  # total N
                dfb = max(G - 1, 0)
                dfw = max(N - G, 0)
                msb = ssb / dfb if dfb > 0 else np.nan
                msw = ssw / dfw if dfw > 0 else np.nan
                k_bar = float(grp_stats["n"].mean()) if G > 0 else np.nan  # average cluster size
                icc1 = (msb - msw) / (msb + (k_bar - 1) * msw) if (np.isfinite(msb) and np.isfinite(msw) and k_bar >= 2) else np.nan

                # Show table
                anova_tbl = pd.DataFrame({
                    "Component": ["Between (Groups)", "Within (Residual)"],
                    "SS": [ssb, ssw],
                    "df": [dfb, dfw],
                    "MS": [msb, msw]
                })
                fig_anova = go.Figure(data=[go.Table(
                    header=dict(values=list(anova_tbl.columns), fill_color="#2c3e50", font=dict(color="white")),
                    cells=dict(values=[anova_tbl[c] for c in anova_tbl.columns], align="left")
                )])
                fig_anova.update_layout(title="One-way ANOVA (Learning Gain ~ Cluster)", template="plotly_white")
                st.plotly_chart(fig_anova, use_container_width=True)

                st.metric("ICC(1) (proxy)", "—" if not np.isfinite(icc1) else f"{icc1:.3f}")
                st.caption("Formula: ICC(1) = (MSB − MSW) / (MSB + (k̄−1)·MSW), where k̄ is the average cluster size.")

                # ---------- B) Cluster size distribution & Pareto (Lorenz-style) ----------
                st.subheader("B.2) Cluster Size Distribution & Coverage")
                fig_sizes = px.histogram(grp_stats, x="n", nbins=min(30, grp_stats["n"].nunique()), title="Cluster Size Distribution")
                fig_sizes.update_layout(template="plotly_white", xaxis_title="Cluster Size (n_i)", yaxis_title="Count of Clusters")
                st.plotly_chart(fig_sizes, use_container_width=True)

                # Lorenz-style cumulative coverage of students by clusters
                g_sorted = grp_stats.sort_values("n", ascending=False).reset_index(drop=True)
                g_sorted["cum_clusters"] = np.arange(1, len(g_sorted) + 1)
                g_sorted["cum_students"] = g_sorted["n"].cumsum()
                g_sorted["prop_clusters"] = g_sorted["cum_clusters"] / g_sorted["cum_clusters"].max()
                g_sorted["prop_students"] = g_sorted["cum_students"] / g_sorted["n"].sum()
                fig_lor = px.line(g_sorted, x="prop_clusters", y="prop_students",
                                title="Cumulative Coverage: %Clusters vs %Students")
                fig_lor.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
                fig_lor.update_layout(template="plotly_white", xaxis_title="% of Clusters", yaxis_title="% of Students")
                st.plotly_chart(fig_lor, use_container_width=True)

                # ---------- C) Caterpillar plot: cluster means with 95% CI ----------
                st.subheader("B.3) Caterpillar Plot — Cluster Means (±95% CI)")
                # SE uses within-group SD if n_i>1; fallback to MSW if necessary
                merged = grp_stats.copy()
                # Standard error for each group mean ~ sqrt( var_i / n_i ), fallback to MSW/n_i
                se_i = np.sqrt(np.where(merged["n"] > 1,
                                        merged["var"] / merged["n"].clip(lower=1),
                                        msw / merged["n"].clip(lower=1)))
                merged["lo"] = merged["mean"] - 1.96 * se_i
                merged["hi"] = merged["mean"] + 1.96 * se_i
                merged = merged.sort_values("mean")
                fig_cat = go.Figure()
                fig_cat.add_trace(go.Scatter(
                    x=merged["mean"], y=merged[cl], mode="markers",
                    error_x=dict(type="data", array=merged["hi"] - merged["mean"],
                                arrayminus=merged["mean"] - merged["lo"]),
                    name="Mean ±95% CI"
                ))
                fig_cat.add_vline(x=grand, line_dash="dash", annotation_text=f"Grand mean={grand:.2f}")
                fig_cat.update_layout(template="plotly_white", title="Caterpillar Plot of Cluster Means",
                                    xaxis_title="Mean Learning Gain (Δ)", yaxis_title="Cluster")
                st.plotly_chart(fig_cat, use_container_width=True)

                # ---------- D) Funnel plot: cluster mean vs size with 95% control limits ----------
                st.subheader("B.4) Funnel Plot — Mean vs Size with Control Limits")
                # Control limits around grand mean using MSW (pooled within variance)
                merged["se_mean"] = np.sqrt(msw / merged["n"].clip(lower=1)) if np.isfinite(msw) else np.nan
                merged["hi_funnel"] = grand + 1.96 * merged["se_mean"]
                merged["lo_funnel"] = grand - 1.96 * merged["se_mean"]
                fig_fun = go.Figure()
                fig_fun.add_trace(go.Scatter(x=merged["n"], y=merged["mean"], mode="markers", name="Clusters"))
                fig_fun.add_trace(go.Scatter(x=merged["n"], y=merged["hi_funnel"], mode="lines", name="+1.96·SE", line=dict(dash="dot")))
                fig_fun.add_trace(go.Scatter(x=merged["n"], y=merged["lo_funnel"], mode="lines", name="-1.96·SE", line=dict(dash="dot")))
                fig_fun.add_hline(y=grand, line_dash="dash", annotation_text="Grand Mean")
                fig_fun.update_layout(template="plotly_white", title="Funnel Plot: Cluster Mean vs Size",
                                    xaxis_title="Cluster Size (n_i)", yaxis_title="Mean Learning Gain (Δ)")
                st.plotly_chart(fig_fun, use_container_width=True)

                # ---------- E) Bootstrap CI for ICC(1) ----------
                st.subheader("B.5) Bootstrap 95% CI for ICC(1)")
                def icc_bootstrap(df, cluster_col, value_col, B=800, seed=42):
                    rng = np.random.default_rng(seed)
                    groups = df[cluster_col].unique().tolist()
                    values_by_g = {g: df.loc[df[cluster_col] == g, value_col].values for g in groups}
                    out = []
                    for _ in range(B):
                        # Cluster bootstrap: resample clusters with replacement, include all their members
                        sample_groups = rng.choice(groups, size=len(groups), replace=True)
                        samp_vals = []
                        samp_lab  = []
                        for g in sample_groups:
                            v = values_by_g[g]
                            samp_vals.append(v)
                            samp_lab.extend([g] * len(v))
                        yb = np.concatenate(samp_vals) if len(samp_vals) else np.array([])
                        if yb.size < 3 or len(set(samp_lab)) < 2:
                            continue
                        # compute ICC on bootstrap sample
                        tmp = pd.DataFrame({cluster_col: samp_lab, value_col: yb})
                        # ANOVA components
                        grand_b = yb.mean()
                        gstat = tmp.groupby(cluster_col)[value_col].agg(n="count", mean="mean", var="var").reset_index()
                        ssb_b = float(((gstat["n"] * (gstat["mean"] - grand_b) ** 2).sum()))
                        ssw_b = float(((gstat["n"] - 1) * gstat["var"]).sum())
                        G_b = int(gstat.shape[0]); N_b = int(tmp.shape[0])
                        dfb_b = max(G_b - 1, 0); dfw_b = max(N_b - G_b, 0)
                        msb_b = ssb_b / dfb_b if dfb_b > 0 else np.nan
                        msw_b = ssw_b / dfw_b if dfw_b > 0 else np.nan
                        kbar_b = float(gstat["n"].mean()) if G_b > 0 else np.nan
                        icc_b = (msb_b - msw_b) / (msb_b + (kbar_b - 1) * msw_b) if (np.isfinite(msb_b) and np.isfinite(msw_b) and kbar_b >= 2) else np.nan
                        if np.isfinite(icc_b):
                            out.append(icc_b)
                    return np.array(out)

                icc_samples = icc_bootstrap(icc_df, cl, "learning_gain", B=800, seed=42)
                if icc_samples.size >= 50:
                    lo, hi = np.percentile(icc_samples, [2.5, 97.5])
                    fig_icc_hist = px.histogram(pd.DataFrame({"ICC1_boot": icc_samples}), x="ICC1_boot", nbins=40,
                                                title="Bootstrap Distribution of ICC(1)")
                    fig_icc_hist.add_vline(x=np.nanmean(icc_samples), line_dash="dash", annotation_text="Mean boot ICC")
                    fig_icc_hist.add_vrect(x0=lo, x1=hi, line_width=0, fillcolor="orange", opacity=0.2,
                                        annotation_text=f"95% CI [{lo:.3f}, {hi:.3f}]")
                    fig_icc_hist.update_layout(template="plotly_white", xaxis_title="ICC(1)")
                    st.plotly_chart(fig_icc_hist, use_container_width=True)
                    st.caption(f"Bootstrap 95% CI for ICC(1): [{lo:.3f}, {hi:.3f}]")
                else:
                    st.caption("Not enough resamples for a stable bootstrap ICC(1) CI.")

                # ---------- F) Design effect & effective sample size ----------
                st.subheader("B.6) Design Effect & Effective Sample Size")
                deff = 1 + (k_bar - 1) * icc1 if (np.isfinite(icc1) and np.isfinite(k_bar)) else np.nan
                n_eff = N / deff if (np.isfinite(deff) and deff > 0) else np.nan
                det = pd.DataFrame({
                    "quantity": ["Total N", "Number of clusters (G)", "Average cluster size (k̄)", "ICC(1)", "Design effect (DEFF)", "Effective N (N/DEFF)"],
                    "value": [N, G, k_bar, icc1, deff, n_eff]
                })
                fig_det = go.Figure(data=[go.Table(
                    header=dict(values=list(det.columns), fill_color="#2c3e50", font=dict(color="white")),
                    cells=dict(values=[det[c] for c in det.columns], align="left")
                )])
                fig_det.update_layout(title="Design Effect & Effective Sample Size", template="plotly_white")
                st.plotly_chart(fig_det, use_container_width=True)

                # ---------- G) Variance homogeneity check (group variances vs size) ----------
                st.subheader("B.7) Cluster Variance vs Size")
                fig_var = px.scatter(grp_stats, x="n", y="var", trendline="ols" if grp_stats.shape[0] > 3 else None,
                                    title="Within-Cluster Variance vs Cluster Size")
                fig_var.update_layout(template="plotly_white", xaxis_title="Cluster Size (n_i)", yaxis_title="Within-Cluster Variance")
                st.plotly_chart(fig_var, use_container_width=True)

                # ---------- H) Compare ICC across multiple candidate clusterings ----------
                st.subheader("B.8) ICC Comparison Across Candidate Clusterings")
                # Allow user to compare ICC for multiple grouping columns (if present)
                candidates = [c for c in ["class_id","section_id","instructor","course_id","group"] if c in lg.columns]
                selected = st.multiselect("Select cluster columns to compare ICC(1):", options=candidates, default=[cl] if cl in candidates else candidates[:1])
                rows_icc = []
                for cc in selected:
                    sub = lg[[cc, "learning_gain"]].dropna()
                    if sub[cc].nunique() >= 2:
                        icc_cc = quick_icc_oneway(sub["learning_gain"], sub[cc])
                        rows_icc.append(dict(cluster_level=cc, ICC1=icc_cc, G=sub[cc].nunique(), N=len(sub)))
                if rows_icc:
                    cmp_tbl = pd.DataFrame(rows_icc)
                    fig_cmp = go.Figure(data=[go.Table(
                        header=dict(values=list(cmp_tbl.columns), fill_color="#2c3e50", font=dict(color="white")),
                        cells=dict(values=[cmp_tbl[c] for c in cmp_tbl.columns], align="left")
                    )])
                    fig_cmp.update_layout(title="ICC(1) Across Candidate Clusterings", template="plotly_white")
                    st.plotly_chart(fig_cmp, use_container_width=True)
        else:
            st.info("The advanced ICC diagnostics require the variables `lg` (learning gains) and the selected cluster column `cl` from the previous block.")

# ===========================================================================
# C) Mixed-Effects (approx) & Effect Size (d_adjusted / Cohen’s d)
# ===========================================================================
with tabs[2]:
    st.subheader("C) Mixed-Effects (approx) & Effect Size")
    st.markdown(r"""
**Planned model (post-test outcome)**  
A three-level mixed model (site / class / student) adjusts for **Pre** as covariate:  
$$
Y_{ijk} = \beta_0 + \beta_1 \text{EDUAI}_{jk} + \beta_2 \text{Pre}_{ijk} + u_k + v_{jk} + \varepsilon_{ijk}
$$
We provide a **safe approximation** here:
- If a binary treatment column (e.g., `treatment` in {0,1}) exists, we run **OLS with cluster-robust SE** (if `statsmodels` is available), adjusting for **Pre**.
- Else, we show **group contrasts** and **Cohen’s d**.
Standardized effect (displayed) follows the plan:  
$$
d_{\text{adjusted}} = \frac{\beta_1}{\sqrt{\tau_j^2 + \tau_k^2 + \sigma^2}}\quad\text{(approximated here with pooled SD)}
$$
""")

    if not has("learning_gains"):
        st.info("Learning gains required.")
    else:
        lg = st.session_state["learning_gains"].copy()
        lg.columns = [c.lower() for c in lg.columns]
        has_pre = "pre" in lg.columns
        has_post = "post" in lg.columns
        has_treat = "treatment" in lg.columns  # 0/1 expected
        cluster_col = next((c for c in ["class_id","section_id","instructor","course_id","group"] if c in lg.columns), None)

        # 1) Regression (if possible)
        if HAS_SM and has_post and has_pre and has_treat:
            st.markdown("**Model:** `post ~ treatment + pre`  (cluster-robust by chosen cluster if available)")
            dfm = lg.dropna(subset=["post","pre","treatment"]).copy()
            if cluster_col and dfm[cluster_col].nunique() > 1:
                model = smf.ols("post ~ treatment + pre", data=dfm).fit(cov_type="cluster", cov_kwds={"groups": dfm[cluster_col]})
                cluster_txt = f"cluster={cluster_col}"
            else:
                model = smf.ols("post ~ treatment + pre", data=dfm).fit(cov_type="HC3")
                cluster_txt = "robust=HC3"
            st.code(model.summary().as_text(), language="text")

            beta1 = model.params.get("treatment", np.nan)
            # approximate denominator with pooled SD of residuals as proxy
            sd_pool = float(np.sqrt(np.nanvar(model.resid, ddof=1)))
            d_adj = beta1 / sd_pool if sd_pool and np.isfinite(sd_pool) else np.nan
            st.metric("Approx. adjusted effect (d_adj)", "—" if not np.isfinite(d_adj) else f"{d_adj:.3f}",
                      help=f"Covariance: {cluster_txt}. This is a proxy for the fully specified multilevel denominator.")

        # 2) Group contrasts (fallback or complement)
        if "group" in lg.columns:
            st.markdown("**Group contrasts (Cohen’s d)**")
            grp = lg.groupby("group")["learning_gain"].agg(["mean","std","count"]).reset_index()
            st.dataframe(grp, use_container_width=True)
            figb = px.bar(grp, x="group", y="mean", error_y="std", title="Mean Learning Gain by Group")
            figb.update_layout(template="plotly_white", xaxis_tickangle=30)
            st.plotly_chart(figb, use_container_width=True)

            # Pairwise Cohen's d
            gvals = grp["group"].tolist()
            rows = []
            for i in range(len(gvals)):
                for j in range(i+1, len(gvals)):
                    gi, gj = gvals[i], gvals[j]
                    di = cohen_d(lg.loc[lg["group"]==gi, "learning_gain"],
                                 lg.loc[lg["group"]==gj, "learning_gain"])
                    # p-value (t-test) if SciPy available
                    if HAS_SCIPY:
                        tstat, pval = stats.ttest_ind(
                            pd.to_numeric(lg.loc[lg["group"]==gi, "learning_gain"], errors="coerce").dropna(),
                            pd.to_numeric(lg.loc[lg["group"]==gj, "learning_gain"], errors="coerce").dropna(),
                            equal_var=False
                        )
                    else:
                        pval = np.nan
                    rows.append(dict(group_i=gi, group_j=gj, cohens_d=di, pval=pval))
            pw = pd.DataFrame(rows)
            if not pw.empty:
                pw["pval_adj_BH_q=.10"] = bh_adjust(pw["pval"].tolist(), q=0.10)
                st.write("Pairwise contrasts (Cohen’s d) with BH-adjusted p-values:")
                st.dataframe(pw, use_container_width=True)

        # ================================================
        # Advanced Add-Ons for Section C) Mixed-Effects & Effect Size
        # Paste this block **right after** your current Section C code
        # Requirements: numpy, pandas, plotly, streamlit
        # Optional (guarded): statsmodels, scipy
        # Reuses: lg (learning_gains), HAS_SM, HAS_SCIPY, cohen_d, bh_adjust
        # ================================================
        import numpy as np
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go
        import streamlit as st

        try:
            import statsmodels.api as sm
            import statsmodels.formula.api as smf
            HAS_SM = True if 'HAS_SM' not in globals() else HAS_SM
        except Exception:
            HAS_SM = False

        try:
            from scipy import stats
            HAS_SCIPY = True if 'HAS_SCIPY' not in globals() else HAS_SCIPY
        except Exception:
            HAS_SCIPY = False

        st.markdown("### Advanced Modeling & Effect Size Diagnostics (Different Figures)")

        # Guard: we reuse `lg` prepared in the block above
        if "lg" in locals():
            CDF = lg.copy()
            CDF.columns = [c.lower() for c in CDF.columns]
            # Make sure numeric
            for c in ["pre","post","learning_gain","treatment"]:
                if c in CDF.columns:
                    CDF[c] = pd.to_numeric(CDF[c], errors="coerce")

            has_pre  = "pre" in CDF.columns
            has_post = "post" in CDF.columns
            has_trt  = "treatment" in CDF.columns and CDF["treatment"].dropna().isin([0,1]).any()

            # ===========================================================
            # C.1 Coefficient Plot (Forest) with 95% CI  — OLS: post ~ treatment + pre
            # ===========================================================
            st.subheader("C.1) Coefficient Forest (OLS with robust SE)")
            if HAS_SM and has_post and has_pre and has_trt:
                dfm = CDF.dropna(subset=["post","pre","treatment"]).copy()
                # Cluster-robust if a cluster column exists
                cluster_col = next((c for c in ["class_id","section_id","instructor","course_id","group"] if c in CDF.columns and CDF[c].nunique()>1), None)
                if cluster_col:
                    model = smf.ols("post ~ treatment + pre", data=dfm).fit(cov_type="cluster", cov_kwds={"groups": dfm[cluster_col]})
                    covar_txt = f"cluster={cluster_col}"
                else:
                    model = smf.ols("post ~ treatment + pre", data=dfm).fit(cov_type="HC3")
                    covar_txt = "robust=HC3"

                # Build coefficient table
                params = model.params
                conf   = model.conf_int()
                se     = model.bse
                out = pd.DataFrame({
                    "term": params.index,
                    "beta": params.values,
                    "lo95": conf[0].values,
                    "hi95": conf[1].values,
                    "SE":   se.values
                })
                # Drop intercept from plot but keep in table
                plot_df = out[out["term"]!="Intercept"].copy()

                fig_coef = go.Figure()
                fig_coef.add_trace(go.Scatter(
                    x=plot_df["beta"], y=plot_df["term"], mode="markers",
                    error_x=dict(type="data", array=plot_df["hi95"]-plot_df["beta"],
                                arrayminus=plot_df["beta"]-plot_df["lo95"]),
                    name="β ± 95% CI"
                ))
                fig_coef.add_vline(x=0, line_dash="dash")
                fig_coef.update_layout(template="plotly_white",
                                    title=f"Coefficient Forest: post ~ treatment + pre  ({covar_txt})",
                                    xaxis_title="Coefficient", yaxis_title="Term")
                st.plotly_chart(fig_coef, use_container_width=True)

                # Full coefficient table
                tcoef = go.Figure(data=[go.Table(
                    header=dict(values=list(out.columns), fill_color="#2c3e50", font=dict(color="white")),
                    cells=dict(values=[out[c] for c in out.columns], align="left")
                )])
                tcoef.update_layout(template="plotly_white", title="Coefficient Table (robust)")
                st.plotly_chart(tcoef, use_container_width=True)
            else:
                st.caption("Need statsmodels + columns: post, pre, treatment (0/1) to show the coefficient forest.")

            # ===========================================================
            # C.2 Added-Variable (Partial Residual) Plots
            # ===========================================================
            st.subheader("C.2) Added-Variable (Partial Residual) Plots")
            st.caption("Visualizes adjusted association after regressing out other predictors.")
            if HAS_SM and has_post and has_pre:
                # Build a base model with any available binary treatment (optional)
                rhs = ["pre"] + (["treatment"] if has_trt else [])
                dfm = CDF.dropna(subset=["post"] + rhs).copy()
                if len(dfm) >= 30:
                    # For term X, partial residual = residual(y|others) vs residual(X|others)
                    def partial_xy(ycol, xcol, Zcols):
                        Y = dfm[ycol]
                        X = dfm[[xcol] + Zcols]
                        # residuals of Y on Z
                        rY = sm.OLS(Y, sm.add_constant(dfm[Zcols])).fit().resid if Zcols else Y - Y.mean()
                        # residuals of X on Z
                        rX = sm.OLS(dfm[xcol], sm.add_constant(dfm[Zcols])).fit().resid if Zcols else dfm[xcol] - dfm[xcol].mean()
                        return pd.DataFrame({"rX": rX, "rY": rY})

                    # Plot for PRE adjusting for TREATMENT (if present) and for TREATMENT adjusting for PRE (if present)
                    if has_pre:
                        Z = ["treatment"] if has_trt else []
                        av_pre = partial_xy("post", "pre", Z)
                        fig_av1 = px.scatter(av_pre, x="rX", y="rY", trendline="ols" if len(av_pre)>30 else None,
                                            title="Added-Variable: PRE (resid POST ~ resid PRE)")
                        fig_av1.update_layout(template="plotly_white", xaxis_title="Residual PRE", yaxis_title="Residual POST")
                        st.plotly_chart(fig_av1, use_container_width=True)
                    if has_trt:
                        Z = ["pre"] if has_pre else []
                        av_trt = partial_xy("post", "treatment", Z)
                        fig_av2 = px.scatter(av_trt, x="rX", y="rY", trendline="ols" if len(av_trt)>30 else None,
                                            title="Added-Variable: TREATMENT (resid POST ~ resid TRT)")
                        fig_av2.update_layout(template="plotly_white", xaxis_title="Residual TREATMENT", yaxis_title="Residual POST")
                        st.plotly_chart(fig_av2, use_container_width=True)
            else:
                st.caption("statsmodels or required columns missing for added-variable plots.")

            # ===========================================================
            # C.3 Interaction Check: post ~ treatment * pre (surface & slices)
            # ===========================================================
            st.subheader("C.3) Interaction Check — post ~ treatment × pre")
            if HAS_SM and has_post and has_pre and has_trt:
                dfm = CDF.dropna(subset=["post","pre","treatment"]).copy()
                try:
                    m_int = smf.ols("post ~ treatment * pre", data=dfm).fit(cov_type="HC3")
                    st.code(m_int.summary().as_text(), language="text")
                    # Slices: predict over range of PRE for treatment=0/1
                    pre_grid = np.linspace(dfm["pre"].quantile(0.05), dfm["pre"].quantile(0.95), 50)
                    pred0 = pd.DataFrame({"pre": pre_grid, "treatment": 0})
                    pred1 = pd.DataFrame({"pre": pre_grid, "treatment": 1})
                    y0 = m_int.predict(pred0); y1 = m_int.predict(pred1)
                    dfp = pd.DataFrame({"pre": np.r_[pre_grid, pre_grid],
                                        "post_hat": np.r_[y0, y1],
                                        "treatment": ["0"]*len(pre_grid) + ["1"]*len(pre_grid)})
                    fig_int = px.line(dfp, x="pre", y="post_hat", color="treatment",
                                    title="Predicted POST across PRE by Treatment",
                                    labels={"treatment":"TRT"})
                    fig_int.update_layout(template="plotly_white")
                    st.plotly_chart(fig_int, use_container_width=True)
                except Exception as e:
                    st.warning(f"Interaction model failed: {e}")
            else:
                st.caption("Need statsmodels + post, pre, treatment for interaction check.")

            # ===========================================================
            # C.4 Influence Diagnostics: Leverage & Cook’s Distance
            # ===========================================================
            st.subheader("C.4) Influence Diagnostics — Leverage & Cook’s D")
            if HAS_SM and has_post and (has_trt or has_pre):
                rhs = ["pre"] + (["treatment"] if has_trt else [])
                dfm = CDF.dropna(subset=["post"] + rhs).copy()
                try:
                    m_diag = smf.ols("post ~ " + " + ".join(rhs), data=dfm).fit()
                    infl = m_diag.get_influence()
                    cooks = infl.cooks_distance[0]
                    hat   = infl.hat_matrix_diag
                    res   = m_diag.resid
                    idx   = np.arange(len(dfm))
                    ddf = pd.DataFrame({"index": idx, "leverage": hat, "cooksD": cooks, "residual": res})
                    fig_cd = px.scatter(ddf, x="leverage", y="cooksD",
                                        title="Influence: Cook’s D vs Leverage",
                                        hover_data=["index","residual"])
                    fig_cd.update_layout(template="plotly_white")
                    st.plotly_chart(fig_cd, use_container_width=True)

                    # Top influential points table
                    topk = ddf.sort_values("cooksD", ascending=False).head(20)
                    fig_top = go.Figure(data=[go.Table(
                        header=dict(values=list(topk.columns), fill_color="#2c3e50", font=dict(color="white")),
                        cells=dict(values=[topk[c] for c in topk.columns], align="left")
                    )])
                    fig_top.update_layout(template="plotly_white", title="Top 20 Observations by Cook’s D")
                    st.plotly_chart(fig_top, use_container_width=True)
                except Exception as e:
                    st.warning(f"Diagnostics failed: {e}")
            else:
                st.caption("Need statsmodels + post and at least one predictor (pre or treatment).")

            # ===========================================================
            # C.5 Randomization/Permutation Test for Treatment Effect on POST
            # ===========================================================
            st.subheader("C.5) Permutation Test — Treatment Effect on POST (adjusting for PRE)")
            if has_post and has_pre and has_trt:
                dfp = CDF.dropna(subset=["post","pre","treatment"]).copy()
                if len(dfp) >= 30:
                    # Residualize POST on PRE, then test mean(resid) difference by T
                    from numpy.random import default_rng
                    rng = default_rng(42)
                    # residualize: post_hat = a + b*pre
                    b = np.polyfit(dfp["pre"].values, dfp["post"].values, deg=1)
                    resid = dfp["post"].values - (b[1] + b[0]*dfp["pre"].values)
                    obs = resid[dfp["treatment"]==1].mean() - resid[dfp["treatment"]==0].mean()

                    B = 2000
                    perm = []
                    trt = dfp["treatment"].values.copy()
                    for _ in range(B):
                        rng.shuffle(trt)
                        perm.append(resid[trt==1].mean() - resid[trt==0].mean())
                    perm = np.array(perm)
                    pval = (np.sum(np.abs(perm) >= np.abs(obs)) + 1) / (B + 1)

                    # Plot permutation distribution
                    h = px.histogram(pd.DataFrame({"perm_diff": perm}), x="perm_diff", nbins=40,
                                    title=f"Permutation Null of Adjusted Mean Difference (p≈{pval:.4f})")
                    h.add_vline(x=obs, line_dash="dash", annotation_text=f"Observed={obs:.3f}")
                    h.update_layout(template="plotly_white")
                    st.plotly_chart(h, use_container_width=True)

                    st.caption(f"Observed adjusted difference (TRT−CTRL) on residual POST: {obs:.3f} (two-sided permutation p≈{pval:.4f}).")
                else:
                    st.caption("Too few rows for a reliable permutation test.")
            else:
                st.caption("Need post, pre, treatment for a permutation test.")

            # ===========================================================
            # C.6 Propensity Score (PS) Diagnostics & Overlap (if covariates exist)
            # ===========================================================
            st.subheader("C.6) Propensity Score Diagnostics & Overlap")
            # Use available simple covariates: pre + optional group/course/topic as dummies
            covars = []
            if has_pre: covars.append("pre")
            for cat in ["group","course_id","topic"]:
                if cat in CDF.columns and CDF[cat].notna().any():
                    covars.append(cat)
            show_ps = HAS_SM and has_trt and len(covars) > 0
            if show_ps:
                dfps = CDF.dropna(subset=["treatment"] + ([c for c in covars if c in CDF.columns])).copy()
                # Build formula for logistic PS: treatment ~ pre + C(group) + C(course_id) + C(topic)
                rhs_terms = []
                if "pre" in covars: rhs_terms.append("pre")
                for cat in ["group","course_id","topic"]:
                    if cat in covars: rhs_terms.append(f"C({cat})")
                form = "treatment ~ " + " + ".join(rhs_terms) if rhs_terms else "treatment ~ 1"
                try:
                    ps_model = smf.logit(form, data=dfps).fit(disp=False)
                    dfps["ps_hat"] = ps_model.predict(dfps)
                    # Overlap histogram
                    fig_ps = px.histogram(dfps, x="ps_hat", color=dfps["treatment"].map({0:"Control",1:"Treatment"}),
                                        barmode="overlay", nbins=30, opacity=0.6,
                                        title="Propensity Score Overlap (Estimated)")
                    fig_ps.update_layout(template="plotly_white", xaxis_title="Propensity Score (P[T=1|X])")
                    st.plotly_chart(fig_ps, use_container_width=True)

                    # Balance table (standardized mean difference on PRE by PS quintiles)
                    dfps["_q"] = pd.qcut(dfps["ps_hat"], q=5, duplicates="drop")
                    bal = (dfps.groupby(["_q","treatment"])["pre"].agg(["mean","std","count"]).reset_index()
                                if "pre" in dfps.columns else pd.DataFrame())
                    if not bal.empty:
                        tbl = go.Figure(data=[go.Table(
                            header=dict(values=list(bal.columns), fill_color="#2c3e50", font=dict(color="white")),
                            cells=dict(values=[bal[c] for c in bal.columns], align="left")
                        )])
                        tbl.update_layout(template="plotly_white", title="Balance by PS Quintiles (PRE only)")
                        st.plotly_chart(tbl, use_container_width=True)
                except Exception as e:
                    st.warning(f"Propensity model failed: {e}")
            else:
                st.caption("Provide treatment plus simple covariates (e.g., pre, group, course_id, topic) to view PS diagnostics.")

            # ===========================================================
            # C.7 Equivalence Test (TOST) for Small Effects on POST
            # ===========================================================
            st.subheader("C.7) Equivalence (TOST) on POST: TRT vs CTRL")
            if HAS_SCIPY and has_post and has_trt:
                dfe = CDF.dropna(subset=["post","treatment"]).copy()
                x = dfe.loc[dfe["treatment"]==1, "post"]
                y = dfe.loc[dfe["treatment"]==0, "post"]
                x = pd.to_numeric(x, errors="coerce").dropna()
                y = pd.to_numeric(y, errors="coerce").dropna()
                if len(x)>=10 and len(y)>=10:
                    # user-specified equivalence margin (Cohen's d-equivalent). Default 0.2 (small).
                    delta = st.number_input("Equivalence margin (Cohen’s d units)", min_value=0.05, max_value=1.0, value=0.2, step=0.05)
                    # Convert to raw-score margin using pooled SD
                    nx, ny = len(x), len(y)
                    sp = np.sqrt(((nx-1)*x.var(ddof=1) + (ny-1)*y.var(ddof=1)) / (nx+ny-2))
                    eps = float(delta * sp)
                    # TOST on mean difference μx−μy within (−eps, +eps)
                    diff = x.mean() - y.mean()
                    se = np.sqrt(x.var(ddof=1)/nx + y.var(ddof=1)/ny)
                    if se > 0:
                        t1 = (diff - (-eps)) / se
                        p1 = 1 - stats.t.cdf(t1, df=nx+ny-2)
                        t2 = (eps - diff) / se
                        p2 = 1 - stats.t.cdf(t2, df=nx+ny-2)
                        p_tost = max(p1, p2)  # conservative
                        st.caption(f"TOST: diff={diff:.3f}, margin=±{eps:.3f} (raw). p≈{p_tost:.4f} (both one-sided tests).")
                    else:
                        st.caption("Cannot compute TOST (zero standard error).")
                else:
                    st.caption("Need ≥10 observations per arm for a stable TOST.")
            else:
                st.caption("SciPy or required columns are missing for TOST.")

            # ===========================================================
            # C.8 Model Fit Diagnostics: Predicted vs Actual & Residual ECDF
            # ===========================================================
            st.subheader("C.8) Model Fit — Predicted vs Actual & Residual ECDF")
            if HAS_SM and has_post and (has_pre or has_trt):
                rhs = []
                if has_trt: rhs.append("treatment")
                if has_pre: rhs.append("pre")
                dfm = CDF.dropna(subset=["post"] + rhs).copy()
                try:
                    m_fit = smf.ols("post ~ " + " + ".join(rhs), data=dfm).fit(cov_type="HC3")
                    dfm["yhat"] = m_fit.predict(dfm)
                    dfm["resid"] = dfm["post"] - dfm["yhat"]
                    fig_pred = px.scatter(dfm, x="yhat", y="post", color="treatment" if has_trt else None,
                                        title="Predicted vs Actual (POST)")
                    fig_pred.add_shape(type="line", x0=dfm["yhat"].min(), y0=dfm["yhat"].min(),
                                    x1=dfm["yhat"].max(), y1=dfm["yhat"].max(), line=dict(dash="dash"))
                    fig_pred.update_layout(template="plotly_white", xaxis_title="Predicted", yaxis_title="Actual")
                    st.plotly_chart(fig_pred, use_container_width=True)

                    fig_res = px.ecdf(dfm, x="resid", title="Residual ECDF (POST model)")
                    fig_res.update_layout(template="plotly_white", xaxis_title="Residual")
                    st.plotly_chart(fig_res, use_container_width=True)
                except Exception as e:
                    st.warning(f"Fit diagnostics failed: {e}")
            else:
                st.caption("Need statsmodels + post + predictors to show fit diagnostics.")

            # ===========================================================
            # C.9 Simple k-Fold Cross-Validation for POST (R²)
            # ===========================================================
            st.subheader("C.9) k-Fold Cross-Validation (R² on POST)")
            if has_post and (has_pre or has_trt):
                rhs = []
                if has_trt: rhs.append("treatment")
                if has_pre: rhs.append("pre")
                dfk = CDF.dropna(subset=["post"] + rhs).copy()
                if len(dfk) >= 30:
                    k = st.slider("Choose k for CV", min_value=3, max_value=10, value=5, step=1)
                    idx = np.arange(len(dfk))
                    rng = np.random.default_rng(123)
                    rng.shuffle(idx)
                    folds = np.array_split(idx, k)
                    r2s = []
                    for f in range(k):
                        test_idx = folds[f]
                        train_idx = np.concatenate([folds[j] for j in range(k) if j != f])
                        tr = dfk.iloc[train_idx]
                        te = dfk.iloc[test_idx]
                        if HAS_SM:
                            mcv = smf.ols("post ~ " + " + ".join(rhs), data=tr).fit()
                            yhat = mcv.predict(te)
                        else:
                            # plain numpy fallback: post = a + b1*treatment + b2*pre
                            Xtr = np.column_stack([np.ones(len(tr))] + [tr[x].values for x in rhs])
                            coef, *_ = np.linalg.lstsq(Xtr, tr["post"].values, rcond=None)
                            Xte = np.column_stack([np.ones(len(te))] + [te[x].values for x in rhs])
                            yhat = Xte @ coef
                        y = te["post"].values
                        ssr = np.sum((y - yhat)**2)
                        sst = np.sum((y - y.mean())**2)
                        r2 = 1 - ssr/sst if sst > 0 else np.nan
                        r2s.append(r2)
                    r2s = [float(x) for x in r2s if np.isfinite(x)]
                    if len(r2s) > 0:
                        fig_r2 = px.box(pd.DataFrame({"R2_CV": r2s}), y="R2_CV",
                                        title=f"{k}-Fold CV R² on POST")
                        fig_r2.update_layout(template="plotly_white")
                        st.plotly_chart(fig_r2, use_container_width=True)
                        st.caption(f"Mean CV R² = {np.mean(r2s):.3f}")
                else:
                    st.caption("Too few rows for stable cross-validation.")
            else:
                st.caption("Need post and at least one predictor (pre or treatment) for CV.")

            # ===========================================================
            # C.10 Subgroup Treatment Effects (Forest by Group/Course)
            # ===========================================================
            st.subheader("C.10) Subgroup Effects — Forest by Group/Course")
            # Compute subgroup differences on POST adjusting for PRE via within-subgroup OLS
            subgroup_candidates = [c for c in ["group","course_id","instructor","section_id"] if c in CDF.columns]
            if has_trt and has_post and has_pre and subgroup_candidates:
                col = st.selectbox("Choose subgrouping variable", subgroup_candidates, index=0)
                rows = []
                for gname, sub in CDF.dropna(subset=["post","pre","treatment"]).groupby(col):
                    if sub["treatment"].nunique() < 2 or len(sub) < 20:
                        continue
                    try:
                        if HAS_SM:
                            m = smf.ols("post ~ treatment + pre", data=sub).fit(cov_type="HC3")
                            b1 = float(m.params.get("treatment", np.nan))
                            se = float(m.bse.get("treatment", np.nan))
                        else:
                            # numpy fallback
                            X = np.column_stack([np.ones(len(sub)), sub["treatment"].values, sub["pre"].values])
                            coef, *_ = np.linalg.lstsq(X, sub["post"].values, rcond=None)
                            # naive SE via residual variance
                            yhat = X @ coef
                            sigma2 = np.sum((sub["post"].values - yhat)**2) / max(len(sub)-X.shape[1], 1)
                            cov = sigma2 * np.linalg.inv(X.T @ X)
                            b1 = float(coef[1]); se = float(np.sqrt(cov[1,1])) if np.isfinite(cov[1,1]) else np.nan
                        lo = b1 - 1.96*se if np.isfinite(se) else np.nan
                        hi = b1 + 1.96*se if np.isfinite(se) else np.nan
                        rows.append(dict(subgroup=gname, beta_trt=b1, se=se, lo95=lo, hi95=hi, n=len(sub)))
                    except Exception:
                        continue
                sgt = pd.DataFrame(rows)
                if not sgt.empty:
                    sgt = sgt.sort_values("beta_trt")
                    fig_sg = go.Figure()
                    fig_sg.add_trace(go.Scatter(
                        x=sgt["beta_trt"], y=sgt["subgroup"], mode="markers",
                        error_x=dict(type="data", array=sgt["hi95"]-sgt["beta_trt"], arrayminus=sgt["beta_trt"]-sgt["lo95"]),
                        name="β_treatment ±95% CI"
                    ))
                    fig_sg.add_vline(x=0, line_dash="dash")
                    fig_sg.update_layout(template="plotly_white", title=f"Subgroup Treatment Effects (adjusted for PRE) by {col}",
                                        xaxis_title="β̂ (treatment)", yaxis_title=col)
                    st.plotly_chart(fig_sg, use_container_width=True)

                    tsg = go.Figure(data=[go.Table(
                        header=dict(values=list(sgt.columns), fill_color="#2c3e50", font=dict(color="white")),
                        cells=dict(values=[sgt[c] for c in sgt.columns], align="left")
                    )])
                    tsg.update_layout(template="plotly_white", title="Subgroup Effect Table")
                    st.plotly_chart(tsg, use_container_width=True)
            else:
                st.caption("Provide treatment, pre, post, and at least one subgroup column (e.g., group, course_id) to view subgroup forests.")
        else:
            st.info("This add-on expects `lg` to be defined in the Section C block above.")
            
# ============================================================
# D) Multiple Outcomes & FDR (BH)
# ============================================================
with tabs[3]:
    st.subheader("D) Multiple Outcomes & FDR")
    st.markdown(r"""
We control multiplicity using **Benjamini–Hochberg (FDR)** across a small family of tests.  
**BH rule** finds the largest \(k\) such that  
$$
p_{(k)} \le \frac{k}{m} q
$$
and declares all \(p_{(i)} \le p_{(k)}\) significant. Here we use \(q=0.10\).
    """)

    tests = []  # collect simple, illustrative tests across modules
    # 1) Learning gain group diff ANOVA
    if has("learning_gains") and "group" in st.session_state["learning_gains"].columns:
        lg = st.session_state["learning_gains"].copy()
        lg.columns = [c.lower() for c in lg.columns]
        if HAS_SCIPY and lg["group"].nunique() >= 2:
            arrays = [pd.to_numeric(lg.loc[lg["group"]==g, "learning_gain"], errors="coerce").dropna().values
                      for g in sorted(lg["group"].dropna().unique())]
            if all(len(a) > 1 for a in arrays):
                F, p = stats.f_oneway(*arrays)
                tests.append(dict(test="ANOVA: gain~group", pval=p))
    # 2) PEI group diff ANOVA
    if has("telemetry_with_pei") and "group" in st.session_state["telemetry_with_pei"].columns:
        tele = st.session_state["telemetry_with_pei"].copy()
        tele.columns = [c.lower() for c in tele.columns]
        if "prompt_evolution_index" in tele.columns and HAS_SCIPY and tele["group"].nunique() >= 2:
            arrays = [pd.to_numeric(tele.loc[tele["group"]==g, "prompt_evolution_index"], errors="coerce").dropna().values
                      for g in sorted(tele["group"].dropna().unique())]
            if all(len(a) > 1 for a in arrays):
                F, p = stats.f_oneway(*arrays)
                tests.append(dict(test="ANOVA: PEI~group", pval=p))
    # 3) Fairness equalized odds gap (simple TPR diff test vs ref)
    if has("fairness_df"):
        fair = st.session_state["fairness_df"].copy()
        fair.columns = [c.lower() for c in fair.columns]
        if {"group","y_true"}.issubset(fair.columns) and (("y_pred" in fair.columns) or ("y_score" in fair.columns)):
            if "y_pred" not in fair.columns and "y_score" in fair.columns:
                fair["_y_pred_thr"] = (pd.to_numeric(fair["y_score"], errors="coerce") >= 0.5).astype(int)
                yhat = "_y_pred_thr"
            else:
                yhat = "y_pred"
            # compute group TPRs and test difference to ref (largest N)
            rows = []
            for g, sub in fair.dropna(subset=["group"]).groupby("group"):
                y = pd.to_numeric(sub["y_true"], errors="coerce").astype(int)
                yp = pd.to_numeric(sub[yhat], errors="coerce").astype(int)
                P = int((y==1).sum())
                TPR = safe_rate(int(((yp==1)&(y==1)).sum()), P)
                rows.append((g, len(sub), P, TPR))
            diag = pd.DataFrame(rows, columns=["group","n","pos","TPR"]).sort_values("n", ascending=False)
            if len(diag) >= 2 and HAS_SCIPY:
                ref = diag.iloc[0]
                # naive z-test for difference in proportions within positives
                for _, r in diag.iloc[1:].iterrows():
                    # For a fair comparison, align on positive cases count; here approximate with available P (may differ)
                    # Use pooled variance from counts of positives (proxy)
                    # If 'pos' is zero in any group, skip test
                    if ref["pos"] > 0 and r["pos"] > 0 and np.isfinite(ref["TPR"]) and np.isfinite(r["TPR"]):
                        p1, p2 = ref["TPR"], r["TPR"]
                        n1, n2 = ref["pos"], r["pos"]
                        p_pool = (p1*n1 + p2*n2) / (n1 + n2)
                        se = np.sqrt(p_pool*(1-p_pool)*(1/n1 + 1/n2))
                        if se > 0:
                            z = (p1 - p2) / se
                            pval = 2*(1 - stats.norm.cdf(abs(z)))
                            tests.append(dict(test=f"TPR gap vs ref ({r['group']})", pval=pval))

    if not tests:
        st.info("Not enough data to assemble a multi-outcome family. Upload/compute gains, PEI, and fairness to use FDR.")
    else:
        df_tests = pd.DataFrame(tests)
        df_tests["pval_adj_BH_q=.10"] = bh_adjust(df_tests["pval"].tolist(), q=0.10)
        st.write("Family of tests and BH-adjusted p-values:")
        st.dataframe(df_tests, use_container_width=True)

        figtab = go.Figure(data=[go.Table(
            header=dict(values=list(df_tests.columns), fill_color="#2c3e50", font=dict(color="white")),
            cells=dict(values=[df_tests[c] for c in df_tests.columns], align="left")
        )])
        figtab.update_layout(title="BH-FDR across outcomes (q=0.10)", template="plotly_white")
        st.plotly_chart(figtab, use_container_width=True)




        # ============================
        # D — Advanced FDR Add-Ons (10 items)
        # Paste right after df_tests is created inside tabs[3]
        # Requires: numpy, pandas, plotly, (optional) scipy
        # Reuses: df_tests, bh_adjust, has(), st.session_state
        # ============================


        # ---- Guard: need df_tests with a 'pval' column
        if "df_tests" in locals() and isinstance(df_tests, pd.DataFrame) and "pval" in df_tests.columns:
            D = df_tests.copy()
            D = D.dropna(subset=["pval"]).reset_index(drop=True)
            m = len(D)

            # === Helper: detect family from test name (gain / PEI / fairness / other)
            def detect_family(s: str) -> str:
                s = (s or "").lower()
                if "gain" in s: return "Gain"
                if "pei" in s: return "PEI"
                if "tpr" in s or "fair" in s or "equalized" in s: return "Fairness"
                return "Other"

            if "family" not in D.columns:
                D["family"] = D["test"].astype(str).apply(detect_family)

            # =====================================================================================
            # 1) P-value Histogram + ECDF with Uniform(0,1) reference
            # =====================================================================================
            st.markdown("#### D.1) P-value Distribution (Histogram + ECDF)")
            fig_p_hist = px.histogram(D, x="pval", nbins=min(30, max(5, D["pval"].nunique())),
                                    title="Histogram of p-values", color="family", barmode="overlay", opacity=0.7)
            fig_p_hist.update_layout(template="plotly_white", xaxis_title="p-value", yaxis_title="Count")
            st.plotly_chart(fig_p_hist, use_container_width=True)

            fig_p_ecdf = px.ecdf(D, x="pval", color="family", title="ECDF of p-values")
            # Add Uniform(0,1) reference line
            fig_p_ecdf.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
            fig_p_ecdf.update_layout(template="plotly_white", xaxis_title="p-value", yaxis_title="F(p)")
            st.plotly_chart(fig_p_ecdf, use_container_width=True)

            # =====================================================================================
            # 2) QQ Plot of p-values vs Uniform(0,1)
            # =====================================================================================
            st.markdown("#### D.2) QQ Plot of p-values vs Uniform(0,1)")
            pp = np.sort(D["pval"].values)
            qq_df = pd.DataFrame({
                "theoretical": np.linspace(0, 1, len(pp), endpoint=False) + 1e-9,
                "empirical": pp
            })
            fig_qq = px.scatter(qq_df, x="theoretical", y="empirical", title="QQ Plot: p-values vs Uniform(0,1)")
            fig_qq.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
            fig_qq.update_layout(template="plotly_white", xaxis_title="Uniform quantiles", yaxis_title="Empirical p-values")
            st.plotly_chart(fig_qq, use_container_width=True)

            # =====================================================================================
            # 3) BH Step-up Curve: p_(i) vs (i/m)·q with threshold k
            # =====================================================================================
            st.markdown("#### D.3) BH Step-up Curve (q = 0.10)")
            q = 0.10
            D_sorted = D.sort_values("pval").reset_index(drop=True)
            D_sorted["rank"] = np.arange(1, len(D_sorted) + 1)
            D_sorted["bh_line"] = (D_sorted["rank"] / m) * q
            # find largest k where p_(k) <= (k/m) q
            k_idx = np.where(D_sorted["pval"].values <= D_sorted["bh_line"].values)[0]
            k_star = int(k_idx.max() + 1) if k_idx.size else 0
            thr_p = float(D_sorted.loc[k_star - 1, "pval"]) if k_star > 0 else np.nan

            fig_bh = go.Figure()
            fig_bh.add_trace(go.Scatter(x=D_sorted["rank"], y=D_sorted["pval"], mode="markers+lines", name="p_(i)"))
            fig_bh.add_trace(go.Scatter(x=D_sorted["rank"], y=D_sorted["bh_line"], mode="lines", name="(i/m)·q", line=dict(dash="dash")))
            if k_star > 0:
                fig_bh.add_vline(x=k_star, line_dash="dot", annotation_text=f"k* = {k_star}")
                fig_bh.add_hline(y=thr_p, line_dash="dot", annotation_text=f"p* = {thr_p:.4f}")
            fig_bh.update_layout(template="plotly_white", title="Benjamini–Hochberg Step-Up", xaxis_title="Rank (i)", yaxis_title="Value")
            st.plotly_chart(fig_bh, use_container_width=True)

            # =====================================================================================
            # 4) Manhattan-style bar of –log10(p) colored by family
            # =====================================================================================
            st.markdown("#### D.4) Manhattan-style Bar of –log10(p) by Test")
            D["minus_log10_p"] = -np.log10(D["pval"].clip(lower=1e-300))
            D["test_ix"] = np.arange(1, len(D) + 1)
            fig_manh = px.bar(D, x="test_ix", y="minus_log10_p", color="family",
                            title="–log10(p) by Test (Manhattan-style)")
            fig_manh.update_layout(template="plotly_white", xaxis_title="Test index", yaxis_title="–log10(p)")
            st.plotly_chart(fig_manh, use_container_width=True)

            # =====================================================================================
            # 5) Holm–Bonferroni vs BH comparison table & decision
            # =====================================================================================
            st.markdown("#### D.5) Holm–Bonferroni vs BH Decisions (q = 0.10)")
            # Holm adjusted p-values
            D_holm = D.sort_values("pval").reset_index(drop=True)
            holm_adj = []
            for i, p in enumerate(D_holm["pval"].values, start=1):
                holm_adj.append(min((m - i + 1) * p, 1.0))
            # enforce monotonicity
            holm_adj = np.maximum.accumulate(holm_adj[::-1])[::-1]
            D_holm["p_holm"] = holm_adj
            # BH adjusted (use your bh_adjust if returns adjusted; here create classic BH q-values)
            # If bh_adjust already added in df_tests as 'pval_adj_BH_q=.10', we recompute generic BH q-value:
            ranks = D_holm["rank"] if "rank" in D_holm.columns else np.arange(1, m+1)
            bh_q = np.minimum.accumulate((m / ranks) * D_holm["pval"].to_numpy()[::-1])[::-1]
            D_holm["p_bh"] = np.clip(bh_q, 0, 1)
            D_holm["BH_signif(q=.10)"] = D_holm["p_bh"] <= q
            D_holm["Holm_signif(q=.10)"] = D_holm["p_holm"] <= q

            fig_holm = go.Figure(data=[go.Table(
                header=dict(values=list(D_holm[["test","pval","p_bh","p_holm","BH_signif(q=.10)","Holm_signif(q=.10)"]].columns),
                            fill_color="#2c3e50", font=dict(color="white")),
                cells=dict(values=[D_holm[c] for c in ["test","pval","p_bh","p_holm","BH_signif(q=.10)","Holm_signif(q=.10)"]],
                        align="left")
            )])
            fig_holm.update_layout(template="plotly_white", title="Holm–Bonferroni vs BH Decisions")
            st.plotly_chart(fig_holm, use_container_width=True)

            # =====================================================================================
            # 6) Storey’s q-value approximation: estimate π0 and q-values
            # =====================================================================================
            st.markdown("#### D.6) Storey–Tibshirani q-values (π₀ estimate)")
            lam = 0.5
            pi0 = min(1.0, np.mean(D["pval"].values > lam) / (1 - lam))
            # q-values: q_i = min_{t >= i} (π0 * m * p_(t) / t)
            Dq = D.sort_values("pval").reset_index(drop=True)
            t_vals = np.arange(1, len(Dq) + 1)
            qvals = (pi0 * m * Dq["pval"].values / t_vals)
            Dq["q_storey"] = np.minimum.accumulate(qvals[::-1])[::-1]
            Dq["q_storey"] = np.clip(Dq["q_storey"], 0, 1)
            st.caption(f"Estimated π₀ ≈ {pi0:.3f} (λ={lam}). Smaller π₀ implies more true signals.")
            fig_q = go.Figure(data=[go.Table(
                header=dict(values=list(Dq[["test","pval","q_storey","family"]].columns),
                            fill_color="#2c3e50", font=dict(color="white")),
                cells=dict(values=[Dq[c] for c in ["test","pval","q_storey","family"]], align="left")
            )])
            fig_q.update_layout(template="plotly_white", title="Storey q-values (approx.)")
            st.plotly_chart(fig_q, use_container_width=True)

            # =====================================================================================
            # 7) Sensitivity: number of rejections vs q (BH curve)
            # =====================================================================================
            st.markdown("#### D.7) Sensitivity: Rejections vs FDR level q (BH)")
            q_grid = np.linspace(0.01, 0.25, 25)
            R = []
            P_sorted = np.sort(D["pval"].values)
            for qx in q_grid:
                line = (np.arange(1, m+1) / m) * qx
                k_idx = np.where(P_sorted <= line)[0]
                k = int(k_idx.max() + 1) if k_idx.size else 0
                R.append(k)
            dfR = pd.DataFrame({"q": q_grid, "rejections": R})
            fig_R = px.line(dfR, x="q", y="rejections", markers=True,
                            title="BH Rejections vs FDR level q")
            fig_R.update_layout(template="plotly_white", xaxis_title="q", yaxis_title="# rejections")
            st.plotly_chart(fig_R, use_container_width=True)

            # =====================================================================================
            # 8) Family-wise view: –log10(p) per family with thresholds
            # =====================================================================================
            st.markdown("#### D.8) –log10(p) by Family (faceted)")
            fig_fac = px.bar(D, x="test", y="minus_log10_p", color="family", facet_col="family",
                            facet_col_wrap=3, title="–log10(p) Faceted by Family")
            fig_fac.update_layout(template="plotly_white", xaxis_title="Test", yaxis_title="–log10(p)")
            st.plotly_chart(fig_fac, use_container_width=True)

            # ========================= D.9) Overlap of Discoveries (robust) =========================
            st.markdown("#### D.9) Overlap of Discoveries by Family (configurable)")

            # Controls
            colq = st.columns(3)
            with colq[0]:
                q_level = st.slider("FDR level q", min_value=0.01, max_value=0.25, value=0.10, step=0.01)
            with colq[1]:
                corr_method = st.selectbox("Correction", ["BH (step-up)", "Storey q-values"], index=0)
            with colq[2]:
                topk_fallback = st.slider("Top-K per family (fallback)", min_value=2, max_value=10, value=5, step=1,
                                        help="Shown if no discoveries at chosen q.")

            # Prepare p-values & families
            D9 = D.sort_values("pval").reset_index(drop=True)
            m9 = len(D9)
            if m9 == 0:
                st.caption("No tests available for overlaps.")
            else:
                # Decide significance set
                if corr_method.startswith("BH"):
                    ranks = np.arange(1, m9 + 1)
                    line = (ranks / m9) * q_level
                    pass_idx = np.where(D9["pval"].to_numpy() <= line)[0]
                    k_star = int(pass_idx.max() + 1) if pass_idx.size else 0
                    sig_mask = np.zeros(m9, dtype=bool)
                    if k_star > 0:
                        sig_mask[:k_star] = True
                    sig_df = D9.loc[sig_mask, ["test","family","pval"]]
                else:
                    # Storey q-values (π0 at λ=.5)
                    lam = 0.5
                    pi0 = min(1.0, np.mean(D9["pval"].to_numpy() > lam) / (1 - lam))
                    t_vals = np.arange(1, m9 + 1)
                    qvals = (pi0 * m9 * D9["pval"].to_numpy() / t_vals)
                    qvals = np.minimum.accumulate(qvals[::-1])[::-1]
                    D9["q_storey"] = np.clip(qvals, 0, 1)
                    sig_df = D9.loc[D9["q_storey"] <= q_level, ["test","family","pval"]]

                # If there are discoveries, show true overlap
                if not sig_df.empty:
                    fam_sets = {fam: set(sig_df.loc[sig_df["family"]==fam, "test"]) for fam in sig_df["family"].unique()}
                    families = sorted(fam_sets.keys())
                    # Pairwise overlap counts
                    mat = np.zeros((len(families), len(families)), dtype=int)
                    for i, fi in enumerate(families):
                        for j, fj in enumerate(families):
                            mat[i, j] = len(fam_sets[fi].intersection(fam_sets[fj]))
                    heat = go.Figure(data=go.Heatmap(z=mat, x=families, y=families, colorscale="Blues"))
                    heat.update_layout(template="plotly_white", title=f"Overlap of Significant Tests (q={q_level:.2f}, {corr_method})",
                                    xaxis_title="Family", yaxis_title="Family")
                    st.plotly_chart(heat, use_container_width=True)

                    cnt = pd.DataFrame({"family": families, "n_signif": [len(fam_sets[f]) for f in families]})
                    fig_cnt = px.bar(cnt, x="family", y="n_signif",
                                    title=f"Significant Tests per Family (q={q_level:.2f}, {corr_method})")
                    fig_cnt.update_layout(template="plotly_white", yaxis_title="# significant")
                    st.plotly_chart(fig_cnt, use_container_width=True)

                    st.dataframe(sig_df.reset_index(drop=True), use_container_width=True)
                else:
                    # No discoveries → near-miss view
                    st.warning(f"No discoveries at q={q_level:.2f} using {corr_method}. Showing near-miss overlap and top-K per family.")
                    # Near-miss threshold: twice the BH line OR top-k by smallest p within each family
                    ranks = np.arange(1, m9 + 1)
                    near_line = (ranks / m9) * (q_level * 2.0)  # “looser” cut for visualization only
                    near_mask = D9["pval"].to_numpy() <= near_line
                    near_df = D9.loc[near_mask, ["test","family","pval"]]
                    # Ensure each family contributes at least topK if near_df is still empty/small
                    if near_df.empty:
                        near_df = (D9
                                .sort_values(["family","pval"])
                                .groupby("family")
                                .head(topk_fallback)
                                .loc[:, ["test","family","pval"]])

                    fam_sets = {fam: set(near_df.loc[near_df["family"]==fam, "test"]) for fam in near_df["family"].unique()}
                    families = sorted(fam_sets.keys())
                    mat = np.zeros((len(families), len(families)), dtype=int)
                    for i, fi in enumerate(families):
                        for j, fj in enumerate(families):
                            mat[i, j] = len(fam_sets[fi].intersection(fam_sets[fj]))

                    heat = go.Figure(data=go.Heatmap(z=mat, x=families, y=families, colorscale="Purples"))
                    heat.update_layout(template="plotly_white",
                                    title=f"Near-Miss Overlap (visual only; q≈{q_level:.2f} loosened)",
                                    xaxis_title="Family", yaxis_title="Family")
                    st.plotly_chart(heat, use_container_width=True)

                    # Top-K per family table
                    topk = (D9.sort_values(["family","pval"])
                            .groupby("family")
                            .head(topk_fallback)
                            .reset_index(drop=True))
                    fig_topk = go.Figure(data=[go.Table(
                        header=dict(values=list(topk[["family","test","pval"]].columns),
                                    fill_color="#2c3e50", font=dict(color="white")),
                        cells=dict(values=[topk[c] for c in ["family","test","pval"]], align="left")
                    )])
                    fig_topk.update_layout(template="plotly_white",
                                        title=f"Top-{topk_fallback} Smallest p per Family (fallback view)")
                    st.plotly_chart(fig_topk, use_container_width=True)
            # =====================================================================================
            # 10) Effect-size table + heatmap (if we can derive effects from source data)
            # =====================================================================================
            st.markdown("#### D.10) Effect-Size Table & Heatmap (derived from modules)")
            eff_rows = []

            # 10a) Gain: max pairwise Cohen's d across groups (largest vs next) as a single summary
            if has("learning_gains") and "group" in st.session_state["learning_gains"].columns:
                lgx = st.session_state["learning_gains"].copy()
                lgx.columns = [c.lower() for c in lgx.columns]
                grps = [g for g in lgx["group"].dropna().unique()]
                if len(grps) >= 2:
                    # choose two largest-N groups
                    order = lgx["group"].value_counts().index.tolist()
                    g1, g2 = order[0], order[1]
                    x = pd.to_numeric(lgx.loc[lgx["group"]==g1, "learning_gain"], errors="coerce").dropna().values
                    y = pd.to_numeric(lgx.loc[lgx["group"]==g2, "learning_gain"], errors="coerce").dropna().values
                    if len(x) >= 5 and len(y) >= 5:
                        nx, ny = len(x), len(y)
                        sp = np.sqrt(((nx-1)*x.var(ddof=1) + (ny-1)*y.var(ddof=1)) / (nx+ny-2))
                        d = (x.mean() - y.mean()) / sp if sp > 0 else np.nan
                        eff_rows.append(dict(outcome="Gain", contrast=f"{g1} vs {g2}", effect=d))

            # 10b) PEI: same approach if group exists
            if has("telemetry_with_pei") and "group" in st.session_state["telemetry_with_pei"].columns:
                telx = st.session_state["telemetry_with_pei"].copy()
                telx.columns = [c.lower() for c in telx.columns]
                if "prompt_evolution_index" in telx.columns and telx["group"].nunique() >= 2:
                    order = telx["group"].value_counts().index.tolist()
                    g1, g2 = order[0], order[1]
                    x = pd.to_numeric(telx.loc[telx["group"]==g1, "prompt_evolution_index"], errors="coerce").dropna().values
                    y = pd.to_numeric(telx.loc[telx["group"]==g2, "prompt_evolution_index"], errors="coerce").dropna().values
                    if len(x) >= 5 and len(y) >= 5:
                        nx, ny = len(x), len(y)
                        sp = np.sqrt(((nx-1)*x.var(ddof=1) + (ny-1)*y.var(ddof=1)) / (nx+ny-2))
                        d = (x.mean() - y.mean()) / sp if sp > 0 else np.nan
                        eff_rows.append(dict(outcome="PEI", contrast=f"{g1} vs {g2}", effect=d))

            # 10c) Fairness: TPR diff vs reference group
            if has("fairness_df"):
                fx = st.session_state["fairness_df"].copy()
                fx.columns = [c.lower() for c in fx.columns]
                if {"group","y_true"}.issubset(fx.columns) and (("y_pred" in fx.columns) or ("y_score" in fx.columns)):
                    if "y_pred" not in fx.columns and "y_score" in fx.columns:
                        fx["_y_pred_thr"] = (pd.to_numeric(fx["y_score"], errors="coerce") >= 0.5).astype(int)
                        yhat = "_y_pred_thr"
                    else:
                        yhat = "y_pred"
                    rows = []
                    for g, sub in fx.dropna(subset=["group"]).groupby("group"):
                        y = pd.to_numeric(sub["y_true"], errors="coerce").astype(int)
                        yp = pd.to_numeric(sub[yhat], errors="coerce").astype(int)
                        P = int((y==1).sum())
                        TP = int(((yp==1)&(y==1)).sum())
                        tpr = TP / P if P > 0 else np.nan
                        rows.append((g, tpr, P))
                    T = pd.DataFrame(rows, columns=["group","TPR","P"]).sort_values("P", ascending=False)
                    if len(T) >= 2 and np.isfinite(T.iloc[0]["TPR"]) and np.isfinite(T.iloc[1]["TPR"]):
                        refg = T.iloc[0]["group"]
                        eff_rows.append(dict(outcome="Fairness (TPR)", contrast=f"{refg} vs next", effect=float(T.iloc[0]["TPR"] - T.iloc[1]["TPR"])))

            if eff_rows:
                E = pd.DataFrame(eff_rows)
                fig_et = go.Figure(data=[go.Table(
                    header=dict(values=list(E.columns), fill_color="#2c3e50", font=dict(color="white")),
                    cells=dict(values=[E[c] for c in E.columns], align="left")
                )])
                fig_et.update_layout(template="plotly_white", title="Derived Effect Sizes (summary)")
                st.plotly_chart(fig_et, use_container_width=True)

                # Heatmap (normalize effect magnitudes)
                pivot = E.pivot_table(index="outcome", columns="contrast", values="effect", aggfunc="mean")
                heat2 = go.Figure(data=go.Heatmap(z=pivot.values,
                                                x=pivot.columns.astype(str),
                                                y=pivot.index.astype(str),
                                                colorscale="RdBu", zmid=0))
                heat2.update_layout(template="plotly_white", title="Effect Size Heatmap (derived)")
                st.plotly_chart(heat2, use_container_width=True)
            else:
                st.caption("Not enough module data to derive effect sizes.")
        else:
            st.info("No df_tests with p-values available yet for advanced FDR visuals.")
# ============================================================
# E) Mediation (PEI / RDS  →  Learning Gain)
# ============================================================
with tabs[4]:
    st.subheader("E) Mediation (Process → Outcome)")
    st.markdown(r"""
**Concept:** Does EDU-AI impact gains **through** process indicators (PEI/RDS)?  
Within-student (simplified) product-of-coefficients approach:  
$$
\text{Indirect} = a \times b,\;\; a:\; M \sim T,\;\; b:\; \Delta \sim M\;(\text{and }T)
$$
We report OLS coefficients with robust SE if possible. (*MSEM is ideal; this is a pragmatic in-app approximation.*)
""")
    # ---- libs for modeling (must exist before any HAS_SM checks) ----
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        HAS_SM = True
    except Exception:
        HAS_SM = False

    import numpy as np
    # We need per-student mediator(s) and outcomes
    if not (has("learning_gains") and has("telemetry_with_pei")):
        st.info("Need both learning gains and telemetry (with PEI/RDS).")
    else:
        lg = st.session_state["learning_gains"].copy()
        tele = st.session_state["telemetry_with_pei"].copy()
        lg.columns = [c.lower() for c in lg.columns]
        tele.columns = [c.lower() for c in tele.columns]
        # join on student_id if possible
        if "student_id" not in lg.columns or "student_id" not in tele.columns:
            st.info("Mediation requires 'student_id' in both tables.")
        else:
            # ---- Recover/compute missing columns in tele; make join selection safe ----
            def ensure_col(df, name, value=None):
                if name not in df.columns:
                    df[name] = value
                return df

            # Recover 'group' if missing (rename or merge from assessments if available)
            if "group" not in tele.columns:
                if "condition" in tele.columns:
                    tele = tele.rename(columns={"condition": "group"})
                elif "arm" in tele.columns:
                    tele = tele.rename(columns={"arm": "group"})
                elif "assessments" in st.session_state and \
                    {"student_id","group"}.issubset(st.session_state["assessments"].columns.str.lower()):
                    # pull from assessments if provided in session_state
                    A = st.session_state["assessments"].copy()
                    A.columns = [c.lower() for c in A.columns]
                    gmap = A[["student_id","group"]].dropna().drop_duplicates("student_id", keep="last")
                    tele = tele.merge(gmap, on="student_id", how="left")
                else:
                    tele = ensure_col(tele, "group", "Unknown")

            # Compute mediator fallbacks if missing
            if "prompt_evolution_index" not in tele.columns and "prompt" in tele.columns:
                STRATEGY = {"plan","debug","optimize","compare","analyze","verify","refactor","test"}
                CONSTRAINT = {"must","include","exactly","at least","no more than","use","ensure","without","limit","constrain"}
                def _pei(p):
                    if not isinstance(p, str): p = ""
                    toks = [t for t in p.split() if t.isalpha()]
                    lex = 0.0 if not toks else (0.2 + 0.75*len(set(toks))/max(1,len(toks)))
                    sv  = sum(1 for w in p.lower().split() if w in STRATEGY)
                    cd  = sum(p.lower().count(c) for c in CONSTRAINT)
                    return round(0.3*lex + 0.35*(min(sv,6)/6) + 0.35*(min(cd,6)/6), 3)
                tele["prompt_evolution_index"] = tele["prompt"].astype(str).apply(_pei)

            if "rds_proxy" not in tele.columns:
                txt = tele.get("ai_response", "").astype(str) + " " + tele.get("prompt","").astype(str)
                s = txt.str.lower()
                cues = ["because","therefore","hence","justify","so that","however","evidence"]
                cue_counts = sum(s.str.count(rf"\b{c}\b") for c in cues)
                tokens = s.str.split().str.len()
                rds = (cue_counts // 2) + (tokens > 40).astype(int)
                tele["rds_proxy"] = rds.clip(lower=0, upper=4)

            # Select only columns that actually exist to avoid KeyError
            want = ["student_id","prompt_evolution_index","rds_proxy","group"]
            have = [c for c in want if c in tele.columns]
            tele_sel = tele[have].drop_duplicates("student_id")
            joined = lg.merge(tele_sel, on="student_id", how="inner")
            st.write("Joined view (head):")
            st.dataframe(joined.head(20), use_container_width=True)

            has_treat = "treatment" in lg.columns
            # Choose mediator
            med = st.selectbox("Choose mediator", [c for c in ["prompt_evolution_index","rds_proxy"] if c in joined.columns], index=0)
            # Regressions (if statsmodels available); else fallbacks
            results = {}
            if HAS_SM:
                # a-path: M ~ T (or ~ group as proxy if no T)
                if has_treat:
                    a_mod = smf.ols(f"{med} ~ treatment", data=joined).fit(cov_type="HC3")
                elif "group" in joined.columns:
                    a_mod = smf.ols(f"{med} ~ C(group)", data=joined).fit(cov_type="HC3")
                else:
                    a_mod = None
                # b-path: Δ ~ M (+T)
                if has_treat:
                    b_mod = smf.ols(f"learning_gain ~ {med} + treatment", data=joined).fit(cov_type="HC3")
                else:
                    b_mod = smf.ols(f"learning_gain ~ {med}", data=joined).fit(cov_type="HC3")
                if a_mod:
                    st.markdown("**a-path model**")
                    st.code(a_mod.summary().as_text(), language="text")
                    a = a_mod.params.get("treatment", np.nan) if has_treat else np.nan
                else:
                    st.info("No treatment column; using b-path only.")
                    a = np.nan
                st.markdown("**b-path model**")
                st.code(b_mod.summary().as_text(), language="text")
                b = b_mod.params.get(med, np.nan)
                indirect = a*b if np.isfinite(a) and np.isfinite(b) else np.nan
                st.metric("Indirect (a×b)", "—" if not np.isfinite(indirect) else f"{indirect:.3f}")
            else:
                st.info("statsmodels not available; showing correlations as a proxy.")
                a = joined["learning_gain"].corr(joined[med])
                st.metric("Corr(Δ, mediator)", "—" if not np.isfinite(a) else f"{a:.3f}")




        # ================================
        # E — Advanced Mediation Add-Ons (10 new figures/tables)
        # Paste this **at the end of your `with tabs[4]:` block**, AFTER your base mediation code
        # Requirements: numpy, pandas, plotly; optional: statsmodels, scipy
        # Reuses: st.session_state, joined (merge of gains+telemetry), med (chosen mediator), has_treat
        # ================================


        # Try optional libs
        try:
            import statsmodels.api as sm
            import statsmodels.formula.api as smf
            HAS_SM = True if 'HAS_SM' not in globals() else HAS_SM
        except Exception:
            HAS_SM = False

        try:
            from scipy import stats
            HAS_SCIPY = True if 'HAS_SCIPY' not in globals() else HAS_SCIPY
        except Exception:
            HAS_SCIPY = False

        st.markdown("### Advanced Mediation Diagnostics (PEI/RDS → Gain)")

        # --- Rebuild context if needed (defensive) ---
        _ok_context = True
        if 'joined' not in locals() or 'med' not in locals():
            _ok_context = False
            # Try to reconstruct
            if "learning_gains" in st.session_state and "telemetry_with_pei" in st.session_state:
                lg_ = st.session_state["learning_gains"].copy()
                te_ = st.session_state["telemetry_with_pei"].copy()
                lg_.columns = [c.lower() for c in lg_.columns]
                te_.columns = [c.lower() for c in te_.columns]
                if "student_id" in lg_.columns and "student_id" in te_.columns:
                    joined = lg_.merge(te_[["student_id","prompt_evolution_index","rds_proxy","group"]].drop_duplicates("student_id"),
                                    on="student_id", how="inner")
                    # pick default mediator
                    med = "prompt_evolution_index" if "prompt_evolution_index" in joined.columns else (
                        "rds_proxy" if "rds_proxy" in joined.columns else None)
                    has_treat = "treatment" in lg_.columns
                    _ok_context = med is not None

        if not _ok_context:
            st.info("Mediation add-ons need merged data (`joined`) and a chosen mediator (`med`).")
        else:
            # Clean numeric
            if "learning_gain" in joined.columns:
                joined["learning_gain"] = pd.to_numeric(joined["learning_gain"], errors="coerce")
            if "pre" in joined.columns:
                joined["pre"] = pd.to_numeric(joined["pre"], errors="coerce")
            joined[med] = pd.to_numeric(joined[med], errors="coerce")
            if "treatment" in joined.columns:
                joined["treatment"] = pd.to_numeric(joined["treatment"], errors="coerce")

            # ===================== helper functions =====================
            def fit_ols_formula(formula, data, cluster=None):
                if not HAS_SM:
                    return None
                try:
                    if cluster is not None and cluster in data.columns and data[cluster].nunique() > 1:
                        return smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data[cluster]})
                    return smf.ols(formula, data=data).fit(cov_type="HC3")
                except Exception:
                    return None

            def sobel_se(a, sa, b, sb):
                # Sobel standard error for a*b
                return np.sqrt((b**2)*(sa**2) + (a**2)*(sb**2))

            def boot_indirect(df, med_col, y_col="learning_gain", t_col="treatment", B=2000, seed=42, cluster=None):
                """Cluster-aware bootstrap (cluster on provided column if available, else row bootstrap)."""
                rng = np.random.default_rng(seed)
                inds = []
                # Bootstrap indices by cluster if cluster column present
                if cluster and cluster in df.columns and df[cluster].nunique() > 1:
                    groups = df[cluster].dropna().unique().tolist()
                    members = {g: df.index[df[cluster]==g].to_numpy() for g in groups}
                    for _ in range(B):
                        # resample clusters, then take all rows in that cluster (with replacement across clusters)
                        chosen = rng.choice(groups, size=len(groups), replace=True)
                        idx = np.concatenate([members[g] for g in chosen]) if len(chosen)>0 else np.array([], dtype=int)
                        inds.append(idx)
                else:
                    n = len(df)
                    for _ in range(B):
                        inds.append(rng.integers(0, n, n))
                ab_vals = []
                for idx in inds:
                    if len(idx) < 10:  # skip tiny resamples
                        continue
                    s = df.iloc[idx]
                    a, b = np.nan, np.nan
                    # a-path
                    if t_col in s.columns and s[t_col].notna().any():
                        m_a = fit_ols_formula(f"{med_col} ~ {t_col}", s, cluster=cluster)
                        if m_a is not None and (t_col in m_a.params.index):
                            a = float(m_a.params[t_col])
                    # b-path
                    rhs = f"{med_col} + {t_col}" if (t_col in s.columns and s[t_col].notna().any()) else med_col
                    m_b = fit_ols_formula(f"{y_col} ~ {rhs}", s, cluster=cluster)
                    if m_b is not None and (med_col in m_b.params.index):
                        b = float(m_b.params[med_col])
                    if np.isfinite(a) and np.isfinite(b):
                        ab_vals.append(a*b)
                return np.array(ab_vals, dtype=float)

            cluster_col = next((c for c in ["class_id","section_id","instructor","course_id","group"] if c in joined.columns), None)

            st.markdown("#### E.1) Mediator & Outcome Distributions by Group/Treatment")
            cols = [c for c in ["group","treatment"] if c in joined.columns]
            if med in joined.columns:
                fig_box = px.box(joined, x=cols[0] if cols else None, y=med, color=cols[0] if cols else None,
                                points="all", title=f"Distribution of Mediator ({med})")
                fig_box.update_layout(template="plotly_white", yaxis_title=med)
                st.plotly_chart(fig_box, use_container_width=True)
            if "learning_gain" in joined.columns:
                fig_hist = px.histogram(joined, x="learning_gain", nbins=30, color=cols[0] if cols else None,
                                        title="Histogram of Learning Gain (Δ)")
                fig_hist.update_layout(template="plotly_white", xaxis_title="Δ (Post−Pre)")
                st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown("#### E.2) Scatter with Trend — Δ vs Mediator")
            if "learning_gain" in joined.columns and med in joined.columns:
                fig_sc = px.scatter(joined, x=med, y="learning_gain",
                                    color=cols[0] if cols else None, trendline="ols",
                                    title=f"Δ ~ {med} (with OLS trend)")
                fig_sc.update_layout(template="plotly_white", xaxis_title=med, yaxis_title="Learning Gain (Δ)")
                st.plotly_chart(fig_sc, use_container_width=True)

            st.markdown(r"""#### E.3) Path Coefficients Table (a, b, c, c′)
        **a**: \(M \sim T\) , **b**: \(\Delta \sim M(+T)\) , **c**: \(\Delta \sim T\) , **c′**: \(\Delta \sim T + M\)
        """)
            rows = []
            if HAS_SM:
                # a-path
                a_beta, a_se = np.nan, np.nan
                if has_treat and "treatment" in joined.columns:
                    a_mod = fit_ols_formula(f"{med} ~ treatment", joined, cluster=cluster_col)
                    if a_mod is not None and ("treatment" in a_mod.params.index):
                        a_beta = float(a_mod.params["treatment"]); a_se = float(a_mod.bse["treatment"])
                # b, c, c'
                b_beta, b_se, c_beta, c_se, cprime_beta, cprime_se = [np.nan]*6
                b_rhs = f"{med} + treatment" if has_treat and "treatment" in joined.columns else med
                b_mod = fit_ols_formula(f"learning_gain ~ {b_rhs}", joined, cluster=cluster_col)
                if b_mod is not None and (med in b_mod.params.index):
                    b_beta = float(b_mod.params[med]); b_se = float(b_mod.bse[med])
                if has_treat and "treatment" in joined.columns:
                    c_mod = fit_ols_formula("learning_gain ~ treatment", joined, cluster=cluster_col)
                    if c_mod is not None and ("treatment" in c_mod.params.index):
                        c_beta = float(c_mod.params["treatment"]); c_se = float(c_mod.bse["treatment"])
                    cprime_mod = fit_ols_formula(f"learning_gain ~ {med} + treatment", joined, cluster=cluster_col)
                    if cprime_mod is not None and ("treatment" in cprime_mod.params.index):
                        cprime_beta = float(cprime_mod.params["treatment"]); cprime_se = float(cprime_mod.bse["treatment"])

                path_tbl = pd.DataFrame({
                    "path": ["a: M~T", "b: Δ~M(+T)", "c: Δ~T", "c′: Δ~T+M"],
                    "beta": [a_beta, b_beta, c_beta, cprime_beta],
                    "SE":   [a_se,   b_se,   c_se,   cprime_se]
                })
                fig_path = go.Figure(data=[go.Table(
                    header=dict(values=list(path_tbl.columns), fill_color="#2c3e50", font=dict(color="white")),
                    cells=dict(values=[path_tbl[c] for c in path_tbl.columns], align="left")
                )])
                fig_path.update_layout(template="plotly_white", title="Path Coefficients Summary")
                st.plotly_chart(fig_path, use_container_width=True)
            else:
                st.caption("statsmodels not available — path coefficient table skipped.")

            st.markdown("#### E.4) Sobel Test (a×b) for Indirect Effect (proxy)")
            if HAS_SM and has_treat and "treatment" in joined.columns:
                if 'a_mod' in locals() and 'b_mod' in locals() and a_mod is not None and b_mod is not None:
                    a_hat = a_mod.params.get("treatment", np.nan); sa = a_mod.bse.get("treatment", np.nan)
                    b_hat = b_mod.params.get(med, np.nan); sb = b_mod.bse.get(med, np.nan)
                    if np.isfinite(a_hat) and np.isfinite(b_hat) and np.isfinite(sa) and np.isfinite(sb):
                        ab = a_hat * b_hat
                        se_ab = sobel_se(a_hat, sa, b_hat, sb)
                        if np.isfinite(se_ab) and se_ab > 0:
                            z = ab / se_ab
                            p = 2*(1 - stats.norm.cdf(abs(z))) if HAS_SCIPY else np.nan
                            st.caption(f"Sobel: indirect={ab:.4f}, SE={se_ab:.4f}, z≈{z:.2f}, p≈{p:.4f}")
                    else:
                        st.caption("Sobel not computed — insufficient estimates.")
                else:
                    st.caption("Sobel not computed — missing a/b models.")
            else:
                st.caption("Need statsmodels + treatment for Sobel test.")

            st.markdown("#### E.5) Bootstrap Distribution of Indirect Effect (a×b)")
            if HAS_SM and has_treat and "treatment" in joined.columns:
                ab_samp = boot_indirect(joined.dropna(subset=[med, "learning_gain", "treatment"]),
                                        med_col=med, y_col="learning_gain", t_col="treatment",
                                        B=1000, seed=123, cluster=cluster_col)
                if ab_samp.size >= 50:
                    lo, hi = np.percentile(ab_samp, [2.5, 97.5])
                    fig_boot = px.histogram(pd.DataFrame({"ab": ab_samp}), x="ab", nbins=40,
                                            title=f"Bootstrap Indirect Effect a×b (95% CI: [{lo:.3f}, {hi:.3f}])")
                    fig_boot.add_vline(x=np.mean(ab_samp), line_dash="dash", annotation_text=f"mean={np.mean(ab_samp):.3f}")
                    fig_boot.add_vrect(x0=lo, x1=hi, line_width=0, fillcolor="orange", opacity=0.2)
                    fig_boot.update_layout(template="plotly_white", xaxis_title="a×b")
                    st.plotly_chart(fig_boot, use_container_width=True)
                else:
                    st.caption("Not enough bootstrap resamples/rows to show distribution.")
            else:
                st.caption("Provide treatment & enable statsmodels for bootstrap indirect effect.")

            st.markdown("#### E.6) Moderator Check — Indirect by Group (Bootstrap CI)")
            if HAS_SM and "group" in joined.columns and has_treat and "treatment" in joined.columns:
                rows = []
                for g, sub in joined.dropna(subset=[med, "learning_gain", "treatment"]).groupby("group"):
                    if len(sub) < 25 or sub["treatment"].nunique() < 2:
                        continue
                    ab = boot_indirect(sub, med_col=med, y_col="learning_gain", t_col="treatment",
                                    B=600, seed=7, cluster=cluster_col)
                    if ab.size >= 50:
                        m, lo, hi = float(np.mean(ab)), float(np.percentile(ab, 2.5)), float(np.percentile(ab, 97.5))
                        rows.append(dict(group=g, indirect=m, lo95=lo, hi95=hi, n=len(sub)))
                if rows:
                    T = pd.DataFrame(rows).sort_values("indirect")
                    fig_for = go.Figure()
                    fig_for.add_trace(go.Scatter(
                        x=T["indirect"], y=T["group"], mode="markers",
                        error_x=dict(type="data", array=T["hi95"]-T["indirect"], arrayminus=T["indirect"]-T["lo95"]),
                        name="Indirect ±95% CI"
                    ))
                    fig_for.add_vline(x=0, line_dash="dash")
                    fig_for.update_layout(template="plotly_white", title="Indirect Effect by Group (Bootstrap)",
                                        xaxis_title="a×b", yaxis_title="Group")
                    st.plotly_chart(fig_for, use_container_width=True)
                else:
                    st.caption("Not enough data per group to estimate moderated mediation.")
            else:
                st.caption("Group column or treatment not available — skipping moderated mediation.")

            # ---- E.7) Quantile Slices — Δ across Mediator Quintiles (patched for Interval -> str) ----
            st.markdown("#### E.7) Quantile Slices — Δ across Mediator Quintiles")

            if med in joined.columns and "learning_gain" in joined.columns:
                qdf = joined.dropna(subset=[med, "learning_gain"]).copy()
                if not qdf.empty:
                    # Create quintile bins
                    qdf["_Q"] = pd.qcut(qdf[med], q=5, duplicates="drop")

                    # Group summary
                    summ = qdf.groupby("_Q", observed=True)["learning_gain"].agg(["mean", "std", "count"]).reset_index()

                    # OPTION A: Use human-readable string labels for x-axis
                    def _iv_to_str(iv):
                        try:
                            return f"[{iv.left:.3g}, {iv.right:.3g})"
                        except Exception:
                            return str(iv)

                    summ["_Q_label"] = summ["_Q"].astype(str)  # or: summ["_Q"].map(_iv_to_str)

                    # Plot with string x-axis (no Interval objects)
                    fig_q = px.bar(
                        summ, x="_Q_label", y="mean", error_y="std",
                        title=f"Learning Gain by Quintiles of {med}"
                    )
                    fig_q.update_layout(template="plotly_white", xaxis_title=f"{med} quintile", yaxis_title="Mean Δ")
                    st.plotly_chart(fig_q, use_container_width=True)

                    # (Optional) Show the table with both interval and label
                    st.dataframe(summ[["_Q_label", "mean", "std", "count"]], use_container_width=True)

                    # OPTION B (numeric x-axis): use bin midpoints instead of labels
                    # summ["_Q_mid"] = summ["_Q"].apply(lambda iv: 0.5*(iv.left + iv.right))
                    # fig_q = px.line(summ, x="_Q_mid", y="mean", markers=True,
                    #                 title=f"Learning Gain by Quintiles of {med} (bin midpoints)")
                    # fig_q.update_layout(template="plotly_white", xaxis_title=f"{med} (quintile midpoints)", yaxis_title="Mean Δ")
                    # st.plotly_chart(fig_q, use_container_width=True)

            
            st.markdown("#### E.8) Partial Residual Plot (Δ ~ mediator | pre)")
            if HAS_SM and "pre" in joined.columns and med in joined.columns and "learning_gain" in joined.columns:
                dfm = joined.dropna(subset=["learning_gain", med, "pre"]).copy()
                if len(dfm) > 20:
                    # residual(Δ | pre) vs residual(med | pre)
                    rY = sm.OLS(dfm["learning_gain"], sm.add_constant(dfm[["pre"]])).fit().resid
                    rX = sm.OLS(dfm[med], sm.add_constant(dfm[["pre"]])).fit().resid
                    pr = pd.DataFrame({"resid_med": rX, "resid_gain": rY})
                    fig_pr = px.scatter(pr, x="resid_med", y="resid_gain", trendline="ols",
                                        title=f"Partial Residuals: Δ ~ {med} | pre")
                    fig_pr.update_layout(template="plotly_white", xaxis_title=f"Residual {med}", yaxis_title="Residual Δ")
                    st.plotly_chart(fig_pr, use_container_width=True)
                else:
                    st.caption("Too few rows for stable partial residuals.")

            st.markdown("#### E.9) Missingness Pattern (Key Columns)")
            keys = ["student_id", med, "learning_gain", "treatment", "pre", "group"]
            cols_exist = [c for c in keys if c in joined.columns]
            if cols_exist:
                miss = joined[cols_exist].isna().mean().reset_index()
                miss.columns = ["column","missing_rate"]
                fig_miss = px.bar(miss, x="column", y="missing_rate",
                                title="Missingness Rate by Column")
                fig_miss.update_layout(template="plotly_white", yaxis_tickformat=".0%")
                st.plotly_chart(fig_miss, use_container_width=True)
                st.dataframe(miss, use_container_width=True)

            st.markdown("#### E.10) Sensitivity Grid (What-if attenuation of a and b)")
            st.caption("Explores how indirect = (α·â)×(β·b̂) changes if true paths were attenuated by factors α, β ∈ [0.5, 1.5].")
            a_hat, b_hat = np.nan, np.nan
            if HAS_SM and has_treat and "treatment" in joined.columns:
                # reuse earlier fits if available; otherwise fit quickly
                try:
                    a_m = fit_ols_formula(f"{med} ~ treatment", joined, cluster=cluster_col)
                    b_m = fit_ols_formula(f"learning_gain ~ {med} + treatment", joined, cluster=cluster_col)
                    if a_m is not None and ("treatment" in a_m.params.index):
                        a_hat = float(a_m.params["treatment"])
                    if b_m is not None and (med in b_m.params.index):
                        b_hat = float(b_m.params[med])
                except Exception:
                    pass
            if np.isfinite(a_hat) and np.isfinite(b_hat):
                alpha = np.linspace(0.5, 1.5, 41)
                beta  = np.linspace(0.5, 1.5, 41)
                A, B = np.meshgrid(alpha, beta, indexing="ij")
                Z = (A * a_hat) * (B * b_hat)
                fig_sens = go.Figure(data=go.Heatmap(
                    z=Z, x=beta, y=alpha, colorscale="RdBu", zmid=0))
                fig_sens.update_layout(template="plotly_white",
                                    title="Sensitivity of Indirect Effect to Path Attenuation",
                                    xaxis_title="β multiplier on b̂", yaxis_title="α multiplier on â")
                st.plotly_chart(fig_sens, use_container_width=True)
            else:
                st.caption("Cannot build sensitivity grid — missing â or b̂ (need treatment and statsmodels).")






        # ============================
        # E) Mediation — robust pre-join builder
        # Paste THIS BLOCK **inside with tabs[4]:**, immediately AFTER your intro markdown
        # and BEFORE the advanced add-ons. It guarantees `joined` and `med` exist.
        # ============================
        import numpy as np
        import pandas as pd
        import streamlit as st
        import plotly.express as px

        # ---------- 0) Helpers ----------
        def _rds_proxy(text: str)->int:
            if not isinstance(text, str): text=""
            cues=["because","therefore","hence","justify","so that","however","evidence"]
            score=sum(c in text.lower() for c in cues)
            if len(text.split())>40: score+=1
            return min(4,max(0,score//2))

        def _compute_pei(prompt: str) -> float:
            STRATEGY = {"plan","debug","optimize","compare","analyze","verify","refactor","test"}
            CONSTRAINT = {"must","include","exactly","at least","no more than","use","ensure","without","limit","constrain"}
            if not isinstance(prompt, str): prompt = ""
            toks = [t for t in prompt.split() if t.isalpha()]
            lex = 0.0 if not toks else (0.2 + 0.75*len(set(toks))/len(toks))
            sv = sum(1 for w in prompt.lower().split() if w in STRATEGY)
            cd = sum(prompt.lower().count(c) for c in CONSTRAINT)
            return round(0.3*lex + 0.35*(min(sv,6)/6) + 0.35*(min(cd,6)/6), 3)

        def _first_nonempty(series: pd.Series):
            # convenient for picking an id-like column automatically
            for name in ["student_id","sid","stu_id","user_id","id"]:
                if name in series.index: 
                    return name
            return None

        # ---------- 1) Pull inputs (assess/gains + telemetry) ----------
        if "learning_gains" not in st.session_state:
            st.warning("Learning gains missing. Compute them on the Assessments page first.")
            st.stop()

        lg_src = st.session_state["learning_gains"].copy()
        lg_src.columns = [c.lower() for c in lg_src.columns]

        # Try to get per-student mediators from telemetry_with_pei if present; else build from tele_df
        tele_ready = "telemetry_with_pei" in st.session_state
        if not tele_ready and "tele_df" in st.session_state:
            tele = st.session_state["tele_df"].copy()
            tele.columns = [c.lower() for c in tele.columns]
            # best-effort compute mediators
            if "prompt_evolution_index" not in tele.columns and "prompt" in tele.columns:
                tele["prompt_evolution_index"] = tele["prompt"].astype(str).apply(_compute_pei)
            if "rds_proxy" not in tele.columns:
                src_col = "reflection_text" if "reflection_text" in tele.columns else ("ai_response" if "ai_response" in tele.columns else None)
                tele["rds_proxy"] = tele[src_col].astype(str).apply(_rds_proxy) if src_col else np.nan
            st.session_state["telemetry_with_pei"] = tele
            tele_ready = True

        if not tele_ready:
            st.warning("Telemetry with PEI/RDS not found. Upload telemetry or compute PEI/RDS on the Telemetry page.")
            st.stop()

        tele_src = st.session_state["telemetry_with_pei"].copy()
        tele_src.columns = [c.lower() for c in tele_src.columns]

        # ---------- 2) Let user map ID columns if 'student_id' is missing ----------
        st.markdown("**ID Mapping for Mediation Join**")

        lg_id_candidates = [c for c in lg_src.columns if c in ["student_id","sid","stu_id","user_id","id"]]
        tele_id_candidates = [c for c in tele_src.columns if c in ["student_id","sid","stu_id","user_id","id"]]

        c1, c2 = st.columns(2)
        with c1:
            lg_id_col = st.selectbox(
                "Choose ID column in learning gains table",
                options=lg_src.columns.tolist(),
                index=lg_src.columns.get_loc(lg_id_candidates[0]) if lg_id_candidates else 0
            )
        with c2:
            tele_id_col = st.selectbox(
                "Choose ID column in telemetry table",
                options=tele_src.columns.tolist(),
                index=tele_src.columns.get_loc(tele_id_candidates[0]) if tele_id_candidates else 0
            )

        # Standardize to 'student_id'
        lg_src = lg_src.rename(columns={lg_id_col: "student_id"})
        tele_src = tele_src.rename(columns={tele_id_col: "student_id"})

        # Coerce ID to string (to avoid merge mismatches like 001 vs 1)
        # Ensure student_id is a single column, not a DataFrame
        if isinstance(lg_src["student_id"], pd.DataFrame):
            lg_src["student_id"] = lg_src["student_id"].iloc[:, 0]

        if isinstance(tele_src["student_id"], pd.DataFrame):
            tele_src["student_id"] = tele_src["student_id"].iloc[:, 0]

        # Now safely coerce and strip whitespace
        lg_src["student_id"] = lg_src["student_id"].astype(str).str.strip()
        tele_src["student_id"] = tele_src["student_id"].astype(str).str.strip()
        # ---------- 3) Reduce telemetry to per-student mediator rows ----------
        mediator_cols = [c for c in ["prompt_evolution_index","rds_proxy"] if c in tele_src.columns]
        if not mediator_cols:
            st.error("No mediator columns found in telemetry (need 'prompt_evolution_index' and/or 'rds_proxy').")
            st.stop()

        # Keep one row per student by averaging mediators; keep 1 group label if present
        agg = tele_src.groupby("student_id")[mediator_cols].mean().reset_index()
        if "group" in tele_src.columns:
            gmap = tele_src[["student_id","group"]].dropna().drop_duplicates("student_id", keep="last")
            agg = agg.merge(gmap, on="student_id", how="left")

        # ---------- 4) Build the join ----------
        # Ensure learning_gains has 'learning_gain'; if not, try to compute Δ from pre/post
        if "learning_gain" not in lg_src.columns and {"pre","post"}.issubset(set(lg_src.columns)):
            lg_src["learning_gain"] = pd.to_numeric(lg_src["post"], errors="coerce") - pd.to_numeric(lg_src["pre"], errors="coerce")

        if "learning_gain" not in lg_src.columns:
            st.error("Learning gains table must contain 'learning_gain' or ('pre' and 'post') to compute Δ.")
            st.stop()

        # Merge
        joined = lg_src.merge(agg, on="student_id", how="inner")

        # ---------- 5) Let user pick mediator and (optional) treatment ----------
        med_options = [c for c in ["prompt_evolution_index","rds_proxy"] if c in joined.columns]
        med = st.selectbox("Choose mediator (M)", options=med_options, index=0)
        has_treat = "treatment" in joined.columns

        # ---------- 6) Display diagnostics ----------
        st.success(f"Joined N = {len(joined)} students (inner join on student_id).")
        show_cols = ["student_id", "learning_gain"]

        if {"pre","post"}.issubset(joined.columns):
            show_cols += ["pre","post"]

        show_cols.append(med)
        if "group" in joined.columns:
            show_cols.append("group")

        st.dataframe(
            joined[show_cols].head(20),
            use_container_width=True
        )

        # ID overlap chart
        counts = pd.DataFrame({
            "source": ["learning_gains","telemetry"],
            "n_unique_ids": [lg_src["student_id"].nunique(), tele_src["student_id"].nunique()]
        })
        fig_ids = px.bar(counts, x="source", y="n_unique_ids", title="Unique IDs by Source (pre-join)")
        fig_ids.update_layout(template="plotly_white", yaxis_title="# unique student_id")
        st.plotly_chart(fig_ids, use_container_width=True)

        # Save to session_state so downstream add-ons can rely on them even on rerun
        st.session_state["med_joined"] = joined
        st.session_state["med_var"] = med
        st.session_state["med_has_treat"] = has_treat

        # Also expose to local scope for immediate use by add-ons placed below
        # (Streamlit executes top-to-bottom on each rerun; this keeps compatibility with earlier add-on code)
        # joined, med, has_treat already defined above
# ============================================================
# F) Dose–Response (Usage → Post or ∆)
# ============================================================
with tabs[5]:
    st.subheader("F) Dose–Response (Usage Intensity → Outcome)")
    st.markdown(r"""
We explore a smooth/curved effect of **usage intensity** on **Post** or **Δ**:  
$$
Y = \beta_0 + f(U) + \varepsilon
$$
In-app approximation: quadratic trend \(Y \approx \beta_0 + \beta_1 U + \beta_2 U^2\) with confidence band by bootstrap.  
""")
    if not (has("learning_gains") and has("telemetry_with_pei")):
        st.info("Need learning gains and telemetry.")
    else:
        lg = st.session_state["learning_gains"].copy()
        tele = st.session_state["telemetry_with_pei"].copy()
        lg.columns = [c.lower() for c in lg.columns]
        tele.columns = [c.lower() for c in tele.columns]
        if "student_id" not in lg.columns or "student_id" not in tele.columns:
            st.info("Require 'student_id' in both tables.")
        else:
            # Build a usage proxy if none present: number of telemetry rows per student
            usage_col = next((c for c in ["usage_minutes","turns","sessions","events","interaction_count"] if c in tele.columns), None)
            if usage_col is None:
                usage = tele.groupby("student_id").size().rename("usage_proxy").reset_index()
            else:
                usage = tele.groupby("student_id")[usage_col].sum().rename("usage_proxy").reset_index()

            dfu = lg.merge(usage, on="student_id", how="inner").dropna(subset=["usage_proxy"])
            y_target = st.selectbox("Outcome", [c for c in ["learning_gain","post"] if c in dfu.columns],
                                    index=0)
            if dfu.empty:
                st.info("No overlap after join.")
            else:
                # Fit quadratic
                X = dfu["usage_proxy"].astype(float).values
                Y = pd.to_numeric(dfu[y_target], errors="coerce").values
                ok = np.isfinite(X) & np.isfinite(Y)
                X, Y = X[ok], Y[ok]
                if len(X) < 10:
                    st.info("Too few points to fit a curve.")
                else:
                    Xc = (X - X.mean()) / (X.std() if X.std() else 1)
                    A = np.column_stack([np.ones_like(Xc), Xc, Xc**2])
                    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
                    pred = A @ coef

                    # Bootstrap CI (light)
                    rng = np.random.default_rng(42)
                    boots = []
                    for _ in range(300):
                        idx = rng.integers(0, len(Xc), len(Xc))
                        bcoef, *_ = np.linalg.lstsq(A[idx], Y[idx], rcond=None)
                        boots.append(A @ bcoef)
                    boots = np.vstack(boots)
                    lo = np.percentile(boots, 2.5, axis=0)
                    hi = np.percentile(boots, 97.5, axis=0)

                    df_plot = pd.DataFrame(dict(usage=X, y=Y, yhat=pred, lo=lo, hi=hi))
                    df_plot = df_plot.sort_values("usage")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_plot["usage"], y=df_plot["y"],
                                             mode="markers", name="Observed"))
                    fig.add_trace(go.Scatter(x=df_plot["usage"], y=df_plot["yhat"],
                                             mode="lines", name="Quadratic fit"))
                    fig.add_trace(go.Scatter(x=df_plot["usage"].tolist()+df_plot["usage"].tolist()[::-1],
                                             y=df_plot["hi"].tolist()+df_plot["lo"].tolist()[::-1],
                                             fill="toself", line=dict(width=0),
                                             name="95% CI", hoverinfo="skip", opacity=0.2))
                    fig.update_layout(template="plotly_white",
                                      title=f"Dose–Response: Usage → {y_target}",
                                      xaxis_title="Usage (proxy)", yaxis_title=y_target)
                    st.plotly_chart(fig, use_container_width=True)



        # ================================
        # F — Advanced Dose–Response Add-Ons (10 NEW figures/tables)
        # Paste this block **at the end of your `with tabs[5]:` section**, AFTER your current curve fit/plot.
        # It is robust: it reconstructs data if needed and only renders when inputs exist.
        # Requirements: numpy, pandas, plotly; optional: statsmodels, scipy
        # ================================
        import numpy as np
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go
        import streamlit as st

        # ---------- Try optional libs ----------
        try:
            import statsmodels.api as sm
            import statsmodels.formula.api as smf
            HAS_SM = True if 'HAS_SM' not in globals() else HAS_SM
        except Exception:
            HAS_SM = False

        try:
            from scipy import stats
            HAS_SCIPY = True if 'HAS_SCIPY' not in globals() else HAS_SCIPY
        except Exception:
            HAS_SCIPY = False

        # ---------- Rebuild context defensively ----------
        ok_ctx = "learning_gains" in st.session_state and "telemetry_with_pei" in st.session_state
        if not ok_ctx:
            st.info("Dose–response add-ons need learning gains and telemetry.")
        else:
            lg_ = st.session_state["learning_gains"].copy()
            te_ = st.session_state["telemetry_with_pei"].copy()
            lg_.columns = [c.lower() for c in lg_.columns]
            te_.columns = [c.lower() for c in te_.columns]
            if "student_id" not in lg_.columns or "student_id" not in te_.columns:
                st.info("Require 'student_id' in both tables.")
            else:
                # Build usage proxy if none present
                usage_col = next((c for c in ["usage_minutes","turns","sessions","events","interaction_count"] if c in te_.columns), None)
                if usage_col is None:
                    usage = te_.groupby("student_id").size().rename("usage_proxy").reset_index()
                else:
                    usage = te_.groupby("student_id")[usage_col].sum().rename("usage_proxy").reset_index()

                dfu = lg_.merge(usage, on="student_id", how="inner").dropna(subset=["usage_proxy"]).copy()
                # Select outcome (prefer ∆; fall back to post if present)
                y_target = "learning_gain" if "learning_gain" in dfu.columns else ("post" if "post" in dfu.columns else None)
                if y_target is None or dfu.empty:
                    st.info("No outcome available (need learning_gain or post) after join.")
                else:
                    # Numerics
                    dfu["usage_proxy"] = pd.to_numeric(dfu["usage_proxy"], errors="coerce")
                    dfu[y_target] = pd.to_numeric(dfu[y_target], errors="coerce")
                    dfu = dfu.dropna(subset=["usage_proxy", y_target]).copy()
                    has_group = "group" in dfu.columns
                    has_treat = "treatment" in dfu.columns

                    # Centered usage for polynomial designs
                    X = dfu["usage_proxy"].to_numpy()
                    Y = dfu[y_target].to_numpy()
                    if len(X) < 10:
                        st.caption("Too few rows for advanced dose–response diagnostics.")
                    else:
                        Xc = (X - X.mean()) / (X.std() if X.std() else 1.0)

                        # ===============================================================
                        # F.1 Distribution diagnostics: usage & outcome (dual axes)
                        # ===============================================================
                        st.markdown("#### F.1) Usage & Outcome Distributions")
                        fig_f1 = go.Figure()
                        hist_u = np.histogram(X, bins=min(40, max(10, int(np.sqrt(len(X))))))
                        fig_f1.add_trace(go.Bar(x=(hist_u[1][:-1]+hist_u[1][1:])/2, y=hist_u[0],
                                                name="Usage histogram", opacity=0.6))
                        fig_f1.update_layout(template="plotly_white", title="Usage Distribution (counts)",
                                            xaxis_title="Usage proxy", yaxis_title="Count")
                        st.plotly_chart(fig_f1, use_container_width=True)

                        fig_f1b = px.histogram(dfu, x=y_target, nbins=30, title=f"{y_target} Distribution")
                        fig_f1b.update_layout(template="plotly_white", xaxis_title=y_target)
                        st.plotly_chart(fig_f1b, use_container_width=True)

                        # ===============================================================
                        # F.2 Decile binning: mean outcome with 95% CI
                        # ===============================================================
                        st.markdown("#### F.2) Outcome by Usage Deciles (mean ± 95% CI)")
                        q = pd.qcut(dfu["usage_proxy"], q=10, duplicates="drop")
                        gb = dfu.groupby(q)[y_target]
                        dec = gb.agg(["mean","std","count"]).reset_index()
                        dec["se"] = dec["std"] / np.sqrt(dec["count"].clip(lower=1))
                        dec["lo95"] = dec["mean"] - 1.96*dec["se"]
                        dec["hi95"] = dec["mean"] + 1.96*dec["se"]
                        dec["mid"]  = dec["usage_proxy"].apply(lambda i: (i.left + i.right)/2)
                        fig_f2 = go.Figure()
                        fig_f2.add_trace(go.Scatter(x=dec["mid"], y=dec["mean"], mode="lines+markers", name="Mean"))
                        fig_f2.add_trace(go.Scatter(x=dec["mid"].tolist()+dec["mid"].tolist()[::-1],
                                                    y=dec["hi95"].tolist()+dec["lo95"].tolist()[::-1],
                                                    fill="toself", line=dict(width=0), opacity=0.2,
                                                    name="95% CI", hoverinfo="skip"))
                        fig_f2.update_layout(template="plotly_white", title="Decile Means with 95% CI",
                                            xaxis_title="Usage (bin mid)", yaxis_title=f"Mean {y_target}")
                        st.plotly_chart(fig_f2, use_container_width=True)

                        # ===============================================================
                        # F.3 Partial residuals (Δ/post ~ usage + usage²)
                        # ===============================================================
                        st.markdown("#### F.3) Partial Residual Plot (quadratic model)")
                        if HAS_SM:
                            dfm = pd.DataFrame({"y": Y, "x": Xc})
                            m = sm.OLS(dfm["y"], sm.add_constant(np.c_[dfm["x"], dfm["x"]**2])).fit()
                            # partial residual for x: r = resid + beta_x * x
                            bx = m.params[1] if len(m.params) > 1 else np.nan
                            r = m.resid + (bx * dfm["x"])
                            pr = pd.DataFrame({"partial_resid_x": r, "x": dfm["x"]})
                            fig_f3 = px.scatter(pr, x="x", y="partial_resid_x", trendline="ols",
                                                title="Partial Residuals vs Centered Usage")
                            fig_f3.update_layout(template="plotly_white", xaxis_title="Centered usage", yaxis_title="Partial residual")
                            st.plotly_chart(fig_f3, use_container_width=True)
                        else:
                            st.caption("statsmodels not available — partial residual skipped.")

                        # ===============================================================
                        # F.4 Piecewise (segmented) quadratic: brute breakpoint search
                        # ===============================================================
                        st.markdown("#### F.4) Segmented Fit (one breakpoint, brute search)")
                        xs = dfu["usage_proxy"].to_numpy()
                        ys = dfu[y_target].to_numpy()
                        order = np.argsort(xs)
                        xs, ys = xs[order], ys[order]
                        # Candidate breakpoints at deciles 30%-70%
                        cand = np.quantile(xs, np.linspace(0.3, 0.7, 9))
                        best = None
                        for bp in cand:
                            left = xs <= bp
                            right = ~left
                            if left.sum() < 10 or right.sum() < 10:
                                continue
                            # Fit simple linear on each side (stable)
                            Xl = np.c_[np.ones(left.sum()), xs[left]]
                            Xr = np.c_[np.ones(right.sum()), xs[right]]
                            bl, *_ = np.linalg.lstsq(Xl, ys[left], rcond=None)
                            br, *_ = np.linalg.lstsq(Xr, ys[right], rcond=None)
                            pred = np.empty_like(ys)
                            pred[left] = Xl @ bl
                            pred[right]= Xr @ br
                            sse = float(np.sum((ys - pred)**2))
                            if (best is None) or (sse < best[0]):
                                best = (sse, bp, bl, br, pred)
                        if best is not None:
                            _, bp, bl, br, pred = best
                            fig_f4 = go.Figure()
                            fig_f4.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name="Observed"))
                            fig_f4.add_trace(go.Scatter(x=xs, y=pred, mode="lines", name=f"Piecewise fit (bp={bp:.2f})"))
                            fig_f4.add_vline(x=bp, line_dash="dash")
                            fig_f4.update_layout(template="plotly_white", title="Piecewise Linear Dose–Response",
                                                xaxis_title="Usage", yaxis_title=y_target)
                            st.plotly_chart(fig_f4, use_container_width=True)

                        # ===============================================================
                        # F.5 Group facets (if group exists): LOESS-like via rolling median
                        # ===============================================================
                        st.markdown("#### F.5) Group-wise Smoothed Curves (faceted)")
                        if has_group and dfu["group"].nunique() >= 2:
                            rows = []
                            for g, sub in dfu.groupby("group"):
                                sub = sub.sort_values("usage_proxy")
                                if len(sub) < 15:
                                    continue
                                # rolling median in windows of ~10% of data
                                w = max(5, int(0.1*len(sub)))
                                med_y = sub[y_target].rolling(w, center=True, min_periods=max(3, w//2)).median()
                                rows.append(pd.DataFrame({
                                    "usage": sub["usage_proxy"], "y_smooth": med_y, "group": g
                                }))
                            if rows:
                                sf = pd.concat(rows, ignore_index=True).dropna()
                                fig_f5 = px.line(sf, x="usage", y="y_smooth", color="group",
                                                title="Smoothed Outcome by Group (rolling median)")
                                fig_f5.update_layout(template="plotly_white", xaxis_title="Usage", yaxis_title=f"{y_target} (smoothed)")
                                st.plotly_chart(fig_f5, use_container_width=True)
                        else:
                            st.caption("No group column for faceted smoothing.")

                        # ===============================================================
                        # F.6 Heteroscedasticity diagnostic: |residual| vs usage
                        # ===============================================================
                        st.markdown("#### F.6) Heteroscedasticity Check (|residual| vs usage)")
                        # Quadratic fit (as in your base)
                        A = np.c_[np.ones_like(Xc), Xc, Xc**2]
                        bcoef, *_ = np.linalg.lstsq(A, Y, rcond=None)
                        yhat = A @ bcoef
                        abs_res = np.abs(Y - yhat)
                        fig_f6 = px.scatter(pd.DataFrame({"usage": X, "abs_resid": abs_res}),
                                            x="usage", y="abs_resid", trendline="ols",
                                            title="Abs Residual vs Usage")
                        fig_f6.update_layout(template="plotly_white", xaxis_title="Usage", yaxis_title="|Residual|")
                        st.plotly_chart(fig_f6, use_container_width=True)

                        # ===============================================================
                        # F.7 Influence (Cook’s D & leverage) for quadratic model
                        # ===============================================================
                        st.markdown("#### F.7) Influence Diagnostics (Cook’s D / Leverage)")
                        if HAS_SM:
                            mdl = sm.OLS(Y, sm.add_constant(np.c_[Xc, Xc**2])).fit()
                            infl = mdl.get_influence()
                            cooks = infl.cooks_distance[0]
                            hat = infl.hat_matrix_diag
                            df_inf = pd.DataFrame({"usage": X, "cooksD": cooks, "leverage": hat, "resid": mdl.resid})
                            fig_f7a = px.scatter(df_inf, x="leverage", y="cooksD", hover_data=["usage","resid"],
                                                title="Cook’s D vs Leverage")
                            fig_f7a.update_layout(template="plotly_white")
                            st.plotly_chart(fig_f7a, use_container_width=True)

                            topk = df_inf.sort_values("cooksD", ascending=False).head(15)
                            fig_f7b = go.Figure(data=[go.Table(
                                header=dict(values=list(topk.columns), fill_color="#2c3e50", font=dict(color="white")),
                                cells=dict(values=[topk[c] for c in topk.columns], align="left")
                            )])
                            fig_f7b.update_layout(template="plotly_white", title="Top 15 by Cook’s D")
                            st.plotly_chart(fig_f7b, use_container_width=True)
                        else:
                            st.caption("statsmodels not available — influence diagnostics skipped.")

                        # ===============================================================
                        # F.8 Counterfactual contrast: predicted Δ at low vs high usage
                        # ===============================================================
                        st.markdown("#### F.8) Counterfactual Contrast (Low vs High Usage)")
                        q10, q90 = np.quantile(X, [0.10, 0.90])
                        # predict using base quadratic (centered)
                        def qpred(u):
                            z = (u - X.mean()) / (X.std() if X.std() else 1.0)
                            return bcoef[0] + bcoef[1]*z + bcoef[2]*(z**2)
                        low_hat, high_hat = qpred(q10), qpred(q90)
                        delta_cf = float(high_hat - low_hat)
                        st.metric("Predicted contrast (90th − 10th usage)", f"{delta_cf:.3f}",
                                help=f"Model: quadratic fit on {y_target}. 10th={q10:.2f}, 90th={q90:.2f}")

                        # ===============================================================
                        # F.9 If treatment exists: usage × treatment interaction
                        # ===============================================================
                        st.markdown("#### F.9) Interaction: Usage × Treatment (prediction slices)")
                        if has_treat and HAS_SM:
                            dfi = dfu.dropna(subset=[y_target, "usage_proxy", "treatment"]).copy()
                            if len(dfi) >= 30 and dfi["treatment"].nunique() == 2:
                                dfi["z"] = (dfi["usage_proxy"] - dfi["usage_proxy"].mean()) / (dfi["usage_proxy"].std() or 1.0)
                                # OLS: y ~ z + z^2 + treatment + z:treatment + z^2:treatment
                                m_int = smf.ols(f"{y_target} ~ z + I(z**2) + treatment + z:treatment + I(z**2):treatment", data=dfi).fit(cov_type="HC3")
                                grid = np.linspace(dfi["z"].quantile(0.05), dfi["z"].quantile(0.95), 60)
                                pred0 = pd.DataFrame({"z": grid, "treatment": 0})
                                pred1 = pd.DataFrame({"z": grid, "treatment": 1})
                                y0 = m_int.predict(pred0); y1 = m_int.predict(pred1)
                                dfp = pd.DataFrame({"z": np.r_[grid, grid],
                                                    "yhat": np.r_[y0, y1],
                                                    "treat": ["0"]*len(grid) + ["1"]*len(grid)})
                                fig_f9 = px.line(dfp, x="z", y="yhat", color="treat",
                                                title="Predicted Outcome by Centered Usage and Treatment",
                                                labels={"treat":"Treatment"})
                                fig_f9.update_layout(template="plotly_white", xaxis_title="Centered usage", yaxis_title=y_target)
                                st.plotly_chart(fig_f9, use_container_width=True)
                            else:
                                st.caption("Need binary treatment with enough rows for interaction slice.")
                        else:
                            st.caption("No treatment or statsmodels unavailable for interaction slices.")

                        # ===============================================================
                        # F.10 Variance-stabilized view: transform Y (Box-Cox-like proxy) and refit
                        # ===============================================================
                        st.markdown("#### F.10) Variance-Stabilized Fit (sqrt transform proxy)")
                        # Simple transform to check stability; use sqrt for nonnegative y, else center-scale
                        if np.nanmin(Y) >= 0:
                            Yt = np.sqrt(Y + 1e-8)
                            ylab = f"sqrt({y_target})"
                        else:
                            Yt = (Y - np.nanmean(Y)) / (np.nanstd(Y) or 1.0)
                            ylab = f"z({y_target})"
                        At = np.c_[np.ones_like(Xc), Xc, Xc**2]
                        coef_t, *_ = np.linalg.lstsq(At, Yt, rcond=None)
                        yhat_t = At @ coef_t
                        df_t = pd.DataFrame({"usage": X, "y_t": Yt, "yhat_t": yhat_t}).sort_values("usage")
                        fig_f10 = go.Figure()
                        fig_f10.add_trace(go.Scatter(x=df_t["usage"], y=df_t["y_t"], mode="markers", name="Transformed outcome"))
                        fig_f10.add_trace(go.Scatter(x=df_t["usage"], y=df_t["yhat_t"], mode="lines", name="Quadratic fit (transformed)"))
                        fig_f10.update_layout(template="plotly_white", title=f"Variance-Stabilized Dose–Response ({ylab})",
                                            xaxis_title="Usage", yaxis_title=ylab)
                        st.plotly_chart(fig_f10, use_container_width=True)
# ============================================================
# G) Fairness Summary — gaps & tables
# ============================================================
with tabs[6]:
    st.subheader("G) Fairness Summary — Accuracy / Parity Gaps")
    st.markdown(r"""
**Key metrics**  
- Selection Rate $SR_g = P(\hat{Y}=1 \mid A=g)$  
- Statistical Parity Difference $SPD = SR_{ref} - SR_g$  
- Disparate Impact $DI = SR_g / SR_{ref}$  
- Equal Opportunity Gap $EOG = |TPR_g - TPR_{ref}|$  
- Equalized Odds Gap $EOD = \frac{|TPR_g - TPR_{ref}| + |FPR_g - FPR_{ref}|}{2}$  
Reference group is chosen as the **largest-N** group for stability.
""")
    if not has("fairness_df"):
        st.info("Upload fairness data first.")
    else:
        df = st.session_state["fairness_df"].copy()
        df.columns = [c.lower() for c in df.columns]
        if "group" not in df.columns:
            st.info("Need a 'group' column.")
        else:
            # Determine predictions
            if "y_pred" in df.columns:
                yhat = "y_pred"
            elif "y_score" in df.columns:
                df["_y_pred_thr"] = (pd.to_numeric(df["y_score"], errors="coerce") >= 0.5).astype(int)
                yhat = "_y_pred_thr"
            else:
                yhat = None

            # Accuracy by group if present
            if "ai_accuracy" in df.columns:
                acc = df.groupby("group")["ai_accuracy"].mean().reset_index()
                st.write("AI Accuracy by group:")
                st.dataframe(acc, use_container_width=True)
                fig = px.bar(acc, x="group", y="ai_accuracy", title="Mean AI Accuracy by Group")
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

            if {"y_true"}.issubset(df.columns) and yhat is not None:
                rows = []
                for g, sub in df.dropna(subset=["group"]).groupby("group"):
                    y = pd.to_numeric(sub["y_true"], errors="coerce").astype(int)
                    yp = pd.to_numeric(sub[yhat], errors="coerce").astype(int)
                    TP = int(((yp==1)&(y==1)).sum()); TN = int(((yp==0)&(y==0)).sum())
                    FP = int(((yp==1)&(y==0)).sum()); FN = int(((yp==0)&(y==1)).sum())
                    n  = len(sub); P, Nn = TP+FN, TN+FP
                    SR  = safe_rate(TP+FP, n)
                    ACC = safe_rate(TP+TN, n)
                    TPR = safe_rate(TP, P)
                    FPR = safe_rate(FP, Nn)
                    rows.append(dict(group=g, N=n, SR=SR, ACC=ACC, TPR=TPR, FPR=FPR))
                gm = pd.DataFrame(rows).sort_values("N", ascending=False)
                if not gm.empty:
                    ref = gm.iloc[0]
                    gm["SPD"] = ref["SR"] - gm["SR"]
                    gm["DI"]  = gm["SR"] / ref["SR"] if ref["SR"] and np.isfinite(ref["SR"]) else np.nan
                    gm["EOG"] = (gm["TPR"] - ref["TPR"]).abs()
                    gm["EOD"] = ((gm["TPR"] - ref["TPR"]).abs() + (gm["FPR"] - ref["FPR"]).abs()) / 2
                    st.caption(f"Reference group (largest N): **{ref['group']}**")
                    st.dataframe(gm, use_container_width=True)

                    # Plot a few
                    for metric, title in [("SPD","Statistical Parity Difference"),
                                          ("DI","Disparate Impact"),
                                          ("EOD","Equalized Odds Gap")]:
                        if metric in gm.columns:
                            figm = px.bar(gm, x="group", y=metric, title=title)
                            figm.update_layout(template="plotly_white")
                            st.plotly_chart(figm, use_container_width=True)

                    st.download_button("Download per-group fairness metrics",
                                       data=gm.to_csv(index=False).encode("utf-8"),
                                       file_name="fairness_metrics.csv",
                                       mime="text/csv")
            else:
                st.info("Provide y_true and (y_pred or y_score) to compute parity metrics.")
        # ================================
        # G — Advanced Fairness Add-Ons (10 NEW figures/tables)
        # Paste this block **at the END of your `with tabs[6]:` section**, AFTER your current metrics.
        # It is robust and only renders when required columns exist.
        # Requirements: numpy, pandas, plotly; optional: scipy
        # ---------- Helpers ----------
        if 'safe_rate' not in globals():
            def safe_rate(num, den):
                num = float(num)
                den = float(den)
                return num/den if den and np.isfinite(den) else np.nan

        def _roc_points(y, s, n_thresh=101):
            """Return DataFrame of ROC points for thresholds in [0,1]."""
            y = pd.to_numeric(y, errors="coerce").astype(int).to_numpy()
            s = pd.to_numeric(s, errors="coerce").to_numpy()
            mask = np.isfinite(y) & np.isfinite(s)
            y, s = y[mask], s[mask]
            if len(y) == 0:
                return pd.DataFrame(columns=["thr","TPR","FPR"])
            thr_list = np.linspace(0, 1, n_thresh)
            rows = []
            P = (y==1).sum()
            N = (y==0).sum()
            for t in thr_list:
                yp = (s >= t).astype(int)
                TP = int(((yp==1)&(y==1)).sum())
                FP = int(((yp==1)&(y==0)).sum())
                TPR = safe_rate(TP, P)
                FPR = safe_rate(FP, N)
                rows.append((t, TPR, FPR))
            return pd.DataFrame(rows, columns=["thr","TPR","FPR"])

        def _pr_points(y, s, n_thresh=101):
            """Precision-Recall curve points."""
            y = pd.to_numeric(y, errors="coerce").astype(int).to_numpy()
            s = pd.to_numeric(s, errors="coerce").to_numpy()
            mask = np.isfinite(y) & np.isfinite(s)
            y, s = y[mask], s[mask]
            if len(y) == 0:
                return pd.DataFrame(columns=["thr","Precision","Recall"])
            thr_list = np.linspace(0, 1, n_thresh)
            rows = []
            P = (y==1).sum()
            for t in thr_list:
                yp = (s >= t).astype(int)
                TP = int(((yp==1)&(y==1)).sum())
                FP = int(((yp==1)&(y==0)).sum())
                Precision = safe_rate(TP, TP+FP)
                Recall = safe_rate(TP, P)
                rows.append((t, Precision, Recall))
            return pd.DataFrame(rows, columns=["thr","Precision","Recall"])

        def _calibration_bins(y, s, bins=10):
            y = pd.to_numeric(y, errors="coerce").astype(int)
            s = pd.to_numeric(s, errors="coerce")
            df = pd.DataFrame({"y": y, "s": s}).dropna()
            if df.empty:
                return pd.DataFrame(columns=["bin_mid","pred_mean","obs_rate","n"])
            q = pd.qcut(df["s"], q=min(bins, df["s"].nunique()), duplicates="drop")
            gr = df.groupby(q)
            out = gr.agg(pred_mean=("s","mean"),
                        obs_rate=("y","mean"),
                        n=("y","size")).reset_index(drop=True)
            out["bin_mid"] = out["pred_mean"]
            return out

        def _confusion_from_pred(y, yp):
            y = pd.to_numeric(y, errors="coerce").astype(int).to_numpy()
            yp = pd.to_numeric(yp, errors="coerce").astype(int).to_numpy()
            mask = np.isfinite(y) & np.isfinite(yp)
            y, yp = y[mask], yp[mask]
            TP = int(((yp==1)&(y==1)).sum())
            TN = int(((yp==0)&(y==0)).sum())
            FP = int(((yp==1)&(y==0)).sum())
            FN = int(((yp==0)&(y==1)).sum())
            return TP, TN, FP, FN

        def _bootstrap_ci_rate(successes, totals, B=1000, seed=123):
            """Bootstrap CI for a proportion given per-group (success, total) pairs."""
            rng = np.random.default_rng(seed)
            successes = np.array(successes, dtype=float)
            totals = np.array(totals, dtype=float)
            if np.any(totals <= 0) or successes.size == 0:
                return np.nan, np.nan, np.nan
            p = successes / totals
            # nonparametric bootstrap over observations
            idx = rng.integers(0, len(p), size=(B, len(p)))
            samp = p[idx].mean(axis=1)
            return float(np.mean(samp)), float(np.percentile(samp, 2.5)), float(np.percentile(samp, 97.5))

        # ---------- Reconstruct local df & yhat if needed ----------
        if 'df' not in locals() and has("fairness_df"):
            df = st.session_state["fairness_df"].copy()
            df.columns = [c.lower() for c in df.columns]

        if df is None or "group" not in df.columns:
            st.info("Cannot render advanced fairness add-ons (need fairness_df with 'group').")
        else:
            # Choose prediction source
            if "y_pred" in df.columns:
                yhat = "y_pred"
            elif "y_score" in df.columns:
                df["_y_pred_thr"] = (pd.to_numeric(df["y_score"], errors="coerce") >= 0.5).astype(int)
                yhat = "_y_pred_thr"
            else:
                yhat = None

            # ===============================================================
            # G.1 ROC by Group (requires y_score)
            # ===============================================================
            st.markdown("#### G.1) ROC Curves by Group")
            if {"y_true","y_score"}.issubset(df.columns):
                roc_traces = []
                for g, sub in df.dropna(subset=["group"]).groupby("group"):
                    roc = _roc_points(sub["y_true"], sub["y_score"])
                    if not roc.empty:
                        roc_traces.append((g, roc))
                if roc_traces:
                    fig_roc = go.Figure()
                    for g, roc in roc_traces:
                        fig_roc.add_trace(go.Scatter(x=roc["FPR"], y=roc["TPR"],
                                                    mode="lines", name=str(g)))
                    fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
                    fig_roc.update_layout(template="plotly_white",
                                        title="ROC by Group", xaxis_title="FPR", yaxis_title="TPR")
                    st.plotly_chart(fig_roc, use_container_width=True)
                else:
                    st.caption("Insufficient y_score per group to draw ROC.")
            else:
                st.caption("Need y_score to draw ROC by group.")

            # ===============================================================
            # G.2 Precision–Recall by Group (requires y_score)
            # ===============================================================
            st.markdown("#### G.2) Precision–Recall Curves by Group")
            if {"y_true","y_score"}.issubset(df.columns):
                pr_traces = []
                for g, sub in df.dropna(subset=["group"]).groupby("group"):
                    pr = _pr_points(sub["y_true"], sub["y_score"])
                    if not pr.empty:
                        pr_traces.append((g, pr))
                if pr_traces:
                    fig_pr = go.Figure()
                    for g, pr in pr_traces:
                        fig_pr.add_trace(go.Scatter(x=pr["Recall"], y=pr["Precision"],
                                                    mode="lines", name=str(g)))
                    fig_pr.update_layout(template="plotly_white",
                                        title="Precision–Recall by Group",
                                        xaxis_title="Recall", yaxis_title="Precision")
                    st.plotly_chart(fig_pr, use_container_width=True)
                else:
                    st.caption("Insufficient y_score per group to draw PR curves.")
            else:
                st.caption("Need y_score to draw PR curves by group.")

            # ===============================================================
            # G.3 Calibration (Reliability) Plots by Group (requires y_score)
            # ===============================================================
            st.markdown("#### G.3) Calibration (Reliability) by Group")
            if {"y_true","y_score"}.issubset(df.columns):
                fig_cal = go.Figure()
                ok_any = False
                for g, sub in df.dropna(subset=["group"]).groupby("group"):
                    cal = _calibration_bins(sub["y_true"], sub["y_score"], bins=10)
                    if not cal.empty:
                        ok_any = True
                        fig_cal.add_trace(go.Scatter(x=cal["pred_mean"], y=cal["obs_rate"],
                                                    mode="lines+markers", name=str(g)))
                if ok_any:
                    fig_cal.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
                    fig_cal.update_layout(template="plotly_white",
                                        title="Reliability Diagram by Group",
                                        xaxis_title="Mean predicted probability",
                                        yaxis_title="Observed positive rate")
                    st.plotly_chart(fig_cal, use_container_width=True)
                else:
                    st.caption("Insufficient data for calibration by group.")
            else:
                st.caption("Need y_score to draw calibration curves.")

            # ===============================================================
            # G.4 Threshold Sweep: Accuracy vs Equalized Odds Gap (utility–fairness front)
            # ===============================================================
            st.markdown("#### G.4) Threshold Sweep — Utility vs Fairness")
            if {"y_true","y_score"}.issubset(df.columns):
                thrs = np.linspace(0.01, 0.99, 55)
                rows = []
                for t in thrs:
                    df["_yp_thr_tmp"] = (pd.to_numeric(df["y_score"], errors="coerce") >= t).astype(int)
                    # Overall accuracy
                    y = pd.to_numeric(df["y_true"], errors="coerce").astype(int)
                    yp = df["_yp_thr_tmp"].astype(int)
                    ACC = safe_rate(((yp==y).sum()), len(df))
                    # Equalized odds gap across groups
                    per_g = []
                    for g, sub in df.dropna(subset=["group"]).groupby("group"):
                        y_g = pd.to_numeric(sub["y_true"], errors="coerce").astype(int)
                        yp_g = sub["_yp_thr_tmp"].astype(int)
                        TP, TN, FP, FN = _confusion_from_pred(y_g, yp_g)
                        P = TP+FN; Nn = TN+FP
                        TPR = safe_rate(TP, P); FPR = safe_rate(FP, Nn)
                        per_g.append((TPR, FPR))
                    if len(per_g) >= 2 and all(np.isfinite(v).all() for v in per_g):
                        TPRs, FPRs = zip(*per_g)
                        eod = (np.nanmax(np.abs(np.subtract.outer(TPRs, TPRs))) +
                            np.nanmax(np.abs(np.subtract.outer(FPRs, FPRs))))/2
                    else:
                        eod = np.nan
                    rows.append((t, ACC, eod))
                sweep = pd.DataFrame(rows, columns=["thr","ACC","EOD"])
                fig_sw = px.scatter(sweep, x="EOD", y="ACC", color="thr",
                                    title="Accuracy vs Equalized Odds Gap across Thresholds",
                                    labels={"EOD":"Equalized Odds Gap","ACC":"Accuracy"})
                fig_sw.update_layout(template="plotly_white")
                st.plotly_chart(fig_sw, use_container_width=True)
            else:
                st.caption("Need y_score to run threshold sweep.")

            # ===============================================================
            # G.5 Confusion Matrix Heatmaps per Group (using current yhat)
            # ===============================================================
            st.markdown("#### G.5) Confusion Heatmaps by Group")
            if {"y_true"}.issubset(df.columns) and (yhat is not None):
                for g, sub in df.dropna(subset=["group"]).groupby("group"):
                    TP, TN, FP, FN = _confusion_from_pred(sub["y_true"], sub[yhat])
                    mat = np.array([[TP, FP],[FN, TN]], dtype=float)
                    mat_norm = mat / mat.sum()
                    fig_cm = go.Figure(data=go.Heatmap(z=mat_norm, x=["Pred=1","Pred=0"], y=["True=1","True=0"],
                                                    colorscale="Blues", zmin=0, zmax=1))
                    fig_cm.update_layout(template="plotly_white",
                                        title=f"Normalized Confusion Matrix — Group {g}")
                    st.plotly_chart(fig_cm, use_container_width=True)
            else:
                st.caption("Need y_true and y_pred/y_score to show confusion heatmaps.")

            # ===============================================================
            # G.6 Bootstrap 95% CI for SPD and EOD
            # ===============================================================
            st.markdown("#### G.6) Bootstrap 95% CI for SPD & EOD (vs reference)")
            if {"y_true"}.issubset(df.columns) and (("y_pred" in df.columns) or ("y_score" in df.columns)):
                # decide y_pred at 0.5 if needed
                if "y_pred" in df.columns:
                    df["_yp_for_boot"] = pd.to_numeric(df["y_pred"], errors="coerce").astype(int)
                else:
                    df["_yp_for_boot"] = (pd.to_numeric(df["y_score"], errors="coerce") >= 0.5).astype(int)

                # compute per-group SR/TPR/FPR
                rows = []
                for g, sub in df.dropna(subset=["group"]).groupby("group"):
                    y = pd.to_numeric(sub["y_true"], errors="coerce").astype(int)
                    yp = sub["_yp_for_boot"].astype(int)
                    n  = len(sub)
                    TP, TN, FP, FN = _confusion_from_pred(y, yp)
                    SR  = safe_rate((yp==1).sum(), n)
                    P = TP+FN; Nn = TN+FP
                    TPR = safe_rate(TP, P)
                    FPR = safe_rate(FP, Nn)
                    rows.append(dict(group=g, n=n, SR=SR, TPR=TPR, FPR=FPR, TP=TP, TN=TN, FP=FP, FN=FN))
                G = pd.DataFrame(rows).sort_values("n", ascending=False)
                if len(G) >= 2:
                    ref = G.iloc[0]["group"]
                    # Bootstrap over rows within groups
                    rng = np.random.default_rng(7)
                    B = 800
                    spd_ci, eod_ci = [], []
                    # Pre-split indices per group for resample
                    idx_map = {g: df.index[df["group"]==g].to_numpy() for g in G["group"]}
                    for _ in range(B):
                        # resample each group with replacement, same size
                        sr = {}
                        tpr = {}
                        fpr = {}
                        ok_any = True
                        for g in G["group"]:
                            idx = rng.choice(idx_map[g], size=len(idx_map[g]), replace=True)
                            sub = df.loc[idx]
                            y = pd.to_numeric(sub["y_true"], errors="coerce").astype(int)
                            yp = sub["_yp_for_boot"].astype(int)
                            TP, TN, FP, FN = _confusion_from_pred(y, yp)
                            n = len(sub); P = TP+FN; Nn = TN+FP
                            sr[g]  = safe_rate((yp==1).sum(), n)
                            tpr[g] = safe_rate(TP, P)
                            fpr[g] = safe_rate(FP, Nn)
                            if not (np.isfinite(sr[g]) and np.isfinite(tpr[g]) and np.isfinite(fpr[g])):
                                ok_any = False
                                break
                        if not ok_any or ref not in sr:
                            continue
                        spd_vals = [sr[ref] - sr[g] for g in sr]
                        # EOD: max avg pairwise gap
                        TPRs = list(tpr.values()); FPRs = list(fpr.values())
                        eod = (np.nanmax(np.abs(np.subtract.outer(TPRs, TPRs))) +
                            np.nanmax(np.abs(np.subtract.outer(FPRs, FPRs))))/2
                        spd_ci.append(np.nanmean(spd_vals))
                        eod_ci.append(eod)
                    if spd_ci and eod_ci:
                        spd_mu, spd_lo, spd_hi = np.mean(spd_ci), np.percentile(spd_ci,2.5), np.percentile(spd_ci,97.5)
                        eod_mu, eod_lo, eod_hi = np.mean(eod_ci), np.percentile(eod_ci,2.5), np.percentile(eod_ci,97.5)
                        fig_ci = go.Figure()
                        fig_ci.add_trace(go.Bar(x=["SPD (ref−others)","EOD"], y=[spd_mu, eod_mu],
                                                error_y=dict(type="data", array=[spd_hi-spd_mu, eod_hi-eod_mu],
                                                            arrayminus=[spd_mu-spd_lo, eod_mu-eod_lo])))
                        fig_ci.update_layout(template="plotly_white", title="Bootstrap CIs for SPD and EOD",
                                            yaxis_title="Estimate (with 95% CI)")
                        st.plotly_chart(fig_ci, use_container_width=True)
                    else:
                        st.caption("Bootstrap did not produce stable estimates.")
                else:
                    st.caption("Need ≥2 groups for CI comparison.")
            else:
                st.caption("Need y_true and y_pred/y_score for bootstrap CIs.")

            # ===============================================================
            # G.7 Thresholds that Equalize TPR (per-group) vs reference (needs y_score)
            # ===============================================================
            st.markdown("#### G.7) Suggested Thresholds to Equalize TPR to Reference")
            if {"y_true","y_score"}.issubset(df.columns):
                # Reference = largest N group
                sizes = df.groupby("group").size().sort_values(ascending=False)
                if not sizes.empty:
                    ref = sizes.index[0]
                    sub_ref = df[df["group"]==ref]
                    roc_ref = _roc_points(sub_ref["y_true"], sub_ref["y_score"])
                    if not roc_ref.empty:
                        # choose ref TPR at default thr=0.5 as target
                        t_ref = 0.5
                        row_ref = roc_ref.iloc[(roc_ref["thr"]-t_ref).abs().argmin()]
                        TPR_target = float(row_ref["TPR"])
                        recs = []
                        for g, sub in df.groupby("group"):
                            roc = _roc_points(sub["y_true"], sub["y_score"])
                            if roc.empty or not np.isfinite(TPR_target):
                                recs.append((g, np.nan, np.nan, np.nan))
                                continue
                            # find threshold in this group closest to target TPR
                            k = (roc["TPR"] - TPR_target).abs().argmin()
                            thr_g = float(roc.iloc[k]["thr"])
                            TPR_g = float(roc.iloc[k]["TPR"]); FPR_g = float(roc.iloc[k]["FPR"])
                            recs.append((g, thr_g, TPR_g, FPR_g))
                        T = pd.DataFrame(recs, columns=["group","thr_match_TPR","TPR_at_thr","FPR_at_thr"]).sort_values("group")
                        fig_thr = go.Figure(data=[go.Table(
                            header=dict(values=list(T.columns), fill_color="#2c3e50", font=dict(color="white")),
                            cells=dict(values=[T[c] for c in T.columns], align="left")
                        )])
                        fig_thr.update_layout(template="plotly_white", title=f"Thresholds Matching Ref TPR (ref={ref}, target at thr=0.5)")
                        st.plotly_chart(fig_thr, use_container_width=True)
                    else:
                        st.caption("Cannot compute reference ROC.")
            else:
                st.caption("Need y_score to suggest equalized TPR thresholds.")

            # ===============================================================
            # G.8 Disparate Impact vs Selection Rate (diagnostic scatter)
            # ===============================================================
            st.markdown("#### G.8) Disparate Impact vs Selection Rate")
            if yhat is not None and "y_true" in df.columns:
                gm_rows = []
                for g, sub in df.dropna(subset=["group"]).groupby("group"):
                    yp = pd.to_numeric(sub[yhat], errors="coerce").astype(int)
                    sr = safe_rate((yp==1).sum(), len(sub))
                    gm_rows.append((g, sr, len(sub)))
                GM = pd.DataFrame(gm_rows, columns=["group","SR","N"]).sort_values("N", ascending=False)
                if len(GM) >= 2:
                    ref = GM.iloc[0]["group"]; SR_ref = float(GM.iloc[0]["SR"])
                    GM["DI"] = GM["SR"] / SR_ref if SR_ref and np.isfinite(SR_ref) else np.nan
                    fig_di = px.scatter(GM, x="SR", y="DI", size="N", text="group",
                                        title="Disparate Impact vs Selection Rate",
                                        labels={"SR":"Selection Rate","DI":"Disparate Impact"})
                    fig_di.update_traces(textposition="top center")
                    fig_di.update_layout(template="plotly_white")
                    st.plotly_chart(fig_di, use_container_width=True)
            else:
                st.caption("Need predictions to compute SR & DI.")

            # ===============================================================
            # G.9 Simpson’s Paradox Diagnostic (overall vs mixture of groups)
            # ===============================================================
            st.markdown("#### G.9) Simpson’s Paradox Check")
            if yhat is not None and "y_true" in df.columns:
                # Overall
                y_all = pd.to_numeric(df["y_true"], errors="coerce").astype(int)
                yp_all = pd.to_numeric(df[yhat], errors="coerce").astype(int)
                TP, TN, FP, FN = _confusion_from_pred(y_all, yp_all)
                P, Nn = TP+FN, TN+FP
                overall = dict(TPR=safe_rate(TP,P), FPR=safe_rate(FP,Nn))
                # Weighted by group sizes
                w_rows = []
                for g, sub in df.dropna(subset=["group"]).groupby("group"):
                    y = pd.to_numeric(sub["y_true"], errors="coerce").astype(int)
                    yp = pd.to_numeric(sub[yhat], errors="coerce").astype(int)
                    TPg, TNg, FPg, FNg = _confusion_from_pred(y, yp)
                    Pg, Nng = TPg+FNg, TNg+FPg
                    w_rows.append((len(sub), safe_rate(TPg,Pg), safe_rate(FPg,Nng)))
                W = pd.DataFrame(w_rows, columns=["n","TPR","FPR"])
                if not W.empty:
                    w_tpr = float(np.average(W["TPR"], weights=W["n"]))
                    w_fpr = float(np.average(W["FPR"], weights=W["n"]))
                    comp = pd.DataFrame({
                        "metric":["TPR","FPR"],
                        "overall":[overall["TPR"], overall["FPR"]],
                        "weighted_by_group":[w_tpr, w_fpr],
                        "abs_diff":[abs(overall["TPR"]-w_tpr), abs(overall["FPR"]-w_fpr)]
                    })
                    fig_sim = go.Figure(data=[go.Table(
                        header=dict(values=list(comp.columns), fill_color="#2c3e50", font=dict(color="white")),
                        cells=dict(values=[comp[c] for c in comp.columns], align="left")
                    )])
                    fig_sim.update_layout(template="plotly_white", title="Simpson’s Paradox Diagnostic")
                    st.plotly_chart(fig_sim, use_container_width=True)
            else:
                st.caption("Need predictions to assess Simpson’s diagnostic.")

            # ===============================================================
            # G.10 Intersectional Fairness (if multiple attributes available)
            # ===============================================================
            st.markdown("#### G.10) Intersectional Fairness (composite groups)")
            sens_candidates = [c for c in ["gender","race","ethnicity","cohort","age_group"] if c in df.columns]
            if len(sens_candidates) >= 2 and {"y_true"}.issubset(df.columns) and (yhat is not None or "y_score" in df.columns):
                combo = df[sens_candidates].astype(str).agg("|".join, axis=1).rename("intersection")
                X = df.copy()
                X["intersection"] = combo
                # build y_pred for current view
                if yhat is None and "y_score" in X.columns:
                    X["_yp_tmp"] = (pd.to_numeric(X["y_score"], errors="coerce") >= 0.5).astype(int)
                    yhat_i = "_yp_tmp"
                else:
                    yhat_i = yhat
                rows = []
                for g, sub in X.dropna(subset=["intersection"]).groupby("intersection"):
                    y = pd.to_numeric(sub["y_true"], errors="coerce").astype(int)
                    yp = pd.to_numeric(sub[yhat_i], errors="coerce").astype(int)
                    TP, TN, FP, FN = _confusion_from_pred(y, yp)
                    n = len(sub); P, Nn = TP+FN, TN+FP
                    SR  = safe_rate((yp==1).sum(), n)
                    ACC = safe_rate((yp==y).sum(), n)
                    TPR = safe_rate(TP, P)
                    FPR = safe_rate(FP, Nn)
                    rows.append(dict(intersection=g, N=n, SR=SR, ACC=ACC, TPR=TPR, FPR=FPR))
                IX = pd.DataFrame(rows).sort_values("N", ascending=False)
                if not IX.empty:
                    ref = IX.iloc[0]
                    IX["SPD"] = ref["SR"] - IX["SR"]
                    IX["DI"]  = IX["SR"] / ref["SR"] if ref["SR"] and np.isfinite(ref["SR"]) else np.nan
                    IX["EOG"] = (IX["TPR"] - ref["TPR"]).abs()
                    IX["EOD"] = ((IX["TPR"] - ref["TPR"]).abs() + (IX["FPR"] - ref["FPR"]).abs())/2
                    # Show top-10 largest EOD gaps
                    top10 = IX.nlargest(10, "EOD")
                    fig_ixt = go.Figure(data=[go.Table(
                        header=dict(values=list(top10.columns), fill_color="#2c3e50", font=dict(color="white")),
                        cells=dict(values=[top10[c] for c in top10.columns], align="left")
                    )])
                    fig_ixt.update_layout(template="plotly_white", title="Intersectional Groups — Top 10 by EOD")
                    st.plotly_chart(fig_ixt, use_container_width=True)

                    # Heatmap of EOD across intersections (cap size)
                    subH = IX.head(20).copy()
                    heat = go.Figure(data=go.Heatmap(
                        z=subH[["SPD","DI","EOG","EOD"]].to_numpy(dtype=float),
                        x=["SPD","DI","EOG","EOD"], y=subH["intersection"].astype(str), colorscale="RdBu", zmid=0
                    ))
                    heat.update_layout(template="plotly_white", title="Intersectional Fairness Metrics Heatmap (Top 20)")
                    st.plotly_chart(heat, use_container_width=True)
            else:
                st.caption("Provide ≥2 sensitive attributes (e.g., gender, race) to analyze intersectional fairness.")                
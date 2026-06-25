"""Retention Copilot - dark dashboard built on native Streamlit components."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.theme import (
    ACCENT_2,
    ACCENT_DEEP,
    MUTED,
    RISK_SCALE,
    TEXT,
    apply_plotly_theme,
    inject_theme,
)
from src.agents.orchestrator import run as run_copilot
from src.data.loader import load_enhanced
from src.data.preprocessor import load_preprocessor
from src.features.engineering import attach_value_risk
from src.models.churn_predictor import load_production
from src.models.segmentation import assign_clusters, load_bundle, segment_summary

st.set_page_config(
    page_title="Retention Copilot",
    layout="wide",
    page_icon="🧬",
    initial_sidebar_state="collapsed",
)
inject_theme()


# ---------------- First-run bootstrap ----------------
# On a fresh deploy (e.g. Streamlit Cloud) the gitignored model files aren't
# present. Build them once before anything tries to load them.
@st.cache_resource(show_spinner="First run: preparing data and training models (~1 min)...")
def _bootstrap() -> bool:
    from src.bootstrap import ensure_artifacts, ensure_faiss

    ensure_artifacts()
    ensure_faiss()
    return True


_bootstrap()


# ---------------- Data ----------------
@st.cache_data(show_spinner=False)
def _load_data() -> pd.DataFrame:
    return load_enhanced()


@st.cache_resource(show_spinner=False)
def _load_models():
    return load_production(), load_preprocessor(), load_bundle()


@st.cache_data(show_spinner=False)
def _scored_portfolio() -> pd.DataFrame:
    df = _load_data().copy()
    bundle, pre, seg_bundle = _load_models()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    X = pre.transform(df.drop(columns=["customerID", "Churn"], errors="ignore"))
    proba = bundle["model"].predict_proba(X)[:, 1]
    out = attach_value_risk(df, proba)
    out["kmeans_cluster"] = assign_clusters(df, seg_bundle).values
    out["persona"] = out["kmeans_cluster"].map(seg_bundle.cluster_personas)
    return out


# ---------------- Hero ----------------
hero_l, hero_r = st.columns([3, 1])
with hero_l:
    st.markdown(
        "**:violet[LIVE]**  ·  RETENTION COPILOT",
        help="Multi-agent AI for churn prediction and retention actions.",
    )
    st.title("Predict churn. Explain it. Save it.")
    st.markdown(
        f"<p style='color:{MUTED}; font-size:1.02rem; max-width:680px; margin-top:-0.4rem;'>"
        "A multi-agent AI copilot that scores churn risk, surfaces the SHAP-driven "
        "drivers, retrieves the right playbook, recommends the next action and "
        "quantifies the ROI of saving each customer."
        "</p>",
        unsafe_allow_html=True,
    )
with hero_r:
    try:
        _scored_preview = _scored_portfolio()
        n_total = len(_scored_preview)
        n_high = int((_scored_preview["risk_tier"] == "High").sum())
        st.metric("High-risk now", f"{n_high:,}", delta=f"of {n_total:,} customers", delta_color="inverse")
    except Exception:
        pass

st.markdown("")

# ---------------- Tabs ----------------
tab_portfolio, tab_customer, tab_kb = st.tabs(
    ["📈  Portfolio", "🧬  Customer Copilot", "📚  Knowledge Base"]
)


# =========================================================================
# Tab 1: Portfolio
# =========================================================================
with tab_portfolio:
    try:
        scored = _scored_portfolio()
    except FileNotFoundError as e:
        st.error(f"{e}\n\nRun `python scripts/train_model.py` first.")
        st.stop()

    arr_at_risk = float(scored.loc[scored["risk_tier"] == "High", "MonthlyCharges"].sum() * 12)
    arr_total = float(scored["MonthlyCharges"].sum() * 12)
    high_risk_count = int((scored["risk_tier"] == "High").sum())
    high_risk_pct = (scored["risk_tier"] == "High").mean()
    avg_risk = scored["churn_probability"].mean()
    total_n = len(scored)

    # ---- KPI row ----
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Customers", f"{total_n:,}", delta="Active base", delta_color="off")
    with k2:
        st.metric(
            "Avg churn risk", f"{avg_risk:.1%}",
            delta=f"{'Elevated' if avg_risk>0.4 else 'Healthy'}",
            delta_color="inverse" if avg_risk > 0.4 else "normal",
        )
    with k3:
        st.metric(
            "High-risk customers", f"{high_risk_count:,}",
            delta=f"{high_risk_pct:.0%} of base", delta_color="inverse",
        )
    with k4:
        st.metric(
            "ARR at risk", f"${arr_at_risk/1000:,.0f}K",
            delta=f"{arr_at_risk/arr_total:.0%} of total ARR", delta_color="inverse",
        )

    st.markdown("")

    # ---- Segment + Risk distribution ----
    cl, cr = st.columns([1.5, 1])

    with cl:
        with st.container(border=True):
            st.subheader("Segment breakdown")
            seg = pd.DataFrame(segment_summary(scored))
            if not seg.empty:
                # Drop any rows with a missing segment label (defensive - would otherwise
                # render as "undefined" in Plotly).
                seg = seg.dropna(subset=["segment"]).copy()
                seg["segment"] = seg["segment"].astype(str)
                seg = seg.sort_values("customers", ascending=True)
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name="Customers",
                    x=seg["customers"].tolist(),
                    y=seg["segment"].tolist(),
                    orientation="h",
                    marker=dict(
                        color=seg["avg_churn_prob"].tolist(),
                        colorscale=RISK_SCALE,
                        cmin=0, cmax=1,
                        line=dict(color="rgba(255,255,255,0.08)", width=1),
                        colorbar=dict(
                            title=dict(text="Avg churn", font=dict(color=MUTED, size=11)),
                            thickness=10, len=0.7,
                            tickfont=dict(color=MUTED, size=10),
                            tickformat=".0%",
                        ),
                    ),
                    text=[f"{c:,}" for c in seg["customers"]],
                    textposition="outside",
                    textfont=dict(color=TEXT, size=12),
                    hovertemplate="<b>%{y}</b><br>Customers: %{x:,}<extra></extra>",
                ))
                fig.update_layout(
                    height=340, xaxis_title="Customers", yaxis_title="",
                    showlegend=False,
                )
                apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

    with cr:
        with st.container(border=True):
            st.subheader("Risk distribution")
            risk_counts = scored["risk_tier"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0)
            fig = go.Figure(go.Pie(
                labels=risk_counts.index.tolist(),
                values=risk_counts.values.tolist(),
                hole=0.66,
                marker=dict(
                    colors=["#f87171", "#fbbf24", "#34d399"],
                    line=dict(color="#0b0b12", width=2),
                ),
                textinfo="label+percent",
                textfont=dict(color=TEXT, size=12, family="Inter"),
                hovertemplate="<b>%{label}</b><br>%{value:,} customers<br>%{percent}<extra></extra>",
            ))
            fig.update_layout(
                height=340, showlegend=False,
                annotations=[dict(
                    text=f"<b style='color:{TEXT};font-size:22px'>{high_risk_count:,}</b><br>"
                         f"<span style='color:{MUTED};font-size:10px;letter-spacing:0.08em'>HIGH RISK</span>",
                    x=0.5, y=0.5, showarrow=False,
                )],
            )
            apply_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

    # ---- Top 50 customers ----
    with st.container(border=True):
        st.subheader("Top 50 customers to call today")
        top = (
            scored.sort_values(["churn_probability", "MonthlyCharges"], ascending=[False, False])
            .head(50).copy()
        )
        top["Churn risk"] = (top["churn_probability"] * 100).round(1)
        top["ARPU"] = top["MonthlyCharges"]
        table = top[[
            "customerID", "segment", "persona", "Churn risk", "ARPU",
            "tenure", "support_ticket_count", "nps_score", "sentiment",
        ]].rename(columns={
            "customerID": "Customer", "segment": "Segment", "persona": "Persona",
            "tenure": "Tenure (mo)", "support_ticket_count": "Tickets",
            "nps_score": "NPS", "sentiment": "Sentiment",
        })
        st.dataframe(
            table, use_container_width=True, hide_index=True, height=440,
            column_config={
                "Churn risk": st.column_config.ProgressColumn(
                    "Churn risk", format="%.1f%%", min_value=0, max_value=100,
                ),
                "ARPU": st.column_config.NumberColumn("ARPU", format="$%.2f"),
                "NPS": st.column_config.ProgressColumn("NPS", format="%d", min_value=0, max_value=10),
            },
        )


# =========================================================================
# Tab 2: Customer Copilot
# =========================================================================
with tab_customer:
    df = _load_data()
    st.subheader("Pick a customer")

    pick_l, pick_r = st.columns([4, 1])
    with pick_l:
        sample_ids = df["customerID"].head(500).tolist()
        cid = st.selectbox("Customer ID", sample_ids, index=0, label_visibility="collapsed")
    with pick_r:
        run_btn = st.button("⚡  Run Copilot", type="primary", use_container_width=True)

    st.caption("Runs Profile → Risk → Explanation → Retention → ROI agents in sequence.")

    if run_btn:
        with st.spinner("Running 5-agent pipeline…"):
            state = run_copilot(cid)

        if "churn_probability" not in state:
            st.error(f"Pipeline failed: {state.get('errors')}")
            st.stop()

        proba = state["churn_probability"]
        seg = state.get("segment", "?")
        persona = state.get("persona", "?")
        risk = state.get("risk_tier", "?")

        # ---- Diagnosis header ----
        with st.container(border=True):
            d1, d2, d3, d4 = st.columns([1.4, 1, 1, 1])
            with d1:
                st.caption("CUSTOMER")
                st.markdown(
                    f"<div style='font-family:JetBrains Mono,monospace; font-size:1.25rem; "
                    f"font-weight:600; color:{TEXT}; margin-top:2px;'>{cid}</div>",
                    unsafe_allow_html=True,
                )
            with d2:
                st.metric("Churn risk", f"{proba:.0%}", delta=risk, delta_color="inverse" if risk == "High" else "off")
            with d3:
                st.metric("Segment", seg.split(" + ")[0], delta=seg.split(" + ")[1] if " + " in seg else "")
            with d4:
                st.metric("Persona", persona)

        # ---- CSM brief ----
        st.markdown("")
        st.info(f"**CSM brief** - {state.get('explanation_text', '(no explanation)')}")

        # ---- SHAP drivers ----
        st.subheader("Top churn drivers (SHAP)")
        drv = state.get("shap_drivers", [])
        if drv:
            dfd = pd.DataFrame(drv).iloc[::-1]
            max_abs = max(abs(dfd["contribution"].min()), abs(dfd["contribution"].max()), 0.01)
            with st.container(border=True):
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=dfd["contribution"], y=dfd["pretty"], orientation="h",
                    marker=dict(
                        color=dfd["contribution"],
                        colorscale=[[0.0, "#34d399"], [0.5, "#1f2937"], [1.0, "#f87171"]],
                        cmin=-max_abs, cmax=max_abs,
                        line=dict(color="rgba(255,255,255,0.08)", width=1),
                    ),
                    text=[f"{c:+.2f}" for c in dfd["contribution"]],
                    textposition="outside",
                    textfont=dict(color=TEXT, size=11, family="JetBrains Mono"),
                    hovertemplate="<b>%{y}</b><br>SHAP %{x:+.3f}<extra></extra>",
                ))
                fig.update_layout(
                    height=max(280, 56 * len(dfd)), showlegend=False,
                    xaxis_title="SHAP contribution  (+ pushes toward churn)", yaxis_title="",
                )
                apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

        # ---- Recommendation + ROI ----
        rec_col, roi_col = st.columns([1.2, 1])

        with rec_col:
            st.subheader("Recommended action plan")
            with st.container(border=True):
                st.markdown(state.get("recommendation_markdown", "*(none)*"))

        with roi_col:
            st.subheader("ROI of chosen action")
            rec = state.get("recommended_roi") or {}
            if rec:
                r1, r2 = st.columns(2)
                with r1:
                    st.metric("Offer cost", f"${rec.get('offer_cost', 0):,.0f}")
                    st.metric("Net value", f"${rec.get('net_value', 0):,.0f}",
                              delta=("profitable" if rec.get("net_value", 0) >= 0 else "underwater"),
                              delta_color="normal" if rec.get("net_value", 0) >= 0 else "inverse")
                with r2:
                    st.metric("Revenue saved", f"${rec.get('expected_revenue_saved', 0):,.0f}",
                              delta="next 12 mo", delta_color="off")
                    roi_val = rec.get("roi_multiple")
                    roi_disp = f"{roi_val:.2f}×" if isinstance(roi_val, (int, float)) else "-"
                    pb = rec.get("payback_months")
                    pb_disp = f"Payback {pb} mo" if pb is not None else "Payback -"
                    st.metric("ROI multiple", roi_disp, delta=pb_disp, delta_color="off")

        # ---- Alternative offers ----
        alt = state.get("roi_estimates", [])
        if alt:
            st.subheader("All offer alternatives - ranked")
            alt_df = pd.DataFrame(alt)[[
                "description", "offer_cost", "expected_revenue_saved",
                "net_value", "roi_multiple", "payback_months",
            ]].rename(columns={
                "description": "Offer", "offer_cost": "Cost",
                "expected_revenue_saved": "Saved", "net_value": "Net",
                "roi_multiple": "ROI ×", "payback_months": "Payback (mo)",
            })
            net_min, net_max = float(alt_df["Net"].min()), float(max(alt_df["Net"].max(), 1))
            st.dataframe(
                alt_df, use_container_width=True, hide_index=True,
                column_config={
                    "Cost": st.column_config.NumberColumn("Cost", format="$%.0f"),
                    "Saved": st.column_config.NumberColumn("Saved", format="$%.0f"),
                    "Net": st.column_config.ProgressColumn(
                        "Net value", format="$%.0f", min_value=net_min, max_value=net_max,
                    ),
                    "ROI ×": st.column_config.NumberColumn("ROI ×", format="%.2f"),
                },
            )

        # ---- Retrieved knowledge ----
        docs = state.get("retrieved_docs", [])
        if docs:
            st.subheader("Knowledge the LLM retrieved")
            for d in docs:
                with st.expander(f"📄  {d['source']}    ·    relevance {d['score']:.3f}"):
                    st.markdown(d["text"])

        # ---- Pipeline trace ----
        with st.expander("⚙️  Pipeline trace (per-agent latency)"):
            tr = pd.DataFrame(state.get("trace", []))
            if not tr.empty and "ms" in tr.columns:
                fig = go.Figure(go.Bar(
                    x=tr["ms"], y=tr["agent"], orientation="h",
                    marker=dict(
                        color=[ACCENT_2 if ok else "#f87171" for ok in tr.get("ok", [True]*len(tr))],
                        line=dict(color="rgba(255,255,255,0.1)", width=1),
                    ),
                    text=[f"{ms} ms" for ms in tr["ms"]],
                    textposition="outside",
                    textfont=dict(color=TEXT, size=11),
                    hovertemplate="<b>%{y}</b><br>%{x} ms<extra></extra>",
                ))
                fig.update_layout(height=220, showlegend=False, xaxis_title="Latency (ms)", yaxis_title="")
                apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
            st.json(state.get("trace", []))

        if state.get("errors"):
            st.error("\n".join(state["errors"]))
    else:
        with st.container(border=True):
            st.markdown(
                "<div style='text-align:center; padding:2.5rem 1rem;'>"
                "<div style='font-size:3rem;'>⚡</div>"
                f"<div style='font-size:1.1rem; font-weight:600; color:{TEXT}; margin-top:0.4rem;'>Ready to run the copilot</div>"
                f"<div style='color:{MUTED}; margin-top:0.4rem;'>Pick a customer above and click <b>Run Copilot</b>. "
                "The 5-agent pipeline produces a full retention brief in seconds.</div>"
                "</div>",
                unsafe_allow_html=True,
            )


# =========================================================================
# Tab 3: Knowledge Base
# =========================================================================
with tab_kb:
    st.subheader("Knowledge base")
    kb_dir = ROOT / "knowledge_base"
    st.caption(
        f"The RAG retriever indexes every `.md` file in `{kb_dir.name}/`. "
        "Edit them to change what the Retention Agent reasons over."
    )
    md_files = sorted(kb_dir.glob("*.md"))
    cols = st.columns(2)
    for i, md in enumerate(md_files):
        with cols[i % 2]:
            with st.expander(f"📄  {md.name}", expanded=False):
                st.markdown(md.read_text(encoding="utf-8"))

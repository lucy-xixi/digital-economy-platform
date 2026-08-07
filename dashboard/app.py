"""
DEIP Streamlit Dashboard

Pages:
  Home · Overview · Indicators · AI Impact · Research Assistant
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytics.correlation import correlate_ai_vs_dei
from src.analytics.diagnostics import diagnose_region_trend
from src.analytics.forecasting import forecast_dei_linear, walk_forward_backtest
from src.analytics.trend_analysis import dei_trend_by_region, region_ranking, trend_summary
from src.ai_agent.report_generator import ResearchReportGenerator
from src.bootstrap import run_pipeline
from src.causal_analysis.effect_estimator import estimate_ai_effect_on_dei
from src.config import (
    DEI_CSV_PATH,
    LINEAGE_DIR,
    PROCESSED_CSV_PATH,
    PROJECT_ROOT,
    QUALITY_REPORT_PATH,
    setup_logging,
)
from src.indicators.sensitivity import weight_sensitivity

logger = setup_logging(__name__)

st.set_page_config(
    page_title="DEIP — Digital Economy Intelligence Platform",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #0b1f33 0%, #12324d 100%);
        color: #e8f1f8;
        border: 1px solid #1e4a6e;
        border-radius: 10px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] label { color: #9fb6c9 !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #f0f7fc !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading DEIP datasets…")
def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed metrics and DEI index; bootstrap pipeline if missing."""
    if not PROCESSED_CSV_PATH.exists() or not DEI_CSV_PATH.exists():
        run_pipeline(force_regenerate=True)
    metrics = pd.read_csv(PROCESSED_CSV_PATH)
    dei = pd.read_csv(DEI_CSV_PATH)
    return metrics, dei


def kpi_row(dei: pd.DataFrame) -> None:
    """Render home-page KPI metrics."""
    latest_year = int(dei["year"].max())
    latest = dei[dei["year"] == latest_year]
    prev = dei[dei["year"] == latest_year - 1] if latest_year - 1 in set(dei["year"]) else None

    latest_dei = float(latest["dei"].mean())
    top_row = latest.sort_values("dei", ascending=False).iloc[0]
    top_region = str(top_row["region"])
    top_dei = float(top_row["dei"])

    if prev is not None and len(prev):
        growth = (latest_dei / float(prev["dei"].mean()) - 1.0) * 100.0
    else:
        growth = float(latest["growth_rate"].mean(skipna=True))

    c1, c2, c3 = st.columns(3)
    c1.metric("Latest DEI (national avg)", f"{latest_dei:.1f}", help=f"Year {latest_year}")
    c2.metric("YoY Growth Rate", f"{growth:.2f}%")
    c3.metric("Top Region", f"{top_region}", help=f"DEI={top_dei:.1f}")


def page_home(metrics: pd.DataFrame, dei: pd.DataFrame) -> None:
    """Home / KPI landing page."""
    st.title("Digital Economy Intelligence Platform")
    st.caption(
        "Enterprise demo — contracted pipeline · hard DQ gate · lineage · "
        "panel stats · tool-grounded research agent"
    )
    kpi_row(dei)

    st.markdown("---")
    left, right = st.columns([1.4, 1])
    with left:
        st.subheader("Control Surfaces")
        st.markdown(
            """
            - **Contracts** `metrics_v1` on raw / clean / DEI stages  
            - **Hard DQ gate** blocks DEI when score / missing / keys fail  
            - **Upsert warehouse** SQLite `ON CONFLICT` keyed by `run_id`  
            - **Lineage** `data/processed/lineage/<run_id>.json`  
            - **CLI** `python -m src.cli run|validate|forecast|report`
            """
        )
    with right:
        st.subheader("Data Quality Gate")
        if QUALITY_REPORT_PATH.exists():
            report = json.loads(QUALITY_REPORT_PATH.read_text(encoding="utf-8"))
            st.metric("Quality Score", f"{report.get('quality_score', '—')}/100")
            gate = report.get("gate_passed")
            st.write(f"Gate passed: **{gate}**")
            st.caption(report.get("summary", ""))
        else:
            st.info("Run the pipeline to generate a Data Quality Report.")

        lineage_files = sorted(LINEAGE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if lineage_files:
            latest = json.loads(lineage_files[0].read_text(encoding="utf-8"))
            st.caption(f"Latest run: `{latest.get('run_id', '')[:12]}…` · {latest.get('status')}")

    st.markdown("---")
    if st.button("Re-run data pipeline", type="secondary"):
        with st.spinner("Running staged pipeline with gate + lineage…"):
            load_datasets.clear()
            run_pipeline(force_regenerate=True)
        st.success("Pipeline completed. Reloading…")
        st.rerun()


def page_overview(dei: pd.DataFrame) -> None:
    """Page 1 — DEI trends and rankings."""
    st.header("Digital Economy Overview")
    wide = dei_trend_by_region(dei)
    fig = px.line(
        wide.reset_index().melt(id_vars="year", var_name="region", value_name="DEI"),
        x="year",
        y="DEI",
        color="region",
        markers=True,
        title="DEI Trend by Region (2018–2025)",
    )
    fig.update_layout(legend_title_text="Region", yaxis_title="DEI (0–100)")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Latest Regional Ranking")
        ranking = region_ranking(dei)
        st.dataframe(ranking, use_container_width=True, hide_index=True)
    with col2:
        st.subheader("Trend Summary")
        summary = trend_summary(dei)
        st.json(
            {
                "period": summary["period"],
                "top_region": summary["top_region"],
                "top_dei": summary["top_dei"],
                "cagr_by_region_pct": summary["cagr_by_region_pct"],
            }
        )


def page_indicators(metrics: pd.DataFrame, dei: pd.DataFrame) -> None:
    """Page 2 — radar + weight sensitivity."""
    st.header("Indicator Analysis")
    pillars = [
        "digital_infrastructure",
        "digital_industry",
        "ai_innovation",
        "digital_application",
    ]
    missing = [p for p in pillars if p not in dei.columns]
    if missing:
        st.error(f"Missing pillar columns: {missing}. Re-run the pipeline.")
        return

    years = sorted(dei["year"].unique())
    year = st.selectbox("Select year", years, index=len(years) - 1)
    subset = dei[dei["year"] == year]

    fig = go.Figure()
    for _, row in subset.iterrows():
        fig.add_trace(
            go.Scatterpolar(
                r=[row[p] for p in pillars] + [row[pillars[0]]],
                theta=[p.replace("_", " ").title() for p in pillars]
                + ["Digital Infrastructure"],
                fill="toself",
                name=row["region"],
                opacity=0.55,
            )
        )
    fig.update_layout(
        title=f"Primary Indicators Radar — {year}",
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        height=560,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Pillar Scores")
    st.dataframe(
        subset[["region", *pillars, "dei"]].sort_values("dei", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Weight Sensitivity (±10%)")
    sens = weight_sensitivity(metrics, year=int(year))
    st.metric("Mean rank stability (Spearman)", f"{sens['mean_rank_stability']}")
    scen_df = pd.DataFrame(
        [
            {
                "scenario": s["scenario"],
                "top_region": s["top_region"],
                "rank_spearman_vs_base": s["rank_spearman_vs_base"],
            }
            for s in sens["scenarios"]
        ]
    )
    st.dataframe(scen_df, use_container_width=True, hide_index=True)


def page_ai_impact(metrics: pd.DataFrame, dei: pd.DataFrame) -> None:
    """Page 3 — panel correlation + forecast intervals + diagnostics."""
    st.header("AI Impact Analysis")
    corr = correlate_ai_vs_dei(metrics, dei)
    scatter_df = pd.DataFrame(corr["scatter_data"])

    fig = px.scatter(
        scatter_df,
        x="ai_patents",
        y="dei",
        color="region",
        hover_data=["year"],
        trendline="ols",
        title="AI Patents vs Digital Economy Index",
        labels={"ai_patents": "AI Patents", "dei": "DEI"},
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pooled Pearson r", f"{corr['pearson_r']:.4f}")
    c2.metric("Within Pearson r", f"{corr.get('within_pearson_r')}")
    c3.metric("Spearman ρ", f"{corr['spearman_rho']}")
    c4.metric("N observations", f"{corr['n_obs']}")
    st.info(corr["interpretation"])
    st.caption(corr.get("note", ""))

    st.subheader("DEI Forecast — OLS + 95% Prediction Intervals")
    fcast = forecast_dei_linear(dei)
    fig2 = go.Figure()
    for region, grp in fcast.groupby("region"):
        g = grp.sort_values("year")
        fig2.add_trace(
            go.Scatter(
                x=g["year"],
                y=g["dei"],
                mode="lines+markers",
                name=str(region),
                line=dict(dash="solid"),
            )
        )
        fut = g[g["type"] == "forecast"]
        if len(fut) and fut["dei_lo"].notna().any():
            fig2.add_trace(
                go.Scatter(
                    x=list(fut["year"]) + list(fut["year"][::-1]),
                    y=list(fut["dei_hi"]) + list(fut["dei_lo"][::-1]),
                    fill="toself",
                    fillcolor="rgba(30,100,180,0.12)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name=f"{region} 95% PI",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
    fig2.update_layout(title="Historical + Forecasted DEI with Prediction Bands", yaxis_title="DEI")
    st.plotly_chart(fig2, use_container_width=True)

    bt = walk_forward_backtest(dei)
    if len(bt):
        st.caption(f"Walk-forward backtest mean AE = {bt['abs_error'].mean():.2f}")

    st.subheader("Residual Diagnostics (dei ~ year)")
    st.dataframe(diagnose_region_trend(dei), use_container_width=True, hide_index=True)


def page_causal(dei: pd.DataFrame) -> None:
    """Page — causal effect of AI Innovation on DEI (human-readable)."""
    from src.causal_analysis.report import format_causal_report, robustness_table

    st.header("因果分析 · AI 创新 → 数字经济指数")
    st.markdown(
        """
        本页回答一个问题：**AI 创新指数提高，会不会“导致”数字经济指数上升？**  
        输出重点不是 JSON，而是：**效应大小 + 置信区间 + 是否显著** 的白话结论。
        """
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        outcome = st.selectbox(
            "结果变量",
            ["dei", "dei_ex_ai"],
            format_func=lambda x: (
                "DEI（综合指数）" if x == "dei" else "dei_ex_ai（剔除 AI 后的溢出效应）"
            ),
        )
    with col_b:
        binary = st.checkbox("二值处理：AI ≥ 中位数", value=False)
    with col_c:
        engine = st.selectbox(
            "估计引擎",
            ["dowhy", "statsmodels", "econml"],
            format_func=lambda x: {
                "dowhy": "DoWhy（推荐）",
                "statsmodels": "OLS + Bootstrap",
                "econml": "EconML LinearDML（需另装）",
            }[x],
        )

    auto = st.checkbox("打开页面自动估计", value=True)

    should_run = auto or st.button("开始估计因果效应", type="primary")
    if not should_run:
        st.markdown(
            """
            **你会看到什么**
            1. 一句话结论（有没有因果影响）
            2. 效应数值 + 95% 置信区间
            3. 区间图（横线不跨 0 → 更支持显著）
            4. 稳健性对照表
            """
        )
        return

    with st.spinner("正在识别因果图并估计效应与置信区间…"):
        result = estimate_ai_effect_on_dei(
            dei,
            outcome=outcome,
            binary_treatment=binary,
            prefer_engine=engine,
            n_boot=300,
        )
    primary = result["primary"]
    level = int(float(primary.get("confidence_level", 0.95)) * 100)

    # KPI strip in Chinese
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("因果效应（点估计）", f"{primary['effect']:.4f}")
    m2.metric(f"{level}% 区间下限", f"{primary['ci_low']:.4f}")
    m3.metric(f"{level}% 区间上限", f"{primary['ci_high']:.4f}")
    m4.metric(
        "是否显著",
        "显著" if result["significant_at_alpha"] else "不显著",
        help="置信区间是否包含 0",
    )

    # Plain-language report
    st.markdown(format_causal_report(result, lang="zh"))

    # Forest plot with Chinese labels
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[primary["effect"]],
            y=["因果效应"],
            mode="markers",
            marker=dict(size=16, color="#1f4e79"),
            error_x=dict(
                type="data",
                symmetric=False,
                array=[primary["ci_high"] - primary["effect"]],
                arrayminus=[primary["effect"] - primary["ci_low"]],
                color="#1f4e79",
                thickness=3,
                width=10,
            ),
            hovertemplate=(
                f"效应=%{{x:.4f}}<br>"
                f"区间=[{primary['ci_low']:.4f}, {primary['ci_high']:.4f}]"
                "<extra></extra>"
            ),
            name="点估计",
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#888", annotation_text="无效应(0)")
    fig.update_layout(
        title="因果效应点估计与置信区间（越过 0 则不显著）",
        xaxis_title="效应大小（对 DEI 的影响，分）",
        height=300,
        showlegend=False,
        margin=dict(l=80, r=30, t=60, b=50),
    )
    st.plotly_chart(fig, use_container_width=True)

    rob = robustness_table(result)
    if rob:
        st.subheader("稳健性对照")
        st.caption("换结果变量 / 二值处理 / 其他引擎后，效应方向是否仍然一致。")
        st.dataframe(pd.DataFrame(rob), use_container_width=True, hide_index=True)

    with st.expander("查看原始 JSON（调试用）"):
        st.json(result)


def page_research(_metrics: pd.DataFrame, _dei: pd.DataFrame) -> None:
    """Page — tool-grounded research agent."""
    st.header("Research Assistant")
    st.caption(
        "Tool-grounded agent: trend · panel correlation · forecast/backtest · "
        "diagnostics · weight sensitivity · causal effect · methodology retrieval"
    )

    query = st.text_area("Research question", value="分析未来数字经济趋势", height=100)

    if st.button("Generate Report", type="primary"):
        with st.spinner("Running analytics tools and assembling ResearchBrief…"):
            brief = ResearchReportGenerator().generate(query=query)
        st.success(f"grounded={brief.grounded} · model={brief.model_name}")
        st.markdown(brief.to_markdown())
        with st.expander("Structured JSON"):
            st.json(brief.model_dump())


def main() -> None:
    """Dashboard entrypoint."""
    try:
        metrics, dei = load_datasets()
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        st.stop()

    st.sidebar.title("DEIP Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Home",
            "Digital Economy Overview",
            "Indicator Analysis",
            "AI Impact Analysis",
            "Causal Analysis",
            "Research Assistant",
        ],
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Project root: `{PROJECT_ROOT.name}`")
    st.sidebar.caption(f"Rows: metrics={len(metrics)} · DEI={len(dei)}")

    routes = {
        "Home": lambda: page_home(metrics, dei),
        "Digital Economy Overview": lambda: page_overview(dei),
        "Indicator Analysis": lambda: page_indicators(metrics, dei),
        "AI Impact Analysis": lambda: page_ai_impact(metrics, dei),
        "Causal Analysis": lambda: page_causal(dei),
        "Research Assistant": lambda: page_research(metrics, dei),
    }
    routes[page]()


if __name__ == "__main__":
    main()

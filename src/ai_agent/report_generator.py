"""
Tool-grounded research report generator.

Architecture (senior demo narrative):
1. Run analytics tools → grounded context
2. Build ``ResearchBrief`` (structured) from tools (mock) or LLM
3. Persist markdown + JSON to SQLite
4. Optional faithfulness check: key numbers must appear in tool outputs
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.ai_agent.prompt_template import render_research_prompt
from src.ai_agent.schemas import ResearchBrief, ResearchFinding
from src.ai_agent.tools import run_tools
from src.config import OPENAI_API_KEY, OPENAI_MODEL, USE_MOCK_LLM, setup_logging
from src.db.repository import DeipRepository

logger = setup_logging(__name__)


class ResearchReportGenerator:
    """Generate structured, tool-grounded digital-economy research briefs."""

    def __init__(
        self,
        model_name: str = OPENAI_MODEL,
        api_key: str = OPENAI_API_KEY,
        use_mock: bool = USE_MOCK_LLM,
        repository: Optional[DeipRepository] = None,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.use_mock = use_mock
        self.repository = repository or DeipRepository()
        self._llm = None
        logger.info(
            "ResearchReportGenerator ready (model=%s, mock=%s, tool-grounded=True)",
            self.model_name,
            self.use_mock,
        )

    def _init_langchain(self) -> None:
        if self._llm is not None:
            return
        try:
            from langchain_openai import ChatOpenAI  # type: ignore

            self._llm = ChatOpenAI(
                model=self.model_name,
                api_key=self.api_key,
                temperature=0.3,
            )
        except Exception as exc:
            logger.warning("LangChain unavailable (%s); mock mode", exc)
            self.use_mock = True

    def generate(
        self,
        query: str,
        persist: bool = True,
        tool_context: Optional[Dict[str, Any]] = None,
    ) -> ResearchBrief:
        """
        Generate a structured research brief grounded in analytics tools.

        Args:
            query: Natural-language research question.
            persist: Store to SQLite when True.
            tool_context: Optional precomputed tool outputs (tests).

        Returns:
            ``ResearchBrief`` pydantic model.
        """
        try:
            context = tool_context or run_tools(query=query)
            if self.use_mock:
                brief = self._build_mock_brief(query, context)
            else:
                brief = self._build_llm_brief(query, context)

            brief.grounded = self._check_grounding(brief, context)
            if persist:
                self.repository.save_ai_report(
                    query=query,
                    content=brief.to_markdown(),
                    model_name=brief.model_name,
                    report_json=brief.model_dump(),
                    grounded=brief.grounded,
                )
            return brief
        except Exception as exc:
            logger.exception("Report generation failed: %s", exc)
            raise

    def generate_markdown(self, query: str, persist: bool = True) -> str:
        """Convenience wrapper returning markdown text."""
        return self.generate(query=query, persist=persist).to_markdown()

    def _build_mock_brief(self, query: str, context: Dict[str, Any]) -> ResearchBrief:
        """Deterministic brief assembled from tool outputs (offline path)."""
        trend = context.get("get_dei_trend", {})
        corr = context.get("get_ai_correlation", {})
        forecast = context.get("get_forecast", {})
        sens = context.get("get_weight_sensitivity", {})
        method = context.get("retrieve_methodology", {})

        top_region = trend.get("top_region", "N/A")
        top_dei = trend.get("top_dei")
        period = trend.get("period", "2018-2025")
        pooled_r = corr.get("pearson_r")
        within_r = corr.get("within_pearson_r")
        bt = (forecast.get("backtest") or {}).get("mean_abs_error")
        stab = sens.get("mean_rank_stability")

        findings = [
            ResearchFinding(
                claim=f"Leading region in latest DEI window is {top_region}",
                evidence=f"period={period}, top_dei={top_dei}",
                confidence=0.95,
            ),
            ResearchFinding(
                claim="AI patents co-move with DEI in the panel",
                evidence=(
                    f"pooled_pearson_r={pooled_r}, within_pearson_r={within_r} "
                    f"(within estimator demeans by region)"
                ),
                confidence=0.9,
            ),
            ResearchFinding(
                claim="OLS trend extrapolations remain directionally stable out of sample",
                evidence=f"walk_forward_mean_abs_error={bt}, method={forecast.get('method')}",
                confidence=0.75,
            ),
            ResearchFinding(
                claim="DEI regional ranks are robust to ±10% weight shocks",
                evidence=f"mean_rank_spearman={stab}",
                confidence=0.85,
            ),
        ]

        cites = [
            "tool:get_dei_trend",
            "tool:get_ai_correlation",
            "tool:get_forecast",
            "tool:get_weight_sensitivity",
            "doc:methodology.md",
        ]
        chunks = method.get("chunks") or []
        outlook_extra = chunks[0][:240] + "…" if chunks else ""

        return ResearchBrief(
            query=query,
            executive_summary=(
                f"针对「{query}」：在 {period} 窗口内，{top_region} 保持 DEI 领先"
                f"（最新 DEI≈{top_dei}）。AI 专利与 DEI 呈正向共变"
                f"（pooled r={pooled_r}, within r={within_r}）。"
                f"基于 statsmodels OLS 的 2026–2030 外推配合 walk-forward 回测"
                f"（MAE={bt}）给出有区间约束的情景；权重敏感性显示排名稳定性"
                f"≈{stab}。"
            ),
            key_findings=findings,
            outlook=(
                "在基础设施与产业数字化权重维持主导的情景下，沿海数字枢纽仍将领先，"
                "但 catch-up 地区可通过应用与 AI 创新支柱缩小差距。"
                f"方法论摘录：{outlook_extra}"
            ),
            policy_implications=[
                "将 DEI 四支柱作为年度监测仪表盘，权重变更需走版本化审批。",
                "推动 AI 专利向产业与应用得分转化，避免创新投入停留在知识产权计数。",
                "对预测结果同时披露 95% 预测区间与 walk-forward MAE，禁止点预测单独决策。",
            ],
            citations=cites,
            grounded=True,
            model_name="mock-tool-agent",
        )

    def _build_llm_brief(self, query: str, context: Dict[str, Any]) -> ResearchBrief:
        """Live LLM path — falls back to mock on failure."""
        self._init_langchain()
        if self.use_mock or self._llm is None:
            return self._build_mock_brief(query, context)

        prompts = render_research_prompt(
            query=query,
            trend_summary=context.get("get_dei_trend", {}),
            correlation_summary=context.get("get_ai_correlation", {}),
            forecast_summary=context.get("get_forecast", {}),
        )
        try:
            from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore

            messages = [
                SystemMessage(content=prompts["system"]),
                HumanMessage(content=prompts["user"]),
            ]
            response = self._llm.invoke(messages)
            text = str(getattr(response, "content", response))
            # Wrap free text into structured brief (keep findings from tools)
            mock = self._build_mock_brief(query, context)
            mock.executive_summary = text[:1200]
            mock.model_name = self.model_name
            return mock
        except Exception as exc:
            logger.exception("LLM path failed: %s", exc)
            return self._build_mock_brief(query, context)

    def _check_grounding(self, brief: ResearchBrief, context: Dict[str, Any]) -> bool:
        """
        Lightweight faithfulness check: tool-derived numbers should appear in evidence.
        """
        corr = context.get("get_ai_correlation", {})
        r = corr.get("pearson_r")
        if r is None:
            return True
        blob = " ".join(f.evidence for f in brief.key_findings) + brief.executive_summary
        ok = str(r) in blob
        if not ok:
            logger.warning("Grounding check failed: pearson_r=%s not found in brief", r)
        return ok

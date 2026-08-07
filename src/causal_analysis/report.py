"""
Human-readable formatting for causal effect results.

Turns the estimator JSON payload into plain-language Chinese (and English labels)
for Dashboard / CLI — so users don't have to parse raw JSON.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _outcome_label(outcome: str) -> str:
    return {
        "dei": "数字经济综合指数 DEI",
        "dei_ex_ai": "剔除 AI 支柱后的 DEI（溢出效应）",
    }.get(outcome, outcome)


def _treatment_label(treatment: str) -> str:
    return {
        "ai_innovation": "AI 创新指数（连续）",
        "ai_high": "高 AI 创新组（高于中位数 = 1）",
    }.get(treatment, treatment)


def _significance_cn(significant: bool, alpha: float) -> str:
    if significant:
        return f"显著（{int((1 - alpha) * 100)}% 置信区间不包含 0）"
    return f"不显著（{int((1 - alpha) * 100)}% 置信区间包含 0，尚不能排除“无因果效应”）"


def format_causal_report(result: Dict[str, Any], lang: str = "zh") -> str:
    """
    Render a causal summary dict as markdown.

    Args:
        result: Output of ``estimate_ai_effect_on_dei``.
        lang: ``zh`` (default) or ``en``.

    Returns:
        Markdown string suitable for Streamlit / CLI.
    """
    if lang != "zh":
        return _format_en(result)
    return _format_zh(result)


def _estimand_label(raw: Any) -> str:
    text = str(raw or "")
    if "backdoor" in text.lower() or "ATE" in text:
        return "ATE（后门调整 / Backdoor）"
    if len(text) > 80:
        return "ATE（后门调整）"
    return text or "ATE"


def _format_zh(result: Dict[str, Any]) -> str:
    primary = result.get("primary") or {}
    ident = result.get("identification") or {}
    effect = primary.get("effect")
    lo = primary.get("ci_low")
    hi = primary.get("ci_high")
    alpha = float(result.get("alpha") or 0.05)
    sig = bool(result.get("significant_at_alpha"))
    treatment = str(primary.get("treatment", ident.get("treatment", "")))
    outcome = str(primary.get("outcome", ident.get("outcome", "")))
    level = int(float(primary.get("confidence_level") or (1 - alpha)) * 100)

    direction = "正向" if (effect is not None and effect >= 0) else "负向"
    unit_hint = (
        "AI 创新指数每提高 1 分，DEI 大约变化"
        if treatment == "ai_innovation"
        else "相对低 AI 组，高 AI 组的 DEI 大约高出"
    )

    note = ident.get("note") or ""
    if "mechanical weight" in note:
        note = (
            "全量 DEI 已包含 AI 支柱（权重 20%），因此主估计可能含构造性成分；"
            "请同时查看稳健性中的 dei_ex_ai（溢出）结果。"
        )

    lines: List[str] = [
        "## 因果分析结果（可读版）",
        "",
        "### 一句话结论",
        (
            f"**AI 创新对数字经济指数存在{direction}因果影响："
            f"效应估计 = `{effect}`，{level}% 置信区间 = `[{lo}, {hi}]`，"
            f"{_significance_cn(sig, alpha)}。**"
        ),
        "",
        "### 这是什么意思？",
        f"- **处理变量（原因）**：{_treatment_label(treatment)}",
        f"- **结果变量（结果）**：{_outcome_label(outcome)}",
        f"- **效应解读**：{unit_hint} **{effect}** 分。",
        (
            f"- **不确定性**：真实效应大概率落在 **[{lo}, {hi}]** 之间"
            f"（{level}% 置信区间）。"
        ),
        f"- **统计判断**：{_significance_cn(sig, alpha)}。",
        "",
        "### 如何理解图中的区间",
        "- 圆点 = 点估计（最可能的效应大小）",
        "- 横线 = 置信区间（估计的不确定性范围）",
        "- 若横线**不跨越 0**：更支持“确有因果影响”",
        "- 若横线**跨越 0**：证据尚不足以认定显著因果影响",
        "",
        "### 识别设定（简要）",
        f"- 估计量：{_estimand_label(primary.get('estimand'))}",
        f"- 方法：{primary.get('method', '')}",
        f"- 引擎：{primary.get('engine', '')}",
        f"- 样本量：{primary.get('n_obs', '—')}",
        f"- 控制变量：{', '.join(ident.get('common_causes') or [])}",
        f"- 说明：{note}",
    ]

    if primary.get("p_value") is not None:
        lines.append(f"- p 值：`{primary.get('p_value')}`")
    if primary.get("std_error") is not None:
        lines.append(f"- 标准误：`{primary.get('std_error')}`")

    robustness = result.get("robustness") or []
    if robustness:
        lines.extend(["", "### 稳健性检查（换设定后效应是否还在）", ""])
        lines.append("| 设定 | 效应 | 置信区间下限 | 置信区间上限 | 引擎 | 说明 |")
        lines.append("|---|---:|---:|---:|---|---|")
        for row in robustness:
            lines.append(
                "| {treatment}→{outcome} | {effect} | {ci_low} | {ci_high} | {engine} | {notes} |".format(
                    treatment=row.get("treatment"),
                    outcome=row.get("outcome"),
                    effect=row.get("effect"),
                    ci_low=row.get("ci_low"),
                    ci_high=row.get("ci_high"),
                    engine=row.get("engine"),
                    notes=(row.get("notes") or "").replace("|", "/"),
                )
            )

    lines.extend(
        [
            "",
            "### 阅读提示",
            "- 全量 DEI 本身包含 AI 支柱（权重 20%），部分效应可能来自指数构造；"
            "可对照 **dei_ex_ai（溢出）** 结果。",
            "- 因果结论依赖“控制年份/地区后无剩余混杂”的假设，模拟数据仅作方法演示。",
            "",
            f"> {result.get('interpretation', '')}",
        ]
    )
    return "\n".join(lines)


def _format_en(result: Dict[str, Any]) -> str:
    primary = result.get("primary") or {}
    return (
        f"Causal effect of {primary.get('treatment')} on {primary.get('outcome')}: "
        f"{primary.get('effect')} "
        f"(CI [{primary.get('ci_low')}, {primary.get('ci_high')}]). "
        f"{result.get('interpretation', '')}"
    )


def robustness_table(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten robustness rows into dashboard-friendly columns (Chinese headers)."""
    rows = []
    for r in result.get("robustness") or []:
        rows.append(
            {
                "设定": f"{r.get('treatment')} → {r.get('outcome')}",
                "效应估计": r.get("effect"),
                "CI下限": r.get("ci_low"),
                "CI上限": r.get("ci_high"),
                "是否跨0": (
                    "否（更显著）"
                    if not (float(r.get("ci_low", 0)) <= 0 <= float(r.get("ci_high", 0)))
                    else "是（不显著）"
                ),
                "引擎": r.get("engine"),
                "说明": {
                    "Spillover onto non-AI pillars (leave-one-out DEI).": "溢出到非 AI 支柱",
                    "Binary ATE: above-median AI Innovation vs below.": "二值 ATE：高 AI vs 低 AI",
                }.get(r.get("notes") or "", r.get("notes")),
            }
        )
    return rows

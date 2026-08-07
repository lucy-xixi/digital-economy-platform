"""
Structured output schemas for the DEIP research agent.

Pydantic models enforce executive-summary / findings / outlook / policy shape
so mock and live LLM paths return interview-grade, evaluable artifacts.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ResearchFinding(BaseModel):
    """A single grounded finding with optional numeric citation."""

    claim: str
    evidence: str = Field(description="Numeric or qualitative evidence from tools")
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)


class ResearchBrief(BaseModel):
    """Structured digital-economy research brief."""

    query: str
    executive_summary: str
    key_findings: List[ResearchFinding]
    outlook: str
    policy_implications: List[str]
    citations: List[str] = Field(default_factory=list)
    grounded: bool = True
    model_name: str = "mock-llm"

    def to_markdown(self) -> str:
        """Render brief as markdown for Streamlit / CLI."""
        findings = "\n".join(
            f"- **{f.claim}** — {f.evidence} _(confidence={f.confidence:.2f})_"
            for f in self.key_findings
        )
        policies = "\n".join(f"{i}. {p}" for i, p in enumerate(self.policy_implications, 1))
        cites = "\n".join(f"- `{c}`" for c in self.citations) or "- (none)"
        return f"""# Digital Economy Research Brief

> model={self.model_name} · grounded={self.grounded}

## 1. Executive Summary
{self.executive_summary}

## 2. Key Findings
{findings}

## 3. Outlook (2026–2030)
{self.outlook}

## 4. Policy Implications
{policies}

## Citations
{cites}
"""

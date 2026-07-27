"""디베이트 결과 스키마 + 근거(grounding) 검증."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Suitability = Literal["pass", "warn", "fail"]

PRINCIPLES = [
    "적합성",
    "적정성",
    "설명의무",
    "불공정영업금지",
    "부당권유금지",
    "광고규제",
]

# 인용 패턴: [약관 제5조], [FCPA-19], [CASE-001], [월 여유자금 37만원], 제19조, 40%
_CITATION_PATTERNS = [
    re.compile(r"\[[^\]]{2,60}\]"),
    re.compile(r"제\s?\d+조"),
    re.compile(r"(FCPA|GUIDE|CASE)-[A-Z0-9\-]+"),
    re.compile(r"\d[\d,\.]*\s?(만원|원|%p|%|개월|회)"),
]


def count_citations(text: str) -> int:
    return sum(len(p.findall(text)) for p in _CITATION_PATTERNS)


def is_grounded(text: str, min_citations: int = 1) -> bool:
    return count_citations(text) >= min_citations


def normalize_suitability(v: Any) -> Suitability:
    s = str(v).strip().lower()
    if s in {"pass", "적합", "가능", "ok"}:
        return "pass"
    if s in {"fail", "부적합", "불가", "reject"}:
        return "fail"
    if s in {"warn", "warning", "주의", "경고", "조건부"}:
        return "warn"
    # 혼합 표기 방어
    if "부적합" in s or "fail" in s:
        return "fail"
    if "적합" in s or "pass" in s:
        return "pass"
    return "warn"


class Turn(BaseModel):
    role: Literal["advocate", "skeptic", "persona", "judge", "single"]
    content: str
    citations: int = 0
    grounded: bool = True
    model: str = ""


class Verdict(BaseModel):
    suitability: Suitability
    intent_score: int = Field(ge=0, le=100)
    violated_principles: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    self_confidence: float = 0.5
    summary: str = ""

    @field_validator("violated_principles")
    @classmethod
    def _keep_known(cls, v: list[str]) -> list[str]:
        return [p for p in v if any(p.startswith(k) or k in p for k in PRINCIPLES)]

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> "Verdict":
        def pick(*keys: str, default: Any = None) -> Any:
            for k in keys:
                if k in obj and obj[k] not in (None, ""):
                    return obj[k]
            return default

        score = pick("가입의향점수", "intent_score", "score", default=50)
        try:
            score = int(round(float(score)))
        except (TypeError, ValueError):
            score = 50
        conf = pick("confidence", "신뢰도", default=0.5)
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 0.5

        def as_list(v: Any) -> list[str]:
            if v is None:
                return []
            if isinstance(v, str):
                return [v]
            return [str(x) for x in v]

        return cls(
            suitability=normalize_suitability(pick("적합성", "suitability", default="warn")),
            intent_score=max(0, min(100, score)),
            violated_principles=as_list(pick("위반원칙", "violated_principles", default=[])),
            evidence=as_list(pick("근거", "evidence", default=[])),
            risks=as_list(pick("위험요인", "risks", default=[])),
            recommendations=as_list(pick("개선권고", "recommendations", default=[])),
            self_confidence=max(0.0, min(1.0, conf)),
            summary=str(pick("요약", "summary", default="")),
        )


class DebateResult(BaseModel):
    product_id: str
    product_name: str
    persona_id: str
    segment: str
    mode: Literal["debate", "single"] = "debate"
    seed: int = 0
    temperature: float = 0.7
    turns: list[Turn] = Field(default_factory=list)
    verdict: Verdict
    persona_intent_first: int | None = None
    persona_intent_final: int | None = None
    grounding_doc_ids: list[str] = Field(default_factory=list)
    ungrounded_turns: int = 0
    elapsed_sec: float = 0.0

    def transcript(self) -> str:
        label = {
            "advocate": "옹호자",
            "skeptic": "회의론자",
            "persona": "페르소나",
            "judge": "심판",
            "single": "단발판정",
        }
        return "\n\n".join(f"### {label[t.role]}\n{t.content}" for t in self.turns)

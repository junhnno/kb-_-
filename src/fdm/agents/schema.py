"""디베이트 결과 스키마 + 근거(grounding) 검증."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from ..concerns import (
    OTHER,
    SEVERITY_ORDER,
    TIER_LABEL,
    TIER_MARK,
    TIER_ORDER,
    severity_rank,
    UNANCHORED_SEVERITY_CAP,
    cap_severity,
    concern_tier,
    is_cross_checked,
    tier_sort_key,
    type_ids,
    type_label,
    verify_with,
)

Suitability = Literal["pass", "warn", "fail"]

PRINCIPLES = [
    "적합성",
    "적정성",
    "설명의무",
    "불공정영업금지",
    "부당권유금지",
    "광고규제",
]

# 심판 응답의 필수 키. 각 튜플은 허용 별칭 묶음이며, 하나라도 있으면 충족으로 본다.
# llm.chat_json(required_keys=...)이 이걸로 검사해 누락 시 재요청한다.
#
# `근거`는 일부러 빼 두었다. 근거 인용을 요구하지 않는 비교군(naive arm)에서는
# `"근거": []`가 정당한 응답인데, 이를 필수로 두면 재요청을 반복하다 예외로 죽는다.
# 근거 유무는 실패 조건이 아니라 **품질 지표**(evidence_empty_rate)로 측정한다.
REQUIRED_VERDICT_KEYS: tuple[tuple[str, ...], ...] = (
    ("적합성", "판정", "suitability"),
    ("가입의향점수", "intent_score", "score"),
)

# 백엔드에 구조화 출력으로 강제할 JSON 스키마.
# Ollama는 format에, vLLM은 response_format={"type":"json_schema"}에 실린다.
VERDICT_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "적합성": {"type": "string", "enum": ["pass", "warn", "fail"]},
        "가입의향점수": {"type": "integer", "minimum": 0, "maximum": 100},
        "위반원칙": {"type": "array", "items": {"type": "string", "enum": PRINCIPLES}},
        "근거": {"type": "array", "items": {"type": "string"}},
        "우려": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "유형": {"type": "string", "enum": list(type_ids())},
                    "심각도": {"type": "string", "enum": SEVERITY_ORDER},
                    "내용": {"type": "string"},
                    "앵커": {"type": "string"},
                },
                "required": ["유형", "심각도", "내용"],
            },
        },
        "개선권고": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "요약": {"type": "string"},
    },
    "required": ["적합성", "가입의향점수", "위반원칙", "근거", "우려", "요약"],
}

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


class Concern(BaseModel):
    """구조화된 우려 하나.

    자유 텍스트가 아니라 유형·심각도·앵커를 갖는다. 이렇게 해야
    (1) 채점이 키워드 추측 없이 정확해지고
    (2) 계산값과의 모순 판정이 유형으로 바로 되고
    (3) '어디를 확인해야 하는가'를 산출물에 붙일 수 있다.
    """

    type: str = Field(description="concern_taxonomy.json의 유형 id 또는 'other'")
    severity: str = Field(default="주의", description="경미 | 주의 | 중대 | 치명")
    statement: str = Field(description="우려 내용 한 문장")
    anchor: str = Field(
        default="", description="계산값·약관 조항·페르소나 수치 등 근거. 없으면 심각도가 제한된다"
    )
    verify_with: str = Field(default="", description="실제 데이터로 확인하는 방법 (코드가 채움)")
    sources: list[str] = Field(
        default_factory=list,
        description="이 우려를 제기한 방식(single/debate). 둘 다면 교차 확인된 것",
    )

    @field_validator("type")
    @classmethod
    def _known_type(cls, v: str) -> str:
        return v if v in type_ids() else OTHER

    @field_validator("severity")
    @classmethod
    def _known_severity(cls, v: str) -> str:
        return v if v in SEVERITY_ORDER else "주의"

    def finalize(self) -> "Concern":
        """앵커 없는 우려의 심각도를 제한하고 확인 방법을 채운다."""
        if not self.anchor.strip():
            self.severity = cap_severity(self.severity, UNANCHORED_SEVERITY_CAP)
        if not self.verify_with:
            self.verify_with = verify_with(self.type)
        return self

    # --- 계층 (교차확인 × 심각도) — 데이터는 이미 있고 표시만 없었다 ---

    @property
    def cross_checked(self) -> bool:
        return is_cross_checked(self.sources)

    @property
    def tier(self) -> str:
        return concern_tier(self.severity, self.sources)

    @property
    def tier_label(self) -> str:
        return TIER_LABEL[self.tier]

    @property
    def sort_key(self) -> tuple:
        return tier_sort_key(self.severity, self.sources)

    def headline(self) -> str:
        """산출물 한 줄 표기: [계층·심각도·교차확인] 유형 — 내용"""
        cross = "교차확인" if self.cross_checked else "단독"
        return (
            f"{TIER_MARK[self.tier]} **[{self.tier_label}]** "
            f"`{self.severity}·{cross}` {type_label(self.type)} — {self.statement}"
        )


def sort_concerns(concerns: list[Concern]) -> list[Concern]:
    """계층 → 심각도 → 교차확인 순. 분석가가 위에서부터 읽으면 되는 순서."""
    return sorted(concerns, key=lambda c: c.sort_key)


def group_by_tier(concerns: list[Concern]) -> dict[str, list[Concern]]:
    """계층별로 묶는다. 빈 계층도 키는 남겨 '없음'을 표시할 수 있게 한다."""
    out: dict[str, list[Concern]] = {t: [] for t in TIER_ORDER}
    for c in sort_concerns(concerns):
        out[c.tier].append(c)
    return out


def stamp_sources(concerns: list[Concern], source: str) -> list[Concern]:
    """단독 실행 결과에 '어느 방식이 제기했나'를 표시한다.

    교차확인 계층의 입력이다. 이 표시가 없으면 sources가 비어 모든 우려가
    단독으로 취급되고, ensemble의 교차확인 신호가 산출물에서 사라진다.
    """
    for c in concerns:
        if not c.sources:
            c.sources = [source]
    return concerns


def merge_concerns(groups: dict[str, list[Concern]]) -> list[Concern]:
    """여러 방식이 제기한 우려를 유형 기준으로 합친다.

    측정 근거: 단발과 디베이트는 **서로 다른 우려를 놓친다**.
    디베이트는 숨은 비용·감당능력에 강하고(회의론자가 파고듦),
    단발은 성향 부적합·끼워팔기·한도 같은 분류적 위험에 강하다.
    두 목록의 합집합이 정답 우려를 30/30 잡았다(단독은 22~26).

    같은 유형이 겹치면 심각도가 높은 쪽을 남기되, 앵커가 있는 쪽을 우선한다.
    """
    by_type: dict[str, Concern] = {}
    for source, concerns in groups.items():
        for c in concerns:
            c = c.model_copy(deep=True)
            c.sources = [source]
            prev = by_type.get(c.type)
            if prev is None:
                by_type[c.type] = c
                continue
            merged_sources = sorted(set(prev.sources) | set(c.sources))
            # 앵커 유무 → 심각도 순으로 우선한다
            better = c if (bool(c.anchor) > bool(prev.anchor)) else prev
            if bool(c.anchor) == bool(prev.anchor):
                better = c if severity_rank(c.severity) > severity_rank(prev.severity) else prev
            better = better.model_copy(deep=True)
            better.sources = merged_sources
            by_type[c.type] = better
    # 계층(교차확인 × 심각도) 순으로 정렬한다
    return sort_concerns(list(by_type.values()))


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
    concerns: list[Concern] = Field(
        default_factory=list, description="구조화된 우려 (유형·심각도·앵커). 채점과 산출물의 기본 단위"
    )
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
            """리스트를 기대하지만 문자열·딕셔너리로 오는 경우까지 흡수한다.

            심판이 `"근거": {"광고표시": "...", "설명의무": "..."}` 처럼 딕셔너리로
            답하는 일이 실제로 관측됐다. 이때 `[str(x) for x in v]`는 **키만** 남겨
            내용을 버리므로, `키: 값` 형태로 평탄화한다.
            """
            if v is None:
                return []
            if isinstance(v, str):
                return [v]
            if isinstance(v, dict):
                return [f"{k}: {val}" for k, val in v.items()]
            if isinstance(v, list):
                out: list[str] = []
                for x in v:
                    if isinstance(x, dict):
                        out += [f"{k}: {val}" for k, val in x.items()]
                    else:
                        out.append(str(x))
                return out
            return [str(v)]

        # 구조화 우려 파싱. 구형(위험요인 문자열 배열)도 받아 other 유형으로 흡수한다.
        concerns: list[Concern] = []
        raw_concerns = pick("우려", "concerns", default=None)
        if isinstance(raw_concerns, list):
            for item in raw_concerns:
                if isinstance(item, dict):
                    concerns.append(
                        Concern(
                            type=str(item.get("유형") or item.get("type") or OTHER),
                            severity=str(item.get("심각도") or item.get("severity") or "주의"),
                            statement=str(item.get("내용") or item.get("statement") or ""),
                            anchor=str(item.get("앵커") or item.get("anchor") or ""),
                        ).finalize()
                    )
                elif isinstance(item, str):
                    concerns.append(Concern(type=OTHER, statement=item).finalize())
        legacy_risks = as_list(pick("위험요인", "risks", default=[]))
        if not concerns:
            concerns = [Concern(type=OTHER, statement=r).finalize() for r in legacy_risks]

        return cls(
            suitability=normalize_suitability(
                pick("적합성", "판정", "suitability", default="warn")
            ),
            intent_score=max(0, min(100, score)),
            violated_principles=as_list(pick("위반원칙", "violated_principles", default=[])),
            evidence=as_list(pick("근거", "evidence", default=[])),
            # risks는 하위호환용 파생값. 우려의 문장만 뽑아 둔다.
            risks=[c.statement for c in concerns] or legacy_risks,
            concerns=concerns,
            recommendations=as_list(pick("개선권고", "recommendations", default=[])),
            self_confidence=max(0.0, min(1.0, conf)),
            summary=str(pick("요약", "판단요지", "summary", default="")),
        )


class DebateResult(BaseModel):
    product_id: str
    product_name: str
    persona_id: str
    segment: str
    mode: Literal["debate", "single", "ensemble"] = "debate"
    seed: int = 0
    temperature: float = 0.7
    turns: list[Turn] = Field(default_factory=list)
    verdict: Verdict
    persona_intent_first: int | None = None
    persona_intent_final: int | None = None
    grounding_doc_ids: list[str] = Field(default_factory=list)
    ungrounded_turns: int = 0
    judge_schema_repaired: bool = Field(
        default=False,
        description="심판이 요구 스키마를 어겨 교정 재요청이 필요했는가 (품질 지표)",
    )
    label_disagreement: bool = Field(
        default=False,
        description="병행(ensemble)에서 단발과 디베이트의 라벨이 갈렸는가. 추가 검토 신호",
    )
    dropped_concerns: list[str] = Field(
        default_factory=list,
        description="사실팩 계산값과 모순되어 기각된 우려 (사유 포함). 환각 발생량 지표",
    )
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

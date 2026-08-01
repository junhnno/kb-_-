"""멀티시드 반복 → 합의도 기반 신뢰도 스코어.

같은 (상품, 페르소나) 조합을 시드·온도를 바꿔 N회 돌린 뒤
  - 적합성 라벨의 최빈값 비율 (label_agreement)
  - 가입의향점수의 표준편차 (score_std)
를 합쳐 0~1 신뢰도를 만든다. 저신뢰 조합은 리포트에 '추가 검증 필요'로 플래그된다.
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field

from ..agents.schema import Concern, DebateResult, Suitability, merge_concerns

# 표준편차가 이 값 이상이면 점수 안정성 0점으로 본다 (0~100 스케일)
SCORE_STD_CEIL = 25.0
LOW_THRESHOLD = 0.55
HIGH_THRESHOLD = 0.8


class ConsensusResult(BaseModel):
    product_id: str
    product_name: str = ""
    persona_id: str
    segment: str
    n_runs: int
    mode: str = "debate"

    modal_suitability: Suitability
    label_counts: dict[str, int] = Field(default_factory=dict)
    label_agreement: float
    intent_mean: float
    intent_std: float
    intent_min: int
    intent_max: int
    judge_self_confidence: float = 0.5
    ungrounded_turns: int = 0

    confidence: float
    confidence_level: Literal["high", "medium", "low"]
    needs_review: bool

    evidence: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    grounding_doc_ids: list[str] = Field(default_factory=list)
    concerns: list[Concern] = Field(
        default_factory=list,
        description="여러 런의 우려를 유형 기준으로 합친 것. sources가 교차확인 계층의 입력",
    )
    concern_run_ratio: dict[str, float] = Field(
        default_factory=dict,
        description="유형 → 몇 비율의 런에서 제기됐나. **재현성이지 정확성이 아니다**",
    )

    @property
    def adopted(self) -> bool:
        return self.intent_mean >= 60


def _top_items(seq: list[str], k: int) -> list[str]:
    """여러 런에 걸쳐 반복 등장한 항목을 우선해 대표 근거/위험을 뽑는다."""
    counts = Counter(s.strip() for s in seq if s and s.strip())
    return [s for s, _ in counts.most_common(k)]


def _merge_run_concerns(results: list[DebateResult]) -> tuple[list[Concern], dict[str, float]]:
    """여러 시드 런의 우려를 하나로 합친다.

    주의: 시드 반복은 **재현성** 신호이고, 교차확인(단발·디베이트가 모두 제기)은
    **정확성** 신호다. 실측상 재현성은 정확성과 무관했으므로(저신뢰 오답률 40% vs
    전체 42%), 반복 횟수를 sources에 섞어 계층을 부풀리면 안 된다.
    sources는 제기 '방식'만 담고, 반복 비율은 별도 필드로 낸다.
    """
    groups: dict[str, list[Concern]] = {}
    for r in results:
        for c in r.verdict.concerns:
            for src in c.sources or [r.mode]:
                groups.setdefault(src, []).append(c)
    merged = merge_concerns(groups) if groups else []

    n = len(results)
    raised = Counter(c.type for r in results for c in {x.type: x for x in r.verdict.concerns}.values())
    ratio = {t: round(cnt / n, 3) for t, cnt in raised.items()}
    return merged, ratio


def aggregate(results: list[DebateResult]) -> ConsensusResult:
    if not results:
        raise ValueError("결과가 비어 있다")
    labels = [r.verdict.suitability for r in results]
    scores = [r.verdict.intent_score for r in results]
    counts = Counter(labels)
    modal, modal_n = counts.most_common(1)[0]
    agreement = modal_n / len(labels)
    std = statistics.pstdev(scores) if len(scores) > 1 else 0.0

    score_stability = max(0.0, 1.0 - std / SCORE_STD_CEIL)
    self_conf = statistics.fmean([r.verdict.self_confidence for r in results])
    # 가중치: 라벨 합의 0.5, 점수 안정성 0.3, 심판 자기신뢰 0.2
    confidence = 0.5 * agreement + 0.3 * score_stability + 0.2 * self_conf
    # 근거 없는 발화가 있으면 감점
    ungrounded = sum(r.ungrounded_turns for r in results)
    confidence *= 1.0 - min(0.2, 0.04 * ungrounded)
    confidence = round(max(0.0, min(1.0, confidence)), 3)

    level: Literal["high", "medium", "low"] = (
        "high" if confidence >= HIGH_THRESHOLD else ("low" if confidence < LOW_THRESHOLD else "medium")
    )

    r0 = results[0]
    merged_concerns, concern_ratio = _merge_run_concerns(results)
    return ConsensusResult(
        product_id=r0.product_id,
        product_name=r0.product_name,
        persona_id=r0.persona_id,
        segment=r0.segment,
        n_runs=len(results),
        mode=r0.mode,
        modal_suitability=modal,
        label_counts=dict(counts),
        label_agreement=round(agreement, 3),
        intent_mean=round(statistics.fmean(scores), 1),
        intent_std=round(std, 2),
        intent_min=min(scores),
        intent_max=max(scores),
        judge_self_confidence=round(self_conf, 3),
        ungrounded_turns=ungrounded,
        confidence=confidence,
        confidence_level=level,
        needs_review=level == "low",
        evidence=_top_items([e for r in results for e in r.verdict.evidence], 6),
        risks=_top_items([x for r in results for x in r.verdict.risks], 6),
        recommendations=_top_items([x for r in results for x in r.verdict.recommendations], 5),
        grounding_doc_ids=sorted({d for r in results for d in r.grounding_doc_ids}),
        concerns=merged_concerns,
        concern_run_ratio=concern_ratio,
    )


def seed_plan(n: int, base_seed: int = 0, base_temp: float = 0.8) -> list[tuple[int, float]]:
    """멀티시드 계획: 시드와 온도를 함께 흔든다."""
    temps = [base_temp, base_temp - 0.2, base_temp + 0.1, base_temp - 0.4, base_temp + 0.2]
    return [
        (base_seed + 1000 * i, round(max(0.1, min(1.2, temps[i % len(temps)])), 2))
        for i in range(n)
    ]

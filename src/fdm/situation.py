"""판매 정황과 모순되는 우려를 걸러낸다.

왜 필요한가 (실측된 실패, 2026-08-01 pass12 실행):
  정답셋의 `facts`(판매 정황)는 **주어진 사실**인데 모델이 이를 읽지 않고
  정반대 주장을 한다. 22건 중 최소 7건에서 같은 패턴이 나왔다.

    CASE-014 정황 "중도해지 시 불이익을 설명했다"      → 출력 "중도해지 이자손실 설명이 부족"
    CASE-008 정황 "중도해지 불이익을 표로 설명했다"     → 출력 "중도해지 불이익 설명 부족"
    CASE-020 정황 "확정금리와 총 상환액을 안내했다"     → 출력 "금리 변동 가능성을 설명하지 않아"

  이건 판단의 차이가 아니라 **입력과의 직접 모순**이다. 정황에 "설명했다"고 적힌
  항목을 "설명 안 했다"고 하는 것은 사실관계 오류이므로 기각할 수 있다.

왜 유형(type) 기반 스크리닝으로는 안 잡히나:
  `facts.CONTRADICTION_RULES`는 우려의 `type`으로 판정한다. 그런데 CASE-020은
  금리 변동을 문제 삼으면서 유형을 `explanation_insufficient`로 달았다.
  유형이 어긋나면 규칙이 발동하지 않는다. 그래서 여기서는 **문장 텍스트**로 본다.

안전장치:
  - 정황이 비어 있으면 아무것도 하지 않는다.
  - 정황이 그 항목을 "설명했다"고 **명시**한 경우에만 발동한다.
    (정황에 언급이 없으면 모름이지 아니오가 아니다 — 작업원칙 3)
  - 설명 부족을 주장하는 문장에만 적용한다. 같은 항목의 다른 우려
    (예: "중도해지 이율이 낮다")는 건드리지 않는다.
"""

from __future__ import annotations

import re

# 정황에서 "이 항목을 고객에게 알렸다"고 볼 수 있는 표현
DISCLOSURE_MARKERS = (
    "설명", "안내", "고지", "교부", "제시", "표로", "보여주", "알렸", "확인 서명", "서명을 받",
)

# 부정 표현. 이게 같은 절에 있으면 '설명했다'가 아니라 '설명 안 했다'는 뜻이다.
#
# 이 목록이 없으면 정반대로 읽는다(실측: 초기 구현이 CASE-004의
# "…시나리오는 안내받지 못했다"를 '안내함'으로, CASE-007의 "설명하지 않았다"를
# '설명함'으로 판정해, gold=fail 4건에서 정답 우려를 기각할 뻔했다).
NEGATION_MARKERS = (
    "못했", "못한", "못하", "않았", "않고", "않은", "않아", "않는",
    "없이", "없었", "누락", "미고지", "미설명", "안 했", "생략",
)

# 부분·형식적 이행. 설명이 있긴 했으나 충분하지 않았다는 신호이므로 기각 근거가 못 된다.
# (CASE-010 "안내가 형식적으로만 이루어졌다", CASE-011 "구두 설명은 '대부분 연장된다'는 취지")
QUALIFIER_MARKERS = (
    "형식적", "만 이루어", "취지", "대부분", "만으로", "라며", "라고만", "위주로",
)

# 우려 문장이 "그 항목의 설명이 모자랐다"고 주장하는 표현
INSUFFICIENCY_MARKERS = (
    "부족", "미흡", "누락", "미고지", "설명하지", "안내하지", "고지하지",
    "않아", "않았", "없어", "없이", "미제공", "불충분", "부재",
)

# 항목별 키워드. 정황과 우려 문장 양쪽에서 같은 어휘를 찾는다.
TOPIC_LABEL: dict[str, str] = {
    "early_termination": "중도해지·중도상환 불이익",
    "rate_structure": "금리 구조(고정/변동)",
    "rate_display": "기본금리·우대금리 표시",
    "preferential": "우대조건",
    "principal_loss": "원금 손실 가능성",
    "deposit_protection": "예금자보호 한도",
    "fee": "수수료·연회비",
    "maturity": "만기 후 이율·기한연장",
    "tax": "과세·세제",
}

TOPICS: dict[str, tuple[str, ...]] = {
    "early_termination": ("중도해지", "중도상환", "해지이율", "만기 전 해지"),
    "rate_structure": ("금리 변동", "변동금리", "확정금리", "고정금리", "금리 재산정"),
    "rate_display": ("기본금리", "최고금리", "우대금리", "우대이율", "표시금리"),
    "preferential": ("우대조건", "우대 조건", "실적 조건"),
    "principal_loss": ("원금 손실", "원금손실", "원금 비보장", "손실 가능"),
    "deposit_protection": ("예금자보호", "보호 한도", "예금보험"),
    "fee": ("수수료", "연회비", "부대비용"),
    "maturity": ("만기 후 이율", "만기후이율", "만기 일시상환", "기한연장"),
    "tax": ("과세", "비과세", "세제", "원천징수"),
}


def _clauses(text: str) -> list[str]:
    """문장 단위로 자른다.

    나열 구분자('·', ',')로는 자르지 않는다. "금리·중도해지이율·만기 후 이율을 설명하고"처럼
    항목을 나열한 뒤 마지막에 동사가 오는 문형이 흔해서, 나열을 쪼개면 앞 항목들이
    동사와 분리돼 설명 사실을 놓친다(실측: CASE-006에서 early_termination을 놓쳤다).
    """
    return [c for c in re.split(r"[.。\n]|(?<=다)\s+", text) if c.strip()]


def disclosed_topics(situation: str) -> set[str]:
    """정황이 '설명했다'고 **명시**한 항목들.

    부정문·부분이행 표현이 섞인 절은 통째로 제외한다. 절 안에서 어느 항목이
    부정 대상인지까지는 가르지 않는다 — 애매하면 기각하지 않는 쪽이 안전하다
    (정답 우려를 지우는 비용이 오탐 하나를 남기는 비용보다 크다).
    """
    if not (situation or "").strip():
        return set()
    found: set[str] = set()
    for clause in _clauses(situation):
        if not any(m in clause for m in DISCLOSURE_MARKERS):
            continue
        if any(n in clause for n in NEGATION_MARKERS):
            continue
        if any(q in clause for q in QUALIFIER_MARKERS):
            continue
        for topic, kws in TOPICS.items():
            if any(k in clause for k in kws):
                found.add(topic)
    return found


def claims_insufficient_disclosure(text: str) -> set[str]:
    """이 문장이 '설명이 부족했다'고 주장하는 항목들."""
    if not (text or "").strip():
        return set()
    out: set[str] = set()
    if not any(m in text for m in INSUFFICIENCY_MARKERS):
        return out
    for topic, kws in TOPICS.items():
        if any(k in text for k in kws):
            out.add(topic)
    return out


def contradicts_situation(concern, situation: str) -> str | None:
    """정황과 모순되면 모순 항목명을, 아니면 None을 돌려준다."""
    disclosed = disclosed_topics(situation)
    if not disclosed:
        return None
    text = f"{concern.statement} {concern.anchor}"
    hit = claims_insufficient_disclosure(text) & disclosed
    return sorted(hit)[0] if hit else None


def screen_by_situation(concerns: list, situation: str) -> tuple[list, list[str]]:
    """정황이 '설명했다'고 명시한 항목의 '설명 부족' 주장을 기각한다.

    반환: (유지, 기각[사유 포함])
    """
    if not (situation or "").strip():
        return concerns, []
    kept, dropped = [], []
    for c in concerns:
        topic = contradicts_situation(c, situation)
        if topic:
            dropped.append(f"{c.statement} [기각: 판매 정황에 '{topic}' 설명 사실이 명시됨]")
        else:
            kept.append(c)
    return kept, dropped

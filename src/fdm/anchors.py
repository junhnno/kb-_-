"""앵커(근거) 실재 검증 — 모델이 지어낸 인용을 걸러낸다.

왜 필요한가:
  `Concern.anchor`는 지금까지 **비어 있는지만** 검사했다(`schema.Concern.finalize`).
  내용은 아무도 안 봤으므로, 모델이 존재하지 않는 `[약관 제10조]`를 써넣어도
  '중대'·'치명' 심각도를 그대로 받는다. 산출물이 "여기를 보라"고 지목하는
  도구인 이상, 지목한 곳이 없으면 분석가의 시간을 직접 낭비시킨다.
  (HANDOFF_PROMPT.md 6-5 항목)

무엇을 검증하나:
  프롬프트가 인용을 세 종류로 제한한다(`prompts.COMMON_RULES`).
    (a) 약관/설명서 조항 ID  — 예: [약관 제5조]      → 상품의 clauses에 실재해야 한다
    (b) 법령·문서 ID         — 예: [FCPA-19]         → 코퍼스에 실재해야 한다
    (c) 페르소나의 재무 수치 — 예: [월 여유자금 37만원] → 사실팩 계산값과 맞아야 한다

판정 3값 (작업원칙 3: 모름과 아니오를 구분한다):
  verified     — 실재하는 ID를 가리키거나, 사실팩의 수치와 일치한다
  fabricated   — ID 꼴을 갖췄는데 실재하지 않는다. **확정적으로 틀린 인용**
  unverifiable — 검증할 거리가 없다(자유 서술). 틀렸다는 뜻이 아니다

지우지 않고 심각도만 낮춘다(작업원칙 4). 놓치면 책임이고 많으면 무시되므로,
기각보다 계층 강등이 안전하다.
"""

from __future__ import annotations

import re

from .concerns import cap_severity, severity_rank
from .facts import FactPack
from .products.schema import Product

VERIFIED = "verified"
FABRICATED = "fabricated"
BORROWED = "borrowed"
UNVERIFIABLE = "unverifiable"

# 지어낸 인용의 심각도 상한. 계층상 T4('접어두기')로 내려가 상위 목록에서 빠진다.
FABRICATED_SEVERITY_CAP = "경미"
# 남의 조정례를 근거로 재활용한 경우의 상한. 문서 자체는 실재하므로 '지어냄'보다는 덜 엄하게 본다.
BORROWED_SEVERITY_CAP = "주의"

# [FCPA-19], [CASE-001] 같은 문서 ID
DOC_ID_RE = re.compile(r"\b([A-Z]{2,}-\d+)\b")
# "약관 제5조", "설명서 제2항", "약관제10조"(공백 없음)까지 잡는다
CLAUSE_RE = re.compile(r"(약관|상품설명서|설명서|광고)\s*제\s*(\d+)\s*([조항호])")
# 수치. 1,234.5 형태의 자릿수 구분과 소수점, 앞의 음수 부호를 허용한다
NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")

# 수치 일치 허용 오차. 사실팩은 반올림해 프롬프트에 넣으므로(예: 부담률 8%)
# 정확히 같기를 요구하면 정당한 인용도 미검증으로 떨어진다.
NUMERIC_REL_TOL = 0.02
NUMERIC_ABS_TOL = 0.5


def _normalize_clause(kind: str, num: str, unit: str) -> str:
    """'약관제 5 조' 같은 표기 흔들림을 '약관 제5조'로 통일한다."""
    return f"{kind} 제{int(num)}{unit}"


def clause_keys(product: Product) -> set[str]:
    """상품이 실제로 가진 조항 ID의 정규화 집합."""
    keys: set[str] = set()
    for c in product.clauses:
        for m in CLAUSE_RE.finditer(c.id):
            keys.add(_normalize_clause(*m.groups()))
    return keys


def known_numbers(facts: FactPack | None) -> set[float]:
    """사실팩이 프롬프트에 제시한 수치들. 앵커의 숫자를 이 집합과 대조한다.

    비율은 프롬프트에 백분율로 찍히므로(예: 부담률 8%) 100배 값도 함께 넣는다.
    """
    if facts is None:
        return set()
    out: set[float] = set()

    def add(v: float | int | None, *, as_pct: bool = False) -> None:
        if v is None:
            return
        out.add(float(v))
        if as_pct:
            out.add(round(float(v) * 100, 4))

    for v in (
        facts.monthly_income, facts.monthly_surplus, facts.monthly_debt_service,
        facts.liquid_assets, facts.dsr_pct, facts.age, facts.deposit_exposure,
        facts.payment_min, facts.payment_max, facts.stressed_rate_pct,
        facts.stressed_dsr_pct,
    ):
        add(v)
    for v in (
        facts.surplus_ratio, facts.burden_min, facts.burden_max,
        facts.burden_effective, facts.lump_sum_burden, facts.joint_attainment,
    ):
        add(v, as_pct=True)
    return out


def _numbers_in(text: str) -> list[float]:
    out = []
    for raw in NUMBER_RE.findall(text):
        try:
            out.append(float(raw.replace(",", "")))
        except ValueError:
            continue
    return out


def _matches_known(value: float, known: set[float]) -> bool:
    for k in known:
        if abs(value - k) <= max(NUMERIC_ABS_TOL, abs(k) * NUMERIC_REL_TOL):
            return True
    return False


def verify_anchor(
    anchor: str,
    *,
    clause_ids: set[str],
    doc_ids: set[str],
    case_doc_ids: set[str] | None = None,
    facts: FactPack | None = None,
) -> str:
    """앵커 하나를 검증해 VERIFIED / FABRICATED / BORROWED / UNVERIFIABLE 중 하나를 반환한다.

    ID 꼴을 갖춘 인용이 하나라도 실재하지 않으면 FABRICATED다 — 나머지가
    맞더라도, 없는 조항을 가리키는 순간 산출물의 신뢰가 깨지기 때문이다.

    조정례(case) 문서만 가리키면 BORROWED다. 문서는 실재하지만 **남의 사건**이라
    이 상품·페르소나·정황에 대한 근거가 못 된다. 프롬프트가 허용한 인용도
    "(b) 법령·감독기준 문서"까지이지 조정례가 아니다.
    (실측 2026-08-01: CASE-013/015/017/018이 전부 [CASE-009]를 자기 근거로 달았다)
    """
    text = (anchor or "").strip()
    if not text:
        return UNVERIFIABLE
    cases = case_doc_ids or set()

    refs_found = False
    borrowed_only = True
    # (a) 약관·설명서 조항
    for m in CLAUSE_RE.finditer(text):
        refs_found = True
        borrowed_only = False
        if _normalize_clause(*m.groups()) not in clause_ids:
            return FABRICATED
    # (b) 법령·문서 ID
    for doc_id in DOC_ID_RE.findall(text):
        refs_found = True
        if doc_id not in doc_ids:
            return FABRICATED
        if doc_id not in cases:
            borrowed_only = False
    if refs_found:
        return BORROWED if borrowed_only else VERIFIED

    # (c) 재무 수치 — 사실팩이 제시한 값과 맞는지
    known = known_numbers(facts)
    if known:
        nums = _numbers_in(text)
        if nums and any(_matches_known(n, known) for n in nums):
            return VERIFIED

    # 검증할 거리가 없다. 틀렸다는 뜻이 아니므로 강등하지 않는다.
    return UNVERIFIABLE


CAPS: dict[str, str] = {
    FABRICATED: FABRICATED_SEVERITY_CAP,
    BORROWED: BORROWED_SEVERITY_CAP,
}
REASON: dict[str, str] = {
    FABRICATED: "실재하지 않음",
    BORROWED: "남의 조정례를 근거로 재활용",
}


def apply_anchor_verification(
    concerns: list,
    *,
    product: Product,
    doc_ids: set[str],
    case_doc_ids: set[str] | None = None,
    facts: FactPack | None = None,
) -> tuple[list, list[str]]:
    """우려 목록의 앵커를 검증하고, 부적절한 인용은 심각도를 낮춘다.

    반환: (우려 목록[제자리 수정], 강등 기록 문자열)
    """
    c_ids = clause_keys(product)
    demoted: list[str] = []
    for c in concerns:
        status = verify_anchor(
            c.anchor, clause_ids=c_ids, doc_ids=doc_ids,
            case_doc_ids=case_doc_ids, facts=facts,
        )
        c.anchor_status = status
        cap = CAPS.get(status)
        if not cap:
            continue
        before = c.severity
        c.severity = cap_severity(c.severity, cap)
        if severity_rank(before) > severity_rank(c.severity):
            demoted.append(
                f"{c.statement} [강등 {before}→{c.severity}: 앵커 '{c.anchor}' {REASON[status]}]"
            )
    return concerns, demoted

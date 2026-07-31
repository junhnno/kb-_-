"""공통 사실팩 — LLM에게 넘길 파생 지표를 코드로 계산하고, 계산값과 모순되는 우려를 차단한다.

왜 필요한가 (실측된 실패):
  - 페르소나 DSR 4.1%인데 "DSR 근접 시 위험" (CASE-002)
  - 월 여유자금 185만원인데 "여유자금 부담" (CASE-006)
  - 예금자보호 한도 내 예치인데 "한도 초과 우려" (CASE-006)
  - 우대조건이 없는 ELS 상품인데 "우대조건 충족 불가" (CASE-001)

원인은 추론 능력이 아니라, **파생 지표가 프롬프트에 없어서** LLM이 직접 나눗셈하고
판단해야 했다는 점이다. 그래서 두 가지를 한다.

  1) 계산해서 프롬프트에 주입 (사전 예방)
  2) 결과에서 계산값과 모순되는 우려를 제거 (사후 차단)
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Callable

from pydantic import BaseModel, Field

from .config import BENCHMARK_DIR
from .personas.schema import Persona
from .products.schema import Product

TAXONOMY_PATH = BENCHMARK_DIR / "concern_taxonomy.json"

# 예금자보호 한도 (원금+이자, 1인당 금융회사별)
DEPOSIT_PROTECTION_LIMIT_MANWON = 5000
# 스트레스 DSR: 변동금리 대출에 가산하는 금리 (%p)
STRESS_RATE_ADD_PCT = 3.0


@lru_cache(maxsize=1)
def _taxonomy() -> dict:
    return json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))


def _monthly_payment(principal_manwon: float, annual_rate_pct: float, months: int) -> float:
    """원리금균등 월 상환액. 금리 0이면 원금 균등분할."""
    if months <= 0:
        return 0.0
    r = annual_rate_pct / 100 / 12
    if r <= 0:
        return principal_manwon / months
    return principal_manwon * r / (1 - (1 + r) ** (-months))


class FactPack(BaseModel):
    """상품 × 페르소나에서 계산된 파생 지표. 모든 금액은 만원 단위."""

    product_id: str
    persona_id: str
    category: str

    # 페르소나 재무 (원본)
    monthly_income: int = 0
    monthly_surplus: int = 0
    monthly_debt_service: int = 0
    dsr_pct: float = 0.0
    liquid_assets: int = 0
    surplus_ratio: float = Field(
        default=0.0, description="월 여유자금 / 월소득. 정기 납입이 없는 상품의 여력 판정에 쓴다"
    )

    # 상품 부담 (시나리오별) — 단일 값으로 정하면 자의적이므로 시나리오를 함께 제시한다
    payment_min: float | None = None
    payment_max: float | None = None
    burden_min: float | None = Field(default=None, description="최소 납입액 / 월 여유자금")
    burden_max: float | None = Field(default=None, description="최대 납입액 / 월 여유자금")

    # 대출
    stressed_rate_pct: float | None = None
    stressed_dsr_pct: float | None = None

    # 상품 속성 (규칙 판정용).
    # **None = 모름**이며 '아니오'와 엄격히 구분한다. 모르는 것을 부정 사실로
    # 취급하면 정당한 우려를 기각하고, 틀린 사실을 프롬프트에 주입하게 된다.
    principal_guaranteed: bool | None = None
    rate_is_variable: bool | None = None
    has_preferentials: bool | None = None
    joint_attainment: float | None = Field(
        default=None, description="우대조건 전부 동시 충족 추정 확률 (est_attainment_rate의 곱)"
    )
    has_fees: bool | None = None
    deposit_exposure: int | None = Field(default=None, description="예금자보호 대상 예치 추정액")
    burden_effective: float | None = Field(
        default=None,
        description=(
            "심각도 판정에 쓰는 대표 부담률. 상품군마다 기준이 다르다 — "
            "적금은 **최소 납입액**(최소도 감당 못 하면 진짜 문제), 대출은 월 상환액, "
            "카드는 실적 조건 부담, 예금·펀드는 일시 예치액/금융자산. "
            "최대 납입액으로 계산하면 '월 여유 37만원에 최대 100만원 납입 = 270%' 같은 "
            "비현실적 값이 나와 신호를 오염시킨다."
        ),
    )
    lump_sum_burden: float | None = Field(
        default=None,
        description="일시 예치액 / 금융자산. 월 납입이 없는 상품(예금·펀드)의 감당능력 지표",
    )
    is_lump_sum_repayment: bool | None = None

    # 플래그 — 값으로 표현하면 왜곡되는 상태
    surplus_nonpositive: bool = Field(
        default=False, description="월 여유자금이 0 이하. 이때 부담률은 계산하지 않는다"
    )
    age: int = 0
    is_senior: bool = False

    def prompt_block(self) -> str:
        """프롬프트에 삽입되는 계산 결과. LLM이 직접 나눗셈하지 않게 한다."""
        L = ["[시스템 계산값] — 아래 수치는 코드가 계산한 것이다. 이와 모순되는 주장은 기각된다."]
        L.append(
            f"- 월소득 {self.monthly_income:,}만원 / 월 여유자금 {self.monthly_surplus:,}만원 "
            f"(소득의 {self.surplus_ratio:.0%})"
        )
        L.append(f"- 금융자산 {self.liquid_assets:,}만원")
        if self.surplus_nonpositive:
            L.append("- ⚠ 월 여유자금이 0 이하다. 신규 납입·상환 여력이 없다고 보아야 한다")
        L.append(f"- 기존 월 상환액 {self.monthly_debt_service:,}만원 (DSR {self.dsr_pct:.1f}%)")
        if self.burden_max is not None:
            L.append(
                f"- 이 상품 월 납입/상환 {self.payment_min:,.0f}~{self.payment_max:,.0f}만원 → "
                f"여유자금 대비 부담률 {self.burden_min:.0%}~{self.burden_max:.0%}"
            )
        if self.is_lump_sum_repayment:
            L.append(
                "- 상환구조: 만기 일시상환 (매월 이자만 납부, 원금은 만기에 전액 상환). "
                "만기 시 상환·차환 부담이 별도로 존재한다"
            )
        if self.stressed_dsr_pct is not None:
            L.append(
                f"- 스트레스 금리 {self.stressed_rate_pct:.1f}% 적용 시 DSR {self.stressed_dsr_pct:.1f}% "
                f"(규제 한도 40%)"
            )
        # 모르는 것은 '모름'이라고 쓴다. 단정하면 틀린 사실을 주입하게 된다.
        if self.principal_guaranteed is not None:
            L.append(f"- 원금보장: {'예' if self.principal_guaranteed else '아니오'}")
        if self.rate_is_variable is None:
            L.append("- 금리유형: 상품자료 미기재 (변동 여부 확인 필요)")
        else:
            L.append(f"- 금리유형: {'변동' if self.rate_is_variable else '고정'}")
        if self.has_preferentials:
            att = f"{self.joint_attainment:.1%}" if self.joint_attainment is not None else "추정치 없음"
            L.append(f"- 우대조건 있음. 전부 동시 충족 추정 확률 {att}")
        elif self.has_preferentials is False:
            L.append("- 우대조건 없음 (우대조건 관련 우려는 성립하지 않는다)")
        else:
            L.append("- 우대조건: 상품자료 미기재 (확인 필요)")
        if self.has_fees is False:
            L.append("- 수수료: 없음")
        elif self.has_fees is None:
            L.append("- 수수료: 상품자료 미기재 (확인 필요)")
        if self.deposit_exposure is not None:
            over = self.deposit_exposure > DEPOSIT_PROTECTION_LIMIT_MANWON
            L.append(
                f"- 예금자보호 대상 예치 추정 {self.deposit_exposure:,}만원 / 한도 "
                f"{DEPOSIT_PROTECTION_LIMIT_MANWON:,}만원 → {'초과' if over else '한도 내'}"
            )
        L.append(f"- 연령 {self.age}세{' (고령자 보호 대상)' if self.is_senior else ''}")
        return "\n".join(L)


def build_fact_pack(product: Product, persona: Persona) -> FactPack:
    f = persona.finance
    income = f.monthly_income_manwon if f else 0
    surplus = f.monthly_surplus_manwon if f else 0
    fp = FactPack(
        product_id=product.product_id,
        persona_id=persona.persona_id,
        category=product.category,
        monthly_income=income,
        monthly_surplus=surplus,
        monthly_debt_service=f.monthly_debt_service_manwon if f else 0,
        dsr_pct=f.dsr_pct if f else 0.0,
        liquid_assets=f.financial_assets_manwon if f else 0,
        surplus_nonpositive=surplus <= 0,
        surplus_ratio=round(surplus / income, 3) if income > 0 else 0.0,
        age=persona.age,
        is_senior=persona.age >= 65,
        # 미기재(None)는 None으로 전파한다. bool()로 변환하면 '모름'이 '아니오'가 된다.
        rate_is_variable=None if product.rate_basis is None else product.rate_basis == "변동",
        has_preferentials=None if product.preferentials is None else bool(product.preferentials),
        has_fees=None if product.fees is None else bool(product.fees),
        is_lump_sum_repayment=(
            any("일시상환" in n for n in product.risk_notes) if product.risk_notes else None
        ),
    )

    # 원금보장 여부 — 상품군으로 판정. 대출·카드는 해당 없음(None)
    if product.category in {"saving", "deposit"}:
        fp.principal_guaranteed = True
    elif product.category in {"fund", "pension"}:
        fp.principal_guaranteed = False

    # 우대조건 동시 충족 확률 (추정치가 있는 것만 곱한다)
    rates = [p.est_attainment_rate for p in (product.preferentials or []) if p.est_attainment_rate]
    if rates:
        joint = 1.0
        for r in rates:
            joint *= r
        fp.joint_attainment = round(joint, 4)

    # 월 납입/상환액 시나리오
    if product.category in {"saving"}:
        fp.payment_min = float(product.min_monthly_manwon or 0)
        fp.payment_max = float(product.max_monthly_manwon or product.min_monthly_manwon or 0)
    elif product.category == "loan" and product.limit_manwon:
        rate = product.intr_rate or 5.0
        months = product.save_trm_months or 60

        def _pay(r: float) -> float:
            # 만기 일시상환은 매월 이자만 내고 원금은 만기에 갚는다.
            # 원리금균등으로 계산하면 상환액이 크게 부풀려진다
            # (실측: 12개월 4,000만원 대출이 월 344만원 → 실제 이자만 20.7만원, 17배 과대).
            if fp.is_lump_sum_repayment:
                return product.limit_manwon * r / 100 / 12
            return _monthly_payment(product.limit_manwon, r, months)

        pay = _pay(rate)
        fp.payment_min = round(pay, 1)
        fp.payment_max = round(pay, 1)
        stressed_rate = rate + STRESS_RATE_ADD_PCT
        stressed_pay = _pay(stressed_rate)
        fp.stressed_rate_pct = stressed_rate
        if income > 0:
            fp.stressed_dsr_pct = round(
                100 * (fp.monthly_debt_service + stressed_pay) / income, 1
            )
    elif product.category == "card":
        # 카드는 실적 조건 금액을 '부담'으로 본다 (전월 실적을 채우려면 그만큼 써야 한다)
        fp.payment_min = 0.0
        fp.payment_max = float(product.limit_manwon or 0) / 10 if product.limit_manwon else None

    if fp.payment_max is not None and not fp.surplus_nonpositive:
        fp.burden_min = round((fp.payment_min or 0) / surplus, 3)
        fp.burden_max = round(fp.payment_max / surplus, 3)

    # 예금자보호 노출액 — 예금은 한도, 적금은 만기 원금 추정
    if product.category == "deposit":
        fp.deposit_exposure = product.limit_manwon
    elif product.category == "saving" and product.max_monthly_manwon and product.save_trm_months:
        # 적금 만기 원금은 여러 해에 걸쳐 쌓이므로 즉시 보호 한도 문제가 아니다.
        # 노출액으로 잡지 않는다 (5년 적금에 '한도 초과' 우려가 붙는 것을 막는다).
        fp.deposit_exposure = None

    # 일시 예치형 상품의 감당능력: 월 납입이 없으므로 금융자산 대비로 본다.
    # (실측 사고: 1억 예치가 필요한 상품인데 월 여유자금만 보고 '부담 없음'으로 기각했다)
    if product.category in {"deposit", "fund"} and product.limit_manwon:
        # 상품 한도는 "최대 이만큼까지 가능"이지 이 사람이 넣을 금액이 아니다.
        # 실제로 예치할 수 있는 금액 = min(상품 한도, 보유 금융자산)으로 본다.
        # (한도 1억 상품을 금융자산 1,300만원인 사람에게 = 769% 같은 값이 나오는 것을 막는다)
        placeable = min(product.limit_manwon, fp.liquid_assets)
        fp.lump_sum_burden = round(placeable / max(fp.liquid_assets, 1), 2)
        # 예금자보호 노출액도 실제 예치 가능액 기준으로 다시 잡는다
        fp.deposit_exposure = placeable

    # 상품군별 대표 부담률 (심각도 판정용)
    if product.category == "saving":
        fp.burden_effective = fp.burden_min      # 최소 납입도 버거운가
    elif product.category in {"loan", "card"}:
        fp.burden_effective = fp.burden_max      # 상환액·실적조건은 고정 부담
    elif product.category in {"deposit", "fund"}:
        fp.burden_effective = fp.lump_sum_burden

    return fp


# --------------------------------------------------------------- 모순 판정 규칙
# 반환 True = "이 우려는 계산값과 모순된다(성립하지 않는다)"
# 판정할 근거가 없으면 False를 반환해 우려를 살려둔다 (안전한 방향이 기본값)
CONTRADICTION_RULES: dict[str, Callable[[FactPack], bool]] = {
    # 여유자금 대비 부담이 명백히 낮으면 감당능력 우려는 성립하지 않는다.
    # 정기 납입이 없는 상품(예금·펀드)은 부담률이 계산되지 않으므로,
    # 여유자금이 소득의 20% 이상인지로 판정한다.
    "affordability": lambda f: (
        not f.surplus_nonpositive
        # 일시 예치액이 금융자산의 80%를 넘으면 감당능력 우려는 정당하다
        and (f.lump_sum_burden is None or f.lump_sum_burden < 0.8)
        and (
            (f.burden_max is not None and f.burden_max < 0.3)
            or (f.burden_max is None and f.surplus_ratio >= 0.2)
        )
    ),
    # DSR이 낮고 스트레스 적용해도 여유가 있으면 DSR 우려는 성립하지 않는다
    "dsr_overload": lambda f: (
        f.dsr_pct < 20.0 and (f.stressed_dsr_pct is None or f.stressed_dsr_pct < 30.0)
    ),
    # 우대조건이 아예 없는 상품 (미기재면 판정하지 않는다)
    "preferential_unattainable": lambda f: f.has_preferentials is False,
    # 원금보장 상품에 원금손실 우려.
    # 단 예금자보호 한도를 넘는 예치액이 있으면 초과분은 실제로 손실 위험이 있으므로
    # 기각하지 않는다 (실측 사고: 한도 초과 1억2천만원 예치 건의 정당한 우려를 지웠다).
    "principal_loss_risk": lambda f: (
        f.principal_guaranteed is True
        and not (
            f.deposit_exposure is not None
            and f.deposit_exposure > DEPOSIT_PROTECTION_LIMIT_MANWON
        )
    ),
    # 고정금리 상품에 금리상승 우려 (금리유형 미기재면 판정하지 않는다)
    "rate_variability": lambda f: f.rate_is_variable is False,
    # 예금자보호 한도 내인데 초과 우려
    "deposit_protection_limit": lambda f: (
        f.deposit_exposure is not None and f.deposit_exposure <= DEPOSIT_PROTECTION_LIMIT_MANWON
    ),
    # 카드가 아닌 상품에 리볼빙 우려
    "revolving_debt_spiral": lambda f: f.category != "card",
    # 만기 일시상환 구조가 아닌 상품 (유의사항 미기재면 판정하지 않는다)
    "maturity_refinance_risk": lambda f: f.is_lump_sum_repayment is False,
    # 수수료가 명시적으로 없는 상품 (fees 미기재면 판정하지 않는다)
    "fee_hidden": lambda f: f.has_fees is False,
    # 고령자가 아닌 페르소나
    "senior_vulnerability": lambda f: f.age < 60,
}


# --------------------------------------------------- 계산값 → 심각도 하한(승격)
# "하드 게이트"의 대안이다. 계산값으로 **라벨을 강제하면** 정답셋과 충돌한다
# (실측: 12건 중 6건 발동, 2건이 gold와 충돌 — 정답 라벨은 '판매 행위의 적법성'이고
#  게이트는 '재무 감당능력'이라 축이 다르다).
# 대신 **우려의 심각도 하한**을 걸면 라벨을 건드리지 않으면서 계산값이 LLM 판단을 이긴다.
# 근거: '치명' 우려는 3/3이 정답이었다 — 치명 판정을 계산으로 늘리는 것이 곧 정확도 개선.
SEVERITY_FLOORS: dict[str, Callable[[FactPack], str | None]] = {
    # 예금·펀드는 자산을 **소비**하는 게 아니라 옮기는 것이라 부담률 기준이 다르다.
    # (금융자산 전액 예치 = 100%인데, 이걸 적금 납입 100%와 같이 보면 안 된다)
    # 유동성이 전부 묶이는 경우만 '주의'로 표시하고, 감당능력 '치명'은 붙이지 않는다.
    "affordability": lambda f: (
        None
        if f.category in {"deposit", "fund"}
        else (
            "치명"
            if f.surplus_nonpositive or (f.burden_effective is not None and f.burden_effective > 0.8)
            else ("중대" if f.burden_effective is not None and f.burden_effective > 0.5 else None)
        )
    ),
    # (유동성 묶임은 별도 우려 유형이 필요하지만, 분류 체계·정답셋에 없어 지금은 두지 않는다.
    #  분류에 없는 유형을 만들면 모델이 제기할 때마다 오탐으로 집계된다.)
    "dsr_overload": lambda f: (
        "치명"
        if (f.stressed_dsr_pct or 0) > 50 or f.dsr_pct > 45
        else ("중대" if (f.stressed_dsr_pct or 0) > 40 or f.dsr_pct > 35 else None)
    ),
    "preferential_unattainable": lambda f: (
        "중대"
        if f.joint_attainment is not None and f.joint_attainment < 0.1
        else ("주의" if f.joint_attainment is not None and f.joint_attainment < 0.3 else None)
    ),
    "principal_loss_risk": lambda f: "중대" if f.principal_guaranteed is False else None,
    "deposit_protection_limit": lambda f: (
        "중대"
        if f.deposit_exposure is not None
        and f.deposit_exposure > DEPOSIT_PROTECTION_LIMIT_MANWON
        else None
    ),
    "maturity_refinance_risk": lambda f: "주의" if f.is_lump_sum_repayment else None,
    "senior_vulnerability": lambda f: "중대" if f.age >= 70 else ("주의" if f.is_senior else None),
}


def apply_severity_floors(concerns: list, facts: FactPack) -> list:
    """계산값이 요구하는 최소 심각도로 끌어올린다 (낮추지는 않는다)."""
    from .concerns import severity_rank

    for c in concerns:
        rule = SEVERITY_FLOORS.get(c.type)
        floor = rule(facts) if rule else None
        if floor and severity_rank(floor) > severity_rank(c.severity):
            c.severity = floor
            if not c.anchor:
                c.anchor = "[시스템 계산값]"
    return concerns


def is_contradicted(type_id: str, facts: FactPack) -> bool:
    """유형이 계산값과 모순되는가. 규칙이 없는 유형은 판정하지 않는다(False)."""
    rule = CONTRADICTION_RULES.get(type_id)
    return bool(rule and rule(facts))


def screen_typed_concerns(concerns: list, facts: FactPack) -> tuple[list, list]:
    """구조화된 우려(Concern)를 유형으로 직접 걸러낸다.

    키워드 추측이 없으므로 문자열판(screen_concerns)보다 정확하다.
    반환: (유지, 기각[사유 포함 문자열])
    """
    kept, dropped = [], []
    for c in concerns:
        if is_contradicted(c.type, facts):
            dropped.append(f"{c.statement} [기각: {c.type} 계산값과 모순]")
        else:
            kept.append(c)
    return kept, dropped


def classify_concern(text: str) -> set[str]:
    """우려 문장에서 우려 유형을 추정한다 (taxonomy 키워드 기반)."""
    return {
        t["id"] for t in _taxonomy()["types"] if any(kw in text for kw in t["keywords"])
    }


def screen_concerns(concerns: list[str], facts: FactPack) -> tuple[list[str], list[str]]:
    """계산값과 모순되는 우려를 걸러낸다.

    반환: (유지된 우려, 기각된 우려[사유 포함])
    유형을 특정할 수 없는 우려는 **유지**한다 — 판정 근거가 없으면 지우지 않는다.
    """
    kept: list[str] = []
    dropped: list[str] = []
    for c in concerns:
        types = classify_concern(c)
        reasons = [
            t for t in types if t in CONTRADICTION_RULES and CONTRADICTION_RULES[t](facts)
        ]
        # 모순되지 않는 유형이 하나라도 있으면 살린다 (여러 유형이 걸린 경우 보수적으로)
        if reasons and not (types - set(reasons)):
            dropped.append(f"{c} [기각: {', '.join(reasons)} 계산값과 모순]")
        else:
            kept.append(c)
    return kept, dropped

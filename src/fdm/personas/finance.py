"""페르소나에 재무 프로파일을 부여한다.

Nemotron-Personas-Korea에는 정밀한 소득/자산/부채 수치가 없으므로,
통계청 가계금융복지조사 기반 연령대별 분포 파라미터(data/benchmark/kosis_household_finance.json)에서
로그정규 근사로 합성값을 뽑는다. persona_id 해시를 시드로 써서 재현 가능하다.

주의: 개별 수치는 실제 개인이 아니며, 변수 간 결합분포(joint distribution) 정합성은
검증되지 않았다. 결과는 '탐색·경보용'으로만 사용한다 (CLAUDE.md §6).
"""

from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache

from ..config import BENCHMARK_DIR
from .schema import FinanceProfile, Persona, age_band

KOSIS_PATH = BENCHMARK_DIR / "kosis_household_finance.json"


@lru_cache(maxsize=1)
def load_kosis_params() -> dict:
    with open(KOSIS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _seeded_uniform(*parts: object) -> float:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:12], 16) / float(0x1000000000000)


def _norm_inv(u: float) -> float:
    """표준정규 분위수 근사 (Acklam 간이판)."""
    u = min(max(u, 1e-6), 1 - 1e-6)
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if u < plow:
        q = math.sqrt(-2 * math.log(u))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if u > phigh:
        q = math.sqrt(-2 * math.log(1 - u))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = u - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    )


def _lognormal(mean: float, cv: float, u: float) -> float:
    """평균 mean, 변동계수 cv인 로그정규분포의 u분위수."""
    sigma = math.sqrt(math.log(1 + cv**2))
    mu = math.log(max(mean, 1.0)) - sigma**2 / 2
    return math.exp(mu + sigma * _norm_inv(u))


def _quintile(income: int, cutoffs: list[int]) -> int:
    for i, c in enumerate(cutoffs, start=1):
        if income < c:
            return i
    return 5


def attach_finance(p: Persona, *, overwrite: bool = False) -> Persona:
    if p.finance is not None and not overwrite:
        return p
    params = load_kosis_params()
    band = params["by_age_band"].get(age_band(p.age), params["by_age_band"]["40대"])

    u_inc = _seeded_uniform(p.persona_id, "income")
    income = _lognormal(band["income_mean_manwon"], band["income_cv"], u_inc)
    # 직업군 배수 (거친 보정)
    for kw, mult in params["occupation_multiplier"].items():
        if kw in p.occupation:
            income *= mult
            break
    # 가구원 수가 많으면 가구 경상소득 상향
    income *= 1 + 0.12 * max(0, p.household_size - 1)
    income = int(round(income / 10) * 10)

    # 자산·부채는 소득 분위와 상관을 갖도록 u를 섞는다
    u_mix = 0.6 * u_inc + 0.4 * _seeded_uniform(p.persona_id, "asset")
    fin_assets = int(_lognormal(band["fin_assets_mean_manwon"], band["fin_assets_cv"], u_mix))
    real_assets = int(_lognormal(band["real_assets_mean_manwon"], band["real_assets_cv"], u_mix))

    u_debt = _seeded_uniform(p.persona_id, "debt")
    has_debt = u_debt < band["debt_holder_ratio"]
    debt = (
        int(_lognormal(band["debt_mean_manwon"], band["debt_cv"], u_debt / band["debt_holder_ratio"]))
        if has_debt
        else 0
    )

    monthly_income = max(1, int(round(income / 12)))
    # 월 원리금상환액 ≈ 부채 × 연이자+원금상환율 / 12
    monthly_ds = int(round(debt * params["annual_debt_service_rate"] / 12))
    dsr = round(100 * monthly_ds / monthly_income, 1)
    consumption = params["consumption_ratio_by_band"].get(age_band(p.age), 0.62)
    surplus = int(round(monthly_income * (1 - consumption))) - monthly_ds

    p.finance = FinanceProfile(
        annual_income_manwon=income,
        monthly_income_manwon=monthly_income,
        financial_assets_manwon=fin_assets,
        real_assets_manwon=real_assets,
        debt_manwon=debt,
        monthly_debt_service_manwon=monthly_ds,
        monthly_surplus_manwon=surplus,
        dsr_pct=dsr,
        income_quintile=_quintile(income, params["income_quintile_cutoffs_manwon"]),
    )
    return p

"""페르소나 / 세그먼트 스키마."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

AGE_BANDS = ["20대", "30대", "40대", "50대", "60대이상"]


def age_band(age: int) -> str:
    if age < 30:
        return "20대"
    if age < 40:
        return "30대"
    if age < 50:
        return "40대"
    if age < 60:
        return "50대"
    return "60대이상"


class FinanceProfile(BaseModel):
    """통계청 가계금융복지조사 분포를 참고해 부여한 재무 프로파일 (합성값)."""

    annual_income_manwon: int = Field(description="연 경상소득 (만원)")
    monthly_income_manwon: int
    financial_assets_manwon: int = Field(description="금융자산 (만원)")
    real_assets_manwon: int = Field(description="실물자산 (만원)")
    debt_manwon: int = Field(description="부채 총액 (만원)")
    monthly_debt_service_manwon: int = Field(description="월 원리금상환액 (만원)")
    monthly_surplus_manwon: int = Field(description="월 여유자금 (만원)")
    dsr_pct: float = Field(description="소득 대비 원리금상환비율 (%)")
    income_quintile: int = Field(ge=1, le=5, description="소득 5분위")
    source: str = "KOSIS 가계금융복지조사 분포 참고(합성 부여값)"

    def summary(self) -> str:
        return (
            f"연소득 {self.annual_income_manwon:,}만원(소득 {self.income_quintile}분위), "
            f"월소득 {self.monthly_income_manwon:,}만원, 금융자산 {self.financial_assets_manwon:,}만원, "
            f"부채 {self.debt_manwon:,}만원, 월상환액 {self.monthly_debt_service_manwon:,}만원 "
            f"(DSR {self.dsr_pct:.1f}%), 월 여유자금 {self.monthly_surplus_manwon:,}만원"
        )


class Persona(BaseModel):
    persona_id: str
    age: int
    sex: Literal["남성", "여성", "미상"] = "미상"
    region: str = "미상"
    occupation: str = "미상"
    education: str = "미상"
    marital_status: str = "미상"
    household_size: int = 1
    persona_text: str = ""
    traits: list[str] = Field(default_factory=list)
    finance: FinanceProfile | None = None
    source: str = "nvidia/Nemotron-Personas-Korea"

    @property
    def band(self) -> str:
        return age_band(self.age)

    def prompt_block(self) -> str:
        """디베이트 프롬프트에 삽입되는 페르소나 서술."""
        lines = [
            f"- ID: {self.persona_id}",
            f"- 인구: {self.age}세 {self.sex}, {self.region}, 가구원 {self.household_size}명, {self.marital_status}",
            f"- 직업/학력: {self.occupation} / {self.education}",
        ]
        if self.traits:
            lines.append(f"- 성향: {', '.join(self.traits)}")
        if self.persona_text:
            lines.append(f"- 서술: {self.persona_text.strip()[:600]}")
        if self.finance:
            lines.append(f"- 재무: {self.finance.summary()}")
        return "\n".join(lines)


class Segment(BaseModel):
    """타깃 조건. 필드가 None이면 해당 조건은 무시."""

    name: str
    age_min: int | None = None
    age_max: int | None = None
    regions: list[str] | None = None
    occupations_include: list[str] | None = None
    income_min_manwon: int | None = None
    income_max_manwon: int | None = None
    income_quintiles: list[int] | None = None
    sex: str | None = None
    dsr_min_pct: float | None = None
    monthly_surplus_max_manwon: int | None = None

    def matches(self, p: Persona) -> bool:
        if self.age_min is not None and p.age < self.age_min:
            return False
        if self.age_max is not None and p.age > self.age_max:
            return False
        if self.sex and p.sex != self.sex:
            return False
        if self.regions and not any(r in p.region for r in self.regions):
            return False
        if self.occupations_include and not any(
            o in p.occupation for o in self.occupations_include
        ):
            return False
        f = p.finance
        if f is None:
            return not (
                self.income_min_manwon
                or self.income_max_manwon
                or self.income_quintiles
                or self.dsr_min_pct
                or self.monthly_surplus_max_manwon
            )
        if self.income_min_manwon is not None and f.annual_income_manwon < self.income_min_manwon:
            return False
        if self.income_max_manwon is not None and f.annual_income_manwon > self.income_max_manwon:
            return False
        if self.income_quintiles and f.income_quintile not in self.income_quintiles:
            return False
        if self.dsr_min_pct is not None and f.dsr_pct < self.dsr_min_pct:
            return False
        if (
            self.monthly_surplus_max_manwon is not None
            and f.monthly_surplus_manwon > self.monthly_surplus_max_manwon
        ):
            return False
        return True

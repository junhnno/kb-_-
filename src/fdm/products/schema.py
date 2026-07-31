"""신규 금융상품 정의 스키마.

필드명은 금융감독원 「금융상품 한눈에」 오픈API(finlife.fss.or.kr) 응답을
참고했다(intr_rate=기본금리, intr_rate2=최고우대금리, save_trm=저축기간 등).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ..config import PRODUCT_DIR

Category = Literal["saving", "deposit", "loan", "card", "pension", "fund"]


class Clause(BaseModel):
    """약관/상품설명서 조항. 에이전트가 근거로 인용하는 단위."""

    id: str = Field(description="예: 약관 제5조")
    title: str
    text: str

    def cite(self) -> str:
        return f"[{self.id}({self.title})] {self.text}"


class Preferential(BaseModel):
    """우대조건."""

    name: str
    rate_bonus_pct: float = 0.0
    requirement: str
    est_attainment_rate: float | None = Field(
        default=None, description="상품기획팀이 추정한 달성률(0~1). 비어 있으면 디베이트가 추정."
    )


class Fee(BaseModel):
    name: str
    amount: str
    condition: str = ""


class Product(BaseModel):
    product_id: str
    name: str
    category: Category
    issuer: str = "KB국민은행"
    summary: str = ""

    # 금리·기간·한도
    intr_rate: float | None = Field(default=None, description="기본금리(연 %) 또는 대출 최저금리")
    intr_rate2: float | None = Field(default=None, description="최고우대금리(연 %) 또는 대출 최고금리")
    intr_rate_type: str = "단리"
    # None = 미기재(모름). 기본값을 "고정"으로 두면 변동금리 상품이 조용히
    # 고정으로 단정되고, 사실팩이 정당한 '금리상승 위험' 우려를 기각한다(실측된 사고).
    rate_basis: Literal["고정", "변동"] | None = None
    save_trm_months: int | None = None
    min_monthly_manwon: int | None = None
    max_monthly_manwon: int | None = None
    limit_manwon: int | None = Field(default=None, description="대출/카드 한도 (만원)")

    # None = 미기재(모름) / [] = 명시적으로 없음.
    # 이 구분이 없으면 "데이터를 안 채웠다"가 "그런 조건은 없다"로 둔갑한다.
    preferentials: list[Preferential] | None = None
    fees: list[Fee] | None = None
    taxation: str = "이자소득세 15.4% 원천징수"
    early_termination: str = ""
    risk_notes: list[str] = Field(default_factory=list)

    target_description: str = ""
    target_segments: list[str] = Field(default_factory=list)
    clauses: list[Clause] = Field(default_factory=list)

    # ------------------------------------------------------------------ helpers
    def max_rate(self) -> float | None:
        return self.intr_rate2 or self.intr_rate

    def prompt_block(self) -> str:
        """디베이트 프롬프트에 삽입되는 상품 서술."""
        lines = [f"- 상품명: {self.name} ({self.issuer}, {self.category})"]
        if self.summary:
            lines.append(f"- 개요: {self.summary}")
        if self.intr_rate is not None:
            kind = "대출금리" if self.category == "loan" else "금리"
            basis = self.rate_basis or "금리유형 미기재"
            lines.append(
                f"- {kind}: 기본 연 {self.intr_rate}% / 최대 연 {self.intr_rate2}% "
                f"({basis}, {self.intr_rate_type})"
            )
        if self.save_trm_months:
            lines.append(f"- 기간: {self.save_trm_months}개월")
        if self.min_monthly_manwon or self.max_monthly_manwon:
            lines.append(
                f"- 월 납입: {self.min_monthly_manwon or 0}~{self.max_monthly_manwon or 0}만원"
            )
        if self.limit_manwon:
            lines.append(f"- 한도: {self.limit_manwon:,}만원")
        if self.preferentials:
            lines.append("- 우대조건:")
            lines += [
                f"  · {p.name} (+{p.rate_bonus_pct}%p): {p.requirement}" for p in self.preferentials
            ]
        elif self.preferentials == []:
            lines.append("- 우대조건: 없음")
        if self.fees:
            lines.append(
                "- 수수료: "
                + "; ".join(f"{f.name} {f.amount} {f.condition}".strip() for f in self.fees)
            )
        elif self.fees == []:
            lines.append("- 수수료: 없음")
        if self.early_termination:
            lines.append(f"- 중도해지/상환: {self.early_termination}")
        lines.append(f"- 과세: {self.taxation}")
        if self.risk_notes:
            lines.append("- 유의사항: " + "; ".join(self.risk_notes))
        if self.target_description:
            lines.append(f"- 타깃: {self.target_description}")
        if self.clauses:
            lines.append("- 인용 가능한 약관 조항:")
            lines += [f"  · {c.cite()}" for c in self.clauses]
        return "\n".join(lines)


def load_product(path: str | Path) -> Product:
    p = Path(path)
    if not p.exists():
        p = PRODUCT_DIR / f"{path}.json"
    with open(p, encoding="utf-8") as f:
        return Product(**json.load(f))


def load_all_products() -> list[Product]:
    return [load_product(f) for f in sorted(PRODUCT_DIR.glob("*.json"))]

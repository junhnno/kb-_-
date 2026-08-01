"""세그먼트 시뮬레이션 오케스트레이션 + 가격·조건 민감도 분석."""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ..agents.debate import DebateConfig, run_debate, single_shot
from ..agents.schema import Concern, DebateResult, group_by_tier, merge_concerns
from ..config import DATA_DIR, OUTPUT_DIR
from ..llm import LLMClient
from ..personas.loader import load_personas, sample_cohort
from ..personas.schema import Persona, Segment
from ..products.schema import Product
from ..rag.retriever import get_retriever
from .confidence import ConsensusResult, aggregate, seed_plan

Mode = Literal["debate", "single"]
SEGMENTS_PATH = DATA_DIR / "segments.json"


def load_segments(names: list[str] | None = None) -> list[Segment]:
    with open(SEGMENTS_PATH, encoding="utf-8") as f:
        segs = [Segment(**row) for row in json.load(f)]
    if names:
        wanted = set(names)
        segs = [s for s in segs if s.name in wanted]
    return segs


class SegmentResult(BaseModel):
    segment: str
    n_personas: int
    adoption_rate: float = Field(description="가입의향 평균 60점 이상인 페르소나 비율")
    mean_intent: float
    verdict_mix: dict[str, float] = Field(default_factory=dict)
    mean_confidence: float
    low_confidence_ratio: float
    top_risks: list[str] = Field(default_factory=list)
    top_recommendations: list[str] = Field(default_factory=list)
    cases: list[ConsensusResult] = Field(default_factory=list)
    top_concerns: list[Concern] = Field(
        default_factory=list,
        description="세그먼트 내 페르소나들의 우려를 유형 기준으로 합쳐 계층순 정렬한 것",
    )

    @property
    def tiers(self) -> dict[str, list[Concern]]:
        """계층별 우려. 산출물은 T1부터 위에서 아래로 읽는다."""
        return group_by_tier(self.top_concerns)

    @property
    def flag(self) -> str:
        if self.low_confidence_ratio >= 0.5:
            return "추가 검증 필요"
        if self.verdict_mix.get("fail", 0) >= 0.3:
            return "판매원칙 위험"
        if self.verdict_mix.get("warn", 0) >= 0.5:
            return "조건 보완 권고"
        return "정상"


class SimulationReport(BaseModel):
    product_id: str
    product_name: str
    mode: Mode
    n_seeds: int
    generated_at: str
    backend: str = ""
    model_small: str = ""
    model_judge: str = ""
    segments: list[SegmentResult] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def save(self, path: str | Path | None = None) -> Path:
        p = Path(path) if path else OUTPUT_DIR / f"sim_{self.product_id}_{self.mode}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return p

    @staticmethod
    def load(path: str | Path) -> "SimulationReport":
        return SimulationReport(**json.loads(Path(path).read_text(encoding="utf-8")))


# --------------------------------------------------------------------- 단일 케이스
def run_case(
    product: Product,
    persona: Persona,
    *,
    segment: str,
    n_seeds: int = 3,
    mode: Mode = "debate",
    config: DebateConfig | None = None,
    client: LLMClient | None = None,
    exclude_doc_ids: set[str] | None = None,
) -> ConsensusResult:
    cfg = config or DebateConfig()
    client = client or LLMClient()
    retriever = get_retriever(cfg.use_dense)
    runner = run_debate if mode == "debate" else single_shot

    runs: list[DebateResult] = []
    for seed, temp in seed_plan(n_seeds, base_temp=cfg.temperature_debater):
        c = DebateConfig(**{**cfg.__dict__, "temperature_debater": temp})
        runs.append(
            runner(
                product,
                persona,
                segment=segment,
                seed=seed,
                config=c,
                client=client,
                retriever=retriever,
                exclude_doc_ids=exclude_doc_ids,
            )
        )
    return aggregate(runs)


# ------------------------------------------------------------------- 상품 시뮬레이션
def simulate_product(
    product: Product,
    *,
    segment_names: list[str] | None = None,
    k_personas: int = 4,
    n_seeds: int = 3,
    mode: Mode = "debate",
    workers: int = 4,
    config: DebateConfig | None = None,
    personas: list[Persona] | None = None,
    progress: bool = True,
) -> SimulationReport:
    from ..config import SETTINGS

    cfg = config or DebateConfig()
    client = LLMClient()
    pool = personas if personas is not None else load_personas()
    names = segment_names or product.target_segments or None
    segments = load_segments(names)
    if not segments:
        segments = load_segments()

    jobs: list[tuple[Segment, Persona]] = []
    for seg in segments:
        for p in sample_cohort(pool, seg, k_personas):
            jobs.append((seg, p))

    def work(job: tuple[Segment, Persona]) -> tuple[str, ConsensusResult]:
        seg, persona = job
        cr = run_case(
            product, persona, segment=seg.name, n_seeds=n_seeds, mode=mode, config=cfg, client=client
        )
        if progress:
            print(
                f"  [{seg.name}] {persona.persona_id} ({persona.age}세 {persona.occupation}) "
                f"→ {cr.modal_suitability} / 의향 {cr.intent_mean} / 신뢰도 {cr.confidence}",
                flush=True,
            )
        return seg.name, cr

    results: list[tuple[str, ConsensusResult]] = []
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(work, jobs))
    else:
        results = [work(j) for j in jobs]

    by_segment: dict[str, list[ConsensusResult]] = {}
    for name, cr in results:
        by_segment.setdefault(name, []).append(cr)

    seg_results = []
    for seg in segments:
        crs = by_segment.get(seg.name, [])
        if not crs:
            continue
        n = len(crs)
        mix = Counter(c.modal_suitability for c in crs)
        seg_results.append(
            SegmentResult(
                segment=seg.name,
                n_personas=n,
                adoption_rate=round(sum(1 for c in crs if c.adopted) / n, 3),
                mean_intent=round(sum(c.intent_mean for c in crs) / n, 1),
                verdict_mix={k: round(v / n, 3) for k, v in mix.items()},
                mean_confidence=round(sum(c.confidence for c in crs) / n, 3),
                low_confidence_ratio=round(sum(1 for c in crs if c.needs_review) / n, 3),
                top_risks=_dedup([r for c in crs for r in c.risks])[:5],
                top_recommendations=_dedup([r for c in crs for r in c.recommendations])[:5],
                cases=crs,
                top_concerns=_segment_concerns(crs),
            )
        )

    return SimulationReport(
        product_id=product.product_id,
        product_name=product.name,
        mode=mode,
        n_seeds=n_seeds,
        generated_at=datetime.now().isoformat(timespec="seconds"),
        backend=SETTINGS.backend,
        model_small=SETTINGS.model_small,
        model_judge=SETTINGS.model_judge,
        segments=seg_results,
        notes=[
            "합성 페르소나 기반 결과다. 변수 간 결합분포 정합성은 검증되지 않았으므로 탐색·경보용으로만 사용한다.",
            "저신뢰(low) 세그먼트는 추가 검증 대상이다.",
        ],
    )


def _dedup(items: list[str]) -> list[str]:
    counts = Counter(i.strip() for i in items if i and i.strip())
    return [k for k, _ in counts.most_common()]


def _segment_concerns(crs: list[ConsensusResult]) -> list[Concern]:
    """세그먼트 내 페르소나별 우려를 유형 기준으로 합친다.

    sources는 제기 '방식'(single/debate)만 담으므로, 페르소나 수가 많다고
    교차확인으로 승격되지 않는다 — 계층이 표본 크기에 오염되지 않게 하는 지점이다.
    """
    groups: dict[str, list[Concern]] = {}
    for cr in crs:
        for c in cr.concerns:
            for src in c.sources or [cr.mode]:
                groups.setdefault(src, []).append(c)
    return merge_concerns(groups) if groups else []


# ------------------------------------------------------------------ 민감도 분석
class VariantSpec(BaseModel):
    label: str
    rate_delta_pct: float = 0.0
    max_monthly_delta_manwon: int = 0
    term_months: int | None = None
    drop_preferentials: list[str] = Field(default_factory=list)
    drop_all_fees: bool = False


def apply_variant(product: Product, spec: VariantSpec) -> Product:
    p = product.model_copy(deep=True)
    p.product_id = f"{product.product_id}::{spec.label}"
    if spec.rate_delta_pct:
        if p.intr_rate is not None:
            p.intr_rate = round(p.intr_rate + spec.rate_delta_pct, 3)
        if p.intr_rate2 is not None:
            p.intr_rate2 = round(p.intr_rate2 + spec.rate_delta_pct, 3)
    if spec.max_monthly_delta_manwon and p.max_monthly_manwon is not None:
        p.max_monthly_manwon = max(1, p.max_monthly_manwon + spec.max_monthly_delta_manwon)
    if spec.term_months:
        p.save_trm_months = spec.term_months
    if spec.drop_preferentials:
        p.preferentials = [
            x for x in (p.preferentials or []) if x.name not in spec.drop_preferentials
        ]
    if spec.drop_all_fees:
        p.fees = []
    return p


class SensitivityRow(BaseModel):
    label: str
    variant_id: str
    segment: str
    adoption_rate: float
    mean_intent: float
    mean_confidence: float
    fail_ratio: float


def sensitivity_analysis(
    product: Product,
    specs: list[VariantSpec],
    *,
    segment_names: list[str] | None = None,
    k_personas: int = 3,
    n_seeds: int = 2,
    mode: Mode = "debate",
    workers: int = 4,
) -> list[SensitivityRow]:
    rows: list[SensitivityRow] = []
    base = VariantSpec(label="기준안")
    for spec in [base] + specs:
        variant = apply_variant(product, spec)
        rep = simulate_product(
            variant,
            segment_names=segment_names or product.target_segments or None,
            k_personas=k_personas,
            n_seeds=n_seeds,
            mode=mode,
            workers=workers,
            progress=False,
        )
        for sr in rep.segments:
            rows.append(
                SensitivityRow(
                    label=spec.label,
                    variant_id=variant.product_id,
                    segment=sr.segment,
                    adoption_rate=sr.adoption_rate,
                    mean_intent=sr.mean_intent,
                    mean_confidence=sr.mean_confidence,
                    fail_ratio=sr.verdict_mix.get("fail", 0.0),
                )
            )
    return rows


def default_variants(product: Product) -> list[VariantSpec]:
    """상품군별 기본 민감도 시나리오."""
    if product.category in {"saving", "deposit"}:
        return [
            VariantSpec(label="금리 -0.5%p", rate_delta_pct=-0.5),
            VariantSpec(label="금리 +0.5%p", rate_delta_pct=0.5),
            VariantSpec(
                label="우대조건 최소화",
                drop_preferentials=[p.name for p in (product.preferentials or [])[1:]],
            ),
            VariantSpec(label="기간 12개월", term_months=12),
        ]
    if product.category == "loan":
        return [
            VariantSpec(label="금리 -0.5%p", rate_delta_pct=-0.5),
            VariantSpec(label="금리 +1.0%p", rate_delta_pct=1.0),
            VariantSpec(label="수수료 전면 면제", drop_all_fees=True),
        ]
    return [
        VariantSpec(label="연회비·수수료 면제", drop_all_fees=True),
        VariantSpec(
            label="실적조건 완화",
            drop_preferentials=[p.name for p in (product.preferentials or [])[1:]],
        ),
    ]

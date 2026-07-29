"""외부 벤치마크 대조 + 애블레이션.

(A) 분쟁조정 사례 정답셋 대조 — 적합성 판정 적중률
    arm 3종: single_norag(단발·근거없음) / single(단발·RAG) / debate(3진영 디베이트·RAG)
    평가 시 해당 사례 문서는 검색에서 제외한다(정답 누출 방지).

(B) KOSIS 보유율 대조 — 세그먼트별 가입의향률의 순위 일치도(Spearman) 및 MAE
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ..agents.debate import DebateConfig, run_debate, single_shot
from ..agents.schema import Suitability
from ..config import BENCHMARK_DIR, OUTPUT_DIR
from ..llm import LLMClient
from ..personas.schema import Persona
from ..products.schema import Product
from ..rag.retriever import get_retriever
from .confidence import aggregate
from .simulate import SimulationReport

CASES_PATH = BENCHMARK_DIR / "dispute_cases.json"
HOLDING_PATH = BENCHMARK_DIR / "segment_holding_rates.json"

Arm = Literal["single_norag", "single", "debate"]
ARMS: tuple[Arm, ...] = ("single_norag", "single", "debate")

RISKY = {"warn", "fail"}


class BenchCase(BaseModel):
    case_id: str
    title: str
    label: Suitability
    principle: list[str] = Field(default_factory=list)
    product: Product
    persona: Persona


def load_cases() -> list[BenchCase]:
    payload = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    out = []
    for c in payload["cases"]:
        out.append(
            BenchCase(
                case_id=c["case_id"],
                title=c["title"],
                label=c["label"],
                principle=c.get("principle", []),
                product=Product(**c["product"]),
                persona=Persona(**c["persona"]),
            )
        )
    return out


class CaseOutcome(BaseModel):
    case_id: str
    title: str
    gold: Suitability
    pred: Suitability
    exact: bool
    risk_correct: bool = Field(description="위험(warn/fail) 대 정상(pass) 이분류 일치")
    principle_recall: float = Field(
        description="정답 위반원칙 중 예측이 잡아낸 비율 (시드별 재현율의 평균)"
    )
    principle_recall_union: float = Field(
        default=0.0, description="시드 합집합 기준 재현율 (참고용 상한)"
    )
    judge_schema_repairs: int = Field(
        default=0, description="심판 응답 스키마 교정이 필요했던 횟수"
    )
    confidence: float
    intent_mean: float
    n_llm_calls: int
    elapsed_sec: float


class ArmScore(BaseModel):
    arm: Arm
    n: int
    accuracy: float
    risk_accuracy: float
    macro_f1: float
    principle_recall: float
    principle_recall_union: float = 0.0
    judge_schema_repairs: int = 0
    mean_confidence: float
    total_llm_calls: int
    total_elapsed_sec: float
    confusion: dict[str, dict[str, int]] = Field(default_factory=dict)
    outcomes: list[CaseOutcome] = Field(default_factory=list)


class AblationReport(BaseModel):
    generated_at: str
    backend: str = ""
    model_small: str = ""
    model_judge: str = ""
    n_seeds: int = 1
    arms: list[ArmScore] = Field(default_factory=list)
    delta_accuracy_vs_single: float = 0.0
    delta_risk_accuracy_vs_single: float = 0.0
    caveats: list[str] = Field(default_factory=list)

    def save(self, path: str | Path | None = None) -> Path:
        p = Path(path) if path else OUTPUT_DIR / "ablation.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return p

    @staticmethod
    def load(path: str | Path) -> "AblationReport":
        return AblationReport(**json.loads(Path(path).read_text(encoding="utf-8")))


def _macro_f1(pairs: list[tuple[str, str]]) -> float:
    labels = {"pass", "warn", "fail"}
    f1s = []
    for lab in labels:
        tp = sum(1 for g, p in pairs if g == lab and p == lab)
        fp = sum(1 for g, p in pairs if g != lab and p == lab)
        fn = sum(1 for g, p in pairs if g == lab and p != lab)
        if tp == 0 and (fp or fn):
            f1s.append(0.0)
            continue
        if tp == 0:
            continue  # 정답·예측 모두 없는 클래스는 제외
        prec, rec = tp / (tp + fp), tp / (tp + fn)
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return round(statistics.fmean(f1s), 3) if f1s else 0.0


def _principle_recall(gold: list[str], pred: list[str]) -> float:
    if not gold:
        return 1.0
    hit = sum(1 for g in gold if any(g in p or p in g for p in pred))
    return hit / len(gold)


def run_case_arm(
    case: BenchCase,
    arm: Arm,
    *,
    n_seeds: int = 1,
    config: DebateConfig | None = None,
    client: LLMClient | None = None,
) -> CaseOutcome:
    cfg = config or DebateConfig()
    client = client or LLMClient()
    retriever = get_retriever(cfg.use_dense)
    exclude = {case.case_id}  # 정답 누출 방지

    runs = []
    for i in range(n_seeds):
        seed = 7000 + 137 * i
        if arm == "debate":
            runs.append(
                run_debate(
                    case.product, case.persona, segment=case.case_id, seed=seed,
                    config=cfg, client=client, retriever=retriever, exclude_doc_ids=exclude,
                )
            )
        else:
            runs.append(
                single_shot(
                    case.product, case.persona, segment=case.case_id, seed=seed,
                    config=cfg, client=client, retriever=retriever, exclude_doc_ids=exclude,
                    with_rag=(arm == "single"),
                )
            )
    cr = aggregate(runs)
    pred = cr.modal_suitability
    n_calls = sum(len(r.turns) for r in runs)
    return CaseOutcome(
        case_id=case.case_id,
        title=case.title,
        gold=case.label,
        pred=pred,
        exact=pred == case.label,
        risk_correct=(pred in RISKY) == (case.label in RISKY),
        # 시드별로 각각 재현율을 구한 뒤 평균한다. 여러 시드의 위반원칙을 합집합으로
        # 모으면 "시드를 늘릴수록 재현율이 올라가는" 지표가 되어 arm 간 비교가 왜곡된다.
        principle_recall=round(
            statistics.fmean(
                _principle_recall(case.principle, r.verdict.violated_principles) for r in runs
            ),
            3,
        ),
        # 참고용: 합집합 기준(=시드 중 한 번이라도 잡았는가). 상한을 보여준다.
        principle_recall_union=round(
            _principle_recall(
                case.principle, [x for r in runs for x in r.verdict.violated_principles]
            ),
            3,
        ),
        judge_schema_repairs=sum(1 for r in runs if r.judge_schema_repaired),
        confidence=cr.confidence,
        intent_mean=cr.intent_mean,
        n_llm_calls=n_calls,
        elapsed_sec=round(sum(r.elapsed_sec for r in runs), 2),
    )


def run_ablation(
    *,
    arms: tuple[Arm, ...] = ARMS,
    n_seeds: int = 1,
    limit: int | None = None,
    config: DebateConfig | None = None,
    progress: bool = True,
) -> AblationReport:
    from ..config import SETTINGS

    cases = load_cases()[: limit or None]
    client = LLMClient()
    scores: list[ArmScore] = []

    for arm in arms:
        outcomes = []
        for case in cases:
            oc = run_case_arm(case, arm, n_seeds=n_seeds, config=config, client=client)
            outcomes.append(oc)
            if progress:
                mark = "O" if oc.exact else "X"
                print(
                    f"  [{arm}] {case.case_id} gold={oc.gold} pred={oc.pred} {mark} "
                    f"(신뢰도 {oc.confidence})",
                    flush=True,
                )
        pairs = [(o.gold, o.pred) for o in outcomes]
        confusion: dict[str, dict[str, int]] = {}
        for g, p in pairs:
            confusion.setdefault(g, Counter())[p] = confusion.setdefault(g, Counter()).get(p, 0) + 1
        scores.append(
            ArmScore(
                arm=arm,
                n=len(outcomes),
                accuracy=round(sum(o.exact for o in outcomes) / len(outcomes), 3),
                risk_accuracy=round(sum(o.risk_correct for o in outcomes) / len(outcomes), 3),
                macro_f1=_macro_f1(pairs),
                principle_recall=round(statistics.fmean(o.principle_recall for o in outcomes), 3),
                principle_recall_union=round(
                    statistics.fmean(o.principle_recall_union for o in outcomes), 3
                ),
                judge_schema_repairs=sum(o.judge_schema_repairs for o in outcomes),
                mean_confidence=round(statistics.fmean(o.confidence for o in outcomes), 3),
                total_llm_calls=sum(o.n_llm_calls for o in outcomes),
                total_elapsed_sec=round(sum(o.elapsed_sec for o in outcomes), 1),
                confusion={k: dict(v) for k, v in confusion.items()},
                outcomes=outcomes,
            )
        )

    by_arm = {s.arm: s for s in scores}
    d_acc = d_risk = 0.0
    if "debate" in by_arm and "single" in by_arm:
        d_acc = round(by_arm["debate"].accuracy - by_arm["single"].accuracy, 3)
        d_risk = round(by_arm["debate"].risk_accuracy - by_arm["single"].risk_accuracy, 3)

    return AblationReport(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        backend=SETTINGS.backend,
        model_small=SETTINGS.model_small,
        model_judge=SETTINGS.model_judge,
        n_seeds=n_seeds,
        arms=scores,
        delta_accuracy_vs_single=d_acc,
        delta_risk_accuracy_vs_single=d_risk,
        caveats=[
            f"정답셋 {len(cases)}건은 조정결정례 유형을 참조해 재구성한 가공 샘플이다. 실제 원문으로 교체 시 수치는 달라진다.",
            "표본이 작아 신뢰구간이 넓다. 결론은 '방향성'으로만 해석해야 한다.",
            "평가 시 해당 사례 문서를 검색 대상에서 제외해 정답 누출을 막았다.",
        ],
    )


# ------------------------------------------------------------- (B) KOSIS 보유율 대조
class HoldingComparison(BaseModel):
    segment: str
    category: str
    actual_holding_rate: float
    simulated_adoption_rate: float
    abs_error: float


class HoldingReport(BaseModel):
    product_id: str
    product_name: str
    category: str
    mode: str
    n_segments: int
    mae: float
    spearman: float | None
    rows: list[HoldingComparison] = Field(default_factory=list)
    note: str = (
        "'보유율'과 '가입의향률'은 정의가 달라 절대값 차이(MAE)보다 세그먼트 간 순위 일치도(Spearman)가 1차 지표다."
    )


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None

    def rank(v: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (
        sum((a - mx) ** 2 for a in rx) ** 0.5 * sum((b - my) ** 2 for b in ry) ** 0.5
    )
    return round(num / den, 3) if den else None


def compare_holding_rates(report: SimulationReport, category: str) -> HoldingReport:
    payload = json.loads(HOLDING_PATH.read_text(encoding="utf-8"))
    table = {
        (r["segment"], r["category"]): r["rate"] for r in payload["holding_rates"]
    }
    rows = []
    for sr in report.segments:
        actual = table.get((sr.segment, category))
        if actual is None:
            continue
        rows.append(
            HoldingComparison(
                segment=sr.segment,
                category=category,
                actual_holding_rate=actual,
                simulated_adoption_rate=sr.adoption_rate,
                abs_error=round(abs(actual - sr.adoption_rate), 3),
            )
        )
    mae = round(statistics.fmean(r.abs_error for r in rows), 3) if rows else 0.0
    rho = _spearman(
        [r.actual_holding_rate for r in rows], [r.simulated_adoption_rate for r in rows]
    )
    return HoldingReport(
        product_id=report.product_id,
        product_name=report.product_name,
        category=category,
        mode=report.mode,
        n_segments=len(rows),
        mae=mae,
        spearman=rho,
        rows=rows,
    )

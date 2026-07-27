"""CLI. 예: `uv run fdm simulate 01_youth_step_saving --seeds 3`"""

from __future__ import annotations

import json
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .agents.debate import DebateConfig, run_debate
from .config import OUTPUT_DIR, SETTINGS
from .eval.benchmark import compare_holding_rates, run_ablation
from .eval.simulate import (
    SimulationReport,
    default_variants,
    load_segments,
    sensitivity_analysis,
    simulate_product,
)
from .llm import LLMClient, LLMError
from .personas.loader import filter_segment, load_personas, sample_cohort
from .products.schema import load_all_products, load_product
from .report import build_report, save_report

# Windows 기본 콘솔 인코딩(cp949)에서는 조정례 제목의 '—', 근거 인용의 '…', 우대조건의 '·'
# 같은 문자가 UnicodeEncodeError를 내며 프로그램을 죽인다(rich도 막아주지 못한다).
# 출력 스트림을 UTF-8로 다시 열고, 그래도 표현 불가한 문자는 '?'로 대체한다.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, OSError):  # 리다이렉트된 스트림 등
        pass

app = typer.Typer(add_completion=False, help="합성 페르소나 기반 금융상품 설계·검증 에이전트")
console = Console()


@app.command()
def doctor() -> None:
    """실행 환경 점검 (백엔드 연결, 데이터 로딩)."""
    console.print(f"backend=[bold]{SETTINGS.backend}[/] base_url={SETTINGS.base_url}")
    console.print(f"토론모델={SETTINGS.model_small} / 심판모델={SETTINGS.model_judge}")
    try:
        res = LLMClient().chat(
            role="advocate", system="너는 테스트용 응답기다.", user="OK 라고만 답하라.", temperature=0.0
        )
        console.print(f"[green]LLM 응답 OK[/] ({res.model}): {res.text[:80]!r}")
    except LLMError as e:
        console.print(f"[red]LLM 실패[/]: {e}")

    products = load_all_products()
    personas = load_personas()
    from .rag.corpus import load_corpus

    docs = load_corpus()
    console.print(
        f"상품 {len(products)}건 / 페르소나 {len(personas)}명 "
        f"(출처: {personas[0].source if personas else '-'}) / RAG 문서 {len(docs)}건"
    )


@app.command("products")
def products_cmd() -> None:
    """등록된 상품 목록."""
    t = Table("ID", "상품명", "상품군", "기본금리", "최고금리", "타깃 세그먼트")
    for p in load_all_products():
        t.add_row(
            p.product_id, p.name, p.category, str(p.intr_rate), str(p.intr_rate2),
            ", ".join(p.target_segments),
        )
    console.print(t)


@app.command("segments")
def segments_cmd(limit: int = 2000) -> None:
    """세그먼트별 페르소나 수 및 평균 재무지표."""
    personas = load_personas(limit=limit)
    t = Table("세그먼트", "인원", "평균 연소득(만원)", "평균 DSR(%)", "평균 월여유(만원)")
    for seg in load_segments():
        pool = filter_segment(personas, seg)
        if not pool:
            t.add_row(seg.name, "0", "-", "-", "-")
            continue
        f = [p.finance for p in pool if p.finance]
        t.add_row(
            seg.name,
            str(len(pool)),
            f"{sum(x.annual_income_manwon for x in f) / len(f):,.0f}",
            f"{sum(x.dsr_pct for x in f) / len(f):.1f}",
            f"{sum(x.monthly_surplus_manwon for x in f) / len(f):,.0f}",
        )
    console.print(t)


@app.command()
def debate(
    product: str = typer.Argument(..., help="상품 파일명 또는 경로"),
    segment: Optional[str] = typer.Option(None, help="세그먼트명 (기본: 상품의 첫 타깃)"),
    seed: int = 0,
    full: bool = typer.Option(False, help="전체 발화 출력"),
) -> None:
    """단일 페르소나 디베이트 1회 실행 (프롬프트 확인용)."""
    prod = load_product(product)
    seg_name = segment or (prod.target_segments[0] if prod.target_segments else "청년_사회초년생")
    seg = load_segments([seg_name])[0]
    personas = load_personas()
    cohort = sample_cohort(personas, seg, 1)
    if not cohort:
        console.print(f"[red]세그먼트 {seg_name}에 해당하는 페르소나가 없다[/]")
        raise typer.Exit(1)
    persona = cohort[0]
    console.print(f"[bold]페르소나[/]\n{persona.prompt_block()}\n")
    res = run_debate(prod, persona, segment=seg_name, seed=seed)
    if full:
        console.print(res.transcript())
    v = res.verdict
    console.print(
        f"\n[bold]판정[/] 적합성={v.suitability} 가입의향={v.intent_score} "
        f"위반원칙={v.violated_principles} 자기신뢰={v.self_confidence}"
    )
    console.print(f"근거: {v.evidence}")
    console.print(f"위험: {v.risks}")
    console.print(f"권고: {v.recommendations}")
    console.print(f"인용문서: {res.grounding_doc_ids} / 무근거 발화 {res.ungrounded_turns}건 / {res.elapsed_sec}s")


@app.command()
def simulate(
    product: str = typer.Argument(...),
    seeds: int = typer.Option(3, help="멀티시드 반복 횟수"),
    personas_per_segment: int = typer.Option(4),
    mode: str = typer.Option("debate", help="debate | single"),
    workers: int = typer.Option(4),
    segments: Optional[str] = typer.Option(None, help="쉼표구분 세그먼트명"),
    with_ablation: bool = typer.Option(False, help="애블레이션도 함께 실행"),
    with_sensitivity: bool = typer.Option(False, help="민감도 분석도 함께 실행"),
) -> None:
    """상품을 세그먼트 전반에 시뮬레이션하고 검증 리포트를 생성한다."""
    prod = load_product(product)
    seg_names = [s.strip() for s in segments.split(",")] if segments else None
    console.print(f"[bold]시뮬레이션 시작[/] {prod.name} (mode={mode}, seeds={seeds})")
    sim = simulate_product(
        prod,
        segment_names=seg_names,
        k_personas=personas_per_segment,
        n_seeds=seeds,
        mode=mode,  # type: ignore[arg-type]
        workers=workers,
    )
    path = sim.save()
    console.print(f"시뮬레이션 결과 저장: {path}")

    sens = None
    if with_sensitivity:
        console.print("[bold]민감도 분석[/]")
        sens = sensitivity_analysis(
            prod, default_variants(prod), segment_names=seg_names,
            k_personas=max(2, personas_per_segment // 2), n_seeds=max(1, seeds - 1),
            mode=mode, workers=workers,  # type: ignore[arg-type]
        )
        (OUTPUT_DIR / f"sensitivity_{prod.product_id}.json").write_text(
            json.dumps([r.model_dump() for r in sens], ensure_ascii=False, indent=2), encoding="utf-8"
        )

    abl = None
    if with_ablation:
        console.print("[bold]애블레이션[/]")
        abl = run_ablation(n_seeds=max(1, seeds - 1))
        console.print(f"애블레이션 저장: {abl.save()}")

    holding = compare_holding_rates(sim, prod.category)
    text = build_report(prod, sim, ablation=abl, holding=holding, sensitivity=sens)
    rpath = save_report(text, prod.product_id)
    console.print(f"[green]리포트 생성[/]: {rpath}")


@app.command()
def ablation(
    seeds: int = typer.Option(2), limit: Optional[int] = typer.Option(None, help="사례 수 제한")
) -> None:
    """분쟁조정 정답셋으로 단발 vs 디베이트 적중률을 비교한다."""
    rep = run_ablation(n_seeds=seeds, limit=limit)
    t = Table("조건", "적중률", "위험탐지", "macroF1", "위반원칙재현율", "LLM호출")
    for a in rep.arms:
        t.add_row(a.arm, f"{a.accuracy:.1%}", f"{a.risk_accuracy:.1%}", f"{a.macro_f1:.3f}",
                  f"{a.principle_recall:.1%}", str(a.total_llm_calls))
    console.print(t)
    console.print(
        f"디베이트 − 단발(RAG): 적중률 {rep.delta_accuracy_vs_single:+.1%}, "
        f"위험탐지 {rep.delta_risk_accuracy_vs_single:+.1%}"
    )
    console.print(f"저장: {rep.save()}")


@app.command("report")
def report_cmd(sim_path: str, product: str) -> None:
    """저장된 시뮬레이션 JSON에서 리포트만 다시 생성한다."""
    sim = SimulationReport.load(sim_path)
    prod = load_product(product)
    holding = compare_holding_rates(sim, prod.category)
    path = save_report(build_report(prod, sim, holding=holding), prod.product_id)
    console.print(f"저장: {path}")


@app.command()
def rag(query: str, k: int = 5) -> None:
    """근거 검색 동작 확인."""
    from .rag.retriever import get_retriever

    for h in get_retriever().retrieve(query, k=k):
        console.print(f"[{h.score:.2f}] {h.doc.doc_id} {h.doc.title}")


if __name__ == "__main__":
    app()

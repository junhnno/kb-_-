"""검증 리포트 생성 (Markdown). 근거·신뢰도를 반드시 표시한다."""

from __future__ import annotations

from pathlib import Path

from .config import OUTPUT_DIR
from .eval.benchmark import AblationReport, HoldingReport
from .eval.simulate import SensitivityRow, SimulationReport
from .products.schema import Product

LEVEL_MARK = {"high": "높음", "medium": "보통", "low": "낮음(추가 검증 필요)"}


def build_report(
    product: Product,
    sim: SimulationReport,
    *,
    ablation: AblationReport | None = None,
    holding: HoldingReport | None = None,
    sensitivity: list[SensitivityRow] | None = None,
) -> str:
    L: list[str] = []
    L.append(f"# 상품 사전검증 리포트 — {product.name}")
    L.append(
        f"\n- 상품ID: `{product.product_id}` / 상품군: {product.category} / 발행: {product.issuer}"
        f"\n- 생성시각: {sim.generated_at}"
        f"\n- 실행환경: backend=`{sim.backend}`, 토론모델=`{sim.model_small}`, 심판모델=`{sim.model_judge}`"
        f"\n- 모드: {sim.mode} / 멀티시드: {sim.n_seeds}회"
    )

    L.append("\n## 1. 요약 판정")
    risky = [s for s in sim.segments if s.verdict_mix.get("fail", 0) >= 0.3]
    lowconf = [s for s in sim.segments if s.low_confidence_ratio >= 0.5]
    if risky:
        L.append(
            f"- **판매원칙 위험 세그먼트 {len(risky)}건**: "
            + ", ".join(f"{s.segment}(fail {s.verdict_mix.get('fail', 0):.0%})" for s in risky)
        )
    else:
        L.append("- 판매원칙 위험(fail 비율 30% 이상) 세그먼트는 없다.")
    if lowconf:
        L.append(
            "- **저신뢰 세그먼트**: " + ", ".join(s.segment for s in lowconf) + " → 추가 검증 필요"
        )
    best = max(sim.segments, key=lambda s: s.mean_intent, default=None)
    if best:
        L.append(f"- 가입의향 최고 세그먼트: {best.segment} (평균 {best.mean_intent}점)")

    L.append("\n## 2. 세그먼트별 결과")
    L.append("\n| 세그먼트 | n | 가입의향(평균) | 가입률 | pass/warn/fail | 신뢰도 | 상태 |")
    L.append("|---|---|---|---|---|---|---|")
    for s in sim.segments:
        mix = "/".join(f"{s.verdict_mix.get(k, 0):.0%}" for k in ("pass", "warn", "fail"))
        L.append(
            f"| {s.segment} | {s.n_personas} | {s.mean_intent} | {s.adoption_rate:.0%} | "
            f"{mix} | {s.mean_confidence:.2f} | {s.flag} |"
        )

    L.append("\n## 3. 주요 위험요인 및 개선권고 (디베이트 근거 기반)")
    for s in sim.segments:
        L.append(f"\n### {s.segment}")
        if s.top_risks:
            L.append("**위험요인**")
            L += [f"- {r}" for r in s.top_risks]
        if s.top_recommendations:
            L.append("\n**개선권고**")
            L += [f"- {r}" for r in s.top_recommendations]
        low = [c for c in s.cases if c.needs_review]
        if low:
            L.append(
                f"\n> 저신뢰 케이스 {len(low)}/{len(s.cases)}건 "
                f"(라벨 합의도 {min(c.label_agreement for c in low):.0%} 이하) — 추가 검증 필요"
            )

    L.append("\n## 4. 인용된 근거")
    doc_ids = sorted({d for s in sim.segments for c in s.cases for d in c.grounding_doc_ids})
    L.append(", ".join(f"`{d}`" for d in doc_ids) or "(없음)")
    L.append("\n대표 근거 문장:")
    seen: set[str] = set()
    for s in sim.segments:
        for c in s.cases:
            for e in c.evidence[:2]:
                if e not in seen:
                    seen.add(e)
                    L.append(f"- {e}")
            if len(seen) >= 12:
                break

    if sensitivity:
        L.append("\n## 5. 가격·조건 민감도")
        L.append("\n| 시나리오 | 세그먼트 | 가입률 | 가입의향 | fail비율 | 신뢰도 |")
        L.append("|---|---|---|---|---|---|")
        for r in sensitivity:
            L.append(
                f"| {r.label} | {r.segment} | {r.adoption_rate:.0%} | {r.mean_intent} | "
                f"{r.fail_ratio:.0%} | {r.mean_confidence:.2f} |"
            )

    if ablation:
        L.append("\n## 6. 애블레이션 — 디베이트가 정확성을 높이는가")
        L.append(f"\n정답셋: 분쟁조정 사례 {ablation.arms[0].n if ablation.arms else 0}건, 시드 {ablation.n_seeds}회")
        L.append("\n| 조건 | 적합성 적중률 | 위험탐지 정확도 | macro F1 | 위반원칙 재현율 | 평균 신뢰도 | LLM 호출수 |")
        L.append("|---|---|---|---|---|---|---|")
        name = {
            "single_norag": "단발 질문 (RAG 없음)",
            "single": "단발 질문 + RAG",
            "debate": "3진영 디베이트 + RAG",
        }
        for a in ablation.arms:
            L.append(
                f"| {name.get(a.arm, a.arm)} | {a.accuracy:.1%} | {a.risk_accuracy:.1%} | "
                f"{a.macro_f1:.3f} | {a.principle_recall:.1%} | {a.mean_confidence:.2f} | {a.total_llm_calls} |"
            )
        L.append(
            f"\n- 디베이트 − 단발(RAG) 적중률 차이: **{ablation.delta_accuracy_vs_single:+.1%}**"
            f" / 위험탐지 차이: **{ablation.delta_risk_accuracy_vs_single:+.1%}**"
        )
        L += [f"- {c}" for c in ablation.caveats]

    if holding:
        L.append("\n## 7. KOSIS 보유율 대조 (외적 타당성)")
        L.append(
            f"\n- 세그먼트 {holding.n_segments}개, MAE {holding.mae:.3f}, "
            f"Spearman ρ {holding.spearman if holding.spearman is not None else 'n/a'}"
        )
        L.append("\n| 세그먼트 | 실제 보유율 | 시뮬 가입률 | 오차 |")
        L.append("|---|---|---|---|")
        for r in holding.rows:
            L.append(
                f"| {r.segment} | {r.actual_holding_rate:.0%} | {r.simulated_adoption_rate:.0%} | {r.abs_error:.2f} |"
            )
        L.append(f"\n> {holding.note}")

    L.append("\n## 8. 한계 (정직한 고지)")
    L += [
        "- 합성 페르소나(Nemotron-Personas-Korea)는 개별 변수 분포는 실제와 정합하나 변수 조합(joint distribution) 정합성은 검증되지 않았다. 본 결과는 **탐색·경보용**이며 확정 근거로 쓸 수 없다.",
        "- 디베이트는 LLM의 내적 충실성을 높이지만 외적 타당성(실제 시장 일치)을 자동 보장하지 않는다. 반드시 실데이터·조정례와 병용해야 한다.",
        "- 본 도구는 상품기획팀용 세그먼트 설계·검증 도구다. 개별 소비자에 대한 상품 추천·판매권유가 아니다.",
        "- 재무 프로파일 수치는 가계금융복지조사 분포에서 합성한 값으로 실제 개인이 아니다.",
    ]
    return "\n".join(L) + "\n"


def save_report(text: str, product_id: str, path: str | Path | None = None) -> Path:
    p = Path(path) if path else OUTPUT_DIR / f"report_{product_id}.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p

"""3진영 디베이트 오케스트레이터.

진행 순서 (CLAUDE.md §3)
  1. 옹호자가 페르소나에게 상품을 설명·권유
  2. 페르소나 1차 반응
  3. 회의론자가 반박
  4. 페르소나 재반응 (양가감정 허용)
  5. 심판이 전체 대화 + RAG 근거로 최종 판정

애블레이션 비교군으로 `single_shot()`(디베이트 없이 1회 판정)을 같은 컨텍스트로 제공한다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..config import SETTINGS
from ..facts import apply_severity_floors, build_fact_pack, screen_typed_concerns
from ..llm import LLMClient, extract_json
from ..personas.schema import Persona
from ..products.schema import Product
from ..rag.retriever import Retriever, get_retriever
from . import prompts as P
from .schema import (
    REQUIRED_VERDICT_KEYS,
    VERDICT_JSON_SCHEMA,
    DebateResult,
    Turn,
    Verdict,
    count_citations,
    is_grounded,
    stamp_sources,
)


@dataclass
class DebateConfig:
    # 기본값을 설정(.env)에서 읽는다. FDM_TEMP_DEBATER로 온도를 쓸어보며 측정할 수 있다.
    temperature_debater: float = field(default_factory=lambda: SETTINGS.temp_debater)
    temperature_judge: float = field(default_factory=lambda: SETTINGS.temp_judge)
    k_docs: int = 6
    use_dense: bool = False
    # 검색 대상 문서 종류. None이면 전체(법조문·가이드라인·조정례).
    #
    # 조정례 문서는 제목에 `— 판정 {label}`이, 본문에 판단요지가 들어 있다. 평가 시
    # 자기 사례만 제외되므로 **이웃 사례의 정답 라벨이 프롬프트에 노출된다.**
    # 실측: 검색 결과의 72%가 조정례이고 그중 74%가 fail/warn이며, 깨끗한 대출
    # 케이스(CASE-019/020)는 검색된 조정례 3건이 전부 fail이었다.
    # ("law", "guideline")로 두면 조정례를 빼고 조문만 근거로 쓴다 — 이 노출이
    # 과잉경고의 원인인지 분리해 재기 위한 장치다.
    retrieve_kinds: tuple[str, ...] | None = None
    enforce_grounding: bool = True  # 근거 없는 발화는 1회 재요청
    use_fact_pack: bool = True  # 파생 지표를 계산해 주입 (사전 예방)
    screen_contradictions: bool = True  # 계산값과 모순되는 우려를 기각 (사후 차단)


def _query_for(product: Product, persona: Persona) -> str:
    parts = [
        product.name,
        product.category,
        product.summary,
        f"{persona.age}세 {persona.occupation}",
        " ".join(product.risk_notes[:3]),
    ]
    if persona.finance:
        parts.append(f"소득 {persona.finance.annual_income_manwon}만원 DSR {persona.finance.dsr_pct}%")
    return " ".join(p for p in parts if p)


def _turn(
    client: LLMClient,
    role: str,
    system: str,
    user: str,
    *,
    temperature: float,
    seed: int,
    json_mode: bool = False,
    enforce_grounding: bool = True,
) -> Turn:
    res = client.chat(
        role=role, system=system, user=user, temperature=temperature, seed=seed, json_mode=json_mode
    )
    text = res.text
    if enforce_grounding and not is_grounded(text):
        retry = client.chat(
            role=role,
            system=system,
            user=user
            + "\n\n[경고] 직전 답변에 근거 인용이 없었다. 약관 조항 ID, 법령 ID, 또는 페르소나의 재무 수치를 대괄호로 인용해 다시 작성하라.",
            temperature=max(0.0, temperature - 0.2),
            seed=seed + 1,
            json_mode=json_mode,
        )
        if is_grounded(retry.text):
            text = retry.text
    return Turn(
        role=role,  # type: ignore[arg-type]
        content=text,
        citations=count_citations(text),
        grounded=is_grounded(text),
        model=res.model,
    )


def _persona_intent(turn_text: str) -> int | None:
    obj = extract_json(turn_text)
    if not obj:
        return None
    for k in ("가입의향점수", "intent_score", "score"):
        if k in obj:
            try:
                return max(0, min(100, int(round(float(obj[k])))))
            except (TypeError, ValueError):
                return None
    return None


def run_debate(
    product: Product,
    persona: Persona,
    *,
    segment: str = "-",
    seed: int = 0,
    config: DebateConfig | None = None,
    client: LLMClient | None = None,
    retriever: Retriever | None = None,
    exclude_doc_ids: set[str] | None = None,
    situation: str = "",
) -> DebateResult:
    cfg = config or DebateConfig()
    client = client or LLMClient()
    retriever = retriever or get_retriever(cfg.use_dense)
    t0 = time.time()

    hits = retriever.retrieve(
        _query_for(product, persona),
        k=cfg.k_docs,
        kinds=cfg.retrieve_kinds,
        exclude_ids=exclude_doc_ids,
    )
    grounding = (
        "\n".join(f"{i+1}. {h.doc.cite()}" for i, h in enumerate(hits)) or "(검색된 근거 없음)"
    )
    facts = build_fact_pack(product, persona) if cfg.use_fact_pack else None
    ctx = P.context_block(
        product.prompt_block(),
        persona.prompt_block(),
        grounding,
        facts.prompt_block() if facts else "",
        situation,
    )

    turns: list[Turn] = []

    t_adv = _turn(
        client, "advocate", P.ADVOCATE_SYSTEM, P.advocate_user(ctx),
        temperature=cfg.temperature_debater, seed=seed,
        enforce_grounding=cfg.enforce_grounding,
    )
    turns.append(t_adv)

    t_p1 = _turn(
        client, "persona", P.PERSONA_SYSTEM, P.persona_first_user(ctx, t_adv.content),
        temperature=cfg.temperature_debater, seed=seed + 11, json_mode=True,
        enforce_grounding=False,
    )
    turns.append(t_p1)

    t_skp = _turn(
        client, "skeptic", P.skeptic_system(), P.skeptic_user(ctx, t_adv.content, t_p1.content),
        temperature=cfg.temperature_debater, seed=seed + 22,
        enforce_grounding=cfg.enforce_grounding,
    )
    turns.append(t_skp)

    t_p2 = _turn(
        client, "persona", P.PERSONA_SYSTEM,
        P.persona_second_user(ctx, t_adv.content, t_skp.content, t_p1.content),
        temperature=cfg.temperature_debater, seed=seed + 33, json_mode=True,
        enforce_grounding=False,
    )
    turns.append(t_p2)

    transcript = "\n\n".join(
        f"### {r}\n{t.content}"
        for r, t in zip(["옹호자", "페르소나(1차)", "회의론자", "페르소나(재반응)"], turns)
    )
    judge_obj = client.chat_json(
        role="judge",
        system=P.judge_system(),
        user=P.judge_user(ctx, transcript),
        temperature=cfg.temperature_judge,
        seed=seed + 44,
        json_schema=VERDICT_JSON_SCHEMA,
        required_keys=REQUIRED_VERDICT_KEYS,
    )
    verdict = Verdict.from_json(judge_obj)
    judge_repaired = client.last_json_repaired

    # 사후 차단: 계산값과 모순되는 우려를 기각한다.
    dropped: list[str] = []
    if facts is not None and cfg.screen_contradictions:
        verdict.concerns, dropped = screen_typed_concerns(verdict.concerns, facts)
        # 계산값이 요구하는 최소 심각도로 승격 (라벨은 건드리지 않는다)
        verdict.concerns = apply_severity_floors(verdict.concerns, facts)
        verdict.risks = [c.statement for c in verdict.concerns]
    stamp_sources(verdict.concerns, "debate")
    turns.append(
        Turn(
            role="judge",
            content=verdict.summary or str(judge_obj)[:500],
            citations=count_citations(" ".join(verdict.evidence)),
            grounded=bool(verdict.evidence),
            model=client.model_for("judge"),
        )
    )

    return DebateResult(
        product_id=product.product_id,
        product_name=product.name,
        persona_id=persona.persona_id,
        segment=segment,
        mode="debate",
        seed=seed,
        temperature=cfg.temperature_debater,
        turns=turns,
        verdict=verdict,
        persona_intent_first=_persona_intent(t_p1.content),
        persona_intent_final=_persona_intent(t_p2.content),
        grounding_doc_ids=[h.doc.doc_id for h in hits],
        ungrounded_turns=sum(1 for t in turns if not t.grounded),
        judge_schema_repaired=judge_repaired,
        dropped_concerns=dropped,
        elapsed_sec=round(time.time() - t0, 2),
    )


def single_shot(
    product: Product,
    persona: Persona,
    *,
    segment: str = "-",
    seed: int = 0,
    config: DebateConfig | None = None,
    client: LLMClient | None = None,
    retriever: Retriever | None = None,
    exclude_doc_ids: set[str] | None = None,
    with_rag: bool = True,
    use_rules: bool = True,
    situation: str = "",
) -> DebateResult:
    """애블레이션 비교군: 디베이트 없이 1회 판정.

    with_rag=False  → 근거 검색 없음
    use_rules=False → COMMON_RULES(인용 강제·동조 금지) 없음 = naive arm
    """
    cfg = config or DebateConfig()
    client = client or LLMClient()
    t0 = time.time()

    doc_ids: list[str] = []
    grounding = "(근거 검색 미사용 — 애블레이션 조건)"
    if with_rag:
        retriever = retriever or get_retriever(cfg.use_dense)
        hits = retriever.retrieve(
            _query_for(product, persona),
            k=cfg.k_docs,
            kinds=cfg.retrieve_kinds,
            exclude_ids=exclude_doc_ids,
        )
        doc_ids = [h.doc.doc_id for h in hits]
        grounding = "\n".join(f"{i+1}. {h.doc.cite()}" for i, h in enumerate(hits)) or "(없음)"

    facts = build_fact_pack(product, persona) if cfg.use_fact_pack else None
    ctx = P.context_block(
        product.prompt_block(),
        persona.prompt_block(),
        grounding,
        facts.prompt_block() if facts else "",
        situation,
    )
    obj = client.chat_json(
        role="single",
        system=P.single_shot_system() if use_rules else P.NAIVE_SYSTEM,
        user=P.single_shot_user(ctx),
        temperature=cfg.temperature_judge,
        seed=seed,
        json_schema=VERDICT_JSON_SCHEMA,
        required_keys=REQUIRED_VERDICT_KEYS,
    )
    verdict = Verdict.from_json(obj)
    repaired = client.last_json_repaired
    dropped: list[str] = []
    if facts is not None and cfg.screen_contradictions:
        verdict.concerns, dropped = screen_typed_concerns(verdict.concerns, facts)
        # 계산값이 요구하는 최소 심각도로 승격 (라벨은 건드리지 않는다)
        verdict.concerns = apply_severity_floors(verdict.concerns, facts)
        verdict.risks = [c.statement for c in verdict.concerns]
    stamp_sources(verdict.concerns, "single")
    turn = Turn(
        role="single",
        content=verdict.summary or str(obj)[:500],
        citations=count_citations(" ".join(verdict.evidence)),
        grounded=bool(verdict.evidence),
        model=client.model_for("single"),
    )
    return DebateResult(
        product_id=product.product_id,
        product_name=product.name,
        persona_id=persona.persona_id,
        segment=segment,
        mode="single",
        seed=seed,
        temperature=cfg.temperature_judge,
        turns=[turn],
        verdict=verdict,
        grounding_doc_ids=doc_ids,
        ungrounded_turns=0 if turn.grounded else 1,
        judge_schema_repaired=repaired,
        dropped_concerns=dropped,
        elapsed_sec=round(time.time() - t0, 2),
    )

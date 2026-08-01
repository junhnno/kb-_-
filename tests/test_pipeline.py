"""mock 백엔드로 파이프라인 전 구간을 확인하는 스모크 테스트."""

from __future__ import annotations

import pytest

from fdm.agents.debate import run_debate, single_shot
from fdm.agents.schema import (
    PRINCIPLES,
    REQUIRED_VERDICT_KEYS,
    Verdict,
    count_citations,
    is_grounded,
    normalize_suitability,
)
from fdm.config import SETTINGS
from fdm.eval.benchmark import load_cases, run_ablation
from fdm.eval.confidence import aggregate
from fdm.eval.simulate import VariantSpec, apply_variant, load_segments, simulate_product
from fdm.llm import extract_json
from fdm.personas.loader import filter_segment, load_personas, sample_cohort
from fdm.products.schema import load_all_products, load_product
from fdm.rag.retriever import get_retriever
from fdm.report import build_report


@pytest.fixture(autouse=True)
def _mock_backend():
    old = SETTINGS.backend
    SETTINGS.backend = "mock"
    yield
    SETTINGS.backend = old


@pytest.fixture(scope="module")
def personas():
    return load_personas(limit=200)


# ------------------------------------------------------------------- 페르소나
def test_personas_have_finance(personas):
    assert len(personas) >= 100
    for p in personas[:20]:
        assert p.finance is not None
        assert p.finance.annual_income_manwon > 0
        assert 1 <= p.finance.income_quintile <= 5
        assert p.finance.monthly_income_manwon > 0


def test_finance_is_deterministic():
    a = load_personas(limit=30)[0]
    b = load_personas(limit=30)[0]
    assert a.persona_id == b.persona_id
    assert a.finance.annual_income_manwon == b.finance.annual_income_manwon


def test_segment_filter_respects_conditions(personas):
    seg = next(s for s in load_segments() if s.name == "고DSR_차주")
    pool = filter_segment(personas, seg)
    assert pool, "고DSR 세그먼트가 비어 있으면 임계값 설정을 재검토해야 한다"
    assert all(p.finance.dsr_pct >= seg.dsr_min_pct for p in pool)
    assert all(p.age >= seg.age_min for p in pool)


def test_sample_cohort_is_reproducible(personas):
    seg = load_segments(["청년_사회초년생"])[0]
    a = [p.persona_id for p in sample_cohort(personas, seg, 3)]
    b = [p.persona_id for p in sample_cohort(personas, seg, 3)]
    assert a == b


# ---------------------------------------------------------------------- 상품
def test_products_load_and_render():
    products = load_all_products()
    assert len(products) == 5
    p = load_product("01_youth_step_saving")
    block = p.prompt_block()
    assert "약관 제5조" in block  # 인용 가능한 조항이 프롬프트에 들어간다
    assert "우대조건" in block


# ----------------------------------------------------------------------- RAG
def test_retrieval_finds_relevant_law():
    hits = get_retriever().retrieve("최고금리만 강조한 광고 오인 우대금리", k=3)
    ids = {h.doc.doc_id for h in hits}
    assert ids & {"FCPA-22", "GUIDE-RATE-AD"}


def test_retrieval_excludes_ids():
    hits = get_retriever().retrieve("리볼빙 수수료 미고지", k=8, exclude_ids={"CASE-007"})
    assert "CASE-007" not in {h.doc.doc_id for h in hits}


# ------------------------------------------------------------------ JSON 파싱
@pytest.mark.parametrize(
    "raw",
    [
        '{"적합성": "pass", "가입의향점수": 70}',
        '```json\n{"적합성": "warn", "가입의향점수": 50}\n```',
        '<think>고민 중</think>\n앞말 {"적합성": "fail", "가입의향점수": 10} 뒷말',
        '설명: {"적합성": "pass", "근거": ["중괄호 } 포함 문자열"], "가입의향점수": 80}',
    ],
)
def test_extract_json_variants(raw):
    obj = extract_json(raw)
    assert obj is not None and "적합성" in obj


def test_chat_json_requires_keys_and_repairs(monkeypatch):
    """심판이 다른 스키마로 답하면 조용히 기본값으로 넘기지 말고 재요청해야 한다."""
    from fdm.llm import LLMClient, LLMError

    client = LLMClient()
    calls: list[str] = []

    # 1차: 스키마 위반(판정/판단요지), 2차: 요구 스키마 준수
    replies = [
        '{"판정": "warn", "판단요지": "설명 부족"}',
        '{"적합성": "warn", "가입의향점수": 35, "근거": ["[FCPA-19] 설명의무"]}',
    ]

    def fake_chat(**kwargs):
        calls.append(kwargs["user"][:20])
        from fdm.llm import LLMResponse

        return LLMResponse(text=replies[len(calls) - 1], model="fake")

    monkeypatch.setattr(client, "chat", fake_chat)
    obj = client.chat_json(
        role="judge", system="s", user="u", required_keys=REQUIRED_VERDICT_KEYS
    )
    assert len(calls) == 2, "필수 키 누락 시 교정 재요청이 있어야 한다"
    assert obj["가입의향점수"] == 35
    assert client.last_json_repaired is True

    # 2회 시도 후에도 스키마를 못 지키면 예외를 던진다 (기본값으로 넘어가면 안 된다)
    calls.clear()
    replies[:] = ['{"판정": "warn"}', '{"판정": "warn"}']
    monkeypatch.setattr(client, "chat", fake_chat)
    with pytest.raises(LLMError, match="스키마 검증 실패"):
        client.chat_json(role="judge", system="s", user="u", required_keys=REQUIRED_VERDICT_KEYS)


def test_verdict_absorbs_dict_evidence():
    """근거가 리스트 대신 딕셔너리로 와도 내용을 잃지 않아야 한다."""
    v = Verdict.from_json(
        {
            "판정": "warn",
            "가입의향점수": 40,
            "근거": {"광고표시": "최고금리만 강조", "설명의무": "중도해지 설명 부족"},
            "판단요지": "광고 개선 권고",
        }
    )
    assert v.suitability == "warn"  # "판정" 별칭 인식
    assert v.summary == "광고 개선 권고"  # "판단요지" 별칭 인식
    assert any("최고금리만 강조" in e for e in v.evidence)  # 키만 남지 않았다
    assert len(v.evidence) == 2


def test_mock_judge_fills_principles():
    """mock이 위반원칙을 채워야 벤치마크 재현율 로직을 LLM 없이 검증할 수 있다."""
    from fdm.llm import _mock_reply

    seen_nonempty = 0
    for i in range(12):
        obj = extract_json(_mock_reply("judge", "sys", f"user{i}", i, 0.7))
        assert "위반원칙" in obj
        assert isinstance(obj["위반원칙"], list)
        assert all(p in PRINCIPLES for p in obj["위반원칙"])
        if obj["적합성"] == "pass":
            assert obj["위반원칙"] == [], "적합 판정에 위반원칙이 있으면 모순이다"
        else:
            seen_nonempty += 1
        assert 0.0 <= obj["confidence"] <= 1.0
        assert obj["개선권고"]
    assert seen_nonempty > 0


def test_debt_capped_by_income_multiple(personas):
    """무직·학생에게 소득 대비 과도한 부채가 붙지 않아야 한다 (결합분포 보정)."""
    from fdm.personas.finance import load_kosis_params

    params = load_kosis_params()
    caps = params["debt_cap_income_multiple_by_occupation"]
    checked = 0
    for p in personas:
        f = p.finance
        if f is None or f.debt_manwon == 0:
            continue
        multiple = params["debt_cap_income_multiple_default"]
        for kw, m in caps.items():
            if kw in p.occupation:
                multiple = m
                break
        cap = max(params["debt_cap_floor_manwon"], int(f.annual_income_manwon * multiple))
        assert f.debt_manwon <= cap, f"{p.persona_id} {p.occupation}: 부채 {f.debt_manwon} > 상한 {cap}"
        checked += 1
    assert checked > 20, "부채 보유 페르소나가 너무 적어 검증이 무의미하다"


def test_fact_pack_computes_derived_indicators():
    """LLM이 나눗셈하지 않도록 파생 지표를 코드가 계산한다."""
    from fdm.facts import build_fact_pack
    from fdm.eval.benchmark import load_cases

    case = next(c for c in load_cases() if c.case_id == "CASE-005")  # DSR 38%, 월여유 -18
    f = build_fact_pack(case.product, case.persona)
    assert f.dsr_pct == 38.0
    assert f.surplus_nonpositive is True, "월 여유자금이 음수면 플래그로 표현해야 한다"
    assert f.burden_max is None, "여유자금이 0 이하면 부담률을 계산하지 않는다(5000% 같은 값 방지)"
    assert f.stressed_dsr_pct is not None and f.stressed_dsr_pct > f.dsr_pct
    block = f.prompt_block()
    assert "시스템 계산값" in block and "DSR" in block


def test_structured_concerns_are_typed_and_capped():
    """우려는 유형·심각도를 갖고, 앵커 없으면 심각도가 제한되어야 한다."""
    from fdm.agents.schema import Concern
    from fdm.concerns import type_ids

    v = Verdict.from_json(
        {
            "적합성": "warn",
            "가입의향점수": 40,
            "근거": ["[약관 제5조]"],
            "우려": [
                {"유형": "preferential_unattainable", "심각도": "치명",
                 "내용": "달성 곤란", "앵커": "동시 충족 8%"},
                {"유형": "early_termination_penalty", "심각도": "치명",
                 "내용": "중도해지 불이익", "앵커": ""},          # 앵커 없음 → 강등
                {"유형": "존재하지_않는_유형", "심각도": "중대", "내용": "기타"},
            ],
        }
    )
    assert len(v.concerns) == 3
    assert v.concerns[0].severity == "치명", "앵커가 있으면 심각도 유지"
    assert v.concerns[1].severity == "주의", "앵커 없으면 '주의' 상한"
    assert v.concerns[2].type == "other", "모르는 유형은 other로"
    assert all(c.type in type_ids() for c in v.concerns)
    # 확인 방법이 자동으로 채워진다 (산출물의 '어디를 봐야 하나')
    assert v.concerns[0].verify_with
    # risks는 하위호환 파생값
    assert v.risks == [c.statement for c in v.concerns]

    # 구형 응답(위험요인 문자열 배열)도 흡수한다
    old = Verdict.from_json({"적합성": "warn", "가입의향점수": 40, "위험요인": ["부담 큼"]})
    assert len(old.concerns) == 1 and old.concerns[0].type == "other"


def test_typed_screening_uses_type_not_keywords():
    """유형 기반 기각은 표현이 달라도 정확히 걸러야 한다."""
    from fdm.agents.schema import Concern
    from fdm.eval.benchmark import load_cases
    from fdm.facts import build_fact_pack, screen_typed_concerns

    case = next(c for c in load_cases() if c.case_id == "CASE-002")  # DSR 4.1%
    f = build_fact_pack(case.product, case.persona)
    concerns = [
        Concern(type="dsr_overload", statement="재정 여건상 상환 여력이 빠듯함", anchor="x"),
        Concern(type="preferential_unattainable", statement="우대 달성 곤란", anchor="8%"),
    ]
    kept, dropped = screen_typed_concerns(concerns, f)
    # 키워드('DSR')가 문장에 없어도 유형으로 기각된다 — 키워드 방식으로는 불가능했다
    assert len(dropped) == 1 and "dsr_overload" in dropped[0]
    assert len(kept) == 1 and kept[0].type == "preferential_unattainable"


def test_lump_sum_loan_payment_is_interest_only():
    """만기 일시상환을 원리금균등으로 계산하면 상환 부담이 크게 부풀려진다.

    실측 사고: 12개월 4,000만원 대출의 월 상환액이 344만원으로 계산되어
    DSR 117.8%가 나왔고, 모델이 이를 근거로 판정을 fail로 올렸다(실제는 월 이자 20.7만원).
    """
    from fdm.facts import build_fact_pack
    from fdm.eval.benchmark import load_cases

    case = next(c for c in load_cases() if c.case_id == "CASE-011")
    f = build_fact_pack(case.product, case.persona)
    assert f.is_lump_sum_repayment is True
    expected = case.product.limit_manwon * case.product.intr_rate / 100 / 12
    assert abs(f.payment_max - expected) < 1.0, f"이자만 계산해야 한다: {f.payment_max}"
    assert f.stressed_dsr_pct < 60, f"DSR이 비현실적으로 높다: {f.stressed_dsr_pct}%"
    assert "만기 일시상환" in f.prompt_block()


def test_contradiction_screening_drops_hallucinated_concerns():
    """실측된 환각 4건이 계산값으로 기각되는지 확인한다."""
    from fdm.facts import build_fact_pack, screen_concerns
    from fdm.eval.benchmark import load_cases

    cases = {c.case_id: c for c in load_cases()}

    # CASE-002: 페르소나 DSR 4.1%인데 "DSR 근접" 주장 (실측된 환각)
    c2 = cases["CASE-002"]
    f2 = build_fact_pack(c2.product, c2.persona)
    kept, dropped = screen_concerns(
        ["DSR 근접 시 추가 자산 운용 위험", "우대조건 달성 가능성 낮음"], f2
    )
    assert any("DSR" in d for d in dropped), "DSR 4.1%인데 DSR 우려는 기각돼야 한다"
    assert any("우대조건" in k for k in kept), "실제 우대조건이 있는 상품이므로 유지돼야 한다"

    # CASE-006: 월여유 185만원, 보호한도 내 예치, 원금보장 상품 (실측된 환각 3종)
    c6 = cases["CASE-006"]
    f6 = build_fact_pack(c6.product, c6.persona)
    kept, dropped = screen_concerns(
        [
            "여유자금 부족으로 납입 부담",
            "예금자보호 한도 초과 위험",
            "원금 손실 가능성",
            "중도해지 시 불이익",
        ],
        f6,
    )
    assert len(dropped) == 3, f"환각 3건이 기각돼야 한다: kept={kept} dropped={dropped}"
    assert any("중도해지" in k for k in kept), "중도해지는 계산으로 반박할 수 없으므로 유지"

    # CASE-001: ELS(우대조건 없음)인데 "우대조건 충족 불가" (실측된 환각)
    c1 = cases["CASE-001"]
    f1 = build_fact_pack(c1.product, c1.persona)
    assert f1.has_preferentials is False
    assert f1.principal_guaranteed is False
    kept, dropped = screen_concerns(["우대조건 충족 불가", "원금 손실 가능성"], f1)
    assert any("우대조건" in d for d in dropped)
    assert any("원금 손실" in k for k in kept), "원금 비보장 상품이므로 원금손실 우려는 정당"


def test_unknown_fields_never_reject_concerns():
    """'미기재'를 '없음'으로 단정하면 안 된다.

    실측 사고: 변동금리 상품인데 rate_basis가 미기재라 '고정'으로 단정되어
    정당한 '금리상승 위험' 우려가 기각됐다. fees·preferentials도 같은 문제였다.
    """
    from fdm.facts import build_fact_pack, screen_concerns
    from fdm.personas.schema import FinanceProfile, Persona
    from fdm.products.schema import Product

    bare = Product(  # 금리유형·수수료·우대조건을 아무것도 기재하지 않은 상품
        product_id="X-1", name="정보 부족 상품", category="loan",
        intr_rate=5.0, intr_rate2=9.0, save_trm_months=60, limit_manwon=5000,
    )
    persona = Persona(
        persona_id="X-P-1", age=40, occupation="사무직",
        finance=FinanceProfile(
            annual_income_manwon=5000, monthly_income_manwon=417,
            financial_assets_manwon=2000, real_assets_manwon=0,
            debt_manwon=6000, monthly_debt_service_manwon=55,
            monthly_surplus_manwon=60, dsr_pct=13.2, income_quintile=3,
        ),
    )
    f = build_fact_pack(bare, persona)
    assert f.rate_is_variable is None, "미기재는 None(모름)이어야 한다"
    assert f.has_fees is None
    assert f.has_preferentials is None

    kept, dropped = screen_concerns(
        ["금리 상승 시 상환액 증가 위험", "수수료 부담", "우대조건 달성 곤란"], f
    )
    assert not dropped, f"모르는 사실로 우려를 기각하면 안 된다: {dropped}"
    assert len(kept) == 3

    # 프롬프트에도 단정 대신 '미기재'라고 써야 한다
    block = f.prompt_block()
    assert "미기재" in block and "고정" not in block

    # 반대로 명시적으로 '없음'이면 기각한다
    explicit = bare.model_copy(update={"rate_basis": "고정", "fees": [], "preferentials": []})
    f2 = build_fact_pack(explicit, persona)
    kept2, dropped2 = screen_concerns(
        ["금리 상승 시 상환액 증가 위험", "수수료 부담", "우대조건 달성 곤란"], f2
    )
    assert len(dropped2) == 3, f"명시적 부정은 기각해야 한다: kept={kept2}"


def test_screening_keeps_unclassifiable_concerns():
    """유형을 특정할 수 없는 우려는 지우지 않는다 (안전한 방향이 기본값)."""
    from fdm.facts import build_fact_pack, screen_concerns
    from fdm.eval.benchmark import load_cases

    case = next(c for c in load_cases() if c.case_id == "CASE-006")
    f = build_fact_pack(case.product, case.persona)
    kept, dropped = screen_concerns(["담당 직원의 태도가 불친절했다"], f)
    assert kept and not dropped


def test_verdict_normalization():
    v = Verdict.from_json({"적합성": "부적합", "가입의향점수": "12.6", "위반원칙": ["설명의무", "존재하지않는원칙"]})
    assert v.suitability == "fail"
    assert v.intent_score == 13
    assert v.violated_principles == ["설명의무"]
    assert normalize_suitability("조건부") == "warn"


def test_grounding_detection():
    assert is_grounded("[약관 제5조]에 따라 우대금리는 만기에 판정된다")
    assert is_grounded("월 여유자금 37만원으로 납입이 가능하다")
    assert not is_grounded("대체로 유리한 상품이라고 생각한다")
    assert count_citations("[FCPA-19] 제19조 40% 12개월") >= 3


# -------------------------------------------------------------------- 디베이트
def test_debate_runs_five_turns(personas):
    product = load_product("01_youth_step_saving")
    res = run_debate(product, personas[0], segment="테스트", seed=1)
    assert [t.role for t in res.turns] == ["advocate", "persona", "skeptic", "persona", "judge"]
    assert res.verdict.suitability in {"pass", "warn", "fail"}
    assert 0 <= res.verdict.intent_score <= 100
    assert res.grounding_doc_ids
    assert res.persona_intent_first is not None


def test_single_shot_norag_has_no_docs(personas):
    product = load_product("01_youth_step_saving")
    res = single_shot(product, personas[0], with_rag=False)
    assert res.mode == "single"
    assert res.grounding_doc_ids == []


# -------------------------------------------------------------------- 신뢰도
def test_confidence_high_when_runs_agree(personas):
    product = load_product("02_senior_time_deposit")
    runs = [run_debate(product, personas[0], seed=5) for _ in range(3)]  # 같은 시드 → 동일 결과
    cr = aggregate(runs)
    assert cr.label_agreement == 1.0
    assert cr.intent_std == 0.0
    assert cr.confidence_level in {"high", "medium"}
    assert not cr.needs_review


def test_confidence_drops_when_runs_disagree(personas):
    product = load_product("04_cashback_card")
    runs = [run_debate(product, personas[i], seed=i * 97) for i in range(5)]
    for r in runs:  # 라벨을 강제로 흩뜨려 저신뢰 경로를 검증
        r.verdict.suitability = ["pass", "warn", "fail", "warn", "pass"][runs.index(r)]
        r.verdict.intent_score = [10, 50, 90, 30, 70][runs.index(r)]
    cr = aggregate(runs)
    assert cr.label_agreement < 0.6
    assert cr.confidence < 0.8


# -------------------------------------------------------- 시뮬레이션·민감도·리포트
def test_simulate_and_report(personas):
    product = load_product("01_youth_step_saving")
    sim = simulate_product(
        product, k_personas=2, n_seeds=2, mode="debate", workers=2, personas=personas, progress=False
    )
    assert sim.segments
    for s in sim.segments:
        assert 0 <= s.adoption_rate <= 1
        assert abs(sum(s.verdict_mix.values()) - 1.0) < 1e-6
        assert s.flag in {"정상", "조건 보완 권고", "판매원칙 위험", "추가 검증 필요"}
    text = build_report(product, sim)
    assert "세그먼트별 결과" in text
    assert "한계" in text


def test_apply_variant_changes_rates():
    product = load_product("01_youth_step_saving")
    v = apply_variant(product, VariantSpec(label="금리-0.5", rate_delta_pct=-0.5))
    assert v.intr_rate == round(product.intr_rate - 0.5, 3)
    assert v.product_id.endswith("::금리-0.5")
    assert product.intr_rate == 3.7  # 원본 불변


# ------------------------------------------------------------------ 애블레이션
def test_benchmark_cases_are_valid():
    cases = load_cases()
    assert len(cases) >= 10
    for c in cases:
        assert c.label in {"pass", "warn", "fail"}
        assert c.persona.finance is not None
        # 판매 정황이 모델에 전달되어야 한다. 없으면 '판매 행위의 적법성'을 묻는
        # 정답 라벨을 '상품 위험도'로만 판단하게 된다 (실측된 불일치).
        assert c.facts, f"{c.case_id}: facts 누락"


def test_situation_reaches_the_prompt():
    """판매 정황이 실제로 프롬프트에 들어가는지 확인 (전달 누락 회귀 방지)."""
    from fdm.agents import prompts as P

    ctx = P.context_block("상품", "페르소나", "근거", situation="직원이 재진단을 유도했다")
    assert "판매 정황" in ctx and "재진단을 유도" in ctx
    # 정황이 없으면 해당 절이 아예 나오지 않아야 한다 (상품 설계 시뮬레이션용)
    assert "판매 정황" not in P.context_block("상품", "페르소나", "근거")


def test_ablation_smoke():
    rep = run_ablation(n_seeds=1, limit=2, progress=False)
    assert {a.arm for a in rep.arms} == {
        "naive", "single_norag", "single", "debate", "ensemble"
    }
    for a in rep.arms:
        assert 0 <= a.accuracy <= 1
        assert 0 <= a.risk_accuracy <= 1
        # 우려 단위 채점을 위해 산출 텍스트가 보존되어야 한다
        assert all(o.risks or o.evidence for o in a.outcomes)
    debate = next(a for a in rep.arms if a.arm == "debate")
    single = next(a for a in rep.arms if a.arm == "single")
    # 디베이트는 호출 수가 많다(비용) — 리포트에서 정확도와 함께 제시된다
    assert debate.total_llm_calls > single.total_llm_calls
    for o in next(a for a in rep.arms if a.arm == "single_norag").outcomes:
        assert o.case_id


def test_refined_rules_keep_legitimate_concerns():
    """실측된 오기각 2건이 더 이상 기각되지 않아야 한다.

    ① 예금자보호 한도를 넘는 예치 → 초과분 원금손실 우려는 정당
    ② 일시 예치액이 금융자산을 크게 넘음 → 감당능력 우려는 정당
    """
    from fdm.agents.schema import Concern
    from fdm.eval.benchmark import load_cases
    from fdm.facts import build_fact_pack, screen_typed_concerns

    case = next(c for c in load_cases() if c.case_id == "CASE-010")  # 고액 예금, 한도 초과
    f = build_fact_pack(case.product, case.persona)
    assert f.deposit_exposure > 5000, "보호 한도를 넘는 예치 상품이어야 한다"
    assert f.lump_sum_burden is not None and f.lump_sum_burden > 0.8

    kept, dropped = screen_typed_concerns(
        [
            Concern(type="principal_loss_risk", statement="한도 초과분 비보호", anchor="1억"),
            Concern(type="affordability", statement="예치 부담", anchor="금융자산 대비"),
        ],
        f,
    )
    assert not dropped, f"정당한 우려가 기각됐다: {dropped}"

    # 반대로 한도 내 예금(CASE-006)에서는 원금손실 우려를 여전히 기각한다
    ok = next(c for c in load_cases() if c.case_id == "CASE-006")
    f2 = build_fact_pack(ok.product, ok.persona)
    kept2, dropped2 = screen_typed_concerns(
        [Concern(type="principal_loss_risk", statement="원금 손실", anchor="x")], f2
    )
    assert len(dropped2) == 1, "한도 내 원금보장 상품은 여전히 기각해야 한다"


def test_skeptic_prompt_requires_independent_scan():
    """회의론자가 옹호자 논점에만 묶이지 않도록 유형 목록과 2부 지시가 있어야 한다."""
    from fdm.agents import prompts as P

    s = P.skeptic_system()
    assert "독립적으로 추가 제기" in s
    assert "우려 유형 목록" in s and "deposit_protection_limit" in s
    assert "2부" in s


def test_merge_concerns_unions_and_marks_sources():
    """병행은 두 방식의 우려를 합치고, 교차 확인된 것을 위로 올려야 한다."""
    from fdm.agents.schema import Concern, merge_concerns

    merged = merge_concerns(
        {
            "single": [
                Concern(type="tying_sale", severity="중대", statement="끼워팔기", anchor="제20조"),
                Concern(type="affordability", severity="주의", statement="부담 있음", anchor=""),
            ],
            "debate": [
                Concern(type="fee_hidden", severity="중대", statement="연회비", anchor="3만원"),
                Concern(type="affordability", severity="치명", statement="감당 불가", anchor="여유 -81만원"),
            ],
        }
    )
    types = {c.type: c for c in merged}
    assert set(types) == {"tying_sale", "affordability", "fee_hidden"}, "합집합이어야 한다"
    # 양쪽이 제기한 우려는 교차 확인 표시가 붙고 맨 위로 온다
    assert types["affordability"].sources == ["debate", "single"]
    assert merged[0].type == "affordability"
    # 겹칠 때 앵커 있는 쪽을 남긴다
    assert types["affordability"].anchor == "여유 -81만원"
    assert types["affordability"].severity == "치명"
    # 한쪽만 제기한 것도 보존된다
    assert types["tying_sale"].sources == ["single"]
    assert types["fee_hidden"].sources == ["debate"]


def test_ensemble_arm_runs_and_flags_disagreement():
    """병행 arm이 두 방식을 모두 돌리고 라벨 불일치를 기록해야 한다."""
    from fdm.eval.benchmark import load_cases, run_case_arm

    case = load_cases()[0]
    oc = run_case_arm(case, "ensemble", n_seeds=1)
    assert oc.n_llm_calls >= 6, "단발 1회 + 디베이트 5회 이상이어야 한다"
    assert oc.concerns, "합쳐진 우려가 있어야 한다"
    assert all("sources" in c for c in oc.concerns)


# ------------------------------------------------- 우려 계층 (교차확인 × 심각도)
def test_concern_tier_matrix():
    """계층은 교차확인과 심각도 두 축으로만 결정된다."""
    from fdm.concerns import concern_tier

    both, solo = ["single", "debate"], ["single"]
    assert concern_tier("치명", both) == "T1"
    assert concern_tier("치명", solo) == "T2"
    assert concern_tier("중대", both) == "T2"
    assert concern_tier("중대", solo) == "T3"
    # 주의 이하는 관측 정확도 0%라 교차확인이 붙어도 올라가지 않는다
    assert concern_tier("주의", both) == "T4"
    assert concern_tier("경미", both) == "T4"


def test_stamp_sources_does_not_overwrite_existing():
    """병행이 이미 표시한 sources를 단독 실행 표시가 덮어쓰면 교차확인이 사라진다."""
    from fdm.agents.schema import Concern, stamp_sources

    kept = Concern(type="affordability", statement="x", sources=["single", "debate"])
    fresh = Concern(type="fee_hidden", statement="y")
    stamp_sources([kept, fresh], "debate")
    assert kept.sources == ["single", "debate"]
    assert fresh.sources == ["debate"]


def test_debate_result_concerns_carry_source(personas):
    product = load_product("01_youth_step_saving")
    res = run_debate(product, personas[0], segment="테스트", seed=1)
    assert res.verdict.concerns, "우려가 있어야 한다"
    assert all(c.sources == ["debate"] for c in res.verdict.concerns)


def test_multiseed_repetition_does_not_inflate_cross_check(personas):
    """시드 반복은 재현성이지 정확성이 아니다. 계층을 부풀리면 안 된다.

    실측: 저신뢰 오답률 40% vs 전체 42% — 재현성은 정확성과 무관했다.
    """
    product = load_product("02_senior_time_deposit")
    runs = [run_debate(product, personas[0], seed=5) for _ in range(3)]
    cr = aggregate(runs)
    assert cr.concerns, "우려가 시뮬레이션 경로까지 전달돼야 한다"
    # 같은 debate 방식을 3회 돌렸을 뿐이므로 교차확인이 아니다
    assert all(c.sources == ["debate"] for c in cr.concerns)
    assert all(not c.cross_checked for c in cr.concerns)
    assert all(c.tier in {"T2", "T3", "T4"} for c in cr.concerns), "T1은 교차확인에서만 나온다"
    # 반복 비율은 별도 필드로 나온다
    assert cr.concern_run_ratio
    assert all(0.0 < v <= 1.0 for v in cr.concern_run_ratio.values())


def test_report_shows_tiers_and_flags_missing_cross_check():
    product = load_product("01_youth_step_saving")
    sim = simulate_product(product, n_seeds=1, k_personas=1, mode="debate")
    assert any(s.top_concerns for s in sim.segments), "세그먼트에 우려가 모여야 한다"
    text = build_report(product, sim)
    assert "## 3. 우려 계층" in text
    assert "즉시 조치" in text and "접어두기" in text
    # 단독 실행에서 교차확인이 불가능하다는 사실을 숨기지 않는다
    assert "교차확인이 성립하지 않는다" in text


def test_tier_basis_carries_sample_size_caveat():
    """계층 근거 수치는 12건·시드1회 잠정치다. 표본 고지 없이 산출물에 나가면 안 된다."""
    from fdm.concerns import TIER_BASIS, TIER_CAVEAT, TIER_ORDER

    assert set(TIER_BASIS) == set(TIER_ORDER)
    assert "12건" in TIER_CAVEAT and "가공" in TIER_CAVEAT
    product = load_product("01_youth_step_saving")
    sim = simulate_product(product, n_seeds=1, k_personas=1, mode="debate")
    assert TIER_CAVEAT in build_report(product, sim), "리포트가 표본 고지를 함께 실어야 한다"


def test_principle_recall_excludes_pass_cases():
    """gold=pass 케이스의 principle은 '쟁점'이지 '위반'이 아니다 (결함 15번).

    이걸 재현율에 넣으면 깨끗한 케이스에서 위반을 주장해야 점수를 받는다.
    실측: CASE-008에서 모델이 '적합성 위반' 주장을 뺐는데 100%→0%로 떨어져
    옳은 자제가 성능 하락으로 기록됐다.
    """
    from fdm.eval.benchmark import has_violation, load_cases

    assert has_violation("fail") and has_violation("warn")
    assert not has_violation("pass")
    # 정답셋의 pass 케이스에도 principle이 붙어 있다 — 그래서 제외가 필요하다
    pass_cases = [c for c in load_cases() if c.label == "pass"]
    assert pass_cases, "pass 케이스가 있어야 이 방어가 의미 있다"
    assert any(c.principle for c in pass_cases), "pass인데 principle이 붙어 있는 상황을 전제한다"


def test_arm_score_ignores_pass_in_principle_recall():
    from fdm.eval.benchmark import run_ablation

    r = run_ablation(arms=("single",), n_seeds=1, progress=False)
    arm = r.arms[0]
    applicable = [o for o in arm.outcomes if o.principle_applicable]
    skipped = [o for o in arm.outcomes if not o.principle_applicable]
    assert applicable and skipped, "두 종류가 다 있어야 한다"
    assert all(o.gold != "pass" for o in applicable)
    # arm 평균은 대상 케이스만 반영한다
    expected = sum(o.principle_recall for o in applicable) / len(applicable)
    assert abs(arm.principle_recall - expected) < 0.002
    # pass 케이스의 과잉 위반주장은 별도로 측정된다
    assert arm.false_principle_rate >= 0.0


# ------------------------------------------------- 정답셋 확대 (6-3)
def test_constructed_pass_cases_stay_out_of_rag():
    """구성한 pass 케이스가 RAG 코퍼스에 들어가면 안 된다.

    케이스 문서에는 `판정 {label}`이 붙어 있고 평가 시 자기 사례만 제외되므로,
    pass 케이스를 코퍼스에 넣으면 이웃 문서 라벨이 pass로 쏠린다. 그러면
    pass 정확도 상승이 모델 개선인지 RAG 사전확률 이동인지 구분할 수 없다.
    """
    from fdm.rag.corpus import load_corpus

    case_docs = [d for d in load_corpus() if d.kind == "case"]
    ids = {d.doc_id for d in case_docs}
    constructed = {c.case_id for c in load_cases() if c.case_id not in ids}
    assert constructed, "구성 케이스가 코퍼스에서 빠져 있어야 한다"
    # 코퍼스의 라벨 분포는 조정례 12건 그대로여야 한다
    from collections import Counter

    labels = Counter(d.title.split("판정 ")[-1] for d in case_docs)
    assert labels == {"fail": 5, "warn": 5, "pass": 2}, f"코퍼스 라벨 분포가 바뀌었다: {labels}"


def test_pass_cases_trip_no_severity_floor():
    """pass 케이스가 사실팩 승격 규칙에 걸리면 '깨끗한 케이스'가 아니다.

    걸리는 케이스를 pass 정답으로 두면 과잉경고를 재는 게 아니라
    규칙과 정답의 불일치를 재게 된다.
    """
    from fdm.facts import SEVERITY_FLOORS, build_fact_pack

    for c in load_cases():
        if c.label != "pass":
            continue
        fp = build_fact_pack(c.product, c.persona)
        tripped = {k: v(fp) for k, v in SEVERITY_FLOORS.items() if v(fp)}
        assert not tripped, f"{c.case_id}가 승격 규칙에 걸린다: {tripped}"


def test_gold_covers_every_case():
    """정답 우려 매핑에 빠진 케이스가 있으면 조용히 0점 처리된다."""
    import json as _json

    from fdm.config import BENCHMARK_DIR

    gold = _json.loads((BENCHMARK_DIR / "concern_taxonomy.json").read_text(encoding="utf-8"))["gold"]
    ids = {c.case_id for c in load_cases()}
    assert set(gold) == ids, f"불일치: 정답셋에만 {ids - set(gold)} / gold에만 {set(gold) - ids}"
    for c in load_cases():
        if c.label == "pass":
            assert gold[c.case_id] == [], f"{c.case_id}는 pass인데 정답 우려가 있다"


def test_lawonly_arm_excludes_dispute_cases():
    """law-only arm은 조정례를 검색하지 않아야 한다.

    조정례 문서 제목에 `— 판정 {label}`이 붙어 있어 이웃 사례의 정답이 노출된다.
    이 arm은 그 노출의 기여분을 분리해 재는 장치이므로, 조정례가 한 건이라도
    섞이면 측정 자체가 무의미해진다.
    """
    from fdm.agents.debate import _query_for
    from fdm.eval.benchmark import LAW_ONLY_KINDS
    from fdm.rag.retriever import get_retriever

    r = get_retriever(False)
    for c in load_cases():
        hits = r.retrieve(
            _query_for(c.product, c.persona), k=5,
            kinds=LAW_ONLY_KINDS, exclude_ids={c.case_id},
        )
        assert hits, f"{c.case_id}: 조문만으로도 근거가 검색돼야 한다"
        kinds = {h.doc.kind for h in hits}
        assert "case" not in kinds, f"{c.case_id}에 조정례가 섞였다: {kinds}"


def test_lawonly_config_does_not_leak_into_other_arms():
    """arm별 cfg 복제가 안 되면 한 번 law-only를 돌린 뒤 다른 arm까지 오염된다."""
    from fdm.agents.debate import DebateConfig
    from fdm.eval.benchmark import run_case_arm

    cfg = DebateConfig()
    case = load_cases()[0]
    run_case_arm(case, "single_lawonly", n_seeds=1, config=cfg)
    assert cfg.retrieve_kinds is None, "원본 config가 변경됐다"

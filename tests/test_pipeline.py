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


def test_ablation_smoke():
    rep = run_ablation(n_seeds=1, limit=2, progress=False)
    assert {a.arm for a in rep.arms} == {"single_norag", "single", "debate"}
    for a in rep.arms:
        assert 0 <= a.accuracy <= 1
        assert 0 <= a.risk_accuracy <= 1
    debate = next(a for a in rep.arms if a.arm == "debate")
    single = next(a for a in rep.arms if a.arm == "single")
    # 디베이트는 호출 수가 많다(비용) — 리포트에서 정확도와 함께 제시된다
    assert debate.total_llm_calls > single.total_llm_calls
    for o in next(a for a in rep.arms if a.arm == "single_norag").outcomes:
        assert o.case_id

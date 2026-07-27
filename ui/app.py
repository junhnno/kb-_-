"""Streamlit 대시보드: 세그먼트 히트맵 + 신뢰도 + 근거 + 애블레이션.

실행: uv run streamlit run ui/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fdm.config import OUTPUT_DIR, SETTINGS  # noqa: E402
from fdm.eval.benchmark import AblationReport, compare_holding_rates  # noqa: E402
from fdm.eval.simulate import SimulationReport, default_variants, sensitivity_analysis, simulate_product  # noqa: E402
from fdm.products.schema import load_all_products, load_product  # noqa: E402
from fdm.report import build_report, save_report  # noqa: E402

st.set_page_config(page_title="금융상품 사전검증 에이전트", layout="wide")

LEVEL_COLOR = {"high": "#2e7d32", "medium": "#ef6c00", "low": "#c62828"}
FLAG_EMOJI = {"정상": "🟢", "조건 보완 권고": "🟡", "판매원칙 위험": "🔴", "추가 검증 필요": "⚪"}


@st.cache_data(show_spinner=False)
def list_products() -> dict[str, str]:
    return {p.name: p.product_id for p in load_all_products()}


def product_files() -> dict[str, Path]:
    return {p.stem: p for p in sorted((ROOT / "data" / "products").glob("*.json"))}


def load_sim_files() -> dict[str, Path]:
    return {p.name: p for p in sorted(OUTPUT_DIR.glob("sim_*.json"), reverse=True)}


st.title("합성 페르소나 기반 금융상품 사전검증")
st.caption(
    "상품기획팀용 세그먼트 설계·검증 도구입니다. 개별 소비자에 대한 상품 추천이 아니며, "
    "결과는 합성 페르소나 기반 **탐색·경보용** 지표입니다."
)

with st.sidebar:
    st.subheader("실행 환경")
    st.write(f"backend: `{SETTINGS.backend}`")
    st.write(f"토론모델: `{SETTINGS.model_small}`")
    st.write(f"심판모델: `{SETTINGS.model_judge}`")
    st.divider()
    st.subheader("새 시뮬레이션 실행")
    pf = product_files()
    sel_file = st.selectbox("상품", list(pf.keys()))
    n_seeds = st.slider("멀티시드 반복", 1, 7, 3)
    k_personas = st.slider("세그먼트별 페르소나 수", 1, 10, 3)
    mode = st.radio("모드", ["debate", "single"], horizontal=True)
    run_sens = st.checkbox("민감도 분석 포함", value=False)
    if st.button("실행", type="primary"):
        prod = load_product(pf[sel_file])
        with st.spinner("디베이트 실행 중… (LLM 호출이 많아 시간이 걸립니다)"):
            rep = simulate_product(
                prod, k_personas=k_personas, n_seeds=n_seeds, mode=mode, workers=4, progress=False
            )
            rep.save()
            if run_sens:
                rows = sensitivity_analysis(
                    prod, default_variants(prod), k_personas=max(2, k_personas // 2),
                    n_seeds=max(1, n_seeds - 1), mode=mode,
                )
                (OUTPUT_DIR / f"sensitivity_{prod.product_id}.json").write_text(
                    json.dumps([r.model_dump() for r in rows], ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        st.success("완료. 아래에서 결과를 선택하세요.")
        st.cache_data.clear()

sims = load_sim_files()
if not sims:
    st.info("결과가 없습니다. 사이드바에서 시뮬레이션을 실행하거나 `uv run fdm simulate <상품>`을 실행하세요.")
    st.stop()

sel = st.selectbox("결과 파일", list(sims.keys()))
sim = SimulationReport.load(sims[sel])

c1, c2, c3, c4 = st.columns(4)
c1.metric("상품", sim.product_name)
c2.metric("모드", sim.mode)
c3.metric("멀티시드", f"{sim.n_seeds}회")
c4.metric(
    "평균 신뢰도",
    f"{sum(s.mean_confidence for s in sim.segments) / max(1, len(sim.segments)):.2f}",
)

seg_df = pd.DataFrame(
    [
        {
            "세그먼트": s.segment,
            "페르소나수": s.n_personas,
            "가입의향(평균)": s.mean_intent,
            "가입률": s.adoption_rate,
            "pass": s.verdict_mix.get("pass", 0.0),
            "warn": s.verdict_mix.get("warn", 0.0),
            "fail": s.verdict_mix.get("fail", 0.0),
            "신뢰도": s.mean_confidence,
            "저신뢰비율": s.low_confidence_ratio,
            "상태": s.flag,
        }
        for s in sim.segments
    ]
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["세그먼트 히트맵", "케이스 상세·근거", "민감도", "벤치마크·애블레이션", "리포트"]
)

with tab1:
    st.subheader("세그먼트 × 지표 히트맵")
    metrics = ["가입의향(평균)", "가입률", "pass", "warn", "fail", "신뢰도"]
    long = seg_df.melt(id_vars=["세그먼트"], value_vars=metrics, var_name="지표", value_name="값")
    long["표시"] = long.apply(
        lambda r: f"{r['값']:.0f}" if r["지표"] == "가입의향(평균)" else f"{r['값']:.2f}", axis=1
    )
    # 지표마다 스케일이 달라 지표별 정규화 후 색을 매긴다
    long["정규값"] = long.groupby("지표")["값"].transform(
        lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else 0.5
    )
    heat = (
        alt.Chart(long)
        .mark_rect()
        .encode(
            x=alt.X("지표:N", sort=metrics),
            y=alt.Y("세그먼트:N"),
            color=alt.Color("정규값:Q", scale=alt.Scale(scheme="redyellowgreen"), legend=None),
            tooltip=["세그먼트", "지표", "값"],
        )
        .properties(height=40 * max(1, len(seg_df)))
    )
    text = heat.mark_text(baseline="middle", fontSize=12).encode(
        text="표시:N", color=alt.value("black")
    )
    st.altair_chart(heat + text, width="stretch")

    st.subheader("세그먼트 요약")
    show = seg_df.copy()
    show["상태"] = show["상태"].map(lambda f: f"{FLAG_EMOJI.get(f, '')} {f}")
    st.dataframe(show, width="stretch", hide_index=True)
    low = [s for s in sim.segments if s.low_confidence_ratio >= 0.5]
    if low:
        st.warning(
            "저신뢰 세그먼트(추가 검증 필요): " + ", ".join(s.segment for s in low)
        )

with tab2:
    st.subheader("페르소나별 판정과 근거")
    seg_pick = st.selectbox("세그먼트", [s.segment for s in sim.segments])
    seg = next(s for s in sim.segments if s.segment == seg_pick)
    for c in seg.cases:
        color = LEVEL_COLOR[c.confidence_level]
        with st.expander(
            f"{c.persona_id} — {c.modal_suitability.upper()} / 의향 {c.intent_mean} / "
            f"신뢰도 {c.confidence:.2f} ({c.confidence_level})"
            + ("  ⚠ 추가 검증 필요" if c.needs_review else "")
        ):
            a, b = st.columns(2)
            a.markdown(
                f"**라벨 분포**: {c.label_counts}  \n"
                f"**합의도**: {c.label_agreement:.0%}  \n"
                f"**의향 범위**: {c.intent_min}~{c.intent_max} (σ={c.intent_std})"
            )
            b.markdown(
                f"<span style='color:{color}'><b>신뢰도 {c.confidence:.2f}</b></span>  \n"
                f"심판 자기신뢰: {c.judge_self_confidence:.2f}  \n"
                f"무근거 발화: {c.ungrounded_turns}건",
                unsafe_allow_html=True,
            )
            st.markdown("**근거**")
            st.write(c.evidence or "—")
            st.markdown("**위험요인**")
            st.write(c.risks or "—")
            st.markdown("**개선권고**")
            st.write(c.recommendations or "—")
            st.caption("인용 문서: " + ", ".join(c.grounding_doc_ids))

with tab3:
    spath = OUTPUT_DIR / f"sensitivity_{sim.product_id}.json"
    if not spath.exists():
        st.info("민감도 분석 결과가 없습니다. 사이드바에서 '민감도 분석 포함'을 켜고 실행하세요.")
    else:
        rows = pd.DataFrame(json.loads(spath.read_text(encoding="utf-8")))
        st.subheader("가격·조건 변경에 따른 반응")
        chart = (
            alt.Chart(rows)
            .mark_bar()
            .encode(
                x=alt.X("label:N", title="시나리오", sort=None),
                y=alt.Y("mean_intent:Q", title="평균 가입의향"),
                color=alt.Color("segment:N", title="세그먼트"),
                column=alt.Column("segment:N", title=None),
                tooltip=["label", "segment", "mean_intent", "adoption_rate", "fail_ratio"],
            )
            .properties(height=260)
        )
        st.altair_chart(chart)
        st.dataframe(rows, width="stretch", hide_index=True)

with tab4:
    st.subheader("KOSIS 보유율 대조 (외적 타당성)")
    prod = next((p for p in load_all_products() if p.product_id == sim.product_id), None)
    if prod:
        hold = compare_holding_rates(sim, prod.category)
        m1, m2 = st.columns(2)
        m1.metric("MAE", f"{hold.mae:.3f}")
        m2.metric("Spearman ρ", "n/a" if hold.spearman is None else f"{hold.spearman:.3f}")
        if hold.rows:
            hdf = pd.DataFrame([r.model_dump() for r in hold.rows]).rename(
                columns={
                    "segment": "세그먼트",
                    "actual_holding_rate": "실제 보유율",
                    "simulated_adoption_rate": "시뮬 가입률",
                }
            )
            hlong = hdf.melt(
                id_vars=["세그먼트"],
                value_vars=["실제 보유율", "시뮬 가입률"],
                var_name="구분",
                value_name="비율",
            )
            st.altair_chart(
                alt.Chart(hlong)
                .mark_bar()
                .encode(
                    x=alt.X("세그먼트:N"),
                    y=alt.Y("비율:Q"),
                    color=alt.Color("구분:N"),
                    xOffset=alt.XOffset("구분:N"),
                    tooltip=["세그먼트", "구분", "비율"],
                )
                .properties(height=300),
                width="stretch",
            )
        st.caption(hold.note)
    else:
        st.info("상품 정의를 찾을 수 없어 보유율 대조를 건너뜁니다(민감도 변형 상품일 수 있음).")

    st.divider()
    st.subheader("애블레이션 — 디베이트가 정확성을 높이는가")
    apath = OUTPUT_DIR / "ablation.json"
    if not apath.exists():
        st.info("애블레이션 결과가 없습니다. `uv run fdm ablation` 을 실행하세요.")
    else:
        abl = AblationReport.load(apath)
        name = {
            "single_norag": "단발(RAG 없음)",
            "single": "단발+RAG",
            "debate": "3진영 디베이트+RAG",
        }
        adf = pd.DataFrame(
            [
                {
                    "조건": name.get(a.arm, a.arm),
                    "적합성 적중률": a.accuracy,
                    "위험탐지 정확도": a.risk_accuracy,
                    "macro F1": a.macro_f1,
                    "위반원칙 재현율": a.principle_recall,
                    "LLM 호출수": a.total_llm_calls,
                }
                for a in abl.arms
            ]
        )
        st.altair_chart(
            alt.Chart(
                adf.melt(
                    id_vars=["조건"],
                    value_vars=["적합성 적중률", "위험탐지 정확도", "macro F1", "위반원칙 재현율"],
                    var_name="지표",
                    value_name="값",
                )
            )
            .mark_bar()
            .encode(x="지표:N", y="값:Q", color="조건:N", xOffset="조건:N", tooltip=["조건", "지표", "값"])
            .properties(height=320),
            width="stretch",
        )
        st.dataframe(adf, width="stretch", hide_index=True)
        st.metric("디베이트 − 단발(RAG) 적중률 차이", f"{abl.delta_accuracy_vs_single:+.1%}")
        for c in abl.caveats:
            st.caption("· " + c)

with tab5:
    prod = next((p for p in load_all_products() if p.product_id == sim.product_id), None)
    if prod is None:
        st.info("상품 정의를 찾을 수 없습니다.")
    else:
        hold = compare_holding_rates(sim, prod.category)
        abl = (
            AblationReport.load(OUTPUT_DIR / "ablation.json")
            if (OUTPUT_DIR / "ablation.json").exists()
            else None
        )
        text = build_report(prod, sim, ablation=abl, holding=hold)
        st.markdown(text)
        if st.button("리포트 저장 (outputs/)"):
            st.success(f"저장: {save_report(text, prod.product_id)}")
        st.download_button("Markdown 다운로드", text, file_name=f"report_{prod.product_id}.md")

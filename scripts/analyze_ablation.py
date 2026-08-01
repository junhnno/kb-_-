"""애블레이션 결과 분석.

세 가지를 본다.
  1. 라벨 지표 (적중률·macro F1·warn율)
  2. 오분류 방향 — 보수 편향 vs 순응 편향 vs 노이즈 판별
  3. 우려 단위 채점 — 라벨로는 안 보이는 '무엇을 짚었는가'

사용: uv run python scripts/analyze_ablation.py [outputs/ablation.json]
"""

from __future__ import annotations

import collections
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAXONOMY = ROOT / "data" / "benchmark" / "concern_taxonomy.json"

NAME = {
    "naive": "순진(규칙·RAG없음)",
    "single_norag": "단발(RAG없음)",
    "single_lawonly": "단발+조문만(조정례X)",
    "debate_lawonly": "디베이트+조문만(조정례X)",
    "ensemble": "병행(단발+디베이트)",
    "single": "단발+RAG",
    "debate": "디베이트+RAG",
}
ORDER = ["pass", "warn", "fail"]
SEVERITY = {"pass": 0, "warn": 1, "fail": 2}


def scored_text(o: dict) -> str:
    """채점 대상 텍스트.

    `evidence`는 제외한다. 검색된 문서 원문이 그대로 인용되어 있어서
    모델이 제기한 우려가 아닌 참조 문구가 검출된다
    (실측: ELS 케이스에서 GUIDE-RATE-AD 인용문 때문에 '적금 광고 오인'이 잡혔다).
    """
    return " ".join(o.get("risks", []) + o.get("recommendations", [])) + " " + o.get("summary", "")


def detect_concerns(text: str, types: list[dict]) -> set[str]:
    """[구형] 산출 텍스트에서 우려 유형을 키워드로 검출.

    표현이 조금만 달라져도 놓치므로 측정 노이즈가 크다(동일 조건 재실행에서 12%p 변동).
    구조화 우려(concerns)가 있는 결과 파일에서는 쓰지 않는다.
    """
    found = set()
    for t in types:
        if any(kw in text for kw in t["keywords"]):
            found.add(t["id"])
    return found


def concern_types(o: dict, types: list[dict]) -> tuple[set[str], str]:
    """우려 유형 집합과 채점 방식을 돌려준다.

    구조화 우려가 있으면 유형을 그대로 쓴다(정확). 없으면 키워드로 추측한다(구형 파일 호환).
    """
    cs = o.get("concerns") or []
    if cs:
        return {c.get("type") for c in cs if c.get("type") and c.get("type") != "other"}, "typed"
    return detect_concerns(scored_text(o), types), "keyword"


def top_k_types(o: dict, k: int) -> set[str]:
    """심각도 상위 k개 우려의 유형 (분류 도구는 상위 몇 건의 정밀도가 중요하다)."""
    cs = [c for c in (o.get("concerns") or []) if c.get("type") != "other"]
    if not cs:
        return set()
    ranked = sorted(cs, key=lambda c: SEVERITY_RANK.get(c.get("severity", "주의"), 1), reverse=True)
    return {c["type"] for c in ranked[:k]}


SEVERITY_RANK = {"경미": 0, "주의": 1, "중대": 2, "치명": 3}


def principle_metrics(arm: dict) -> tuple[float, float]:
    """(위반원칙 재현율, pass 케이스 과잉 위반주장) — outcomes에서 직접 계산한다.

    저장된 `arm['principle_recall']`을 쓰지 않는 이유: 결함 15번 수정 이전에
    만들어진 결과 파일은 pass 케이스를 포함해 평균해 두었다. 여기서 다시 계산해야
    구 파일과 신 파일을 같은 기준으로 비교할 수 있다.
    """
    applicable = [o for o in arm["outcomes"] if o["gold"] != "pass"]
    recall = statistics.fmean([o["principle_recall"] for o in applicable]) if applicable else 0.0
    clean = [o for o in arm["outcomes"] if o["gold"] == "pass"]
    false_claims = (
        statistics.fmean([len(o.get("predicted_principles", [])) for o in clean]) if clean else 0.0
    )
    return recall, false_claims


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs" / "ablation.json"
    a = json.loads(path.read_text(encoding="utf-8"))
    tax = json.loads(TAXONOMY.read_text(encoding="utf-8"))
    types, gold_map = tax["types"], tax["gold"]

    print(f"backend={a['backend']}  심판모델={a['model_judge']}  시드={a['n_seeds']}")
    if a["backend"] == "mock":
        print("\n*** 경고: mock 백엔드다. 아래 수치는 난수이며 해석 불가. ***")

    # ---------------------------------------------------------------- 1. 라벨 지표
    print("\n=== 1. 라벨 지표 ===")
    hdr = (
        f"{'조건':<20}{'적중률':>8}{'위험탐지':>9}{'macroF1':>9}{'원칙재현':>9}"
        f"{'과잉위반':>9}{'warn율':>8}{'pass율':>8}{'호출':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for arm in a["arms"]:
        preds = [o["pred"] for o in arm["outcomes"]]
        n = len(preds)
        pr, fp = principle_metrics(arm)
        print(
            f"{NAME.get(arm['arm'], arm['arm']):<20}"
            f"{arm['accuracy']:>7.1%}{arm['risk_accuracy']:>9.1%}{arm['macro_f1']:>9.3f}"
            f"{pr:>9.1%}{fp:>9.2f}"
            f"{preds.count('warn') / n:>8.1%}{preds.count('pass') / n:>8.1%}"
            f"{arm['total_llm_calls']:>6}"
        )
    print("* 원칙재현 = 실제 위반이 있는 케이스(gold≠pass)만의 평균.")
    print("  gold=pass의 principle은 '쟁점'이지 '위반'이 아니라 채점 대상이 아니다(결함 15번).")
    print("* 과잉위반 = gold=pass에서 주장한 위반원칙 평균 개수. 낮을수록 좋다.")

    # ------------------------------------------------------- 2. 오분류 방향 (편향 판별)
    print("\n=== 2. 오분류 방향 (편향 vs 노이즈) ===")
    print("보수화=정답보다 엄하게(pass→warn) / 관대화=정답보다 느슨하게(fail→pass)\n")
    for arm in a["arms"]:
        strict = lenient = 0
        dirs: collections.Counter = collections.Counter()
        for o in arm["outcomes"]:
            if o["gold"] == o["pred"]:
                continue
            dirs[f"{o['gold']}→{o['pred']}"] += 1
            if SEVERITY[o["pred"]] > SEVERITY[o["gold"]]:
                strict += 1
            else:
                lenient += 1
        total = strict + lenient
        if total == 0:
            verdict = "오류 없음"
        elif strict >= total * 0.7:
            verdict = "보수 편향"
        elif lenient >= total * 0.7:
            verdict = "순응 편향"
        else:
            verdict = "노이즈(양방향)"
        print(
            f"{NAME.get(arm['arm'], arm['arm']):<20} 오류 {total:>2}건  "
            f"보수화 {strict}  관대화 {lenient}  → {verdict}"
        )
        if dirs:
            print(f"{'':<20} {dict(sorted(dirs.items()))}")

    # ------------------------------------------------------------- 3. 혼동행렬
    print("\n=== 3. 혼동행렬 (행=정답, 열=예측) ===")
    for arm in a["arms"]:
        m = {g: collections.Counter() for g in ORDER}
        for o in arm["outcomes"]:
            m[o["gold"]][o["pred"]] += 1
        print(f"\n[{NAME.get(arm['arm'], arm['arm'])}]")
        print(f"{'':>6}" + "".join(f"{p:>7}" for p in ORDER))
        for g in ORDER:
            print(f"{g:>6}" + "".join(f"{m[g][p]:>7}" for p in ORDER))

    # -------------------------------------------------------- 4. 우려 단위 채점
    print("\n=== 4. 우려 단위 채점 (키워드 기반 대리 지표) ===")
    print("정답 우려를 몇 개 짚었나(recall) / 짚은 것 중 정답 비율(precision)\n")
    hdr2 = (
        f"{'조건':<20}{'recall':>9}{'prec':>8}{'F1':>8}{'prec@3':>9}"
        f"{'평균검출':>9}{'과잉경고':>9}{'앵커율':>8}"
    )
    print(hdr2)
    print("-" * len(hdr2))
    per_arm_detail: dict[str, dict[str, set[str]]] = {}
    modes: set[str] = set()
    severity_dist: dict[str, collections.Counter] = {}
    for arm in a["arms"]:
        recalls, precisions, p3s, counts, false_alarms = [], [], [], [], []
        anchored, total_concerns = 0, 0
        sev: collections.Counter = collections.Counter()
        detail = {}
        for o in arm["outcomes"]:
            gold = set(gold_map.get(o["case_id"], []))
            found, mode = concern_types(o, types)
            modes.add(mode)
            detail[o["case_id"]] = found
            counts.append(len(found))
            for c in o.get("concerns") or []:
                total_concerns += 1
                sev[c.get("severity", "?")] += 1
                if (c.get("anchor") or "").strip():
                    anchored += 1
            if gold:
                recalls.append(len(found & gold) / len(gold))
                precisions.append(len(found & gold) / len(found) if found else 0.0)
                t3 = top_k_types(o, 3)
                if t3:
                    p3s.append(len(t3 & gold) / len(t3))
            else:
                # 정답 우려가 없는 케이스(깨끗한 상품)에서 몇 개를 만들어냈나
                false_alarms.append(len(found))
        per_arm_detail[arm["arm"]] = detail
        severity_dist[arm["arm"]] = sev
        r = statistics.fmean(recalls) if recalls else 0.0
        p = statistics.fmean(precisions) if precisions else 0.0
        f1 = 2 * r * p / (r + p) if r + p else 0.0
        p3 = statistics.fmean(p3s) if p3s else 0.0
        fa = statistics.fmean(false_alarms) if false_alarms else 0.0
        anchor_rate = anchored / total_concerns if total_concerns else 0.0
        print(
            f"{NAME.get(arm['arm'], arm['arm']):<20}{r:>8.1%}{p:>8.1%}{f1:>8.3f}"
            f"{p3:>9.1%}{statistics.fmean(counts):>9.1f}{fa:>9.1f}{anchor_rate:>8.0%}"
        )

    mode_label = {"typed": "유형 일치(정확)", "keyword": "키워드 추측(구형)"}
    print(f"\n* 채점 방식: {', '.join(mode_label.get(m, m) for m in sorted(modes))}")
    if "keyword" in modes:
        print("  ⚠ 키워드 채점은 표현 변화에 취약하다. 구조화 우려가 있는 결과로 재측정할 것")
    print("* recall/precision은 정답 우려가 있는 10건 기준")
    print("* prec@3 = 심각도 상위 3개 우려의 정밀도 (분류 도구는 상위 몇 건이 중요)")
    n_clean = sum(1 for cid in {o["case_id"] for arm in a["arms"] for o in arm["outcomes"]}
                  if not gold_map.get(cid))
    print(f"* 과잉경고 = 정답 우려가 없는 깨끗한 케이스 {n_clean}건에서 만들어낸 우려 수 평균")
    print("* 앵커율 = 수치·조항 근거를 댄 우려의 비율 (낮으면 일반론이 많다는 뜻)")

    if any(severity_dist.values()):
        print("\n--- 심각도 분포 ---")
        for arm_id, sev in severity_dist.items():
            if sev:
                tot = sum(sev.values())
                dist = "  ".join(f"{k} {sev[k]}({sev[k]/tot:.0%})" for k in ["치명", "중대", "주의", "경미"] if sev[k])
                print(f"  {NAME.get(arm_id, arm_id):<20} {dist}")

    # ------------------------------------------------- 5. gold=pass 케이스 상세
    print("\n=== 5. gold=pass 케이스 (보수 편향이 드러나는 곳) ===")
    pass_ids = sorted(
        {o["case_id"] for arm in a["arms"] for o in arm["outcomes"] if o["gold"] == "pass"}
    )
    for cid in pass_ids:
        cells = []
        for arm in a["arms"]:
            o = next((x for x in arm["outcomes"] if x["case_id"] == cid), None)
            n_found = len(per_arm_detail[arm["arm"]].get(cid, set()))
            cells.append(f"{NAME.get(arm['arm'], arm['arm'])}={o['pred'] if o else '-'}(우려{n_found})")
        print(f"  {cid}: " + "  ".join(cells))

    # ---------------------------------------------------------- 6. 케이스별 상세
    print("\n=== 6. 케이스별 예측 ===")
    cases = [o["case_id"] for o in a["arms"][0]["outcomes"]]
    print(f"{'케이스':<11}{'정답':>5}" + "".join(f"{NAME.get(x['arm'], x['arm']):>20}" for x in a["arms"]))
    for cid in cases:
        base = next(o for o in a["arms"][0]["outcomes"] if o["case_id"] == cid)
        row = ""
        for arm in a["arms"]:
            o = next((x for x in arm["outcomes"] if x["case_id"] == cid), None)
            mark = "O" if o and o["exact"] else "X"
            row += f"{(o['pred'] if o else '-') + ' ' + mark:>20}"
        print(f"{cid:<11}{base['gold']:>5}{row}")

    print("\n=== 7. 스키마 교정 발생 ===")
    for arm in a["arms"]:
        print(f"  {NAME.get(arm['arm'], arm['arm']):<20} {arm.get('judge_schema_repairs', 0)}건")

    tier_accuracy(a, gold_map)
    return 0


def tier_accuracy(a: dict, gold_map: dict) -> None:
    """계층(교차확인 × 심각도)이 실제로 정확도 순으로 정렬되는가.

    산출물이 T1부터 읽으라고 말하려면, T1이 T3보다 실제로 맞아야 한다.
    concerns.py의 TIER_BASIS 문구는 여기서 나온 수치를 적은 것이므로
    **재측정 후 이 출력과 TIER_BASIS가 어긋나면 TIER_BASIS를 고쳐야 한다.**
    """
    sys.path.insert(0, str(ROOT / "src"))
    from fdm.concerns import TIER_LABEL, TIER_ORDER, concern_tier

    print("\n=== 8. 계층별 정확도 (교차확인 × 심각도) ===")
    print("정답 우려가 있는 케이스만 분모에 넣는다. 교차확인은 ensemble에서만 성립한다.")
    for arm in a["arms"]:
        outcomes = [o for o in arm["outcomes"] if gold_map.get(o["case_id"])]
        rows = []
        for o in outcomes:
            g = set(gold_map[o["case_id"]])
            for c in o.get("concerns") or []:
                t = c.get("type")
                if not t or t == "other":
                    continue
                rows.append((concern_tier(c.get("severity", "주의"), c.get("sources", [])), t in g))
        if not rows:
            continue
        print(f"\n[{NAME.get(arm['arm'], arm['arm'])}]")
        prev = None
        for tier in TIER_ORDER:
            hits = [ok for t, ok in rows if t == tier]
            if not hits:
                print(f"  {tier} {TIER_LABEL[tier]:6}   0/  0 = —")
                continue
            acc = sum(hits) / len(hits)
            # 서열이 뒤집히면 계층 정의가 데이터와 어긋난 것이다
            warn = "  ← 상위 계층보다 높다(서열 역전)" if prev is not None and acc > prev else ""
            print(f"  {tier} {TIER_LABEL[tier]:6} {sum(hits):3}/{len(hits):3} = {acc:5.1%}{warn}")
            prev = acc


if __name__ == "__main__":
    raise SystemExit(main())


def _dropped_summary(path: str = "outputs/ablation_facts.json") -> None:
    """사실팩이 기각한 우려를 나열한다 (환각 발생량 직접 측정)."""
    import json as _json
    from pathlib import Path as _P

    a = _json.loads(_P(path).read_text(encoding="utf-8"))
    print("\n=== 8. 사실팩이 기각한 우려 (환각) ===")
    for arm in a["arms"]:
        total = sum(len(o.get("dropped_concerns", [])) for o in arm["outcomes"])
        print(f"\n[{NAME.get(arm['arm'], arm['arm'])}] 총 {total}건 기각")
        for o in arm["outcomes"]:
            for d in o.get("dropped_concerns", []):
                print(f"   {o['case_id']}: {d}")

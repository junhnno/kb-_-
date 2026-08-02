"""오탐·정확도 개선 스위치 A/B 실행기.

한 번에 여러 조건을 순차 실행하고, 끝나면 비교표를 찍는다.
각 조건은 **별도 프로세스**로 돌린다 — 설정(SETTINGS)과 코퍼스(load_corpus)가
모듈 로드 시점에 캐시되므로, 한 프로세스 안에서 환경변수를 바꿔봐야 반영되지 않는다.

실행 예:
    uv run python scripts/run_ab_experiment.py --seeds 1
    uv run python scripts/run_ab_experiment.py --seeds 1 --arms single,debate,ensemble
    uv run python scripts/run_ab_experiment.py --only baseline,hide_label

각 조건의 결과는 outputs/ab_<조건명>.json 으로 따로 저장돼 덮어쓰이지 않는다.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

# 조건명 -> 이 조건에서 켤 환경변수.
# 한 번에 하나씩만 켠다. 두 개를 같이 켜면 어느 쪽이 기여했는지 구분할 수 없다.
CONDITIONS: dict[str, dict[str, str]] = {
    "baseline": {},
    # 라벨(적중률)을 바꿀 수 있는 유일한 개입. 디베이트 pass 2/12 문제를 겨냥한다.
    "verify_claims": {"FDM_VERIFY_CLAIMS": "1"},
    # 아래 셋은 **사후 필터**라 우려 지표만 바꾸고 라벨은 그대로다.
    "screen_situation": {"FDM_SCREEN_SITUATION": "1"},
    "hide_label": {"FDM_CASE_HIDE_LABEL": "1"},
    "verify_anchors": {"FDM_VERIFY_ANCHORS": "1"},
    "drop_unanchored": {"FDM_DROP_UNANCHORED": "1"},
    # 개별 효과를 먼저 확인한 뒤에만 쓸 것. 한 번에 켜면 기여를 나눌 수 없다.
    "all": {
        "FDM_VERIFY_CLAIMS": "1",
        "FDM_SCREEN_SITUATION": "1",
        "FDM_CASE_HIDE_LABEL": "1",
        "FDM_VERIFY_ANCHORS": "1",
        "FDM_DROP_UNANCHORED": "1",
    },
}

RUNNER = """
import json, sys
from fdm.eval.benchmark import run_ablation
arms = tuple(a for a in sys.argv[1].split(",") if a)
rep = run_ablation(arms=arms, n_seeds=int(sys.argv[2]), seed_base=int(sys.argv[3]), progress=True)
rep.save(sys.argv[4])
print("SAVED", sys.argv[4])
"""


def run_condition(name: str, arms: str, seeds: int, seed_base: int) -> Path | None:
    env = dict(os.environ)
    env.update(CONDITIONS[name])
    env.setdefault("PYTHONIOENCODING", "utf-8")
    out_path = OUT / f"ab_{name}.json"

    flags = ", ".join(f"{k}={v}" for k, v in CONDITIONS[name].items()) or "(없음)"
    print(f"\n{'='*70}\n[{name}] 시작 — 스위치: {flags}\n{'='*70}", flush=True)
    t0 = time.time()

    log_path = OUT / f"ab_{name}.log"
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [sys.executable, "-c", RUNNER, arms, str(seeds), str(seed_base), str(out_path)],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True,
        )
        # 블로킹으로 기다리지 않고 로그를 흘려 보여준다 (1시간 블랙박스 방지)
        shown = 0
        while proc.poll() is None:
            time.sleep(20)
            text = log_path.read_text(encoding="utf-8", errors="replace")
            if len(text) > shown:
                sys.stdout.write(text[shown:])
                sys.stdout.flush()
                shown = len(text)
        text = log_path.read_text(encoding="utf-8", errors="replace")
        if len(text) > shown:
            sys.stdout.write(text[shown:])

    mins = (time.time() - t0) / 60
    if proc.returncode != 0:
        print(f"[{name}] 실패 (returncode={proc.returncode}) — 로그: {log_path}", flush=True)
        return None
    print(f"[{name}] 완료 ({mins:.1f}분) → {out_path}", flush=True)
    return out_path


def _summarize_concerns(raw: dict[str, dict]) -> None:
    """우려 단위 지표 (recall/precision/과잉경고/앵커 검증).

    라벨 지표만 보면 놓치는 것이 있다. 앵커 검증(FDM_VERIFY_ANCHORS)은 우려의
    **심각도**만 낮추고 라벨은 건드리지 않으므로, 위 표에서는 변화가 0으로 보인다.
    효과가 나타나는 곳은 여기다.

    채점 로직은 scripts/analyze_ablation.py를 그대로 재사용한다 — 같은 계산을
    두 벌 두면 반드시 어긋난다.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from analyze_ablation import concern_types  # type: ignore
    except ImportError:
        return  # 분석 스크립트 구조가 바뀌었으면 조용히 건너뛴다 (본 표는 이미 찍혔다)

    tax = json.loads(
        (ROOT / "data" / "benchmark" / "concern_taxonomy.json").read_text(encoding="utf-8")
    )
    types, gold_map = tax["types"], tax["gold"]

    print(f"\n\n{'='*88}\n우려 단위 지표 — 앵커 검증·오탐 필터의 효과는 여기서 보인다\n{'='*88}")
    arms = list({a["arm"] for rep in raw.values() for a in rep["arms"]})
    for arm in sorted(arms):
        print(f"\n### {arm}")
        print(f"{'조건':<16}{'recall':>10}{'precision':>11}{'과잉경고':>10}"
              f"{'앵커율':>9}{'검증됨':>9}{'지어냄':>9}")
        for name, rep in raw.items():
            a = next((x for x in rep["arms"] if x["arm"] == arm), None)
            if not a:
                continue
            rec, prec, false_alarm = [], [], []
            n_c = n_anchored = n_verified = n_fab = 0
            for o in a["outcomes"]:
                gold = set(gold_map.get(o["case_id"], []))
                found, _ = concern_types(o, types)
                for c in o.get("concerns") or []:
                    n_c += 1
                    n_anchored += bool((c.get("anchor") or "").strip())
                    st = c.get("anchor_status")
                    n_verified += st == "verified"
                    n_fab += st == "fabricated"
                if gold:
                    rec.append(len(found & gold) / len(gold))
                    prec.append(len(found & gold) / len(found) if found else 0.0)
                else:
                    false_alarm.append(len(found))

            def avg(xs: list[float]) -> float:
                return sum(xs) / len(xs) if xs else 0.0

            def pct(n: int) -> str:
                return f"{n / n_c:.0%}" if n_c else "-"

            print(f"{name:<16}{avg(rec):>10.1%}{avg(prec):>11.1%}{avg(false_alarm):>10.1f}"
                  f"{pct(n_anchored):>9}{pct(n_verified):>9}{pct(n_fab):>9}")

    print("\n* 과잉경고 = 정답 우려가 없는 깨끗한 케이스에서 만들어낸 우려 수 평균 (낮을수록 좋다)")
    print("* 검증됨/지어냄 = FDM_VERIFY_ANCHORS를 켠 조건에서만 채워진다.")
    print("  '지어냄' 비율이 높다면, 모델이 없는 조항을 인용해 근거를 꾸며내고 있었다는 뜻이다.")


def summarize(paths: dict[str, Path]) -> None:
    """조건 x arm 별 지표 비교표. baseline 대비 증감을 함께 찍는다."""
    data: dict[str, dict[str, dict]] = {}
    raw: dict[str, dict] = {}
    for name, p in paths.items():
        if not p or not p.exists():
            continue
        rep = json.loads(p.read_text(encoding="utf-8"))
        raw[name] = rep
        data[name] = {a["arm"]: a for a in rep["arms"]}

    if not data:
        print("\n비교할 결과가 없습니다.")
        return

    arms = list(next(iter(data.values())).keys())
    print(f"\n\n{'='*88}\n결과 비교 (괄호 = baseline 대비 증감)\n{'='*88}")

    for arm in arms:
        print(f"\n### {arm}")
        w = 18  # 증감 표기까지 들어가므로 넉넉히
        print(f"{'조건':<16}{'적중률':>{w}}{'위험탐지':>{w}}{'macroF1':>{w}}"
              f"{'원칙재현':>{w}}{'과잉위반':>{w}}")
        base = data.get("baseline", {}).get(arm)
        for name, by_arm in data.items():
            a = by_arm.get(arm)
            if not a:
                continue

            def cell(key: str, pct: bool = True) -> str:
                v = a.get(key)
                if v is None:
                    return "-"
                s = f"{v:.1%}" if pct else f"{v:.3f}"
                if base and name != "baseline" and base.get(key) is not None:
                    d = v - base[key]
                    s += f"({d:+.1%})" if pct else f"({d:+.3f})"
                return s

            print(f"{name:<16}{cell('accuracy'):>{w}}{cell('risk_accuracy'):>{w}}"
                  f"{cell('macro_f1', False):>{w}}{cell('principle_recall'):>{w}}"
                  f"{cell('false_principle_rate', False):>{w}}")

    _summarize_concerns(raw)

    print("\n" + "-" * 88)
    print("해석 주의:")
    print(" - 적중률만 오르고 위험탐지가 내려갔다면, 그냥 전체를 관대하게 만든 것일 수 있다.")
    print(" - 과잉위반(깨끗한 케이스에서 주장한 위반원칙 수)이 같이 내려가야 진짜 개선이다.")
    print(" - 정답셋 22건·시드 소수 기준이라 1~2건 차이(약 4.5%p)는 노이즈와 구분되지 않는다.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed-base", type=int, default=7000)
    ap.add_argument("--arms", default="single,debate,ensemble",
                    help="쉼표 구분. 기본은 비교에 필요한 3종만 (naive/single_norag는 생략해 시간 절약)")
    ap.add_argument("--only", default="baseline,screen_situation",
                    help=f"실행할 조건. 선택지: {','.join(CONDITIONS)}")
    args = ap.parse_args()

    names = [n.strip() for n in args.only.split(",") if n.strip()]
    unknown = [n for n in names if n not in CONDITIONS]
    if unknown:
        sys.exit(f"모르는 조건: {unknown}. 선택지: {list(CONDITIONS)}")

    OUT.mkdir(exist_ok=True)
    print(f"조건 {names} / arms={args.arms} / seeds={args.seeds}")
    print("주의: 한 조건씩 순차 실행됩니다. 동시에 다른 LLM 작업을 돌리지 마세요(GPU 경합).")

    paths = {}
    for name in names:
        paths[name] = run_condition(name, args.arms, args.seeds, args.seed_base)

    summarize(paths)


if __name__ == "__main__":
    main()

"""「금융상품 한눈에」 오픈API로 실제 예/적금 상품을 받아 상품 정의 JSON 초안을 만든다.

    FINLIFE_API_KEY=... uv run python scripts/fetch_finlife.py --kind saving --top 3

발급: https://finlife.fss.or.kr → 오픈API → 인증키 신청
용도: 신상품 정의를 쓸 때 '현재 시장 금리 수준'을 비교 기준으로 삼기 위함이다.
받은 초안은 약관 조항(clauses)·우대조건 서술이 비어 있으므로 사람이 채워야 한다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm.config import PRODUCT_DIR  # noqa: E402

BASE = "https://finlife.fss.or.kr/finlifeapi"
ENDPOINT = {"deposit": "depositProductsSearch.json", "saving": "savingProductsSearch.json"}
# 금융회사 그룹: 020000=은행, 030300=상호저축은행
TOP_FIN_GRP = "020000"


def fetch(kind: str, key: str, page: int = 1) -> dict:
    url = f"{BASE}/{ENDPOINT[kind]}"
    params = {"auth": key, "topFinGrpNo": TOP_FIN_GRP, "pageNo": str(page)}
    r = httpx.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def to_draft(base_row: dict, options: list[dict], kind: str) -> dict:
    """API 응답(baseList/optionList)을 Product 스키마 초안으로 변환."""
    opts = [o for o in options if o.get("fin_prdt_cd") == base_row.get("fin_prdt_cd")]
    best = max(opts, key=lambda o: float(o.get("intr_rate2") or 0), default={})
    return {
        "product_id": f"FL-{base_row.get('fin_prdt_cd', 'NA')}",
        "name": base_row.get("fin_prdt_nm", ""),
        "category": kind,
        "issuer": base_row.get("kor_co_nm", ""),
        "summary": (base_row.get("etc_note") or "")[:300],
        "intr_rate": float(best.get("intr_rate") or 0) or None,
        "intr_rate2": float(best.get("intr_rate2") or 0) or None,
        "intr_rate_type": best.get("intr_rate_type_nm", "단리"),
        "save_trm_months": int(best.get("save_trm") or 0) or None,
        "max_monthly_manwon": None,
        "preferentials": [
            {
                "name": "공시 우대조건(요약)",
                "rate_bonus_pct": round(
                    float(best.get("intr_rate2") or 0) - float(best.get("intr_rate") or 0), 3
                ),
                "requirement": (base_row.get("spcl_cnd") or "")[:500],
            }
        ],
        "risk_notes": [
            "공시 API 원문 기반 초안이다. 우대조건 달성률·약관 조항은 사람이 채워야 한다."
        ],
        "target_description": base_row.get("join_member", ""),
        "target_segments": [],
        "clauses": [],
        "_source": "finlife.fss.or.kr 금융상품 한눈에 오픈API",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=list(ENDPOINT), default="saving")
    ap.add_argument("--top", type=int, default=3, help="최고금리 상위 N건만 저장")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    key = os.environ.get("FINLIFE_API_KEY", "").strip()
    if not key:
        print("FINLIFE_API_KEY가 없다. .env에 인증키를 넣어라.", file=sys.stderr)
        return 1

    try:
        payload = fetch(args.kind, key)
    except httpx.HTTPError as e:
        print(f"API 호출 실패: {e}", file=sys.stderr)
        return 1

    result = payload.get("result", {})
    if result.get("err_cd") not in (None, "000"):
        print(f"API 오류: {result.get('err_cd')} {result.get('err_msg')}", file=sys.stderr)
        return 1

    base_list = result.get("baseList", [])
    opt_list = result.get("optionList", [])
    drafts = [to_draft(b, opt_list, args.kind) for b in base_list]
    drafts.sort(key=lambda d: d.get("intr_rate2") or 0, reverse=True)
    drafts = drafts[: args.top]

    outdir = Path(args.outdir) if args.outdir else PRODUCT_DIR / "_finlife_drafts"
    outdir.mkdir(parents=True, exist_ok=True)
    for d in drafts:
        path = outdir / f"{d['product_id']}.json"
        path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{d['intr_rate2']}% {d['issuer']} {d['name']} → {path}")

    print(
        f"\n{len(drafts)}건 저장. data/products/ 로 옮기기 전에 clauses·preferentials·"
        "target_segments를 채워라 (디베이트가 인용할 근거가 된다)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

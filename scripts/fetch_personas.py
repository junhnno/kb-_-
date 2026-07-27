"""Nemotron-Personas-Korea를 내려받아 data/personas/nemotron.jsonl로 캐시한다.

    uv run --extra personas python scripts/fetch_personas.py --limit 5000

저장 후에는 네트워크 없이도 loader가 이 파일을 우선 사용한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm.config import PERSONA_DIR  # noqa: E402
from fdm.personas.finance import attach_finance  # noqa: E402
from fdm.personas.loader import load_from_hf  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--with-finance",
        action="store_true",
        help="재무 프로파일까지 부여해 저장 (기본은 로드 시점에 부여)",
    )
    args = ap.parse_args()

    try:
        personas = load_from_hf(limit=args.limit)
    except ImportError:
        print("datasets가 없다. `uv sync --extra personas` 후 다시 실행하라.", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"다운로드 실패: {e}", file=sys.stderr)
        return 1

    if args.with_finance:
        personas = [attach_finance(p) for p in personas]

    PERSONA_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else PERSONA_DIR / "nemotron.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for p in personas:
            f.write(json.dumps(p.model_dump(exclude_none=True), ensure_ascii=False) + "\n")

    ages = [p.age for p in personas]
    print(f"{len(personas)}명 저장: {out}")
    print(f"연령 {min(ages)}~{max(ages)}세, 지역 {len({p.region for p in personas})}종")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

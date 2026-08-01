"""사실팩 on/off × (단발+RAG, 디베이트) 2×2 애블레이션.

상품 데이터(preferentials)를 보강했으므로 baseline도 함께 재측정해야 공정한 비교가 된다.
결과는 outputs/ablation_nofacts.json / ablation_facts.json 으로 나눠 저장한다.

사용: uv run python scripts/run_factpack_ablation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fdm.agents.debate import DebateConfig  # noqa: E402
from fdm.eval.benchmark import run_ablation  # noqa: E402

ARMS = ("single", "debate")
CONDITIONS = {
    "nofacts": DebateConfig(use_fact_pack=False, screen_contradictions=False),
    "facts": DebateConfig(use_fact_pack=True, screen_contradictions=True),
}


def main() -> int:
    for name, cfg in CONDITIONS.items():
        print(f"\n{'=' * 60}\n조건: {name} (사실팩 {'ON' if cfg.use_fact_pack else 'OFF'})\n{'=' * 60}", flush=True)
        rep = run_ablation(arms=ARMS, n_seeds=1, config=cfg, progress=True)
        out = ROOT / "outputs" / f"ablation_{name}.json"
        rep.save(out)
        print(f"\n저장: {out}")
        for a in rep.arms:
            print(
                f"  {a.arm:8} 적중률 {a.accuracy:.1%}  macroF1 {a.macro_f1:.3f}  "
                f"근거없음 {a.evidence_empty_rate:.0%}  실패 {len(a.failed_cases)}건"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

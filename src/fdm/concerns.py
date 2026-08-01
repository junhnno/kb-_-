"""우려 분류 체계와 심각도 규칙 (facts.py와 agents/schema.py가 공유).

우려를 자유 텍스트로 받고 키워드로 유형을 추측하면 표현이 조금만 달라져도 검출에
실패한다(실측: 동일 조건 재실행에서 recall이 71.8%↔59.7%로 12%p 흔들림).
그래서 **생성 단계에서 유형을 지정**하게 하고, 채점은 유형 일치로 한다.
"""

from __future__ import annotations

import json
from functools import lru_cache

from .config import BENCHMARK_DIR

TAXONOMY_PATH = BENCHMARK_DIR / "concern_taxonomy.json"

# 심각도 서열. 0~100 연속값은 8B 모델에서 70~80에 뭉치므로 4단계 서열을 쓴다.
SEVERITY_ORDER = ["경미", "주의", "중대", "치명"]

# 분류 체계에 없는 우려를 억지로 끼워맞추지 않도록 탈출구를 둔다.
OTHER = "other"

# 계산 앵커(수치·조항)를 대지 못한 우려의 심각도 상한.
# 근거 없이 '치명'을 주장하는 것을 막는다. 일반론(모든 적금에 있는 중도해지 조항 등)이
# 상위 목록을 차지하는 것을 방지하는 장치이기도 하다.
UNANCHORED_SEVERITY_CAP = "주의"


@lru_cache(maxsize=1)
def load_taxonomy() -> dict:
    return json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def type_ids() -> tuple[str, ...]:
    return tuple(t["id"] for t in load_taxonomy()["types"]) + (OTHER,)


@lru_cache(maxsize=1)
def _by_id() -> dict[str, dict]:
    return {t["id"]: t for t in load_taxonomy()["types"]}


def type_label(type_id: str) -> str:
    t = _by_id().get(type_id)
    return t["label"] if t else "분류 외"


def verify_with(type_id: str) -> str:
    """이 우려를 실제 데이터로 확인하는 방법. 산출물의 '어디를 봐야 하나'에 해당."""
    t = _by_id().get(type_id)
    return t.get("verify_with", "") if t else ""


def severity_rank(severity: str) -> int:
    return SEVERITY_ORDER.index(severity) if severity in SEVERITY_ORDER else 1


def cap_severity(severity: str, cap: str) -> str:
    """severity를 cap 이하로 낮춘다."""
    return severity if severity_rank(severity) <= severity_rank(cap) else cap


def prompt_type_list() -> str:
    """심판 프롬프트에 넣을 유형 목록."""
    lines = [f"  - {t['id']}: {t['label']}" for t in load_taxonomy()["types"]]
    lines.append(f"  - {OTHER}: 위 분류에 없는 우려")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 우려 계층 (교차확인 × 심각도)
# ---------------------------------------------------------------------------
# 실측으로 확인된 두 신호를 곱해 분석가가 볼 순서를 만든다.
#   · 교차확인(단발·디베이트가 모두 제기): 52.8%(19/36) vs 단독 37.5%(9/24)
#   · 심각도: 치명 70.0%(7/10), 중대 42.0%(21/50)
# 계층 서열은 단조롭게 성립한다: T1 62.5% > T2 53.3% > T3 31.8%.
# 지우지 않고 정렬한다 — 놓치면 책임이고 많으면 무시되므로, 기각 대신 계층화한다.
#
# 출처: outputs/sev_A.json (ensemble arm, qwen3:8b, 12건, 시드 1회).
# **표본이 작다.** 이전 값(치명 3/3=100%, 교차확인 54.1% vs 단독 20~30%)은 더 작은
# 표본에서 나와 과장돼 있었다. 정답셋을 30건으로 늘리면 다시 움직인다.
# 수치를 고칠 때는 이 주석과 TIER_BASIS를 함께 고칠 것.

TIER_ORDER = ["T1", "T2", "T3", "T4"]

TIER_LABEL = {
    "T1": "즉시 조치",
    "T2": "우선 검토",
    "T3": "참고",
    "T4": "접어두기",
}

TIER_MARK = {"T1": "🔴", "T2": "🟠", "T3": "🟡", "T4": "⚪"}

# 표시용 근거 문구. 위 주석의 출처와 항상 같이 움직여야 한다.
TIER_BASIS = {
    "T1": "치명 + 교차확인 — 두 신호가 모두 강하다. 관측 정확도 62.5%(5/8)",
    "T2": "치명 단독 또는 중대 + 교차확인 — 한 신호가 강하다. 53.3%(16/30)",
    "T3": "중대 단독 — 신호 하나뿐. 31.8%(7/22)",
    "T4": "주의 이하 — 앵커 없는 일반론이 대부분. 근거가 붙기 전에는 행동 근거로 쓰지 않는다",
}

# 산출물에 함께 표시할 고지. 수치를 근거로 쓰기 전에 표본 크기를 알려야 한다.
TIER_CAVEAT = (
    "계층별 정확도는 12건 정답셋·시드 1회에서 측정한 잠정치다(출처: outputs/sev_A.json). "
    "표본이 작아 서열은 신뢰하되 개별 수치는 확정 근거로 쓸 수 없다. "
    "정답셋 12건 자체가 조정례를 참조해 재구성한 가공 샘플이다."
)

# 교차확인은 두 방식을 모두 돌린 ensemble에서만 성립한다.
CROSS_CHECK_MIN_SOURCES = 2


def is_cross_checked(sources: list[str] | tuple[str, ...]) -> bool:
    return len(set(sources)) >= CROSS_CHECK_MIN_SOURCES


def concern_tier(severity: str, sources: list[str] | tuple[str, ...]) -> str:
    """(심각도, 교차확인 여부) → 계층 id.

    single/debate 단독 실행에서는 sources가 항상 1개라 교차확인이 성립하지 않는다.
    그 경우 계층은 심각도만 반영한 보수적(낮은) 값이 되며, 이는 실제로
    '교차확인 정보가 없다'는 뜻이므로 과대평가보다 안전하다.
    """
    rank = severity_rank(severity)
    cross = is_cross_checked(sources)
    if rank >= severity_rank("치명"):
        return "T1" if cross else "T2"
    if rank >= severity_rank("중대"):
        return "T2" if cross else "T3"
    return "T4"


def tier_sort_key(severity: str, sources: list[str] | tuple[str, ...]) -> tuple:
    """계층 우선 → 심각도 → 교차확인 순으로 정렬하기 위한 키 (오름차순 사용)."""
    tier = concern_tier(severity, sources)
    return (TIER_ORDER.index(tier), -severity_rank(severity), 0 if is_cross_checked(sources) else 1)

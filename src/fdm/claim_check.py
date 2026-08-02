"""근거 검증 레이어 — 심판이 보기 **전에** 회의론자 주장을 코드로 검증한다.

## 왜 필요한가 (2026-08-01 qwen3:14b 실측)

같은 모델·같은 케이스·같은 RAG인데 판정 방식만 다르게 했더니:

| | 단발 | 디베이트 |
|---|---|---|
| 적중률 | 81.8% (18/22) | 45.5% (10/22) |
| **gold=pass 정확도** | **11/12** | **2/12** |

모델을 8B에서 14B로 키웠을 때 단발은 8/12 → 11/12로 좋아졌으나 디베이트는
0/12 → 2/12에 머물렀다. **모델 능력의 문제가 아니라 구조의 문제다.**
같은 모델이 혼자 보면 12건 중 11건을 맞히는데, 회의론자 발언이 앞에 붙으면 2건이 된다.

## 메커니즘

심판의 입력에서 단발과 다른 것은 **디베이트 전문뿐**이다. 회의론자는 설계상
"반드시 우려를 찾아내는" 역할이라 깨끗한 상품에도 3~5개를 만들어낸다. 심판은
"우려가 제기되었다"는 사실 자체에 끌려 pass를 주지 못한다.

프롬프트로 "경중을 따져라", "계산값과 모순되면 채택하지 말라"고 지시하는 방식은
이미 두 번 시도했고 두 번 다 실패했다. 일반 규칙을 주면 모델이 그 추론을 해내야 하는데,
그 추론이 바로 실패 지점이기 때문이다.

## 이 모듈의 접근

**추론을 시키지 않고 결과를 준다.** 회의론자의 주장을 코드가 미리 검증해서
"이 주장은 이래서 기각됨"을 항목별로 적어 심판에게 넘긴다.

설계 원칙: **탐지는 넓게, 판정은 검증된 근거로만.**
회의론자 프롬프트는 손대지 않는다 — 우려 recall 96.7%는 이 시스템의 유일한 강점이라
그걸 깎으면 디베이트를 쓸 이유 자체가 없어진다. 역할을 나눈다.
회의론자는 민감도 높은 탐지기, 심판은 검증된 근거만 채택하는 필터.

검증 근거는 두 가지이며 둘 다 **주어진 입력**이라 판단이 개입하지 않는다.
  (a) 사실팩 계산값 — DSR 4.1%인데 "DSR 부담" 같은 수치 모순
  (b) 판매 정황 — "중도해지 불이익을 설명했다"고 명시됐는데 "설명 부족" 주장

기각 목록을 심판에게 보여주는 것만으로 끝내지 않고, **기각이 많다는 사실 자체가
위험 근거가 약하다는 신호**임을 명시한다. "우려가 제기되었다"는 압력을
"제기된 우려가 기각되었다"로 뒤집는 것이 이 블록의 목적이다.
"""

from __future__ import annotations

import re

from .facts import FactPack, screen_concerns
from .situation import TOPIC_LABEL, claims_insufficient_disclosure, disclosed_topics

# 프롬프트에 넣을 때 주장 한 건의 최대 길이. 컨텍스트(기본 8192)를 아끼되
# 심판이 어떤 주장인지 알아볼 수 있는 선.
MAX_CLAIM_CHARS = 90
# 블록에 싣는 최대 항목 수. 너무 길면 심판이 본문을 못 읽는다.
MAX_ITEMS = 8

# 불릿·번호 머리표. 회의론자 출력 형식이 "3~5개 불릿"이고 2부에는 "[유형id]"가 붙는다.
_BULLET = re.compile(r"^\s*(?:[-*•·]|\d+[.)]|\(\d+\))\s*")


def split_claims(text: str) -> list[str]:
    """회의론자 발화를 주장 단위로 자른다.

    줄 단위로만 자른다. 문장 단위로 더 쪼개면 "옹호자 주장 → 반박 → 근거" 형식의
    한 불릿이 셋으로 갈려, 근거 부분만 떼어놓고 모순 판정을 하게 된다.
    """
    out: list[str] = []
    for raw in (text or "").splitlines():
        line = _BULLET.sub("", raw).strip()
        if len(line) < 8:  # 머리말("1부. 반박") 같은 토막 제거
            continue
        out.append(line)
    return out


def check_claims(
    claims: list[str], facts: FactPack | None, situation: str = ""
) -> list[tuple[str, str]]:
    """주어진 사실과 모순되는 주장만 골라 (주장, 기각사유)로 돌려준다.

    판정 근거가 없으면 기각하지 않는다. 정당한 우려를 지우는 비용이
    오탐 하나를 남기는 비용보다 크다.
    """
    rejected: list[tuple[str, str]] = []
    disclosed = disclosed_topics(situation)

    for claim in claims:
        # (a) 판매 정황과의 모순 — 이미 설명한 항목을 '설명 부족'이라 주장
        hit = claims_insufficient_disclosure(claim) & disclosed
        if hit:
            topic = sorted(hit)[0]
            label = TOPIC_LABEL.get(topic, topic)
            rejected.append((claim, f"판매 정황에 '{label}' 설명 사실이 명시됨"))
            continue
        # (b) 사실팩 계산값과의 모순 (기존 텍스트 기반 스크리너 재사용)
        if facts is not None:
            _, dropped = screen_concerns([claim], facts)
            if dropped:
                rejected.append((claim, _fact_reason(dropped[0], facts)))
    return rejected


def _fact_reason(dropped_line: str, facts: FactPack) -> str:
    """기각 사유를 심판이 바로 이해할 수 있는 수치 문장으로 바꾼다.

    'affordability 계산값과 모순' 같은 내부 id를 그대로 보여주면 심판이
    무엇과 모순인지 알 수 없어 지시를 따르기 어렵다.
    """
    raw = dropped_line.split("[기각:", 1)[-1].rstrip("]").strip()
    type_id = raw.split()[0] if raw else ""
    detail = {
        "affordability": (
            f"월 여유자금 {facts.monthly_surplus:,}만원으로 감당 가능한 수준"
        ),
        "dsr_overload": f"DSR {facts.dsr_pct:.1f}% — 규제 한도(40%)에 미달",
        "preferential_unattainable": "이 상품에는 우대조건이 없음",
        "principal_loss_risk": "원금보장 상품 (예금자보호 한도 내)",
        "rate_variability": "고정금리 상품 — 금리 상승 위험 없음",
        "deposit_protection_limit": (
            f"예치 추정 {facts.deposit_exposure:,}만원 — 보호 한도 내"
            if facts.deposit_exposure is not None
            else "예금자보호 한도 내"
        ),
        "revolving_debt_spiral": f"카드 상품이 아님 (상품군: {facts.category})",
        "maturity_refinance_risk": "만기 일시상환 구조가 아님",
        "fee_hidden": "상품 정의에 수수료 없음",
        "senior_vulnerability": f"연령 {facts.age}세 — 고령자 보호 대상 아님",
    }.get(type_id)
    return f"시스템 계산값과 모순 — {detail}" if detail else f"시스템 계산값과 모순 ({raw})"


def verification_block(rejected: list[tuple[str, str]], n_total: int = 0) -> str:
    """심판 프롬프트에 붙일 검증 결과. 기각이 없으면 빈 문자열.

    기각 목록만 보여주면 심판은 "그래도 나머지가 있으니 warn"으로 흐른다.
    그래서 **살아남은 주장이 몇 건인지**를 함께 알린다. 전부 기각됐다면
    그 사실을 명시적으로 말해줘야 pass를 줄 수 있다.
    """
    if not rejected:
        return ""
    survived = max(0, n_total - len(rejected))
    lines = [
        "[코드 검증 결과 — 아래 주장은 주어진 사실과 직접 모순되어 기각되었다]",
        "이 검증은 LLM 판단이 아니라 상품 정의·페르소나 재무·판매 정황에서 코드가 계산한 것이다.",
        "",
    ]
    for i, (claim, reason) in enumerate(rejected[:MAX_ITEMS], 1):
        text = claim if len(claim) <= MAX_CLAIM_CHARS else claim[:MAX_CLAIM_CHARS] + "…"
        lines.append(f"{i}. \"{text}\"\n   → 기각: {reason}")
    lines += ["", f"[검증 요약] 회의론자 주장 {n_total}건 중 {len(rejected)}건 기각, "
              f"{survived}건 통과."]
    if survived == 0:
        lines += [
            "**검증을 통과한 위험 근거가 하나도 없다.** 회의론자가 제기한 모든 주장이",
            "상품 정의·재무 계산값·판매 정황과 모순되었다. 이 경우 적합성은 pass다.",
        ]
    else:
        lines += [
            f"[지시] 기각 항목은 판정 근거로 삼지 말고, 통과한 {survived}건만으로 판정하라.",
            "기각이 많다는 것은 제기된 우려에 비해 **실제 위험 근거가 약하다**는 뜻이다.",
            "우려가 제기되었다는 사실 자체는 warn/fail의 근거가 되지 않는다.",
        ]
    return "\n".join(lines)


def build_verification(
    skeptic_text: str, facts: FactPack | None, situation: str = ""
) -> tuple[str, list[str]]:
    """회의론자 발화 → (심판에게 넣을 검증 블록, 기각 기록).

    기록은 `DebateResult.dropped_concerns`에 남겨 사후 분석에 쓴다.
    """
    claims = split_claims(skeptic_text)
    rejected = check_claims(claims, facts, situation)
    log = [f"{c} [디베이트 중 기각: {r}]" for c, r in rejected]
    return verification_block(rejected, n_total=len(claims)), log

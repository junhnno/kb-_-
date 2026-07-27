"""Nemotron-Personas-Korea 로더 + 세그먼트 필터.

우선순위:
  1. data/personas/*.jsonl  (로컬 캐시 / 오프라인)
  2. Hugging Face `nvidia/Nemotron-Personas-Korea` (datasets 설치 + 네트워크 필요)
  3. 합성 폴백 (KOSIS 연령·지역 분포로 생성) — 네트워크 없이 파이프라인 확인용
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from pathlib import Path

from ..config import PERSONA_DIR
from .finance import attach_finance, load_kosis_params
from .schema import Persona, Segment

HF_DATASET = "nvidia/Nemotron-Personas-Korea"

# Nemotron 컬럼명 → Persona 필드 (데이터셋 버전차를 흡수하기 위해 후보 리스트로 둔다)
FIELD_CANDIDATES = {
    "age": ["age"],
    "sex": ["sex", "gender"],
    "region": ["region", "province", "administrative_division", "city"],
    "occupation": ["occupation", "professional_role", "job"],
    "education": ["education_level", "education"],
    "marital_status": ["marital_status"],
    "household_size": ["household_size"],
}
TEXT_CANDIDATES = ["persona", "professional_persona", "personality", "description"]
TRAIT_CANDIDATES = ["skills_and_expertise", "hobbies_and_interests", "career_goals_and_ambitions"]

# Nemotron-Personas-Korea는 성별을 '남자'/'여자'로 표기한다. 영문 표기도 함께 받는다.
MALE_TOKENS = {"male", "m", "남", "남성", "남자"}
FEMALE_TOKENS = {"female", "f", "여", "여성", "여자"}

# 이 데이터셋에는 가구원 수 컬럼이 없고 family_type(가구 형태)만 있다.
# 형태별 대표 가구원 수로 환산한다. 부분 문자열 매칭이므로 순서가 우선순위다.
FAMILY_TYPE_SIZE = [
    ("배우자·자녀와 거주", 4),
    ("배우자와 자녀", 4),
    ("자녀와 거주", 2),  # 한부모
    ("배우자와 거주", 2),
    ("부모와 동거", 3),
    ("어머니와 동거", 2),
    ("아버지와 동거", 2),
    ("혼자 거주", 1),
    ("3세대", 5),
    ("기타2세대", 3),
]


def household_size_from(row: dict) -> int | None:
    """family_type → 대표 가구원 수. 매칭 실패 시 None."""
    ft = str(row.get("family_type") or "")
    if not ft:
        return None
    for kw, size in FAMILY_TYPE_SIZE:
        if kw in ft:
            return size
    return None


def _first(row: dict, keys: list[str]) -> object | None:
    for k in keys:
        v = row.get(k)
        if v not in (None, "", []):
            return v
    return None


def _to_int(v: object, default: int) -> int:
    try:
        return int(float(str(v).strip().rstrip("세")))
    except (TypeError, ValueError):
        return default


def row_to_persona(row: dict, idx: int) -> Persona | None:
    age = _to_int(_first(row, FIELD_CANDIDATES["age"]), -1)
    if age < 19 or age > 95:  # 성인 대상 상품 검증이므로 미성년 제외
        return None
    sex_raw = str(_first(row, FIELD_CANDIDATES["sex"]) or "").strip().lower()
    sex = "남성" if sex_raw in MALE_TOKENS else ("여성" if sex_raw in FEMALE_TOKENS else "미상")
    traits: list[str] = []
    for key in TRAIT_CANDIDATES:
        v = row.get(key)
        if isinstance(v, str) and v:
            traits.extend([t.strip() for t in v.split(",")[:3] if t.strip()])
        elif isinstance(v, list):
            traits.extend([str(t) for t in v[:3]])

    return Persona(
        persona_id=str(row.get("uuid") or row.get("id") or f"nemo-{idx:06d}"),
        age=age,
        sex=sex,
        region=str(_first(row, FIELD_CANDIDATES["region"]) or "미상"),
        occupation=str(_first(row, FIELD_CANDIDATES["occupation"]) or "미상"),
        education=str(_first(row, FIELD_CANDIDATES["education"]) or "미상"),
        marital_status=str(_first(row, FIELD_CANDIDATES["marital_status"]) or "미상"),
        household_size=_to_int(
            _first(row, FIELD_CANDIDATES["household_size"]),
            household_size_from(row) or 1,
        ),
        persona_text=str(_first(row, TEXT_CANDIDATES) or ""),
        traits=traits[:6],
    )


# ------------------------------------------------------------------- 소스별 로드
def load_from_jsonl(path: Path) -> list[Persona]:
    out: list[Persona] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            p = Persona(**row) if "persona_id" in row else row_to_persona(row, i)
            if p:
                out.append(p)
    return out


def load_from_hf(limit: int = 5000) -> list[Persona]:
    from datasets import load_dataset  # 선택 의존성

    ds = load_dataset(HF_DATASET, split=f"train[:{limit}]")
    out = []
    for i, row in enumerate(ds):
        p = row_to_persona(dict(row), i)
        if p:
            out.append(p)
    return out


def synthesize(n: int = 400, seed: int = 42) -> list[Persona]:
    """네트워크 없이 쓰는 폴백. 연령·지역 분포만 KOSIS 공표치에 맞춘다."""
    params = load_kosis_params()
    rng = random.Random(seed)
    bands = list(params["age_band_share_adult"].items())
    regions = list(params["region_population_share"].items())
    occupations = [
        "사무직 회사원", "생산직 근로자", "서비스직 종사자", "판매직 종사자", "자영업자",
        "프리랜서 디자이너", "소프트웨어 개발자", "간호사", "초등학교 교사", "공무원",
        "중소기업 관리자", "농업 종사자", "무직", "대학생", "전업주부", "아르바이트",
    ]
    edu = ["고졸", "전문대졸", "대졸", "대학원졸"]
    marital = ["미혼", "기혼", "이혼", "사별"]
    band_range = {"20대": (20, 29), "30대": (30, 39), "40대": (40, 49), "50대": (50, 59), "60대이상": (60, 82)}

    def pick(pairs):
        r, acc = rng.random(), 0.0
        total = sum(w for _, w in pairs)
        for k, w in pairs:
            acc += w / total
            if r <= acc:
                return k
        return pairs[-1][0]

    out = []
    for i in range(n):
        band = pick(bands)
        lo, hi = band_range[band]
        age = rng.randint(lo, hi)
        occ = rng.choice(occupations)
        if age >= 65 and rng.random() < 0.6:
            occ = "무직"
        p = Persona(
            persona_id=f"synth-{i:05d}",
            age=age,
            sex=rng.choice(["남성", "여성"]),
            region=pick(regions),
            occupation=occ,
            education=rng.choice(edu),
            marital_status=rng.choice(marital) if age >= 30 else "미혼",
            household_size=rng.choice([1, 1, 2, 2, 3, 3, 4]),
            persona_text=f"{age}세 {occ}. 금융상품 선택 시 {rng.choice(['안정성', '수익률', '편의성', '수수료'])}을 가장 중시한다.",
            traits=[rng.choice(["보수적 투자성향", "적극적 투자성향", "디지털 친화", "대면 채널 선호", "가격 민감"])],
            source="synthetic-fallback (Nemotron 미설치)",
        )
        out.append(p)
    return out


def load_personas(
    *,
    source: str = "auto",
    limit: int = 2000,
    with_finance: bool = True,
) -> list[Persona]:
    personas: list[Persona] = []
    files = sorted(PERSONA_DIR.glob("*.jsonl")) if PERSONA_DIR.exists() else []

    if source in {"auto", "jsonl"} and files:
        for f in files:
            personas.extend(load_from_jsonl(f))
    if not personas and source in {"auto", "hf"}:
        try:
            personas = load_from_hf(limit=limit)
        except Exception:
            personas = []
    if not personas:
        personas = synthesize(n=min(limit, 400))

    personas = personas[:limit]
    if with_finance:
        personas = [attach_finance(p) for p in personas]
    return personas


def filter_segment(personas: Iterable[Persona], segment: Segment) -> list[Persona]:
    return [p for p in personas if segment.matches(p)]


def sample_cohort(
    personas: Iterable[Persona], segment: Segment, k: int, seed: int = 0
) -> list[Persona]:
    """세그먼트 내에서 k명 코호트를 재현 가능하게 추출."""
    pool = filter_segment(personas, segment)
    if len(pool) <= k:
        return pool
    rng = random.Random(f"{segment.name}:{seed}")
    return rng.sample(pool, k)

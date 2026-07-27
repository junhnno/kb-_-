# 03. `src/fdm/personas/` — 가상 고객 만들기 (452줄)

세 파일로 나뉜다.

| 파일 | 줄수 | 역할 |
|---|---|---|
| `schema.py` | 131 | 데이터 모양 정의 (`Persona`, `FinanceProfile`, `Segment`) |
| `finance.py` | 126 | 재무 수치 **합성** (KOSIS 분포 → 개인 수치) |
| `loader.py` | 195 | 데이터 **로딩** (Nemotron / 로컬 / 폴백) + 세그먼트 필터 |

이 디렉터리가 푸는 문제: **Nemotron-Personas-Korea에는 "27세 서비스직 서울 거주"까지는
있지만 "연소득 3,200만원, 부채 1,200만원, DSR 4.1%"가 없다.**
금융상품 적합성을 판정하려면 재무 수치가 반드시 필요하다. 그래서 통계청 분포에서 만들어낸다.

---

# A. `schema.py` — 데이터 모양

## 블록 1 — 연령대 구분 (9~21줄)

```python
AGE_BANDS = ["20대", "30대", "40대", "50대", "60대이상"]


def age_band(age: int) -> str:
    if age < 30:
        return "20대"
    if age < 40:
        return "30대"
    ...
    return "60대이상"
```

KOSIS 통계가 연령대별로 공표되므로, 개인의 나이를 연령대 키로 바꾸는 함수가 필요하다.
`finance.py`가 이 키로 분포 파라미터를 찾는다(`params["by_age_band"]["30대"]`).

**`if age < 30`을 연달아 쓴 이유**: 구간이 순서대로 배타적이므로 `elif`나 `and` 조건이 필요 없다.
위에서 걸러지면 아래로 안 내려간다. 조건을 짧게 유지하는 흔한 방식이다.

---

## 블록 2 — `FinanceProfile` (24~44줄)

```python
class FinanceProfile(BaseModel):
    """통계청 가계금융복지조사 분포를 참고해 부여한 재무 프로파일 (합성값)."""

    annual_income_manwon: int = Field(description="연 경상소득 (만원)")
    monthly_income_manwon: int
    financial_assets_manwon: int = Field(description="금융자산 (만원)")
    real_assets_manwon: int = Field(description="실물자산 (만원)")
    debt_manwon: int = Field(description="부채 총액 (만원)")
    monthly_debt_service_manwon: int = Field(description="월 원리금상환액 (만원)")
    monthly_surplus_manwon: int = Field(description="월 여유자금 (만원)")
    dsr_pct: float = Field(description="소득 대비 원리금상환비율 (%)")
    income_quintile: int = Field(ge=1, le=5, description="소득 5분위")
    source: str = "KOSIS 가계금융복지조사 분포 참고(합성 부여값)"
```

**필드명에 단위를 박은 이유** (`_manwon`, `_pct`)
금융 코드에서 단위 혼동은 치명적이다. `income = 3200`이 3,200원인지 3,200만원인지
3,200달러인지 이름만 봐서 알 수 있어야 한다. 이 프로젝트는 **전부 만원 단위**로 통일했다.

**`Field(ge=1, le=5)`**
pydantic이 검증한다. 6분위를 넣으면 객체 생성 자체가 실패한다.
LLM이나 외부 JSON에서 값을 받을 때 이 한 줄이 방어선이 된다.

**`source` 필드가 왜 데이터에 들어있나**
숫자만 있으면 나중에 "이게 실제 조사값인가, 합성값인가"를 알 수 없다.
**출처를 데이터 안에 넣어 다니는 것**이 이 프로젝트 전체의 규칙이다
(`Persona.source`, `dispute_cases.json`의 `_meta`, mock의 `[MOCK]` 표시도 같은 원칙).

```python
    def summary(self) -> str:
        return (
            f"연소득 {self.annual_income_manwon:,}만원(소득 {self.income_quintile}분위), "
            f"월소득 {self.monthly_income_manwon:,}만원, … "
            f"(DSR {self.dsr_pct:.1f}%), 월 여유자금 {self.monthly_surplus_manwon:,}만원"
        )
```

**`{value:,}`** — 천 단위 콤마 (`3850` → `3,850`). **`{value:.1f}`** — 소수 1자리.
LLM에게 숫자를 줄 때 사람이 읽는 형식으로 주는 게 낫다. `3850`보다 `3,850만원`이
모델에게도 단위 오해를 줄인다.

---

## 블록 3 — `Persona` (47~78줄)

```python
class Persona(BaseModel):
    persona_id: str
    age: int
    sex: Literal["남성", "여성", "미상"] = "미상"
    region: str = "미상"
    occupation: str = "미상"
    ...
    finance: FinanceProfile | None = None
    source: str = "nvidia/Nemotron-Personas-Korea"
```

**`Literal["남성", "여성", "미상"]`**
이 세 값만 허용한다. Nemotron은 `male` / `M` / `남` 등으로 올 수 있어서
`loader.py`가 변환한 뒤 여기에 넣는다. **경계에서 정규화하고, 내부에서는 한 가지 형태만
쓴다**는 원칙이다.

**`finance: FinanceProfile | None = None`**
페르소나는 재무 프로파일 **없이도 존재할 수 있다.** 로딩 직후에는 `None`이고
`attach_finance()`가 나중에 채운다. 2단계로 나눈 이유는 로딩(I/O)과 합성(계산)의
책임을 분리하기 위해서다 — 테스트도 따로 할 수 있다.

```python
    @property
    def band(self) -> str:
        return age_band(self.age)
```

`p.band`로 연령대를 얻는다. 나이는 저장하고 연령대는 계산한다 — **파생 값을 중복
저장하지 않는다**(둘이 어긋날 여지를 없앤다).

### `prompt_block()` — LLM이 보는 것의 전부 (65~78줄)

```python
    def prompt_block(self) -> str:
        lines = [
            f"- ID: {self.persona_id}",
            f"- 인구: {self.age}세 {self.sex}, {self.region}, 가구원 {self.household_size}명, {self.marital_status}",
            f"- 직업/학력: {self.occupation} / {self.education}",
        ]
        if self.traits:
            lines.append(f"- 성향: {', '.join(self.traits)}")
        if self.persona_text:
            lines.append(f"- 서술: {self.persona_text.strip()[:600]}")
        if self.finance:
            lines.append(f"- 재무: {self.finance.summary()}")
        return "\n".join(lines)
```

**이 함수의 출력이 곧 LLM에게 전달되는 페르소나다.** 프롬프트를 고치고 싶으면
여기를 고친다. 결과물은 이런 모양이다:

```
- ID: synth-00149
- 인구: 23세 여성, 서울특별시, 가구원 1명, 미혼
- 직업/학력: 자영업자 / 대졸
- 성향: 보수적 투자성향
- 서술: 23세 자영업자. 금융상품 선택 시 편의성을 가장 중시한다.
- 재무: 연소득 3,850만원(소득 2분위), 월소득 321만원, … (DSR 0.0%), 월 여유자금 103만원
```

**`if` 로 감싼 이유**: 값이 없으면 그 줄을 아예 넣지 않는다.
`- 성향: ` 같은 빈 줄은 모델에게 노이즈이고 토큰만 먹는다.

**`[:600]` 자르기**: Nemotron의 페르소나 서술은 길 수 있다. 5턴 × 여러 페르소나면
토큰이 폭증하므로 상한을 둔다. **비용 통제는 이런 작은 지점에서 결정된다.**

---

## 블록 4 — `Segment.matches()` (81~131줄)

세그먼트는 "타깃 조건"이다. `data/segments.json`이 이 모델로 로드된다.

```python
class Segment(BaseModel):
    name: str
    age_min: int | None = None
    age_max: int | None = None
    regions: list[str] | None = None
    occupations_include: list[str] | None = None
    income_min_manwon: int | None = None
    income_max_manwon: int | None = None
    income_quintiles: list[int] | None = None
    sex: str | None = None
    dsr_min_pct: float | None = None
    monthly_surplus_max_manwon: int | None = None
```

**모든 필드가 `| None = None`인 설계** — "이 조건은 안 봄"을 표현한다.
`청년_사회초년생`은 `age_min: 19, age_max: 29, income_max_manwon: 4200`만 쓰고
나머지는 비운다.

```python
    def matches(self, p: Persona) -> bool:
        if self.age_min is not None and p.age < self.age_min:
            return False
        if self.age_max is not None and p.age > self.age_max:
            return False
```

**조기 반환(early return) 패턴** — 조건 하나라도 어긋나면 즉시 `False`.
`if A and B and C and …` 로 한 줄에 쓰는 것보다 읽기 쉽고, 어느 조건에서 걸렸는지
디버깅할 때 브레이크포인트를 걸기도 쉽다.

**`self.age_min is not None` 이라고 쓴 이유** — `if self.age_min:` 으로 쓰면
`age_min = 0`일 때 조건을 건너뛴다. 0을 유효한 값으로 쓸 수 있는 숫자 필드에서는
반드시 `is not None`으로 검사해야 한다.

```python
        if self.sex and p.sex != self.sex:
```

반면 `sex`는 문자열이라 빈 문자열이 무의미하므로 `if self.sex:`로 충분하다.
**필드 타입에 따라 검사 방식을 달리 하는 것**이 포인트다.

```python
        if self.regions and not any(r in p.region for r in self.regions):
            return False
        if self.occupations_include and not any(
            o in p.occupation for o in self.occupations_include
        ):
            return False
```

**`in`으로 부분 문자열 매칭**을 한다. `occupations_include: ["사무"]`가
`"사무직 회사원"`에 걸리게 하려는 것이다. 정확히 일치를 요구하면 Nemotron의
자유로운 직업 표기와 맞출 수 없다.

`any(...)`는 **OR 조건**이다. `["사무", "개발", "관리자"]` 중 하나라도 포함되면 통과.

```python
        f = p.finance
        if f is None:
            return not (
                self.income_min_manwon
                or self.income_max_manwon
                or self.income_quintiles
                or self.dsr_min_pct
                or self.monthly_surplus_max_manwon
            )
```

**재무 프로파일이 없는 페르소나를 어떻게 처리할지**에 대한 결정이다.

- 세그먼트가 재무 조건을 **하나도 요구하지 않으면** → 통과 (인구 조건만으로 판정 끝)
- 재무 조건을 **하나라도 요구하면** → 판정 불가이므로 탈락

"판정할 수 없으면 포함하지 않는다"는 **보수적 선택**이다. 반대로 통과시키면
소득 조건이 있는 세그먼트에 소득 미상인 사람이 섞여 결과가 오염된다.

```python
        if self.dsr_min_pct is not None and f.dsr_pct < self.dsr_min_pct:
            return False
```

> 이 `dsr_min_pct` 필드는 나중에 추가한 것이다. 처음엔 `고DSR_차주` 세그먼트를
> 소득분위로만 정의했는데, `fdm segments`로 확인하니 **평균 DSR이 14%** 였다.
> 이름과 실제 내용이 다른 세그먼트였던 것이다. DSR 조건을 넣은 뒤 59%가 되었다.
> **세그먼트를 정의한 뒤에는 반드시 실제 분포를 확인해야 한다**는 교훈이다.

---

# B. `finance.py` — 재무 수치 합성 (수학)

## 블록 1 — 문서화된 한계 (1~14줄)

```python
"""페르소나에 재무 프로파일을 부여한다.

Nemotron-Personas-Korea에는 정밀한 소득/자산/부채 수치가 없으므로,
통계청 가계금융복지조사 기반 연령대별 분포 파라미터에서 로그정규 근사로 합성값을 뽑는다.
persona_id 해시를 시드로 써서 재현 가능하다.

주의: 개별 수치는 실제 개인이 아니며, 변수 간 결합분포(joint distribution) 정합성은
검증되지 않았다. 결과는 '탐색·경보용'으로만 사용한다 (CLAUDE.md §6).
"""
```

**한계를 코드 안에 적어두는 것**도 하나의 기법이다. 이 파일을 열어보는 사람은
반드시 이 경고를 먼저 읽는다.

---

## 블록 2 — 해시 기반 시드 (35~38줄)

```python
def _seeded_uniform(*parts: object) -> float:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:12], 16) / float(0x1000000000000)
```

이 프로젝트에서 **가장 중요한 4줄**일 수 있다.

```python
_seeded_uniform("synth-00149", "income")   # → 항상 같은 값 (예: 0.6231…)
_seeded_uniform("synth-00149", "debt")     # → 다른 값 (0.1104…)
```

- `h[:12]`: 해시 앞 12자리 16진수 = 48비트
- `0x1000000000000` = 2⁴⁸ 로 나눠 0~1 실수로 만든다
- `"|".join(...)`: 구분자를 넣어 `("ab","c")`와 `("a","bc")`가 같은 해시가 되지 않게 한다

**왜 `random.random()`을 쓰지 않나**

| | `random` | 해시 시드 |
|---|---|---|
| 같은 페르소나를 두 번 로드 | 매번 다른 소득 | **항상 같은 소득** |
| 페르소나 200명만 로드 | 순서에 따라 값이 밀림 | 각자 독립적으로 결정 |
| 병렬 실행 | 전역 상태 경쟁 | 안전 |

시뮬레이션 결과를 비교하려면 **같은 페르소나가 항상 같은 재무 상태**여야 한다.
그렇지 않으면 "금리를 0.5%p 낮췄더니 반응이 달라졌다"가 금리 때문인지
소득이 바뀌어서인지 알 수 없다. 민감도 분석의 전제 조건이다.

테스트가 이걸 지킨다:

```python
def test_finance_is_deterministic():
    a = load_personas(limit=30)[0]
    b = load_personas(limit=30)[0]
    assert a.finance.annual_income_manwon == b.finance.annual_income_manwon
```

---

## 블록 3 — 정규분포 분위수 근사 (41~66줄)

```python
def _norm_inv(u: float) -> float:
    """표준정규 분위수 근사 (Acklam 간이판)."""
    u = min(max(u, 1e-6), 1 - 1e-6)
    a = [-3.969683028665376e01, 2.209460984245205e02, ...]
    ...
```

**무엇을 하는 함수인가**: 확률 `u`(0~1)를 받아 표준정규분포에서 그 확률에 해당하는
z값을 돌려준다. `_norm_inv(0.5) = 0`, `_norm_inv(0.975) ≈ 1.96`.

**왜 직접 구현했나**: `scipy.stats.norm.ppf()`가 같은 일을 하지만, scipy는
설치 용량이 크다(수십 MB). 이 함수 하나 때문에 무거운 의존성을 추가하는 건 과하다.
Acklam 근사는 유효자릿수 9자리 정도로 충분히 정확하다.

**`min(max(u, 1e-6), 1 - 1e-6)`**: 정확히 0이나 1이 들어오면
`log(0) = -inf`로 발산한다. 경계를 잘라내는 **방어 코드**다.

계수 배열 `a, b, c, d`와 `plow/phigh` 분기는 근사식의 구현 디테일이라
외울 필요는 없다. "확률 → z값 변환"이라는 역할만 알면 된다.

---

## 블록 4 — 로그정규분포 (69~74줄)

```python
def _lognormal(mean: float, cv: float, u: float) -> float:
    """평균 mean, 변동계수 cv인 로그정규분포의 u분위수."""
    sigma = math.sqrt(math.log(1 + cv**2))
    mu = math.log(max(mean, 1.0)) - sigma**2 / 2
    return math.exp(mu + sigma * _norm_inv(u))
```

**왜 정규분포가 아니라 로그정규분포인가**

소득 분포는 대칭이 아니다. 평균 6,800만원인 집단에서 "평균보다 6,800만원 적은 사람"은
있을 수 없지만(음수 소득), "평균보다 1억 많은 사람"은 있다. 오른쪽으로 긴 꼬리가 있다.
정규분포를 쓰면 음수 소득이 나오고, 고소득 꼬리를 표현할 수 없다.

로그정규분포는 **로그를 취하면 정규분포가 되는** 분포다. 항상 양수이고 오른쪽 꼬리가 있다.

**수식 유도** — 로그정규분포의 평균은 `exp(μ + σ²/2)`다. 우리는 평균 `m`을
맞추고 싶으므로 역산한다:

```
m = exp(μ + σ²/2)
→ ln(m) = μ + σ²/2
→ μ = ln(m) − σ²/2          ← 코드 3번째 줄
```

`σ`는 변동계수(`cv = 표준편차/평균`)에서 나온다:

```
cv² = exp(σ²) − 1
→ σ = √(ln(1 + cv²))         ← 코드 2번째 줄
```

`cv = 0.5`면 "표준편차가 평균의 50%"라는 뜻이다. `kosis_household_finance.json`에
연령대별 `income_cv: 0.5` 같은 값이 들어있다.

**`max(mean, 1.0)`**: 평균이 0이면 `log(0)`이 발산하므로 하한을 둔다.

---

## 블록 5 — `attach_finance()` 본체 (86~126줄)

### ① 소득

```python
    u_inc = _seeded_uniform(p.persona_id, "income")
    income = _lognormal(band["income_mean_manwon"], band["income_cv"], u_inc)

    for kw, mult in params["occupation_multiplier"].items():
        if kw in p.occupation:
            income *= mult
            break

    income *= 1 + 0.12 * max(0, p.household_size - 1)
    income = int(round(income / 10) * 10)
```

연령대 평균에서 뽑은 뒤 두 가지 보정을 얹는다.

- **직업 배수**: `무직 0.45`, `사무 1.05`, `의사 2.0` … `break`가 있으므로
  **첫 번째로 매칭된 키워드만** 적용된다. JSON의 키 순서가 우선순위가 된다.
- **가구원 수**: 가구원 1명 추가마다 +12%. 가계금융복지조사가 **가구 단위 소득**을
  공표하므로, 다인 가구는 소득이 높은 경향을 반영한 것이다.
- `int(round(income / 10) * 10)`: **10만원 단위로 반올림**. `3847.32만원` 같은
  가짜 정밀도를 없앤다. 합성값에 소수점을 남기면 실제 조사값처럼 오해된다.

### ② 자산 — 소득과 상관을 만드는 방법

```python
    u_mix = 0.6 * u_inc + 0.4 * _seeded_uniform(p.persona_id, "asset")
    fin_assets = int(_lognormal(band["fin_assets_mean_manwon"], band["fin_assets_cv"], u_mix))
    real_assets = int(_lognormal(band["real_assets_mean_manwon"], band["real_assets_cv"], u_mix))
```

**독립적으로 뽑으면 안 되는 이유**: 소득 분위수 `u_inc`와 자산 분위수를 완전히
독립으로 뽑으면 "연소득 2천만원인데 금융자산 3억"인 사람이 흔하게 나온다.

`u_mix = 0.6·u_inc + 0.4·u_asset` 로 **소득 분위수를 60% 섞으면** 소득이 높은 사람이
자산도 높은 경향이 생긴다. 상관계수를 정확히 통제하는 정교한 방법(가우시안 코퓰라 등)은
아니지만, 가중 평균으로 대략적인 양의 상관을 만드는 실용적 근사다.

> ⚠️ **이 프로젝트의 가장 약한 지점**이 여기다. 0.6 / 0.4라는 숫자에 실증적 근거가 없다.
> KOSIS 원자료(마이크로데이터)로 실제 상관을 추정해 교체해야 한다.
> 이것이 리포트 §8에 쓰는 "변수 결합분포 미검증" 한계의 실체다.

### ③ 부채 — 보유 여부와 금액을 나눈다

```python
    u_debt = _seeded_uniform(p.persona_id, "debt")
    has_debt = u_debt < band["debt_holder_ratio"]
    debt = (
        int(_lognormal(band["debt_mean_manwon"], band["debt_cv"],
                       u_debt / band["debt_holder_ratio"]))
        if has_debt
        else 0
    )
```

부채는 **많은 사람이 0원**이다(연령대별 보유율 42~71%). 그래서 두 단계로 만든다.

1. `u_debt < 보유율` → 부채가 있는가? (30대는 0.71이므로 71% 확률로 있다)
2. 있으면 금액을 뽑는다

`u_debt / band["debt_holder_ratio"]` 를 보라. `u_debt`는 이미 `0 ~ 보유율` 범위로
좁혀진 값이므로, 보유율로 나눠 **0~1로 다시 펼친다**. 이렇게 하지 않으면
부채가 있는 사람들의 금액이 항상 분포 하위에만 몰린다. **조건부 분포를 만들 때
자주 필요한 재정규화**다.

### ④ 파생 지표

```python
    monthly_income = max(1, int(round(income / 12)))
    monthly_ds = int(round(debt * params["annual_debt_service_rate"] / 12))
    dsr = round(100 * monthly_ds / monthly_income, 1)
    consumption = params["consumption_ratio_by_band"].get(age_band(p.age), 0.62)
    surplus = int(round(monthly_income * (1 - consumption))) - monthly_ds
```

- `monthly_ds`: 부채 × 연 11%(원금+이자 상환률 가정) ÷ 12
- `dsr`: 월 상환액 ÷ 월 소득 × 100. **금융 규제의 핵심 지표**(은행권 40% 한도)
- `surplus`: 월소득 × (1 − 소비성향) − 월상환액
  → **적금 납입 여력을 판단하는 근거**. 디베이트에서 회의론자가 이 값을 인용한다
- `max(1, …)`: 0으로 나누는 것을 막는다

여기서 만든 `monthly_surplus_manwon`이 실제 디베이트에서 이렇게 쓰였다:

> "월 여유자금 103만원은 급여이체는 충족 가능하지만, 신용카드 사용 30만원
> 24개월은 자영업자로서 정기 소비가 불확실해 조건 충족 어려움"

---

# C. `loader.py` — 데이터 로딩과 필터

## 블록 1 — 컬럼명 후보 (25~35줄)

```python
FIELD_CANDIDATES = {
    "age": ["age"],
    "sex": ["sex", "gender"],
    "region": ["region", "province", "administrative_division", "city"],
    "occupation": ["occupation", "professional_role", "job"],
    ...
}

def _first(row: dict, keys: list[str]) -> object | None:
    for k in keys:
        v = row.get(k)
        if v not in (None, "", []):
            return v
    return None
```

**외부 데이터셋의 스키마는 바뀐다.** 버전이 올라가며 `region`이 `province`가 되거나
컬럼이 사라질 수 있다. 후보 목록을 두고 **처음 발견되는 값**을 쓰면 그런 변화를 흡수한다.

`v not in (None, "", [])` — 빈 값 세 종류를 모두 건너뛴다. `if v:`로 쓰면
숫자 0도 빈 값으로 취급되므로 명시적으로 나열했다.

---

## 블록 2 — 방어적 변환 (38~72줄)

```python
def _to_int(v: object, default: int) -> int:
    try:
        return int(float(str(v).strip().rstrip("세")))
    except (TypeError, ValueError):
        return default
```

`"27"`, `27.0`, `"27세"`, `None`, `"미상"` 이 모두 들어올 수 있다.

- `str(v)` → 뭐가 오든 문자열로
- `.rstrip("세")` → `"27세"` → `"27"`
- `float()` → `"27.0"`도 처리
- `int()` → 정수화
- 실패하면 기본값

**외부 데이터를 다룰 때는 "그럴 리 없다"를 가정하지 않는다.**

```python
def row_to_persona(row: dict, idx: int) -> Persona | None:
    age = _to_int(_first(row, FIELD_CANDIDATES["age"]), -1)
    if age < 19 or age > 95:
        return None
```

**나이가 이상하면 `None`을 반환해 버린다.** 금융상품 검증 대상은 성인이고,
파싱 실패한 `-1`도 여기서 함께 걸러진다. `None`을 받은 호출부는 그 행을 건너뛴다.

```python
    sex = "남성" if sex_raw.lower() in {"male", "m", "남", "남성"} else (
        "여성" if sex_raw.lower() in {"female", "f", "여", "여성"} else "미상"
    )
```

표기 변형을 `Literal["남성","여성","미상"]`으로 정규화한다.
**경계에서 정규화**하는 원칙의 실행부다.

```python
    traits: list[str] = []
    for key in TRAIT_CANDIDATES:
        v = row.get(key)
        if isinstance(v, str) and v:
            traits.extend([t.strip() for t in v.split(",")[:3] if t.strip()])
        elif isinstance(v, list):
            traits.extend([str(t) for t in v[:3]])
    ...
    traits=traits[:6],
```

`isinstance`로 타입을 확인하고 각각 다르게 처리한다(문자열이면 쉼표 분리, 리스트면 그대로).
`[:3]`, `[:6]`으로 개수를 제한하는 것은 **토큰 비용 통제**다.

---

## 블록 3 — 3단 폴백 (`load_personas`, 168~192줄)

```python
def load_personas(*, source="auto", limit=2000, with_finance=True) -> list[Persona]:
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
```

```
① data/personas/*.jsonl        (가장 빠르고 재현 가능 — 오프라인)
      ↓ 없으면
② HuggingFace 다운로드          (datasets 설치 + 네트워크 필요)
      ↓ 실패하면
③ synthesize()                 (KOSIS 분포 기반 합성 400명)
```

**설계 의도**: 어떤 환경에서도 프로그램이 **멈추지 않는다.** 네트워크가 없는
심사장에서도, 라이브러리를 안 깐 새 노트북에서도 돌아간다.
③으로 떨어지면 `Persona.source`에 `"synthetic-fallback (Nemotron 미설치)"`가 박혀
결과를 보는 사람이 알 수 있다.

**`except Exception`으로 통째로 잡는 것**은 보통 나쁜 습관이지만, 여기서는
의도적이다. HuggingFace 로딩은 네트워크 오류·인증 오류·스키마 오류·디스크 오류 등
수십 가지로 실패할 수 있고, 무엇이든 **폴백으로 가는 게 정답**이다.

**`attach_finance`를 마지막에 일괄 적용** — 어느 경로로 왔든 재무 프로파일은
같은 방식으로 붙는다. 로딩과 합성의 분리가 여기서 값을 한다.

---

## 블록 4 — `synthesize()` (86~135줄)

폴백용 합성기다. 인구 분포만 KOSIS 공표치에 맞춘다.

```python
    def pick(pairs):
        r, acc = rng.random(), 0.0
        total = sum(w for _, w in pairs)
        for k, w in pairs:
            acc += w / total
            if r <= acc:
                return k
        return pairs[-1][0]
```

**가중 추출(weighted sampling)의 교과서적 구현**이다.
`서울 0.181, 부산 0.063, …` 같은 인구 비중대로 지역을 뽑는다.

- `total`로 나누므로 가중치 합이 1이 아니어도 된다
- `acc`를 누적하며 `r`이 들어간 구간을 찾는다 (누적분포함수 역변환)
- 마지막 `return pairs[-1][0]`은 부동소수 오차로 아무 구간에도 안 걸리는 경우 대비

```python
        if age >= 65 and rng.random() < 0.6:
            occ = "무직"
```

65세 이상은 60% 확률로 무직으로 덮어쓴다. **인구 현실을 반영하는 후처리**다.
이런 규칙을 넣지 않으면 "80세 소프트웨어 개발자"가 흔하게 나온다.

**`rng = random.Random(seed)`** — 전역 `random`이 아니라 독립 인스턴스를 쓴다.
다른 코드의 난수 상태를 오염시키지 않는다.

---

## 블록 5 — 세그먼트 필터와 코호트 추출 (183~195줄)

```python
def filter_segment(personas, segment) -> list[Persona]:
    return [p for p in personas if segment.matches(p)]


def sample_cohort(personas, segment, k, seed=0) -> list[Persona]:
    pool = filter_segment(personas, segment)
    if len(pool) <= k:
        return pool
    rng = random.Random(f"{segment.name}:{seed}")
    return rng.sample(pool, k)
```

**`random.Random(f"{segment.name}:{seed}")`** — 시드에 **문자열**을 넣을 수 있다.
`"청년_사회초년생:0"`을 시드로 쓰면 이 세그먼트는 항상 같은 k명이 뽑힌다.

왜 중요한가: 민감도 분석에서 금리만 바꿔 여러 번 돌릴 때 **매번 다른 사람들이
뽑히면 비교가 무의미해진다.** 세그먼트 이름을 시드에 넣어 고정한다.

**`if len(pool) <= k: return pool`** — 후보가 k명보다 적으면 전부 반환한다.
`rng.sample`은 모집단보다 큰 표본을 요구하면 `ValueError`를 낸다.

테스트가 재현성을 검증한다:

```python
def test_sample_cohort_is_reproducible(personas):
    a = [p.persona_id for p in sample_cohort(personas, seg, 3)]
    b = [p.persona_id for p in sample_cohort(personas, seg, 3)]
    assert a == b
```

---

## 이 디렉터리에서 가져갈 것 6가지

1. **필드명에 단위를 박는다** (`_manwon`, `_pct`) — 금융·과학 코드의 기본
2. **출처를 데이터 안에 들고 다닌다** (`source` 필드) — 합성값인지 실측값인지 잃지 않는다
3. **해시를 난수 시드로 쓰면 재현 가능한 합성 데이터가 된다** — 비교 실험의 전제
4. **경계에서 정규화하고 내부에서는 한 형태만 쓴다** (`Literal`로 강제)
5. **외부 데이터는 컬럼명 후보 + 방어적 변환 + 이상값 제거**로 받는다
6. **폴백 체인**으로 어떤 환경에서도 멈추지 않게 만든다 (단, 폴백임을 표시한다)

---

## 실습

1. **재무 합성 분포 확인** — 로그정규 꼬리가 실제로 나오는지 본다.
   ```bash
   uv run python -c "
   from fdm.personas import load_personas
   import statistics as st
   ps = [p for p in load_personas() if p.band == '30대']
   inc = sorted(p.finance.annual_income_manwon for p in ps)
   print(f'n={len(inc)} 최소{inc[0]} 중위{st.median(inc)} 평균{st.mean(inc):.0f} 최대{inc[-1]}')
   "
   ```
   실제 출력:
   ```
   n=74 최소1140 중위5780.0 평균6757 최대18960
   ```
   **평균(6,757) > 중위값(5,780)** 이면 오른쪽 꼬리가 있는 것이다. 정규분포라면 둘이 같다.
   최대값이 중위값의 3.3배까지 벌어지는 것도 로그정규의 특징이다.

2. **소득-자산 상관 확인** — `u_mix`의 0.6/0.4가 만든 상관을 눈으로 본다.
   ```bash
   uv run python -c "
   from fdm.personas import load_personas
   ps = load_personas()
   lo = [p.finance.financial_assets_manwon for p in ps if p.finance.income_quintile == 1]
   hi = [p.finance.financial_assets_manwon for p in ps if p.finance.income_quintile == 5]
   print(f'1분위 평균 금융자산 {sum(lo)/len(lo):,.0f}만원')
   print(f'5분위 평균 금융자산 {sum(hi)/len(hi):,.0f}만원')
   "
   ```
   실제 출력:
   ```
   1분위 평균 금융자산 3,591만원
   5분위 평균 금융자산 15,473만원      ← 4.3배
   ```

3. **상관을 없애보기** — `finance.py`의 `u_mix`를 `_seeded_uniform(p.persona_id, "asset")`만
   쓰도록 바꾸고 위 2번을 다시 실행하라. 두 분위의 차이가 사라지는가?

4. **세그먼트 조건 실험** — `data/segments.json`의 `고DSR_차주`에서 `dsr_min_pct`를
   30 → 50으로 올리고 `uv run fdm segments`로 인원 변화를 본다.

5. **폴백 확인** — `data/personas/`가 비어 있고 `datasets`가 없으면 ③으로 떨어진다.
   ```bash
   uv run python -c "
   from fdm.personas import load_personas
   print(load_personas(limit=3)[0].source)
   "
   ```

---

**이전** ← [02_llm.md](02_llm.md) | **다음** → `04_products_rag.md`

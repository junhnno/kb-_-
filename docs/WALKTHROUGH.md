# 코드 해설 — 이 프로젝트는 어떻게 돌아가는가

공부용 문서다. 파일을 열어보면서 따라 읽도록 실제 줄 번호와 코드를 인용했다.
읽는 순서는 §5의 번호대로가 좋다. 코드 총량은 약 2,700줄이다.

---

## 1. 한 문단 요약

**"신상품을 출시하기 전에, 가상의 고객 400명에게 미리 팔아보고 문제를 찾는 프로그램"** 이다.

핵심은 고객에게 그냥 물어보지 않는다는 점이다. LLM에게 "이 상품 어때?"라고 물으면
대부분 좋다고 답한다(순응 편향). 그래서 **옹호자·회의론자·심판 세 역할을 서로 싸우게** 만들고,
같은 질문을 **여러 번 반복해 답이 흔들리는 정도로 신뢰도를 매기고**,
**실제 분쟁조정 사례 정답셋과 대조해** 이 방식이 정말 더 정확한지 숫자로 증명한다.

---

## 2. 큰 그림 — 실행 한 번의 데이터 흐름

```
                         [입력]
  data/products/01_youth_step_saving.json     data/segments.json
   (금리 3.7~6.0%, 우대조건 4개, 약관 조항)     (타깃: 청년_사회초년생 등)
              │                                        │
              └────────────────┬───────────────────────┘
                               ▼
   ① 코호트 추출     personas/loader.py  → 400명 중 조건에 맞는 55명 → k명 샘플
                     personas/finance.py → 각자에게 소득·부채·DSR·월여유자금 부여
                               ▼
   ② 근거 검색       rag/retriever.py    → 금소법 조문·조정례 23건 중 상위 6건 (BM25)
                               ▼
   ③ 디베이트        agents/debate.py    → 옹호자→페르소나→회의론자→페르소나→심판 (LLM 5회)
                               ▼         판정 = {적합성, 가입의향점수, 위반원칙, 근거, 위험, 권고}
   ④ 멀티시드 반복   eval/confidence.py  → 같은 조합을 시드·온도 바꿔 N회 → 합의도 = 신뢰도
                               ▼
   ⑤ 세그먼트 집계   eval/simulate.py    → 세그먼트별 가입률·판정분포·저신뢰비율
                               ▼
   ⑥ 검증           eval/benchmark.py   → (A) 조정례 정답셋 적중률 (B) KOSIS 보유율 순위상관
                               ▼
   ⑦ 산출물         report.py / ui/app.py → Markdown 리포트 + Streamlit 히트맵
```

**중요한 감각 하나**: 디베이트 1건 = LLM 호출 5회다. 이게
`세그먼트 수 × 페르소나 수 × 시드 수` 만큼 곱해진다.
2세그먼트 × 2명 × 2시드 = 8건 = **LLM 호출 40회**다. 느린 게 당연하다.

---

## 3. 라이브러리 — 무엇을 왜 썼나

### 실제로 쓴 것

| 라이브러리 | 어디서 | 왜 이걸 골랐나 |
|---|---|---|
| **pydantic** v2 | 모든 스키마 | LLM이 뱉는 JSON은 형식이 자주 깨진다. pydantic이 타입 검증·기본값·범위 제한(`Field(ge=0, le=100)`)을 한 곳에서 처리해준다. `model_dump_json()` / `model_validate()`로 결과 저장·복원도 공짜다 |
| **httpx** | `llm.py` | LLM 서버에 HTTP POST를 보내는 용도. `openai` SDK를 안 쓴 이유는 ollama 네이티브 API(`/api/chat`)와 OpenAI 호환 API 둘 다 써야 해서다. SDK를 쓰면 전자를 못 쓴다 |
| **typer** + **rich** | `cli.py` | 함수에 타입힌트만 붙이면 CLI 옵션이 되는 게 typer다. rich는 터미널 표·색 출력 |
| **pandas** | `ui/app.py` | 세그먼트 결과를 표로 만들고 `melt()`로 히트맵용 long-format 변환 |
| **altair** | `ui/app.py` | 선언형 차트. `mark_rect()` + 정규화 값으로 히트맵을 짧게 그릴 수 있다 |
| **streamlit** | `ui/app.py` | 대시보드. 파이썬 파일 하나가 곧 웹앱 |
| **uv** | 전체 | 패키지 관리자. `uv sync`로 가상환경+설치, `uv run`으로 실행 |
| **pytest** | `tests/` | mock 백엔드로 LLM 없이 전 구간 검증 (21개) |

### 선택 의존성 (없어도 돌아감)

| 라이브러리 | 켜는 법 | 없으면 |
|---|---|---|
| **datasets** (HuggingFace) | `uv sync --extra personas` | Nemotron 대신 합성 페르소나 폴백 |
| **sentence-transformers** (bge-m3) | `uv sync --extra dense` | BM25 단독 검색 |

### 일부러 **안** 쓴 것 — 여기가 공부 포인트다

| 안 쓴 것 | 대신 | 이유 |
|---|---|---|
| **LangGraph / LangChain** | 순수 파이썬 함수 5개 (`debate.py`) | 디베이트는 분기 없는 5단계 직선 흐름이다. 그래프 프레임워크를 얹으면 추상화 비용만 늘고, 프롬프트를 고칠 때 어디를 봐야 하는지가 흐려진다. `run_debate()` 하나를 위에서 아래로 읽으면 전부 보이는 게 낫다 |
| **FAISS / Chroma** (벡터DB) | 직접 구현한 BM25 (`retriever.py`) | 문서가 **23건**이다. 23건에 벡터DB를 붙이는 건 과하다. 게다가 법조문 검색은 "제17조", "중도상환수수료" 같은 **정확한 용어 일치**가 중요해서 키워드 검색이 오히려 잘 맞는다 |
| **konlpy / mecab** (형태소 분석) | 어절 + 문자 2-gram (`tokenize()`) | konlpy는 JVM 설치가 필요해 재현성이 떨어진다. "중도해지"를 `중도/해지/중도/도해/해지`로 쪼개는 2-gram만으로 이 규모에선 충분하다 |
| **openai SDK** | httpx 직접 호출 | 위 표 참고 |
| **파인튜닝** | 프롬프트 + RAG | 법령·상품은 자주 바뀐다. 문서를 갈아끼우는 게 재학습보다 싸고, "근거 표시"는 RAG가 담당해야 설명 가능하다 |

---

## 4. 데이터셋 — 무엇이 진짜이고 무엇이 가공인가

이 구분이 제일 중요하다. **가공 데이터로 낸 숫자를 진짜인 것처럼 발표하면 안 된다.**

| 파일 | 내용 | 진위 | 교체 방법 |
|---|---|---|---|
| `data/personas/*.jsonl` | 합성 고객 프로필 | Nemotron-Personas-Korea (CC BY 4.0, **실제 공개 데이터**). 없으면 합성 폴백 | `scripts/fetch_personas.py` |
| `data/benchmark/kosis_household_finance.json` | 연령대별 소득·자산·부채 평균과 변동계수 | 가계금융복지조사 공표치 **근사** | KOSIS 오픈API |
| `data/rag/laws/fcpa_principles.jsonl` | 금소법 6대 판매원칙 + 감독기준 11건 | 조문을 **요약**한 것 (원문 아님) | law.go.kr 원문 |
| `data/benchmark/dispute_cases.json` | 적합성 정답셋 12건 | ⚠️ **가공 샘플**. 조정례 유형만 참조해 재구성 | fss.or.kr 조정결정례 |
| `data/benchmark/segment_holding_rates.json` | 세그먼트×상품군 보유율 | 근사치 | KOSIS |
| `data/products/*.json` | 신상품 5종 | 가상 상품 (필드명만 「금융상품 한눈에」 API 참고) | `scripts/fetch_finlife.py` |

### 페르소나 한 명은 이렇게 생겼다

```
- ID: synth-00149
- 인구: 23세 여성, 서울특별시, 가구원 1명, 미혼
- 직업/학력: 자영업자 / 대졸
- 성향: 보수적 투자성향
- 재무: 연소득 3,850만원(소득 2분위), 월소득 321만원, 금융자산 3,128만원,
        부채 0만원, 월상환액 0만원 (DSR 0.0%), 월 여유자금 103만원
```

앞의 인구·직업 정보는 Nemotron에서 오고, **뒤의 재무 수치는 우리가 KOSIS 분포로 합성**한다.
Nemotron에는 정밀한 소득·부채 수치가 없기 때문이다. 이 합성 과정이 §5.3이다.

### 상품 정의에서 가장 중요한 필드

```json
"clauses": [
  {"id": "약관 제5조", "title": "우대이율",
   "text": "우대이율은 만기해지 시점에 각 우대조건의 충족 여부를 판정하여 적용하며,
            조건 미충족분은 소급 적용하지 않는다."}
]
```

`clauses`가 **에이전트가 인용할 수 있는 근거의 목록**이다. 이게 비어 있으면
디베이트가 근거 없이 일반론만 떠들게 된다. 새 상품을 추가할 때 여기를 반드시 채워야 한다.

---

## 5. 코드 읽기 — 이 순서로 보면 된다

### 5.1 `config.py` (64줄) — 설정과 역할별 모델 배치

읽을 곳은 두 군데다.

```python
# config.py:56  역할 → 모델 티어
ROLE_MODEL = {
    "advocate": "small", "skeptic": "small", "persona": "small",
    "judge": "judge",
    "single": "judge",  # 애블레이션 비교군도 심판 모델로 → 공정 비교
}
```

**비대칭 배치**의 실체다. 토론자는 작은 모델, 심판은 큰 모델을 쓴다.
최종 판정 품질이 결과를 좌우하므로 심판에 우선 투자한다는 전략이다.
`single`(디베이트 없는 비교군)도 심판 모델을 쓰는 게 핵심인데, 그래야
"디베이트가 좋은가"를 비교할 때 **모델 크기 차이가 아니라 방식 차이만** 남는다.

```python
# config.py:45  사고 모드 (성능에 결정적)
think: bool = ... _env("FDM_THINK", "0") ...
```

§8에서 설명한다.

---

### 5.2 `llm.py` (297줄) — LLM 호출 계층

**백엔드 3종을 한 인터페이스로 감싼다.**

```python
# llm.py:53
model = self.model_for(role)          # 역할 → 모델 이름
if self.s.backend == "mock":  ...     # LLM 없이 결정론적 스텁
if self.s.backend == "ollama": ...    # 네이티브 /api/chat
# 그 외(vllm) → OpenAI 호환 /v1/chat/completions
```

#### (a) 왜 ollama만 네이티브 API를 쓰는가

OpenAI 호환 엔드포인트에는 `think` 파라미터가 없다. Qwen3는 하이브리드 추론 모델이라
기본적으로 `<think>...</think>`를 먼저 길게 생성하는데, 우리는 그걸 잘라 버린다.
실측해보니 **1200 토큰을 전부 사고에 쓰고 본문이 0자로 잘리는** 일까지 있었다.
네이티브 API는 `"think": false`로 이걸 끌 수 있고, 켜더라도 `thinking`과 `content`를
분리해서 돌려주므로 잘림을 감지할 수 있다.

```python
# llm.py:170 근처 — 본문이 비면 사고를 끄고 1회 재시도
if not text and msg.get("thinking"):
    payload["think"] = False
    ...
```

#### (b) LLM의 깨진 JSON 다루기 — `extract_json()`

LLM은 JSON만 달라고 해도 앞뒤에 설명을 붙이거나 코드펜스로 감싼다. 그래서 4단계로 관용 파싱한다.

1. `<think>` 블록 제거
2. ` ```json ... ``` ` 펜스 안쪽 시도
3. 통째로 `json.loads()` 시도
4. **중괄호 균형 스캔** — 문자열 안의 `}`를 세지 않도록 따옴표·이스케이프 상태를 추적하며
   첫 `{`부터 짝이 맞는 `}`까지 잘라 파싱

그래도 실패하면 `chat_json()`이 **"이 텍스트를 유효한 JSON으로 다시 출력하라"** 며
온도 0으로 한 번 더 부른다 (`llm.py:229` 부근).

#### (c) mock 백엔드 — LLM 없이 배관 검증

```python
# llm.py:275 근처
key = sha256(system + user)[:12]
jitter = (rand01(key, seed) - 0.5) * 2 * (20 * temperature)
score = clamp(25 + rand01(key) * 60 + jitter)
```

프롬프트 해시로 점수를 만들되 **시드·온도에 따라 흔들리게** 했다.
그래야 멀티시드 신뢰도 계산 로직(분산이 크면 저신뢰)까지 LLM 없이 검증할 수 있다.
단 **여기서 나온 적중률 숫자는 난수라 아무 의미가 없다.**

---

### 5.3 `personas/` — 가상 고객 만들기

#### `schema.py` (131줄)

`Persona`, `FinanceProfile`, `Segment` 세 모델이 있다.

```python
# schema.py — Segment.matches(): 조건이 None이면 무시
if self.age_min is not None and p.age < self.age_min: return False
if self.dsr_min_pct is not None and f.dsr_pct < self.dsr_min_pct: return False
```

`Persona.prompt_block()`이 프롬프트에 들어갈 문자열을 만든다.
**LLM이 실제로 보는 페르소나는 이 함수의 출력이 전부**다.

#### `finance.py` (126줄) — 재무 수치 합성 (수학이 들어간 곳)

Nemotron에 없는 소득·자산·부채를 KOSIS 분포에서 만들어낸다.

**1) 로그정규분포로 소득 뽑기.** 소득은 정규분포가 아니라 오른쪽으로 긴 꼬리를 가진다.
평균 `m`과 변동계수 `cv`가 주어졌을 때:

```
σ = √(ln(1 + cv²))
μ = ln(m) − σ²/2
소득 = exp(μ + σ · Φ⁻¹(u))          # u ∈ (0,1)
```

`Φ⁻¹`(표준정규 분위수)는 scipy 없이 쓰려고 Acklam 근사식을 직접 넣었다 (`_norm_inv`).

**2) u를 어디서 얻는가 — 재현성의 핵심.**

```python
u = sha256(f"{persona_id}|income") 앞 12자리 / 0x1000000000000
```

난수가 아니라 **persona_id의 해시**다. 그래서 몇 번을 돌려도 synth-00149의 연소득은 항상 3,850만원이다.
(테스트 `test_finance_is_deterministic`가 이걸 검증한다.)

**3) 보정을 차례로 얹는다.**

```
소득 × 직업배수(무직 0.45, 의사 2.0 …) × (1 + 0.12 × (가구원수 − 1))
자산: u_mix = 0.6·u_소득 + 0.4·u_자산  ← 소득과 자산에 상관을 준다
부채: u_debt < 연령대별 부채보유율일 때만 발생
월상환액 = 부채 × 0.11 / 12          ← 연 11%를 원리금상환률로 가정
DSR = 100 × 월상환액 / 월소득
월여유자금 = 월소득 × (1 − 소비성향) − 월상환액
```

> ⚠️ 여기가 이 프로젝트의 **가장 약한 고리**다. 개별 변수 분포는 KOSIS와 맞지만,
> "소득이 낮은데 자산은 많은" 같은 **변수 조합의 현실성은 검증되지 않았다.**
> 그래서 리포트 §8에 이 한계를 항상 명시한다.

#### `loader.py` (195줄) — 3단 폴백

```
data/personas/*.jsonl 있으면 그것
없으면 → HuggingFace nvidia/Nemotron-Personas-Korea 다운로드
그것도 실패하면 → synthesize() 로 KOSIS 연령·지역 분포에 맞춰 400명 생성
```

`FIELD_CANDIDATES`로 컬럼명 후보를 여러 개 두는 이유는 데이터셋 버전마다
`region` / `province` / `city` 처럼 이름이 달라질 수 있어서다.

---

### 5.4 `products/schema.py` (127줄)

`Product.prompt_block()`이 상품을 프롬프트용 텍스트로 바꾼다. 금리·기간·우대조건·수수료·
유의사항·**약관 조항**을 모두 펼쳐 넣는다. 이 함수 출력이 곧 옹호자와 회의론자가 보는 상품 설명이다.

---

### 5.5 `rag/` — 근거 검색

#### `corpus.py` (62줄)

법령 jsonl + 조정례 json을 **하나의 `Doc` 리스트**로 합친다.
조정례는 `[사실관계] … [판단요지] …` 형태의 텍스트로 변환된다. 총 23건.

#### `retriever.py` (126줄) — BM25 직접 구현

**토크나이저** (`retriever.py:25`):

```python
"중도해지수수료" → 어절: [중도해지수수료]
                 + 2-gram: [중도, 도해, 해지, 지수, 수수, 수료]
```

형태소 분석기 없이 한국어 부분일치를 잡는 값싼 방법이다.

**BM25 점수** (`retriever.py:53`):

```
score(D,Q) = Σ_t∈Q  IDF(t) · f(t,D)·(k₁+1) / (f(t,D) + k₁·(1−b+b·|D|/avgdl))
IDF(t) = ln(1 + (N − df + 0.5)/(df + 0.5))          k₁=1.5, b=0.75
```

- `f(t,D)`: 문서 내 등장 횟수 → 많이 나오면 점수↑, 단 `k₁`이 포화시킴
- `|D|/avgdl`: 긴 문서가 유리해지는 걸 `b`로 보정
- `IDF`: 흔한 단어는 가치를 낮춤

**정답 누출 차단** (`retriever.py:77`) — 이 프로젝트에서 가장 중요한 세 줄:

```python
hits = [... for d, s in ... if not (exclude_ids and d.doc_id in exclude_ids)]
```

CASE-007을 평가할 때 검색 결과에 CASE-007(정답 라벨 포함)이 끼면
적중률이 부풀려진다. `eval/benchmark.py`가 평가할 때 반드시 자기 사례를 제외한다.
**벤치마크를 만들 때 가장 흔히 저지르는 실수**이니 기억해둘 것.

---

### 5.6 `agents/` — 디베이트 본체

#### `prompts.py` (148줄) — 프롬프트 설계 4원칙

```
1) 인용 강제: 근거는 세 종류만 허용
   (a) 약관 조항 ID [약관 제5조]
   (b) 법령 문서 ID [FCPA-19]
   (c) 페르소나 재무 수치 [월 여유자금 37만원]
2) 반박 강제: 회의론자는 "옹호자 주장 중 최소 2개를 직접 인용해" 반박
3) 양가감정 허용: 페르소나는 "가입은 하겠지만 …" 같은 답을 낼 수 있다
4) 심판 규칙: 근거 인용 없는 주장은 채택하지 않는다 (옹호자·회의론자 동일 적용)
```

2번이 **순응 편향을 깨는 장치**다. 인용 없이 "동의한다"고 하면 프롬프트 위반이 된다.

#### `debate.py` (248줄) — 5단계 오케스트레이션

```python
# debate.py:97  run_debate()
hits  = retriever.retrieve(_query_for(product, persona), k=6, exclude_ids=…)
ctx   = P.context_block(상품, 페르소나, 검색된 근거)      # 5턴 모두가 공유하는 컨텍스트

t_adv = _turn("advocate", …, seed=seed)                  # ① 옹호
t_p1  = _turn("persona",  …, seed=seed+11, json_mode=True)  # ② 1차 반응
t_skp = _turn("skeptic",  …, seed=seed+22)               # ③ 반박
t_p2  = _turn("persona",  …, seed=seed+33, json_mode=True)  # ④ 재반응
judge = client.chat_json("judge", …, temperature=0.2, seed=seed+44)  # ⑤ 판정
```

**온도 설계에 주목**: 토론자는 0.8(다양한 논점이 나오게), 심판은 0.2(판정은 일관되게).
시드는 `seed, +11, +22, +33, +44`로 어긋나게 줘서 다섯 턴이 같은 난수를 쓰지 않게 한다.

**근거 강제 재요청** (`debate.py:48`):

```python
if enforce_grounding and not is_grounded(text):
    retry = client.chat(..., user + "[경고] 직전 답변에 근거 인용이 없었다. …",
                        temperature=temperature - 0.2, seed=seed + 1)
```

재요청은 **1회만** 한다. 무한 재시도는 비용이 폭발하고, 실패는 실패대로
`ungrounded_turns`에 기록해 신뢰도에서 감점하는 게 낫기 때문이다.

`single_shot()` (`debate.py:192`)은 애블레이션 비교군이다. 같은 컨텍스트를 주되
디베이트 없이 1회 판정하며, `with_rag=False`면 근거 검색도 끈다.

#### `schema.py` (139줄) — 판정 정규화와 근거 검사

LLM은 `"적합성": "부적합"`, `"pass"`, `"조건부"` 등 제멋대로 답한다.

```python
normalize_suitability("부적합") → "fail"
normalize_suitability("조건부") → "warn"
Verdict.from_json({"가입의향점수": "12.6"}) → intent_score=13
```

한글 키(`가입의향점수`)와 영문 키(`intent_score`)를 모두 받고, 범위를 벗어나면 자른다.

```python
# 인용 패턴 4종
[대괄호 내용]  |  제N조  |  (FCPA|GUIDE|CASE)-ID  |  숫자+단위(만원/%/개월)
```

> ⚠️ **알려진 한계**: 지금은 "인용 **형태**가 있는가"만 본다. `[약관 제10조]`처럼
> 실제로 존재하지 않는 조항을 지어내도 통과한다. 실측에서 실제로 관측된 문제다.
> ID 실재 여부 검증은 미구현 과제로 남아 있다.

---

### 5.7 `eval/confidence.py` (119줄) — 신뢰도 스코어

같은 조합을 N회 돌린 결과를 하나로 합친다.

```python
# confidence.py:63  aggregate()
label_agreement = 최빈 라벨 개수 / N                    # 예: fail 4회/5회 = 0.8
std             = 가입의향점수의 모표준편차
score_stability = max(0, 1 − std / 25.0)                # SCORE_STD_CEIL = 25

confidence = 0.5·label_agreement + 0.3·score_stability + 0.2·심판자기신뢰
confidence × = (1 − min(0.2, 0.04 × 무근거발화수))       # 최대 20% 감점

high ≥ 0.80  |  0.55 ≤ medium < 0.80  |  low < 0.55 → needs_review = True
```

**왜 이렇게 나눴나**

- 라벨 합의(0.5): pass/warn/fail이 매번 바뀌면 그 판정은 못 믿는다 — 가장 무겁게
- 점수 안정성(0.3): 라벨이 같아도 의향점수가 20↔80으로 튀면 불안정하다
- 심판 자기신뢰(0.2): 모델 자기보고라 과신 경향이 있어 가장 가볍게
- 무근거 감점: 근거 없이 낸 결론은 신뢰도를 깎는다

`seed_plan()` (`confidence.py:113`)은 시드와 온도를 **함께** 흔든다.

```python
temps = [0.8, 0.6, 0.9, 0.4, 1.0]   # base_temp 기준 변주
반환 [(0, 0.8), (1000, 0.6), (2000, 0.9), (3000, 0.4), (4000, 1.0)]
```

시드만 바꾸면 표면적 변동만 보게 된다. 온도까지 흔들어야
"조건을 바꿔도 결론이 유지되는가"라는 진짜 강건성을 측정할 수 있다.

---

### 5.8 `eval/simulate.py` (312줄) — 세그먼트 전개와 민감도

**잡 전개** (`simulate_product()`):

```python
for seg in segments:                       # 상품의 target_segments
    for p in sample_cohort(pool, seg, k):  # 세그먼트당 k명
        jobs.append((seg, p))
ThreadPoolExecutor(max_workers=workers).map(work, jobs)
```

각 잡이 `run_case()` → `seed_plan(n_seeds)` → 디베이트 N회 → `aggregate()`.
스레드로 병렬화하는 이유는 **대기 시간이 전부 네트워크 I/O**(LLM 응답 대기)라
GIL이 문제되지 않기 때문이다. 실측 4동시에 2.04배.

**세그먼트 집계 지표**:

```python
adoption_rate = 가입의향 평균 60점 이상인 페르소나 비율
verdict_mix   = {pass: 0.67, warn: 0.33, fail: 0.0}
flag          = 저신뢰비율 ≥ 0.5 → "추가 검증 필요"
                fail ≥ 0.3      → "판매원칙 위험"
                warn ≥ 0.5      → "조건 보완 권고"
```

**민감도 분석** (`VariantSpec` / `apply_variant`): 상품을 깊은 복사해 금리를 ±0.5%p 바꾸거나
우대조건을 빼고 같은 시뮬레이션을 다시 돌린다. `product_id`에 `::금리 -0.5%p`가 붙어 구분된다.
"금리를 0.5%p 낮추면 어느 세그먼트가 먼저 이탈하는가"를 보는 도구다.

---

### 5.9 `eval/benchmark.py` (343줄) — 정확성 실증

#### (A) 애블레이션 — 3개 arm

| arm | 디베이트 | RAG | 의미 |
|---|---|---|---|
| `single_norag` | ✗ | ✗ | 맨몸 LLM |
| `single` | ✗ | ✓ | 근거만 준 경우 |
| `debate` | ✓ | ✓ | 우리 방식 |

`single_norag → single` 차이가 **RAG의 기여**, `single → debate` 차이가 **디베이트의 기여**다.
이렇게 나눠야 "디베이트 덕분"인지 "근거를 줬기 때문"인지 구분된다.

**지표 4종**

```
accuracy        : 3분류(pass/warn/fail) 정확 일치율
risk_accuracy   : {warn,fail} vs {pass} 이분류 정확도  ← 실무에서 더 중요
                  (경고를 놓치는 것이 warn/fail을 헷갈리는 것보다 치명적)
macro_f1        : 클래스별 F1의 평균. 라벨 불균형 보정
principle_recall: 정답 위반원칙 중 예측이 잡아낸 비율 (부분 문자열 매칭)
total_llm_calls : 비용. 정확도와 **함께** 봐야 한다
```

`macro_f1` 구현에서 정답·예측 모두에 없는 클래스는 평균에서 제외한다.
0으로 세면 F1이 부당하게 낮아지기 때문이다.

#### (B) KOSIS 보유율 대조

```python
mae      = 평균 |실제 보유율 − 시뮬 가입률|
spearman = 순위 상관계수 (동점은 평균 순위, n<3이면 None)
```

**Spearman을 1차 지표로 보는 이유**: "보유율"(이미 가진 사람 비율)과
"가입의향률"(가입하겠다는 비율)은 애초에 정의가 다르다. 절대값이 맞을 리 없다.
대신 **세그먼트 간 순서**(시니어가 청년보다 정기예금을 더 가진다)가 맞는지를 본다.

---

### 5.10 `report.py` · `cli.py` · `ui/app.py`

- **`report.py`** (149줄): 8개 절짜리 Markdown 생성. §8 "한계"는 **항상** 붙는다 —
  지울 수 없게 코드에 박아뒀다.
- **`cli.py`** (208줄): typer 명령 7개 (`doctor` / `products` / `segments` / `rag` /
  `debate` / `simulate` / `ablation` / `report`). 함수 시그니처가 곧 CLI 옵션이다.
- **`ui/app.py`** (309줄): 5탭 대시보드. 히트맵은 지표마다 스케일이 달라
  `groupby("지표").transform(min-max 정규화)` 후 색을 매기고, 셀에는 원래 값을 쓴다.

---

## 6. 실행 추적 — `fdm debate` 한 번에 벌어지는 일

`uv run fdm debate 01_youth_step_saving --full` 실행 시:

| # | 위치 | 하는 일 |
|---|---|---|
| 1 | `cli.py:debate` | 상품 JSON 로드 → 첫 타깃 세그먼트 선택 |
| 2 | `loader.load_personas()` | 400명 로드 + 각자 재무 프로파일 부여 |
| 3 | `sample_cohort(…, 1)` | 세그먼트 조건 통과자 중 1명 (해시 시드라 항상 동일) |
| 4 | `retriever.retrieve(k=6)` | BM25로 근거 6건 |
| 5 | `_turn("advocate")` | **LLM 1회** — 옹호 |
| 6 | `_turn("persona", json)` | **LLM 2회** — 1차 반응 (JSON) |
| 7 | `_turn("skeptic")` | **LLM 3회** — 반박 |
| 8 | `_turn("persona", json)` | **LLM 4회** — 재반응 |
| 9 | `chat_json("judge")` | **LLM 5회** — 최종 판정 |
| 10 | 출력 | 판정·근거·위험·권고·인용문서·무근거 발화 수·소요 시간 |

### 실제 관측된 디베이트 효과

qwen3:8b 실행에서 회의론자가 이렇게 반박했다:

> **우대조건 달성 가능성 부족** [약관 제5조]
> 월 여유자금 103만원은 급여이체는 충족 가능하지만, 신용카드 사용 30만원 24개월은
> **자영업자로서 정기 소비가 불확실**해 조건 충족 어려움. [CASE-002]처럼 우대조건 달성이
> 사실상 어렵다면 최고금리 적용 불가.

그 결과 페르소나의 가입의향이 **65 → 55로 하락**했다.
회의론자가 (1) 약관 조항, (2) 페르소나의 직업, (3) 유사 조정례를 한꺼번에 엮었다 —
프롬프트의 인용 강제가 의도대로 작동한 사례다.

---

## 7. 숫자 읽는 법

| 숫자 | 의미 | 주의 |
|---|---|---|
| 가입의향 55 | 100점 만점 추정 의향 | 절대값은 못 믿는다. **세그먼트 간 비교**로만 쓸 것 |
| 신뢰도 0.34 (low) | 반복 시 판정이 흔들림 | 결과가 틀렸다는 뜻이 아니라 **더 봐야 한다**는 뜻 |
| fail 33% | 페르소나 3명 중 1명이 부적합 판정 | n이 3이면 통계가 아니다. 방향만 본다 |
| 적중률 41.7% | 12건 중 5건 맞춤 | 12건 표본의 신뢰구간은 ±25%p 수준. **차이가 20%p 미만이면 결론 내지 말 것** |
| Spearman 0.6 | 세그먼트 순위가 대체로 맞음 | n<3이면 `None` |

---

## 8. 성능 — 측정치

RTX 5070 Laptop 8GB + `qwen3:8b` 실측.

| 조치 | 효과 |
|---|---|
| 사고 모드 끄기 (`FDM_THINK=0`, 기본값) | 디베이트 1건 99초 → **47초** |
| `--workers 4` | 4동시에 **2.04배** 처리량 |
| `FDM_KEEP_ALIVE=30m` | 모델 재적재 대기 제거 |

**사고 모드 실측 (동일 프롬프트)**

| | 소요 | 생성 토큰 | 사고 | 본문 |
|---|---|---|---|---|
| thinking ON | 22.7초 | 1200 (상한 도달) | 4,556자 | **0자** ← 잘림 |
| thinking OFF | 9.2초 | 201 | 0자 | 282자 |

느린 것보다 **본문이 통째로 사라지는 것**이 더 큰 문제였다. 그래서 기본값을 끔으로 뒀다.

---

## 9. 직접 해볼 실험

공부하면서 손으로 확인해볼 것들. 앞의 셋은 LLM 없이(`FDM_BACKEND=mock`) 즉시 된다.

1. **검색기 감각 익히기** — `uv run fdm rag "중도상환수수료"` 와 `"리볼빙 수수료 미고지"`를
   비교하고, `retriever.py`의 2-gram을 꺼보면(어절만 사용) 결과가 얼마나 나빠지는지 본다.
2. **세그먼트 조건 바꾸기** — `data/segments.json`의 `고DSR_차주`에서 `dsr_min_pct`를
   30 → 50으로 올리고 `uv run fdm segments`로 인원과 평균 DSR 변화를 확인한다.
3. **신뢰도 공식 흔들기** — `confidence.py`의 `SCORE_STD_CEIL`을 25 → 10으로 낮추면
   저신뢰 판정이 얼마나 늘어나는가? 테스트가 깨지는가?
4. **프롬프트 하나만 바꾸기** — `prompts.py`의 `SKEPTIC_SYSTEM`에서
   "최소 2개를 직접 인용해 반박하라"를 지우고 실제 모델로 디베이트를 돌려보라.
   페르소나 의향이 1차→재반응에서 **덜 떨어지는지** 관찰한다. 이게 순응 편향 실험이다.
5. **RAG 없이 판정** — `single_shot(..., with_rag=False)`와 `True`를 같은 사례에 돌려
   근거 유무가 판정을 바꾸는지 본다.
6. **정답 누출 재현** — `benchmark.py`의 `exclude = {case.case_id}`를 지우고 애블레이션을
   돌리면 적중률이 얼마나 부풀려지는가? (누출이 왜 위험한지 체감하는 실험)
7. **온도 실험** — `DebateConfig(temperature_judge=0.9)`로 심판 온도를 올리면
   멀티시드 합의도가 얼마나 무너지는가?
8. **새 상품 추가** — `data/products/`에 JSON을 하나 더 만들되 `clauses`를 비워두고 돌려보라.
   근거 인용이 사라지고 `ungrounded_turns`가 늘어나는 것을 확인한다.

---

## 10. 알려진 약점 (정직하게)

| 약점 | 위치 | 영향 |
|---|---|---|
| **인용 ID 실재 검증 없음** | `agents/schema.py` | 존재하지 않는 `[약관 제10조]`를 지어내도 통과. 실측에서 관측됨 |
| **정답셋이 가공 샘플** | `data/benchmark/dispute_cases.json` | 애블레이션 수치는 잠정치. 조정례 원문으로 교체 필요 |
| **변수 결합분포 미검증** | `personas/finance.py` | 개별 분포는 맞아도 조합은 비현실적일 수 있다 |
| **표본 12건** | 애블레이션 | 신뢰구간이 너무 넓다. 사례를 50건 이상으로 늘려야 유의미 |
| **심판이 같은 모델 계열** | `config.py` | 토론자와 심판이 같은 모델이면 같은 편향을 공유한다. 심판만 다른 계열(EXAONE)로 두는 게 낫다 |
| **법조문이 요약본** | `data/rag/laws/` | 원문이 아니라 인용의 법적 정확성이 떨어진다 |

이 표를 그대로 발표 자료의 "한계와 향후 과제"로 쓸 수 있다.
심사에서는 한계를 숨긴 팀보다 **한계를 정확히 아는 팀**이 신뢰를 얻는다.

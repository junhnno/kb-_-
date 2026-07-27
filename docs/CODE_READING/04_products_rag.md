# 04. `products/` + `rag/` — 상품 정의와 근거 검색 (315줄)

| 파일 | 줄수 | 역할 |
|---|---|---|
| `products/schema.py` | 127 | 상품 정의 스키마 + 프롬프트 변환 |
| `rag/corpus.py` | 62 | 법령·조정례를 하나의 문서 리스트로 |
| `rag/retriever.py` | 126 | BM25 검색 (+ 선택적 dense 하이브리드) |

두 디렉터리를 함께 보는 이유: **디베이트가 인용할 수 있는 근거는 두 곳에서만 온다.**
상품의 약관 조항(`products`)과 법령·조정례(`rag`)다. 이 둘이 프롬프트의 재료다.

---

# A. `products/schema.py`

## 블록 1 — `Clause`: 인용의 단위 (20~29줄)

```python
class Clause(BaseModel):
    """약관/상품설명서 조항. 에이전트가 근거로 인용하는 단위."""

    id: str = Field(description="예: 약관 제5조")
    title: str
    text: str

    def cite(self) -> str:
        return f"[{self.id}({self.title})] {self.text}"
```

**이 작은 클래스가 프로젝트의 핵심 설계 결정을 담고 있다.**

상품을 그냥 텍스트로 주면 LLM은 "이 상품은 조건이 까다롭다"처럼 **출처 없는 주장**을 한다.
조항을 ID·제목·본문으로 쪼개서 주면 `[약관 제5조]`처럼 **가리킬 대상**이 생긴다.

`cite()`의 출력:

```
[약관 제5조(우대이율)] 우대이율은 만기해지 시점에 각 우대조건의 충족 여부를 판정하여
적용하며, 조건 미충족분은 소급 적용하지 않는다.
```

이 형식이 프롬프트에 들어가고, 모델은 답변에서 `[약관 제5조]`만 따와 인용한다.
`agents/schema.py`의 인용 검출 정규식이 이 대괄호 패턴을 찾는다.

> ⚠️ 지금은 모델이 `[약관 제10조]`처럼 **없는 조항을 지어내도 통과**한다.
> `Clause.id` 집합과 대조하는 검증이 미구현이다. (실제로 관측된 문제)

---

## 블록 2 — `Preferential`: 우대조건 (32~42줄)

```python
class Preferential(BaseModel):
    name: str
    rate_bonus_pct: float = 0.0
    requirement: str
    est_attainment_rate: float | None = Field(
        default=None, description="상품기획팀이 추정한 달성률(0~1). 비어 있으면 디베이트가 추정."
    )
```

**`est_attainment_rate`가 이 프로젝트의 문제의식이다.**

적금 광고의 "최고 연 6.0%"는 우대조건 4개를 36개월간 전부 충족해야 나온다.
각 조건의 달성률이 0.45 / 0.35 / 0.55 / 0.3 이라면 전부 충족할 확률은
대략 `0.45 × 0.35 × 0.55 × 0.3 ≈ 2.6%`다. 즉 **광고 금리를 받는 사람은 40명 중 1명**이다.

이 필드를 상품 정의에 넣어두면 회의론자가 이걸 근거로 공격할 수 있다.
`None`으로 비워두면 LLM이 페르소나의 소득·직업을 보고 직접 추정한다.

**`rate_bonus_pct`가 대출에서는 음수**다 (`-0.4` = 0.4%p 감면).
적금은 금리가 높아야 좋고 대출은 낮아야 좋으므로, 부호로 방향을 표현했다.

---

## 블록 3 — `Product` 필드 (56~90줄)

```python
class Product(BaseModel):
    product_id: str
    name: str
    category: Category            # Literal["saving","deposit","loan","card","pension","fund"]
    issuer: str = "KB국민은행"
    summary: str = ""

    intr_rate: float | None = Field(default=None, description="기본금리(연 %) 또는 대출 최저금리")
    intr_rate2: float | None = Field(default=None, description="최고우대금리(연 %) 또는 대출 최고금리")
    intr_rate_type: str = "단리"
    rate_basis: Literal["고정", "변동"] = "고정"
    save_trm_months: int | None = None
    ...
```

**필드명이 왜 `intr_rate`, `intr_rate2`처럼 이상한가**
금융감독원 「금융상품 한눈에」 오픈API의 실제 응답 필드명이다.
`scripts/fetch_finlife.py`로 실제 상품을 받아올 때 **변환 없이 그대로 매핑**하려고
API 이름을 그대로 썼다. 예쁜 이름(`base_rate`, `max_rate`)으로 바꾸면 매핑 코드가
한 겹 더 필요하고, 공시 데이터와 대조할 때 헷갈린다.

**`Category`를 `Literal`로 제한**하면 오타(`"savings"` vs `"saving"`)를 pydantic이 잡는다.
`eval/benchmark.py`가 이 값으로 KOSIS 보유율 표를 조회하므로 오타가 나면 대조가 조용히 실패한다.

---

## 블록 4 — `prompt_block()`: 상품 → 프롬프트 (99~127줄)

```python
    def prompt_block(self) -> str:
        lines = [f"- 상품명: {self.name} ({self.issuer}, {self.category})"]
        if self.summary:
            lines.append(f"- 개요: {self.summary}")
        if self.intr_rate is not None:
            kind = "대출금리" if self.category == "loan" else "금리"
            lines.append(
                f"- {kind}: 기본 연 {self.intr_rate}% / 최대 연 {self.intr_rate2}% "
                f"({self.rate_basis}, {self.intr_rate_type})"
            )
        ...
        if self.preferentials:
            lines.append("- 우대조건:")
            lines += [
                f"  · {p.name} (+{p.rate_bonus_pct}%p): {p.requirement}" for p in self.preferentials
            ]
        ...
        if self.clauses:
            lines.append("- 인용 가능한 약관 조항:")
            lines += [f"  · {c.cite()}" for c in self.clauses]
        return "\n".join(lines)
```

**`kind = "대출금리" if self.category == "loan" else "금리"`**
같은 필드가 상품군에 따라 다른 의미를 갖는다. 적금의 `intr_rate2`는 "최고 우대금리"(좋음),
대출의 `intr_rate2`는 "최고 금리"(나쁨)다. 레이블을 바꿔 모델이 오해하지 않게 한다.

**`"- 인용 가능한 약관 조항:"` 이라는 문구**가 중요하다. 단순히 조항을 나열하는 게 아니라
**"이걸 인용해라"라고 지시**하는 셈이다. 프롬프트 엔지니어링은 이런 작은 레이블에서 갈린다.

**리스트에 `+=`로 여러 줄 추가**하는 패턴(`lines += [...]`)은 `extend()`와 같다.
컴프리헨션과 조합하면 조건부 다중 줄 생성이 간결해진다.

---

## 블록 5 — 로더 (117~127줄)

```python
def load_product(path: str | Path) -> Product:
    p = Path(path)
    if not p.exists():
        p = PRODUCT_DIR / f"{path}.json"
    with open(p, encoding="utf-8") as f:
        return Product(**json.load(f))
```

**경로도 받고 이름도 받는다.** CLI에서 `fdm debate 01_youth_step_saving`처럼
짧은 이름을 쓸 수 있게 하는 편의 기능이다. 파일이 없으면 `data/products/이름.json`으로 해석한다.

**`Product(**json.load(f))`** — JSON 딕셔너리를 그대로 pydantic 모델에 펼쳐 넣는다.
필드가 빠졌거나 타입이 틀리면 **여기서 즉시 에러**가 난다. 잘못된 상품 정의를 들고
디베이트까지 갔다가 이상한 결과를 얻는 것보다 훨씬 낫다.

**`encoding="utf-8"` 필수** — Windows 기본 인코딩(cp949)으로 열면 한글 JSON이 깨진다.
이 프로젝트의 모든 파일 I/O에 명시되어 있다.

---

# B. `rag/corpus.py` — 문서 통합

## 블록 1 — `Doc` (18~28줄)

```python
@dataclass(frozen=True)
class Doc:
    doc_id: str
    title: str
    text: str
    kind: str  # law | guideline | case
    principle: tuple[str, ...] = ()

    def cite(self, max_chars: int = 420) -> str:
        body = self.text if len(self.text) <= max_chars else self.text[:max_chars] + "…"
        return f"[{self.doc_id}] {self.title}: {body}"
```

**`frozen=True`** — 불변 객체가 된다. 필드를 바꾸려 하면 에러다.
검색 인덱스가 참조하는 문서가 도중에 변경되면 인덱스와 실제 내용이 어긋나므로
불변으로 막았다. 부수적으로 해시 가능해져 `set`에 넣을 수 있다.

**`principle: tuple[str, ...]`** — 리스트가 아니라 튜플인 이유도 불변성이다.
`frozen=True`인 dataclass에 가변 필드(list)를 넣으면 "겉은 불변, 속은 가변"이 된다.

**`cite(max_chars=420)`** — 프롬프트에 넣을 때 420자로 자른다.
문서 6건 × 420자 ≈ 2,500자가 매 턴 컨텍스트에 들어간다. 자르지 않으면
법조문 전문이 컨텍스트를 다 차지한다. **RAG에서 청크 길이는 비용과 정확도의 교환**이다.

---

## 블록 2 — `load_corpus()` (31~62줄)

```python
@lru_cache(maxsize=1)
def load_corpus() -> list[Doc]:
    docs: list[Doc] = []

    for path in sorted(LAWS_DIR.glob("*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                docs.append(Doc(...))
```

**`@lru_cache(maxsize=1)`** — 두 번째 호출부터는 파일을 다시 읽지 않고 캐시를 반환한다.
디베이트마다 23건을 다시 파싱하면 낭비다. `maxsize=1`은 인자가 없는 함수라 캐시 슬롯이 하나면 충분하다.

**JSONL(줄 단위 JSON)을 쓴 이유**: 문서를 한 줄씩 추가·삭제하기 쉽고, 파일 하나가
깨져도 그 줄만 문제가 된다. 법조문처럼 계속 추가되는 데이터에 적합하다.
빈 줄은 건너뛴다(`if not line: continue`) — 파일 끝의 개행 때문에 필요하다.

```python
    if CASES_PATH.exists():
        with open(CASES_PATH, encoding="utf-8") as f:
            payload = json.load(f)
        for c in payload["cases"]:
            docs.append(
                Doc(
                    doc_id=c["case_id"],
                    title=f"조정사례(가공) {c['title']} — 판정 {c['label']}",
                    text=f"[사실관계] {c['facts']}\n[판단요지] {c['decision_gist']}",
                    kind="case",
                    ...
                )
            )
```

**여기에 함정이 있다.** 조정례 문서의 `title`에 **정답 라벨(`판정 warn`)이 들어간다.**

의도된 것이다 — 디베이트할 때는 유사 사례의 결론을 참고하는 게 정당하다.
하지만 **애블레이션에서 그 사례 자체를 평가할 때 이 문서가 검색되면 정답을 그대로 보여주는 것**이 된다.
그래서 `retriever.retrieve(exclude_ids=...)`가 반드시 필요하다. 다음 절에서 본다.

**하나의 데이터 파일을 두 용도로 쓴다**: `dispute_cases.json`이 RAG 코퍼스이면서
동시에 벤치마크 정답셋이다. 데이터를 중복 관리하지 않는 대신, **누출에 주의해야 한다.**

---

# C. `rag/retriever.py` — BM25 직접 구현

## 블록 1 — 한국어 토크나이저 (21~33줄)

```python
_WORD = re.compile(r"[0-9A-Za-z가-힣%]+")
_HANGUL = re.compile(r"[가-힣]{2,}")


def tokenize(text: str) -> list[str]:
    toks = [t.lower() for t in _WORD.findall(text)]
    grams: list[str] = []
    for t in toks:
        if _HANGUL.fullmatch(t) and len(t) > 2:
            grams += [t[i : i + 2] for i in range(len(t) - 1)]
    return toks + grams
```

**동작 예시**

```python
tokenize("중도상환수수료 1.2%")
# → ['중도상환수수료', '1', '2%',                     ← 어절 ('.'이 경계, '%'는 문자에 포함)
#    '중도','도상','상환','환수','수수','수료']        ← 7글자 → 2-gram 6개
```

`'.'`은 문자 클래스 `[0-9A-Za-z가-힣%]`에 없으므로 경계가 되어 `1.2%`가 `1`과 `2%`로 쪼개진다.
`%`는 클래스에 있으므로 숫자에 붙어 남는다 — `"40%"` 같은 표현을 하나의 토큰으로 잡으려는 의도다.

**왜 2-gram을 섞나**

한국어는 조사가 붙고 복합어가 길다. `"중도상환수수료"`와 `"중도상환"`은
어절 단위로는 다른 토큰이라 매칭되지 않는다. 2-gram을 넣으면
`중도`, `도상`, `상환`이 공통으로 잡혀 **부분 일치**가 가능해진다.

**왜 어절도 함께 남기나**

2-gram만 쓰면 정확한 용어 일치의 가중치가 사라진다. `"제17조"` 같은 짧고
정확한 토큰은 어절로 잡는 게 낫다. **둘을 합치면 정확 일치와 부분 일치를 동시에 얻는다.**

**`len(t) > 2` 조건**: 2글자 단어는 2-gram이 자기 자신이라 중복이므로 건너뛴다.

**형태소 분석기를 안 쓴 이유**: konlpy는 JVM 설치가 필요해 재현성이 나쁘고,
Windows 환경 설정이 까다롭다. 문서 23건 규모에서는 이 값싼 방법으로 충분하다.
(실제로 `"우대금리 최고금리 광고 오인"` 검색 시 관련 문서 2건이 1·2위로 나온다.)

---

## 블록 2 — BM25 인덱스 구축 (40~51줄)

```python
class BM25:
    def __init__(self, docs: list[Doc], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(f"{d.title} {d.text}")) for d in docs]
        self.len = [sum(c.values()) for c in self.tf]
        self.avg_len = (sum(self.len) / len(self.len)) if self.len else 1.0
        df: Counter[str] = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - v + 0.5) / (v + 0.5)) for t, v in df.items()}
```

**미리 계산해두는 것들**

| 변수 | 내용 | 왜 미리 계산하나 |
|---|---|---|
| `tf[i]` | i번 문서의 단어별 등장 횟수 | 검색마다 토큰화하면 낭비 |
| `len[i]` | i번 문서의 총 토큰 수 | 길이 정규화에 필요 |
| `avg_len` | 전체 평균 문서 길이 | 길이 정규화의 기준 |
| `idf[t]` | 단어 t의 역문서빈도 | 검색어마다 재계산할 필요 없음 |

**`Counter(tokenize(...))`** — `{'중도': 3, '상환': 2, ...}` 형태의 빈도표를 만든다.
`Counter.update(c.keys())`는 **키만** 세므로 "이 단어가 등장한 문서 수(df)"가 된다.

**`title`과 `text`를 합쳐 색인**한다. 제목에 핵심어가 있으므로 함께 넣는 게 유리하다.
(제목에 가중치를 더 주는 필드 부스팅은 미구현이다.)

**IDF 공식**

```
idf(t) = ln(1 + (N − df + 0.5) / (df + 0.5))
```

- `N`: 전체 문서 수(23), `df`: t가 등장한 문서 수
- df가 작으면(희귀어) → 분자가 커져 idf↑
- df가 N에 가까우면(모든 문서에 있는 단어) → idf가 0에 가까워진다
- `+0.5`는 **스무딩**. df=0이나 df=N에서 발산하는 걸 막는다
- `1 +` 를 씌운 것도 음수 방지용이다 (BM25+ 변형)

---

## 블록 3 — BM25 점수 계산 (53~67줄)

```python
    def search(self, query: str) -> list[float]:
        q = tokenize(query)
        scores = [0.0] * len(self.docs)
        for term in q:
            idf = self.idf.get(term)
            if not idf:
                continue                       # 코퍼스에 없는 단어는 무시
            for i, c in enumerate(self.tf):
                f = c.get(term)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg_len)
                scores[i] += idf * f * (self.k1 + 1) / denom
        return scores
```

**수식**

```
score(D,Q) = Σ_{t∈Q}  idf(t) · f(t,D)·(k₁+1)
                      ─────────────────────────────────────────
                      f(t,D) + k₁·(1 − b + b·|D|/avgdl)
```

**각 항의 역할을 직관으로 이해하기**

1. **`f(t,D)` (단어 빈도)** — 많이 나오면 점수가 오른다. 단
   `f/(f + k₁·…)` 형태라 **포화(saturation)** 한다. `k₁=1.5`에서
   빈도 1 → 2로 늘 때의 증가가 빈도 10 → 11보다 훨씬 크다.
   "중도상환이 10번 나온 문서"가 "1번 나온 문서"보다 10배 관련 있는 건 아니기 때문이다.
   (단순 TF-IDF는 이 포화가 없어 반복 단어에 취약하다.)

2. **`|D|/avgdl` (길이 정규화)** — 긴 문서는 우연히 단어를 많이 포함한다.
   분모를 키워 불이익을 준다. `b=0.75`는 정규화 강도로,
   `b=0`이면 길이 무시, `b=1`이면 완전 정규화다.

3. **`idf(t)`** — 흔한 단어("금융", "상품")의 기여를 낮춘다.

**`k₁=1.5, b=0.75`** 는 정보검색 문헌의 표준 기본값이다. 튜닝 여지는 있지만
문서 23건에서는 큰 차이가 없다.

**루프 순서에 주목**: 쿼리 단어를 바깥, 문서를 안쪽에 둔다.
쿼리에 없는 단어는 아예 순회하지 않으므로(`if not idf: continue`)
문서 수가 늘어도 쿼리 단어 수만큼만 일이 늘어난다.

---

## 블록 4 — `Retriever.retrieve()`와 정답 누출 차단 (69~99줄)

```python
    def retrieve(
        self,
        query: str,
        k: int = 5,
        *,
        kinds: tuple[str, ...] | None = None,
        exclude_ids: set[str] | None = None,
    ) -> list[Hit]:
        scores = self.bm25.search(query)
        if self.dense is not None:
            d_scores = self.dense(query)
            mx_b = max(scores) or 1.0
            mx_d = max(d_scores) or 1.0
            scores = [0.5 * s / mx_b + 0.5 * d / mx_d for s, d in zip(scores, d_scores)]

        hits = [
            Hit(doc=d, score=s)
            for d, s in zip(self.docs, scores)
            if (kinds is None or d.kind in kinds)
            and not (exclude_ids and d.doc_id in exclude_ids)
        ]
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]
```

### 하이브리드 점수 결합

```python
scores = [0.5 · (bm25/max_bm25) + 0.5 · (dense/max_dense) for …]
```

BM25 점수는 0~20 같은 임의 범위이고 코사인 유사도는 0~1이다.
**스케일이 다른 점수를 그냥 더하면 큰 쪽이 지배한다.** 각각 최댓값으로 나눠
0~1로 맞춘 뒤 반반 섞는다.

`max(scores) or 1.0` — 모든 점수가 0이면 `max`도 0이라 0으로 나누게 된다.
`or 1.0`이 이를 막는다(파이썬에서 `0 or 1.0`은 `1.0`).

### `exclude_ids` — 이 프로젝트에서 가장 중요한 두 줄

```python
            and not (exclude_ids and d.doc_id in exclude_ids)
```

왜 필요한가. `dispute_cases.json`의 문서 제목은 이렇게 생겼다:

```
[CASE-007] 조정사례(가공) 리볼빙 수수료 미고지 — 판정 fail
```

**제목에 정답(`fail`)이 들어 있다.** CASE-007의 적합성을 맞히는 평가에서 이 문서가
검색되면, 모델은 추론할 필요 없이 제목을 베끼면 된다. 적중률이 100%로 나오지만
아무것도 증명하지 못한다. 이것이 **데이터 누출(data leakage)** 이다.

`eval/benchmark.py`가 이렇게 쓴다:

```python
exclude = {case.case_id}     # 자기 자신을 제외
run_debate(..., exclude_doc_ids=exclude)
```

머신러닝 벤치마크에서 가장 흔하고, 가장 발견하기 어려운 실수다.
**"정답이 입력에 섞여 들어갈 경로가 있는가"를 항상 의심해야 한다.**
(4편 실습 3번에서 이걸 일부러 재현해본다.)

`kinds` 파라미터는 "법령만 검색" 같은 필터용으로 만들어뒀지만 현재 호출부에서는 안 쓴다.

---

## 블록 5 — 선택적 dense 검색 (108~126줄)

```python
def _try_load_dense(docs: list[Doc]):
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("BAAI/bge-m3")
        emb = model.encode([f"{d.title} {d.text}" for d in docs], normalize_embeddings=True)

        def score(query: str) -> list[float]:
            q = model.encode([query], normalize_embeddings=True)[0]
            return (emb @ q).tolist()

        return score
    except Exception:
        return None
```

**클로저 패턴**: 함수 안에서 `score` 함수를 정의해 반환한다.
`score`는 `model`과 `emb`를 **기억한 채** 밖으로 나간다.
클래스를 만들지 않고 상태를 감싸는 가벼운 방법이다.

**`normalize_embeddings=True`** — 벡터를 단위 길이로 정규화한다.
그러면 내적(`emb @ q`)이 곧 **코사인 유사도**가 된다. 나눗셈이 사라져 빠르다.

**`emb @ q`** — `@`는 행렬 곱 연산자다. `(23, 1024) @ (1024,) = (23,)`,
즉 23개 문서와 쿼리의 유사도를 한 번에 계산한다.

**`except Exception: return None`** — 라이브러리가 없거나 모델 다운로드가 실패하면
`None`을 반환하고, `retrieve()`는 BM25 단독으로 동작한다.
`config.py`의 dotenv 처리, `loader.py`의 폴백과 같은 원칙이다.

```python
@lru_cache(maxsize=2)
def get_retriever(use_dense: bool = False) -> Retriever:
    return Retriever(use_dense=use_dense)
```

**인덱스 재구축 방지.** 디베이트 수십 건이 같은 검색기를 공유한다.
`maxsize=2`는 `use_dense` True/False 두 경우를 위한 것이다.

> ⚠️ `lru_cache`는 스레드 안전하지 않지만, `Retriever`가 생성 후 읽기만 하므로
> 실질적인 문제는 없다. 다만 최초 호출이 동시에 들어오면 인덱스가 두 번 만들어질 수 있다.

---

## 이 편에서 가져갈 것 5가지

1. **인용의 단위를 데이터 구조로 만든다** (`Clause`, `Doc`) — 근거 표시의 출발점
2. **외부 API 필드명을 그대로 쓰면 매핑 코드가 사라진다** (`intr_rate2`)
3. **BM25의 세 요소** — 빈도 포화(`k₁`), 길이 정규화(`b`), 희귀어 가중(`idf`)
4. **스케일이 다른 점수를 섞을 때는 각각 정규화한다**
5. **정답이 입력으로 새어 들어갈 경로를 항상 의심한다** (`exclude_ids`)

---

## 실습

1. **토크나이저 관찰**
   ```bash
   uv run python -c "
   from fdm.rag.retriever import tokenize
   print(tokenize('중도상환수수료 1.2%'))
   print(tokenize('제17조'))
   "
   ```

2. **검색 품질 비교** — 어떤 쿼리에서 법령이, 어떤 쿼리에서 조정례가 상위에 오는가?
   ```bash
   uv run fdm rag "중도상환수수료"
   ```
   ```bash
   uv run fdm rag "고령자에게 원금손실 상품 권유"
   ```

3. **누출 재현 (중요)** — `exclude_ids`가 없으면 어떻게 되는지 직접 확인한다.
   ```bash
   PYTHONIOENCODING=utf-8 uv run python -c "
   from fdm.rag.retriever import get_retriever
   r = get_retriever()
   q = '리볼빙 수수료 미고지 카드 발급'
   print('=== 제외 없음 ===')
   for h in r.retrieve(q, k=3): print(' ', h.doc.doc_id, h.doc.title[:45])
   print('=== CASE-007 제외 ===')
   for h in r.retrieve(q, k=3, exclude_ids={'CASE-007'}): print(' ', h.doc.doc_id, h.doc.title[:45])
   "
   ```
   실제 출력의 첫 줄:
   ```
   CASE-007 조정사례(가공) 리볼빙(일부결제금액이월약정) 수수료율 미고지 — 판정 fail
   ```
   **제목에 정답 라벨(`fail`)이 그대로 보인다.** 모델은 추론 없이 이걸 베끼면 100% 맞힌다.
   이게 왜 위험한지 체감할 수 있다.

   > 💡 **Windows 인코딩 주의**: `python -c` 로 한글을 `print` 하면 기본 콘솔 인코딩(cp949)에서
   > `—`, `…` 때문에 `UnicodeEncodeError`가 난다. 위처럼 `PYTHONIOENCODING=utf-8` 을 붙이면 된다.
   > `fdm` CLI는 `cli.py`에서 출력 스트림을 UTF-8로 다시 열어 이 문제를 처리한다
   > (이 문서를 쓰면서 발견해 고친 버그다).

4. **2-gram 끄기** — `retriever.py`의 `tokenize()`에서 `return toks + grams`를
   `return toks`로 바꾸고 실습 2를 다시 실행하라. 검색 순위가 어떻게 나빠지는가?

5. **상품 프롬프트 확인** — LLM이 실제로 보는 상품 설명을 그대로 출력한다.
   ```bash
   uv run python -c "
   from fdm.products import load_product
   print(load_product('04_cashback_card').prompt_block())
   "
   ```

6. **`clauses`를 비우면?** — 상품 JSON에서 `clauses`를 `[]`로 만들고
   `uv run fdm debate ...`를 돌려보라. 무근거 발화가 늘어나는지 확인한다.

---

**이전** ← [03_personas.md](03_personas.md) | **다음** → `05_agents.md` (디베이트 엔진)

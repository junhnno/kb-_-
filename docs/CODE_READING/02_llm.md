# 02. `src/fdm/llm.py` — LLM 호출 관문 (297줄)

**모든 LLM 호출이 이 파일 하나를 지나간다.** 디베이트 5턴, 애블레이션 3 arm,
JSON 판정이 전부 `LLMClient.chat()` 또는 `chat_json()`을 부른다.

이 파일이 푸는 문제는 네 가지다.

1. 백엔드가 3개인데(ollama / vllm / mock) 호출부는 하나만 알고 싶다
2. LLM은 JSON을 달라고 해도 깨진 JSON을 준다
3. Qwen3는 사고 모드 때문에 본문이 잘린다
4. LLM 없이도 파이프라인을 돌려봐야 한다

---

## 블록 1 — 반환 타입과 예외 (22~30줄)

```python
@dataclass
class LLMResponse:
    text: str
    model: str
    raw: dict[str, Any] | None = None


class LLMError(RuntimeError):
    pass
```

**왜 문자열을 그냥 반환하지 않고 감쌌나**
`text`만 반환하면 "어느 모델이 답했는지"를 잃는다. 비대칭 배치를 쓰는 프로젝트에서는
`Turn.model` 필드에 이걸 기록해야 리포트에 "심판=EXAONE-32B"라고 쓸 수 있다.
`raw`는 서버 원본 응답으로, 토큰 수·소요 시간 같은 디버깅 정보가 들어 있다.

**`LLMError`를 따로 만든 이유**
호출부에서 "LLM 문제"와 "그 밖의 버그"를 구분해 잡을 수 있다.
`cli.py:doctor`가 이걸 이용한다:

```python
try:
    res = LLMClient().chat(...)
except LLMError as e:          # 서버가 안 떠 있음 → 친절한 안내
    console.print(f"[red]LLM 실패[/]: {e}")
```

`RuntimeError`를 상속했으므로 굳이 잡지 않으면 일반 예외처럼 위로 올라간다.

---

## 블록 2 — 클라이언트와 역할→모델 결정 (33~40줄)

```python
class LLMClient:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or SETTINGS

    def model_for(self, role: str) -> str:
        tier = ROLE_MODEL.get(role, "small")
        return self.s.model_judge if tier == "judge" else self.s.model_small
```

**`settings or SETTINGS`**
기본은 전역 설정을 쓰지만, 인자로 다른 `Settings`를 넣을 수도 있다.
테스트에서 "이 클라이언트만 다른 모델"을 만들 때 쓰는 **의존성 주입** 패턴이다.

**`ROLE_MODEL.get(role, "small")`**
모르는 역할이 오면 작은 모델로 떨어진다(`.get`의 기본값). 새 역할을 추가했다가
`ROLE_MODEL`에 등록하는 걸 잊어도 죽지 않고 안전한 쪽으로 동작한다.

여기서 `role`은 두 가지 일을 겸한다 — **모델 선택**과 **`_mock_reply`의 응답 형태 선택**.

---

## 블록 3 — `chat()` 진입부: 백엔드 3분기 (42~68줄)

```python
    def chat(
        self,
        *,                      # ← 별표 하나. 이 뒤는 모두 키워드 전용 인자
        role: str,
        system: str,
        user: str,
        temperature: float = 0.7,
        seed: int | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        model = self.model_for(role)
        if self.s.backend == "mock":
            return LLMResponse(
                text=_mock_reply(role, system, user, seed, temperature),
                model=f"mock::{model}",
            )
        if self.s.backend == "ollama":
            return self._chat_ollama_native(...)
        # 아래로 내려가면 vllm (OpenAI 호환)
```

### `*` 의 의미 — 실수를 컴파일 단계에서 막는 장치

```python
def chat(self, *, role, system, user, temperature=0.7, ...)
```

`*` 뒤의 인자는 **반드시 이름을 붙여** 호출해야 한다.

```python
client.chat("advocate", sys, usr, 0.8)              # ✗ TypeError
client.chat(role="advocate", system=sys, user=usr)  # ✓
```

`system`과 `user`는 둘 다 문자열이라 순서를 바꿔 넣어도 파이썬은 모른다.
그러면 시스템 프롬프트와 사용자 프롬프트가 뒤바뀐 채 조용히 이상한 답이 나온다.
`*` 하나로 이 부류의 버그를 원천 차단한다. **인자가 4개 이상이고 타입이 겹치면 쓸 가치가 있다.**

### `mock::` 접두어

```python
model=f"mock::{model}"     # → "mock::qwen3:8b"
```

결과 JSON에 `mock::`가 박히므로, 나중에 리포트를 보고 **"이게 스텁으로 만든 숫자였나"**
를 구분할 수 있다. 가짜 데이터에 표시를 남기는 습관이다.

---

## 블록 4 — OpenAI 호환 경로 (69~109줄)

```python
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens or self.s.max_tokens,
            "stream": False,
        }
        if seed is not None:
            payload["seed"] = seed
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if not self.s.think:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
```

**`messages` 구조**가 채팅 API의 표준이다. `system`은 역할·규칙(프롬프트 설계),
`user`는 이번 요청의 내용(상품·페르소나·지시)이다. 이 프로젝트는 대화를 누적하지 않고
**매 턴 필요한 맥락을 `user`에 전부 다시 넣는다**(`prompts.py`의 `ctx`).
그래야 턴별로 온도·시드·모델을 독립적으로 통제할 수 있다.

**`max_tokens or self.s.max_tokens`** — 인자가 `None`이면 설정값을 쓴다.
단 이 관용구는 `0`도 거짓으로 보므로 "0을 의미 있는 값으로 쓰는 설정"에는 쓰면 안 된다.

**`"stream": False`** — 토큰을 한 개씩 흘려받지 않고 완성된 답을 한 번에 받는다.
우리는 화면에 실시간 출력할 필요가 없고, 파싱해서 쓰기만 하므로 이게 단순하다.

**`temperature`** — 0에 가까우면 항상 비슷한 답(재현성↑), 높으면 다양한 답.
디베이트 토론자는 0.8, 심판은 0.2를 쓴다. 신뢰도 측정에서는 이 값을 **일부러 흔든다**.

**`seed`** — 같은 시드 + 같은 프롬프트 + 같은 온도면 (대체로) 같은 답이 나온다.
`test_confidence_high_when_runs_agree`가 이 성질을 이용해 "합의도 1.0"을 검증한다.

### 에러 처리에서 배울 점 (89~105줄)

```python
        try:
            r = httpx.post(url, json=payload, timeout=self.s.timeout)
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            # 서버가 response_format / chat_template_kwargs를 거부한 경우
            if json_mode or "chat_template_kwargs" in payload:
                payload.pop("response_format", None)
                payload.pop("chat_template_kwargs", None)
                r = httpx.post(url, json=payload, timeout=self.s.timeout)
                r.raise_for_status()
            else:
                raise LLMError(f"{url} 호출 실패: {e} / {e.response.text[:300]}") from e
        except httpx.HTTPError as e:
            raise LLMError(f"{url} 연결 실패: {e}. 백엔드가 떠 있는지 …") from e
```

예외를 **두 종류로 나눈** 게 핵심이다.

| 예외 | 뜻 | 대응 |
|---|---|---|
| `HTTPStatusError` | 서버는 살아있는데 4xx/5xx를 줬다 | 선택적 파라미터를 빼고 1회 재시도 |
| `HTTPError` (그 외) | 연결 자체 실패(서버 없음, 타임아웃) | 즉시 `LLMError` + **해결 방법 안내** |

- **점진적 격하(graceful degradation)**: `response_format`이나 `chat_template_kwargs`를
  모르는 서버가 있다. 그럴 때 죽는 대신 그 키만 빼고 다시 시도한다.
  단 **재시도는 1회만** — 무한 재시도는 장애를 증폭시킨다.
- **`raise … from e`**: 원래 예외를 원인으로 연결한다. 트레이스백에
  "During handling… the above exception"이 나와 근본 원인을 잃지 않는다.
- **에러 메시지에 해결책을 넣는다**: `"ollama serve가 떠 있는지 확인하세요.
  또는 FDM_BACKEND=mock 사용."` 사용자가 다음에 뭘 할지 알 수 있어야 좋은 에러다.
- **`e.response.text[:300]`**: 서버 응답 본문을 300자만 붙인다. 전체를 붙이면
  터미널이 HTML 덤프로 뒤덮인다.

```python
        data = r.json()
        text = data["choices"][0]["message"]["content"] or ""
        return LLMResponse(text=_strip_think(text), model=model, raw=data)
```

`content`가 `None`일 수 있어 `or ""`로 받는다. 그리고 `_strip_think()`로
`<think>` 블록을 제거한다 — OpenAI 호환 경로에서는 사고 내용이 본문에 섞여 오기 때문이다.

---

## 블록 5 — Ollama 네이티브 경로 (111~179줄)

**왜 이 함수가 존재하는가.** 실측 데이터부터 보자.

| | 소요 | 생성 토큰 | 사고 | 본문 |
|---|---|---|---|---|
| thinking ON | 22.7초 | 1200 (상한 도달) | 4,556자 | **0자** |
| thinking OFF | 9.2초 | 201 | 0자 | 282자 |

Qwen3는 하이브리드 추론 모델이라 기본적으로 `<think>`를 먼저 길게 쓴다.
사고가 토큰 예산을 다 먹으면 **본문이 한 글자도 안 나온다.**
그런데 Ollama의 OpenAI 호환 엔드포인트에는 이걸 끌 파라미터가 없다.
네이티브 `/api/chat`에는 `think`가 있다. 그래서 백엔드가 ollama일 때만 이 경로를 쓴다.

```python
        base = self.s.ollama_base_url.rstrip("/").removesuffix("/v1")
        url = f"{base}/api/chat"
```

설정에는 `http://localhost:11434/v1`이 들어 있다. 네이티브 API는 `/v1`이 아니라
`/api/chat`이므로 `/v1`을 떼어낸다. `removesuffix()`는 파이썬 3.9+ 문법으로,
`rstrip("/v1")`처럼 **문자 단위로 갉아먹지 않고** 접미사 전체만 정확히 제거한다.
(`rstrip`은 문자 집합으로 동작해서 `.../v1v1v`도 다 지워버린다 — 흔한 함정이다.)

```python
        options: dict[str, Any] = {
            "temperature": temperature,
            "num_predict": max_tokens or self.s.max_tokens,
        }
        if seed is not None:
            options["seed"] = seed
        payload = {
            "model": model,
            "messages": [...],
            "stream": False,
            "think": self.s.think,
            "options": options,
            "keep_alive": self.s.keep_alive,
        }
        if json_mode:
            payload["format"] = "json"
```

**OpenAI 호환 API와 이름이 다른 것들** — 이 대응표가 이 함수의 실질 내용이다.

| 개념 | OpenAI 호환 | Ollama 네이티브 |
|---|---|---|
| 생성 토큰 상한 | `max_tokens` (최상위) | `options.num_predict` |
| 온도·시드 | 최상위 | `options` 안 |
| JSON 강제 | `response_format={"type":"json_object"}` | `format="json"` |
| 사고 모드 | (없음) | `think: bool` |
| 모델 상주 | (없음) | `keep_alive: "30m"` |

**`keep_alive`** 는 성능에 직접 영향을 준다. 기본값이 짧으면 호출 사이에 모델이
VRAM에서 내려가고, 다음 호출에서 5.2GB를 다시 올리며 수십 초를 날린다.
디베이트처럼 호출이 수십 번 이어지는 작업에서는 필수다.

### 이중 안전장치 (152~179줄)

```python
        except httpx.HTTPStatusError as e:
            payload.pop("think", None)          # 구버전 Ollama는 think를 모른다
            try:
                r = httpx.post(url, json=payload, timeout=self.s.timeout)
                r.raise_for_status()
            except httpx.HTTPError:
                raise LLMError(...) from e
```

첫 번째 안전장치: `think`를 모르는 Ollama 버전이면 그 키를 빼고 한 번 더 시도한다.

```python
        data = r.json()
        msg = data.get("message", {})
        text = _strip_think(msg.get("content") or "")
        if not text and msg.get("thinking"):
            # 사고에 토큰을 다 써 본문이 비었다 → 사고를 끄고 1회 재시도
            payload["think"] = False
            r = httpx.post(url, json=payload, timeout=self.s.timeout)
            ...
```

두 번째 안전장치가 더 중요하다. 네이티브 API는 `content`(본문)와 `thinking`(사고)을
**분리해서** 준다. 그래서 "본문은 비었는데 사고는 있다" = **잘림을 정확히 탐지**할 수 있다.
이 경우 사고를 끄고 다시 부른다.

> OpenAI 호환 경로에서는 둘이 한 문자열로 섞여 오므로 이 탐지가 불가능하다.
> 네이티브 API를 쓰는 두 번째 이유다.

---

## 블록 6 — `chat_json()`: JSON 강제 (181~197줄)

```python
    def chat_json(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("json_mode", True)
        res = self.chat(**kwargs)
        obj = extract_json(res.text)
        if obj is not None:
            return obj

        repair = dict(kwargs)
        repair["user"] = (
            "다음 텍스트를 유효한 JSON 객체 하나로만 다시 출력하라. 설명·코드펜스 금지.\n\n"
            + res.text[:4000]
        )
        repair["temperature"] = 0.0
        obj = extract_json(self.chat(**repair).text)
        if obj is None:
            raise LLMError(f"JSON 파싱 실패: {res.text[:300]}")
        return obj
```

**3중 방어**

1. `json_mode=True`로 서버에 JSON을 요청 (모델이 형식을 지키도록 유도)
2. 실패 시 `extract_json()`으로 관용 파싱 (블록 7)
3. 그래도 실패 시 **LLM에게 자기 출력을 고쳐달라고 다시 요청** — 온도 0으로

**`repair["temperature"] = 0.0`** 이 포인트다. 교정은 창의성이 필요한 작업이 아니라
형식 변환이다. 온도를 0으로 두면 가장 확정적으로 답한다.

**`dict(kwargs)`로 복사**하는 이유: 원본 `kwargs`를 수정하면 호출부에 영향이 갈 수 있다.
얕은 복사로 격리한다.

**`res.text[:4000]`** — 교정 요청에 원문을 다 넣으면 컨텍스트를 넘길 수 있어 잘라 넣는다.

---

## 블록 7 — `extract_json()`: 깨진 JSON 구조하기 (214~255줄)

LLM은 이런 것들을 준다.

```
① {"적합성": "pass"}                          정상
② ```json\n{"적합성": "pass"}\n```            코드펜스로 감쌈
③ <think>고민…</think>{"적합성":"pass"}        사고 블록이 앞에
④ 판정: {"근거": ["중괄호 } 포함"]} 입니다.     앞뒤에 설명 + 문자열 안에 }
```

네 경우를 모두 처리한다.

```python
def extract_json(text: str) -> dict[str, Any] | None:
    text = _strip_think(text)                       # ③ 처리
    m = _FENCE.search(text)
    candidates = ([m.group(1)] if m else []) + [text]   # ② 펜스 안쪽을 먼저 시도
    for cand in candidates:
        cand = cand.strip()
        try:
            obj = json.loads(cand)                  # ① 통째로 파싱
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
```

**`isinstance(obj, dict)` 검사**를 왜 하나: `json.loads("42")`는 예외 없이
정수 `42`를 준다. 우리가 원하는 건 객체이므로 타입을 확인해야 한다.

### 중괄호 균형 스캔 (④ 처리) — 이 함수의 핵심

정규식으로 `\{.*\}` 를 쓰면 문자열 안의 `}`에서 잘못 끊긴다. 그래서 문자를 하나씩 보며
**따옴표 안인지 아닌지를 추적**한다.

```python
        start = cand.find("{")
        while start != -1:
            depth, in_str, esc = 0, False, False
            for i in range(start, len(cand)):
                ch = cand[i]
                if in_str:                    # 문자열 내부
                    if esc:      esc = False  # 이스케이프된 문자 통과
                    elif ch == "\\": esc = True
                    elif ch == '"':  in_str = False
                    continue                  # ← 문자열 안의 { } 는 세지 않는다
                if   ch == '"': in_str = True
                elif ch == "{": depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:            # 짝이 맞았다
                        try:
                            obj = json.loads(cand[start : i + 1])
                            if isinstance(obj, dict):
                                return obj
                        except json.JSONDecodeError:
                            break             # 이 시작점은 실패 → 다음 { 부터
            start = cand.find("{", start + 1)
        return None
```

상태 변수 3개로 도는 **작은 상태 기계**다.

- `depth`: 중괄호 깊이. 0으로 돌아오면 객체가 닫혔다
- `in_str`: 문자열 안인가. 안이면 `{`, `}`를 세지 않는다
- `esc`: 직전 문자가 `\`였나. `"근거: \"인용\""` 같은 이스케이프 처리

**왜 while 루프로 시작점을 옮기나**: 첫 `{`부터 파싱이 실패하면(예: 앞에
`{설명}` 같은 유사 객체가 있으면) **다음 `{`부터 다시 시도**한다.

이 패턴은 LLM 출력을 다루는 코드에서 계속 재사용할 수 있는 자산이다.
테스트 `test_extract_json_variants`가 위 네 경우를 모두 검증한다.

---

## 블록 8 — `_strip_think()` (205~211줄)

```python
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

def _strip_think(text: str) -> str:
    out = _THINK.sub("", text)
    if "<think>" in out and "</think>" not in out:   # 닫히지 않은 경우
        out = out.split("<think>")[0]
    return out.strip()
```

- **`.*?`** (비탐욕): `<think>A</think>B<think>C</think>` 에서 A만, C만 각각 지운다.
  탐욕(`.*`)이면 A부터 C까지 통째로 지워 B까지 사라진다.
- **`re.DOTALL`**: `.`이 줄바꿈도 포함하게 한다. 사고 블록은 여러 줄이다.
- **닫히지 않은 블록 처리**: 토큰 상한에 걸려 `</think>`가 안 나온 경우다.
  정규식이 못 잡으므로 `<think>` 앞부분만 남긴다.
- **정규식을 모듈 수준에서 컴파일**: 함수 안에서 매번 컴파일하지 않아 반복 호출이 빠르다.

---

## 블록 9 — mock 백엔드 (259~297줄)

```python
def _rand01(*parts: Any) -> float:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF
```

해시를 0~1 실수로 바꾼다. `random`을 쓰지 않는 이유는 **같은 입력에 항상 같은 값**이
나와야 테스트가 안정적이기 때문이다. 해시 앞 8자리(32비트)를 최대값으로 나눈다.

```python
def _mock_reply(role, system, user, seed, temperature) -> str:
    key = hashlib.sha256((system + user).encode()).hexdigest()[:12]
    jitter = (_rand01(key, seed) - 0.5) * 2 * (20 * temperature)
    base = 25 + _rand01(key) * 60
    score = max(0, min(100, int(round(base + jitter))))
```

**이 4줄의 설계 의도**

- `base = 25 + rand01(key) × 60` → 프롬프트 내용에 따라 25~85점 사이의 고정 점수.
  **프롬프트가 같으면 점수도 같다.**
- `jitter = (rand01(key, seed) − 0.5) × 2 × (20 × temperature)`
  → 시드에 따라 `±20×temperature` 범위로 흔들린다.
  온도 0.8이면 ±16점, 온도 0.2면 ±4점.
- 결과: **온도를 올리면 답이 더 흔들린다** — 실제 LLM의 성질을 모사한다.

이게 왜 중요한가. 신뢰도 스코어는 "여러 번 돌렸을 때 답이 얼마나 일관적인가"를 재는데,
mock이 항상 똑같은 답을 주면 **신뢰도 로직을 검증할 수 없다.**
온도에 반응하는 흔들림을 넣어야 `test_confidence_drops_when_runs_disagree` 같은
테스트가 의미를 갖는다.

```python
    if role in {"judge", "single"}:
        verdict = "pass" if score >= 60 else ("warn" if score >= 40 else "fail")
        return json.dumps({"적합성": verdict, "가입의향점수": score, "근거": [...], ...},
                          ensure_ascii=False)
    if role == "persona":
        return json.dumps({"발화": "…", "가입의향점수": score, "망설임요인": [...]},
                          ensure_ascii=False)
    label = "옹호자" if role == "advocate" else "회의론자"
    return f"[MOCK/{label}] 근거: … (score_hint={score})"
```

역할별로 **실제 프롬프트가 요구하는 형식과 똑같은 모양**을 반환한다.
그래서 파싱 코드·집계 코드가 mock에서도 진짜처럼 동작한다.

- `ensure_ascii=False`: 한글이 `\uXXXX`로 이스케이프되지 않게 한다
- 모든 문자열에 `[MOCK]` 표시: 리포트에 섞여도 가짜임을 알 수 있다
- 옹호자·회의론자는 자유 텍스트 (실제 프롬프트도 불릿 텍스트를 요구한다)

---

## 이 파일에서 가져갈 것 5가지

1. **`*` 키워드 전용 인자** — 같은 타입 인자가 여러 개면 순서 실수를 문법으로 막는다
2. **예외를 종류별로 나누고, 메시지에 해결책을 쓴다** — 재시도는 1회만
3. **깨진 JSON은 상태 기계로 잘라낸다** — 정규식만으론 문자열 안의 `}`를 못 피한다
4. **해시를 난수 대신 쓰면 재현 가능한 가짜 데이터가 된다**
5. **가짜 데이터에는 항상 표시를 남긴다** (`mock::`, `[MOCK]`)

---

## 실습

1. **JSON 파서 깨보기**
   ```bash
   uv run python -c "
   from fdm.llm import extract_json
   print(extract_json('앞말 {\"a\": \"중괄호 } 포함\"} 뒷말'))
   print(extract_json('<think>고민</think>{\"b\": 1}'))
   print(extract_json('그냥 텍스트'))
   "
   ```

2. **mock의 온도 반응 확인** — 온도를 바꾸면 점수 분산이 커지는지 본다.
   ```bash
   uv run python -c "
   from fdm.llm import _mock_reply
   import json
   for t in (0.1, 0.9):
       s = [json.loads(_mock_reply('judge','sys','usr',i,t))['가입의향점수'] for i in range(8)]
       print(f'temperature={t}: {s}')
   "
   ```

3. **네이티브 vs 호환 경로 비교** — `FDM_BACKEND=vllm`으로 두고
   `FDM_VLLM_BASE_URL=http://localhost:11434/v1`을 주면 Ollama를 OpenAI 호환 경로로 부를 수 있다.
   사고 모드가 안 꺼져 느려지는 것을 직접 확인해보라.

---

**이전** ← [01_config.md](01_config.md) | **다음** → `03_personas.md` (예정)

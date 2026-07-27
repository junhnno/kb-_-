# 01. `src/fdm/config.py` — 설정과 경로 (64줄)

가장 먼저 읽어야 하는 파일이다. 다른 모든 모듈이 여기서 경로와 설정을 가져간다.
**이 파일에 새 설정을 추가하는 법**을 익히면 프로젝트 전체를 조작할 수 있다.

---

## 블록 1 — .env 읽기 (1~14줄)

```python
"""전역 설정. 환경변수(.env) 기반."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv은 선택 의존성처럼 다룬다
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass
```

**`from __future__ import annotations`**
타입힌트를 문자열로 미루는 선언이다. 이게 있으면 `int | None`, `list[str]` 같은
신문법을 구버전 파이썬에서도 쓸 수 있고, 순환 import도 피하기 쉬워진다.
이 프로젝트 모든 파일 첫 줄에 있다.

**`load_dotenv()`**
프로젝트 루트의 `.env` 파일을 읽어 `os.environ`에 올린다. 그래서 우리가
`.env`에 `FDM_BACKEND=ollama`라고 쓰면 코드에서 환경변수처럼 읽힌다.

**왜 `try/except`로 감쌌나**
`python-dotenv`가 없어도 프로그램이 죽지 않게 하려는 것이다. 없으면 그냥
`.env`를 못 읽을 뿐, 터미널에서 직접 준 환경변수(`FDM_BACKEND=mock uv run …`)로는 여전히 동작한다.
**의존성 하나가 없다고 전체가 멈추지 않게 만드는 패턴**이다. 이 프로젝트에서
`datasets`, `sentence-transformers`도 같은 방식으로 다룬다.

---

## 블록 2 — 경로 상수 (16~22줄)

```python
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PERSONA_DIR = DATA_DIR / "personas"
PRODUCT_DIR = DATA_DIR / "products"
RAG_DIR = DATA_DIR / "rag"
BENCHMARK_DIR = DATA_DIR / "benchmark"
OUTPUT_DIR = ROOT / "outputs"
```

**`Path(__file__).resolve().parents[2]` 를 손으로 따라가 보자**

```
__file__                     = …/금융 디베이트 모델/src/fdm/config.py
.resolve()                   = 절대경로로 변환 (심볼릭 링크도 해석)
.parents[0]                  = …/src/fdm
.parents[1]                  = …/src
.parents[2]                  = …/금융 디베이트 모델      ← 프로젝트 루트
```

**왜 이렇게 하나**
`"data/products"` 같은 **상대경로를 쓰면 실행 위치에 따라 깨진다.** 터미널에서
`cd src` 후 실행하거나, Streamlit이 다른 디렉터리에서 스크립트를 띄우거나,
pytest가 루트에서 도는 상황이 모두 다르다.
`__file__` 기준으로 잡으면 **어디서 실행해도 같은 곳을 가리킨다.**

**`/` 연산자**
`pathlib`은 `/`로 경로를 잇는다. `DATA_DIR / "personas"`는 Windows에서
`data\personas`, Linux에서 `data/personas`로 알아서 바뀐다. 문자열 `+`로
경로를 잇지 말아야 하는 이유다.

---

## 블록 3 — 환경변수 헬퍼 (25~27줄)

```python
def _env(key: str, default: str) -> str:
    v = os.environ.get(key)
    return v if v not in (None, "") else default
```

`os.environ.get(key, default)`를 그냥 쓰면 될 것 같지만, 그건 **빈 문자열을 값으로 인정한다.**

```python
os.environ["FDM_MODEL_SMALL"] = ""        # .env에 "FDM_MODEL_SMALL=" 로 쓴 경우
os.environ.get("FDM_MODEL_SMALL", "qwen3:8b")   # → ""      빈 모델명! 버그
_env("FDM_MODEL_SMALL", "qwen3:8b")             # → "qwen3:8b"  안전
```

`.env` 파일에서 값을 지우면 `KEY=` 형태로 남는 일이 흔하다. 그래서 이 한 줄이 필요하다.
**함수 이름 앞의 `_`** 는 "이 모듈 내부용"이라는 파이썬 관례다(강제되진 않는다).

---

## 블록 4 — Settings 데이터클래스 (30~50줄)

```python
@dataclass
class Settings:
    backend: str = field(default_factory=lambda: _env("FDM_BACKEND", "mock"))
    ollama_base_url: str = field(
        default_factory=lambda: _env("FDM_OLLAMA_BASE_URL", "http://localhost:11434/v1")
    )
    vllm_base_url: str = field(
        default_factory=lambda: _env("FDM_VLLM_BASE_URL", "http://localhost:8000/v1")
    )
    model_small: str = field(default_factory=lambda: _env("FDM_MODEL_SMALL", "qwen3:8b"))
    model_judge: str = field(default_factory=lambda: _env("FDM_MODEL_JUDGE", "qwen3:8b"))
    timeout: float = field(default_factory=lambda: float(_env("FDM_TIMEOUT", "180")))
    max_tokens: int = field(default_factory=lambda: int(_env("FDM_MAX_TOKENS", "1200")))
    think: bool = field(default_factory=lambda: _env("FDM_THINK", "0") not in ("0", "false", "False"))
    keep_alive: str = field(default_factory=lambda: _env("FDM_KEEP_ALIVE", "30m"))
```

### `field(default_factory=lambda: …)` 가 왜 필요한가 — 중요한 함정

이렇게 쓰면 안 된다:

```python
@dataclass
class Settings:
    backend: str = _env("FDM_BACKEND", "mock")     # ✗ 나쁜 코드
```

기본값이 **클래스가 정의되는 순간(= import 시점) 한 번만** 평가되기 때문이다.
테스트에서 환경변수를 바꾸고 `Settings()`를 새로 만들어도 옛 값이 그대로 남는다.

`default_factory`는 **인스턴스를 만들 때마다** 함수를 호출한다.

```python
os.environ["FDM_BACKEND"] = "ollama"
Settings().backend      # → "ollama"   매번 새로 읽는다
```

가변 기본값(`list`, `dict`)에 `default_factory`가 필수인 건 잘 알려져 있지만,
**"지금 읽어야 하는 값"에도 필요하다**는 게 여기서 배울 점이다.

### 각 설정의 의미

| 필드 | 기본값 | 의미 |
|---|---|---|
| `backend` | `mock` | `mock`(LLM 없음) / `ollama`(로컬) / `vllm`(Colab) |
| `ollama_base_url` | `localhost:11434/v1` | `/v1`이 붙은 OpenAI 호환 주소. 네이티브 호출 시 `llm.py`가 이 `/v1`을 떼어낸다 |
| `model_small` | `qwen3:8b` | 옹호자·회의론자·페르소나용 |
| `model_judge` | `qwen3:8b` | 심판용. Colab에서 EXAONE 32B로 올린다 |
| `timeout` | 180초 | 한 번의 LLM 호출 제한. 넘으면 `LLMError` |
| `max_tokens` | 1200 | 생성 토큰 상한 |
| `think` | **끔** | 하이브리드 추론 모델의 사고 모드 |
| `keep_alive` | `30m` | 모델을 VRAM에 30분 상주 |

**`think` 파싱을 보라**

```python
_env("FDM_THINK", "0") not in ("0", "false", "False")
```

환경변수는 **항상 문자열**이다. `bool("0")`은 `True`라서 `bool(_env(...))`로 쓰면
끄려고 `0`을 넣어도 켜진다. 자주 나오는 버그라 명시적으로 문자열 목록과 비교했다.

---

## 블록 5 — `base_url` 프로퍼티 (48~50줄)

```python
    @property
    def base_url(self) -> str:
        return self.vllm_base_url if self.backend == "vllm" else self.ollama_base_url
```

백엔드에 따라 주소를 고르는 계산 속성이다. 호출부는 `settings.base_url`만 보면 되고
분기를 알 필요가 없다. `@property`는 **괄호 없이 접근하는 메서드**다
(`settings.base_url` ✓, `settings.base_url()` ✗).

---

## 블록 6 — 싱글턴과 역할 매핑 (53~64줄)

```python
SETTINGS = Settings()

# 디베이트 역할 → 모델 배치 (비대칭 배치 전략)
ROLE_MODEL = {
    "advocate": "small",
    "skeptic": "small",
    "persona": "small",
    "judge": "judge",
    "single": "judge",  # 애블레이션의 단발 질문(디베이트 없음)도 심판 모델로 공정 비교
}

OUTPUT_DIR.mkdir(exist_ok=True)
```

**`SETTINGS = Settings()`** — 모듈 수준 인스턴스 하나를 만들어 전역 공유한다.
파이썬 모듈은 처음 import될 때 한 번만 실행되므로 이게 사실상 싱글턴이다.

테스트는 이 인스턴스의 필드를 갈아끼워 백엔드를 바꾼다:

```python
# tests/test_pipeline.py
@pytest.fixture(autouse=True)
def _mock_backend():
    old = SETTINGS.backend
    SETTINGS.backend = "mock"      # 전역 설정을 임시 변경
    yield
    SETTINGS.backend = old         # 원복
```

**`ROLE_MODEL` — 이 프로젝트 전략의 핵심 5줄**

디베이트 5턴 중 4턴은 작은 모델, 심판만 큰 모델을 쓴다. 최종 판정 품질이
결과를 좌우하므로 비용을 심판에 집중시키는 것이다.

가장 중요한 줄은 `"single": "judge"`다. `single`은 애블레이션의 비교군
(디베이트 없이 1회 판정)인데, 여기에 **작은 모델을 쓰면 비교가 오염된다.**
"디베이트가 더 정확했다"는 결론이 실은 "심판 모델이 더 컸다"는 뜻이 되어버린다.
비교군과 실험군의 모델을 같게 맞춰야 **방식의 차이만** 남는다.

**`OUTPUT_DIR.mkdir(exist_ok=True)`** — import 시 결과 폴더를 만든다.
`exist_ok=True`가 없으면 두 번째 실행에서 `FileExistsError`가 난다.

---

## 실습

1. **경로 확인**
   ```bash
   uv run python -c "from fdm.config import ROOT, DATA_DIR; print(ROOT); print(DATA_DIR)"
   ```

2. **설정 덮어쓰기 실험** — 환경변수가 이긴다는 걸 확인한다.
   ```bash
   FDM_MODEL_JUDGE=qwen3:4b uv run fdm doctor
   ```
   출력의 `심판모델=`이 바뀐다.

3. **새 설정 추가 연습** — `Settings`에 아래를 추가하고 `cli.py:doctor`에서 출력해보라.
   ```python
   k_docs: int = field(default_factory=lambda: int(_env("FDM_K_DOCS", "6")))
   ```
   그 다음 `agents/debate.py`의 `DebateConfig.k_docs` 기본값을 이 설정에서 읽게 바꿔보라.
   (지금은 `DebateConfig`에 하드코딩된 `6`이다.)

---

**다음** → [02_llm.md](02_llm.md) — 모든 LLM 호출이 지나가는 관문

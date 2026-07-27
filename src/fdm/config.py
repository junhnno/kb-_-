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

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PERSONA_DIR = DATA_DIR / "personas"
PRODUCT_DIR = DATA_DIR / "products"
RAG_DIR = DATA_DIR / "rag"
BENCHMARK_DIR = DATA_DIR / "benchmark"
OUTPUT_DIR = ROOT / "outputs"


def _env(key: str, default: str) -> str:
    v = os.environ.get(key)
    return v if v not in (None, "") else default


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
    # Qwen3·EXAONE 등 하이브리드 추론 모델의 사고 모드. 켜면 2~3배 느리고
    # 토큰 예산을 사고가 다 써서 본문이 잘리는 일이 생긴다. 기본은 끔.
    think: bool = field(default_factory=lambda: _env("FDM_THINK", "0") not in ("0", "false", "False"))
    keep_alive: str = field(default_factory=lambda: _env("FDM_KEEP_ALIVE", "30m"))

    @property
    def base_url(self) -> str:
        return self.vllm_base_url if self.backend == "vllm" else self.ollama_base_url


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

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
    # 컨텍스트 창. Ollama 기본값 4096은 부족하다 — 디베이트 심판 프롬프트가
    # (상품+페르소나+사실팩+근거 6건+4턴 전문) 실측 3,434토큰이고 생성 1,200을
    # 더하면 4,634라 초과한다. 초과하면 앞부분(상품 정의)이 잘리고 재계산으로 느려진다.
    #
    # 6144로 둔 이유: 최장 호출(약 5,000토큰)을 여유 있게 담으면서 KV 캐시가
    # 8GB VRAM에서 CPU 오프로드를 유발하지 않는 선이다(4096 대비 +약 300MB).
    # VRAM이 큰 환경(Colab 등)에서는 FDM_NUM_CTX=8192 이상으로 올려도 된다.
    num_ctx: int = field(default_factory=lambda: int(_env("FDM_NUM_CTX", "6144")))

    # 토론자(옹호자·회의론자·페르소나) 온도.
    # 0.8이 기본이었으나, 시드를 바꿔 재측정하니 디베이트의 우려 recall이 13.6%p 흔들렸다
    # (단발 계열은 1%p 이내). 토론 3턴이 매번 다른 논점을 내고 심판이 무엇을 채택할지가
    # 달라지기 때문이다. 다양성과 안정성의 교환이므로 값을 바꿔가며 측정할 수 있게 열어 둔다.
    temp_debater: float = field(default_factory=lambda: float(_env("FDM_TEMP_DEBATER", "0.5")))
    # 심판 온도. 낮게 유지한다 — 이 값 덕분에 라벨은 시드에 무관하게 결정론적이었다.
    temp_judge: float = field(default_factory=lambda: float(_env("FDM_TEMP_JUDGE", "0.2")))
    # 앵커 없는 우려를 심각도 캡이 아니라 아예 제거할지. 기본 False(기존 동작 유지).
    # 오탐(precision) 개선 실험용 스위치 — agents.schema.drop_unanchored 참고.
    drop_unanchored: bool = field(
        default_factory=lambda: _env("FDM_DROP_UNANCHORED", "0") not in ("0", "false", "False")
    )
    # 조정례 문서 제목의 `— 판정 {label}`을 감출지. 기본 False(기존 동작 유지).
    #
    # 코퍼스 조정례 12건 중 10건(83.3%)이 fail/warn이고, 그 라벨이 제목에 박힌 채
    # 프롬프트에 인용된다. 깨끗한 상품을 심사할 때도 "판정 fail" 문서 3건이 근거로
    # 붙으면 모델이 그 사전확률에 끌려간다(실측: 과잉경고 상품당 2.92개).
    #
    # 조정례를 통째로 빼는 lawonly arm은 오히려 나빴다(적중률 77.3% → 59.1%) —
    # 판단요지 본문 자체는 유용하다는 뜻이다. 그래서 **본문은 두고 라벨만 가리는**
    # 중간안을 따로 잰다. 이 스위치가 그 실험용이다.
    case_hide_label: bool = field(
        default_factory=lambda: _env("FDM_CASE_HIDE_LABEL", "0") not in ("0", "false", "False")
    )
    # 앵커가 실재하는 조항·문서·계산값을 가리키는지 검증하고, 지어낸 인용은 심각도를
    # 낮출지. 기본 False(기존 동작 유지). anchors.py 참고.
    verify_anchors: bool = field(
        default_factory=lambda: _env("FDM_VERIFY_ANCHORS", "0") not in ("0", "false", "False")
    )
    # 판매 정황이 "설명했다"고 명시한 항목의 '설명 부족' 주장을 기각할지.
    # 기본 False(기존 동작 유지). situation.py 참고.
    screen_situation: bool = field(
        default_factory=lambda: _env("FDM_SCREEN_SITUATION", "0") not in ("0", "false", "False")
    )
    # 회의론자 주장을 심판이 보기 **전에** 코드로 검증해, 기각 사유를 심판 입력에 주입할지.
    # 사후 필터(screen_situation 등)와 달리 **라벨을 바꿀 수 있는** 유일한 개입이다.
    # 기본 False. claim_check.py 참고.
    verify_claims: bool = field(
        default_factory=lambda: _env("FDM_VERIFY_CLAIMS", "0") not in ("0", "false", "False")
    )
    # v2 개입: 성립하지 않는 우려 유형을 **회의론자가 말하기 전에** 차단한다.
    # 사후 기각은 대화록에 허위 우려를 남겨 심판이 anchoring되지만, 이건 원천 차단이다.
    # verify_claims와 함께 켜는 것을 전제로 한다.
    skeptic_guard: bool = field(
        default_factory=lambda: _env("FDM_SKEPTIC_GUARD", "0") not in ("0", "false", "False")
    )

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

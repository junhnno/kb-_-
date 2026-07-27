"""OpenAI 호환 chat completion 클라이언트.

백엔드 3종:
  - ollama : 로컬 개발 (qwen3:8b 등)
  - vllm   : Colab 데모 (EXAONE 4.0 32B 등)
  - mock   : LLM 없이 파이프라인 배관만 확인하는 결정론적 스텁
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

import httpx

from .config import ROLE_MODEL, SETTINGS, Settings


@dataclass
class LLMResponse:
    text: str
    model: str
    raw: dict[str, Any] | None = None


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or SETTINGS

    # ------------------------------------------------------------------ public
    def model_for(self, role: str) -> str:
        tier = ROLE_MODEL.get(role, "small")
        return self.s.model_judge if tier == "judge" else self.s.model_small

    def chat(
        self,
        *,
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
            return self._chat_ollama_native(
                model=model,
                system=system,
                user=user,
                temperature=temperature,
                seed=seed,
                json_mode=json_mode,
                max_tokens=max_tokens,
            )
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
            # vLLM: Qwen3·EXAONE 등 하이브리드 추론 모델의 사고 모드를 끈다.
            # 서버가 이 키를 모르면 무시하거나 400을 내므로 아래에서 한 번 재시도한다.
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        url = f"{self.s.base_url.rstrip('/')}/chat/completions"
        try:
            r = httpx.post(url, json=payload, timeout=self.s.timeout)
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            # 일부 서버는 response_format / chat_template_kwargs를 거부한다 → 한 번만 재시도
            if json_mode or "chat_template_kwargs" in payload:
                payload.pop("response_format", None)
                payload.pop("chat_template_kwargs", None)
                r = httpx.post(url, json=payload, timeout=self.s.timeout)
                r.raise_for_status()
            else:
                raise LLMError(f"{url} 호출 실패: {e} / {e.response.text[:300]}") from e
        except httpx.HTTPError as e:
            raise LLMError(
                f"{url} 연결 실패: {e}. 백엔드가 떠 있는지 확인하세요 "
                f"(ollama serve / vllm serve). 또는 FDM_BACKEND=mock 사용."
            ) from e

        data = r.json()
        text = data["choices"][0]["message"]["content"] or ""
        return LLMResponse(text=_strip_think(text), model=model, raw=data)

    def _chat_ollama_native(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float,
        seed: int | None,
        json_mode: bool,
        max_tokens: int | None,
    ) -> LLMResponse:
        """Ollama 네이티브 /api/chat.

        OpenAI 호환 엔드포인트에는 `think` 파라미터가 없어서 Qwen3 같은 하이브리드
        추론 모델의 사고 모드를 끌 수 없다. 사고 모드가 켜져 있으면 토큰 예산을
        <think>가 다 써버려 본문이 빈 문자열로 잘리는 일이 생긴다(측정: 1200토큰 중
        본문 0자). 네이티브 API는 think를 끌 수 있고, 켜더라도 thinking과 content를
        분리해 돌려주므로 잘림을 감지할 수 있다.
        """
        base = self.s.ollama_base_url.rstrip("/").removesuffix("/v1")
        url = f"{base}/api/chat"
        options: dict[str, Any] = {
            "temperature": temperature,
            "num_predict": max_tokens or self.s.max_tokens,
        }
        if seed is not None:
            options["seed"] = seed
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": self.s.think,
            "options": options,
            "keep_alive": self.s.keep_alive,  # 매 호출마다 모델을 다시 올리지 않는다
        }
        if json_mode:
            payload["format"] = "json"

        try:
            r = httpx.post(url, json=payload, timeout=self.s.timeout)
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            # think를 지원하지 않는 구버전 Ollama → 파라미터를 빼고 한 번 재시도
            payload.pop("think", None)
            try:
                r = httpx.post(url, json=payload, timeout=self.s.timeout)
                r.raise_for_status()
            except httpx.HTTPError:
                raise LLMError(f"{url} 호출 실패: {e} / {e.response.text[:300]}") from e
        except httpx.HTTPError as e:
            raise LLMError(
                f"{url} 연결 실패: {e}. `ollama serve`가 떠 있는지 확인하세요. "
                f"또는 FDM_BACKEND=mock 사용."
            ) from e

        data = r.json()
        msg = data.get("message", {})
        text = _strip_think(msg.get("content") or "")
        if not text and msg.get("thinking"):
            # 사고에 토큰을 다 써 본문이 비었다 → 사고를 끄고 1회 재시도
            payload["think"] = False
            r = httpx.post(url, json=payload, timeout=self.s.timeout)
            r.raise_for_status()
            data = r.json()
            text = _strip_think(data.get("message", {}).get("content") or "")
        return LLMResponse(text=text, model=model, raw=data)

    def chat_json(self, **kwargs: Any) -> dict[str, Any]:
        """JSON 객체를 강제 파싱. 실패 시 1회 교정 재시도."""
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


# ---------------------------------------------------------------- json helpers
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _strip_think(text: str) -> str:
    """Qwen3 등 reasoning 모델의 <think> 블록 제거."""
    out = _THINK.sub("", text)
    # 닫히지 않은 think 블록
    if "<think>" in out and "</think>" not in out:
        out = out.split("<think>")[0]
    return out.strip()


def extract_json(text: str) -> dict[str, Any] | None:
    """자유 텍스트 안의 첫 JSON 객체를 관용적으로 추출."""
    text = _strip_think(text)
    m = _FENCE.search(text)
    candidates = ([m.group(1)] if m else []) + [text]
    for cand in candidates:
        cand = cand.strip()
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        # 중괄호 균형 스캔
        start = cand.find("{")
        while start != -1:
            depth, in_str, esc = 0, False, False
            for i in range(start, len(cand)):
                ch = cand[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(cand[start : i + 1])
                            if isinstance(obj, dict):
                                return obj
                        except json.JSONDecodeError:
                            break
            start = cand.find("{", start + 1)
    return None


# ----------------------------------------------------------------- mock backend
def _rand01(*parts: Any) -> float:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _mock_reply(role: str, system: str, user: str, seed: int | None, temperature: float) -> str:
    """결정론적 스텁. 프롬프트 해시로 값을 만들되 seed/temperature에 따라 흔들려
    멀티시드 신뢰도 계산 로직까지 검증 가능하게 한다."""
    key = hashlib.sha256((system + user).encode()).hexdigest()[:12]
    jitter = (_rand01(key, seed) - 0.5) * 2 * (20 * temperature)
    base = 25 + _rand01(key) * 60
    score = max(0, min(100, int(round(base + jitter))))

    if role in {"judge", "single"}:
        verdict = "pass" if score >= 60 else ("warn" if score >= 40 else "fail")
        return json.dumps(
            {
                "적합성": verdict,
                "가입의향점수": score,
                "근거": [
                    f"[MOCK] 금소법 제17조(적합성원칙) 대조 결과 score={score}",
                    "[MOCK] 페르소나 소득·부채 대비 납입부담 검토",
                ],
                "위험요인": ["[MOCK] 우대금리 조건 미달 가능성", "[MOCK] 중도해지 시 약정금리 미적용"],
                "요약": "[MOCK] 스텁 판정입니다. 실제 LLM 백엔드로 교체하세요.",
            },
            ensure_ascii=False,
        )
    if role == "persona":
        return json.dumps(
            {
                "발화": "[MOCK] 조건은 괜찮아 보이는데 우대조건을 다 채울 수 있을지 걱정됩니다.",
                "가입의향점수": score,
                "망설임요인": ["[MOCK] 자동이체 유지 부담"],
            },
            ensure_ascii=False,
        )
    label = "옹호자" if role == "advocate" else "회의론자"
    return f"[MOCK/{label}] 근거: 상품 약관 및 페르소나 재무지표 기반 주장 (score_hint={score})"

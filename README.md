# 합성 페르소나 기반 금융상품 설계·검증 에이전트 (PoC)

신규 금융상품(적금·예금·대출·카드)을 **출시 전에** 한국 인구분포를 반영한 합성 고객 페르소나 상대로
시뮬레이션해 가입 반응·적합성·세그먼트별 리스크를 사전 검증하는 도구다.

- 대상: 금융사 상품기획팀 (B2B). **개인 고객 상품 추천이 아니다.**
- 차별점: 적대적 3진영 디베이트로 LLM 순응 편향·환각을 억제하고, **멀티시드 신뢰도 스코어**와
  **외부 벤치마크 대조(조정례·KOSIS)** 로 정확성을 실증한다.

> 📘 코드가 어떻게 돌아가는지 파일·함수 단위로 따라 읽는 해설: **[docs/WALKTHROUGH.md](docs/WALKTHROUGH.md)**

## 1. 빠른 시작

```bash
uv sync --extra ui
```

LLM 없이 파이프라인 배관만 확인 (결정론적 mock 백엔드):

```bash
FDM_BACKEND=mock uv run fdm doctor
```

로컬 Qwen3 8B로 실행:

```bash
ollama pull qwen3:8b
```

```bash
cp .env.example .env
```

`.env`에서 `FDM_BACKEND=ollama` 확인 후:

```bash
uv run fdm doctor
```

## 2. 주요 명령

```bash
uv run fdm products
```

```bash
uv run fdm segments
```

```bash
uv run fdm rag "우대금리 최고금리 광고 오인"
```

```bash
uv run fdm debate 01_youth_step_saving --full
```

```bash
uv run fdm simulate 01_youth_step_saving --seeds 3 --personas-per-segment 4 --with-sensitivity --with-ablation
```

```bash
uv run fdm ablation --seeds 2
```

대시보드:

```bash
uv run python -m streamlit run ui/app.py
```

(`uv run streamlit ...` 형태는 Windows 앱 제어 정책이 `streamlit.exe` 실행을 차단하는 환경이 있어
`python -m streamlit` 로 띄우는 쪽이 안전하다.)

결과물은 `outputs/`에 저장된다 (`sim_*.json`, `ablation.json`, `report_*.md`).

## 3. 파이프라인

```
상품 정의(JSON) ─┐
                 ├→ 세그먼트 코호트 추출 → 재무 프로파일 부여 → [3진영 디베이트 + RAG 근거]
페르소나 풀 ─────┘        │
                          ├→ 멀티시드 N회 반복 → 합의도 = 신뢰도 스코어 → 저신뢰 플래그
                          ├→ 외부 벤치마크 대조(조정례 정답셋 / KOSIS 보유율) → 적중률
                          ├→ 가격·조건 민감도 분석
                          └→ 세그먼트 히트맵 + 검증 리포트(근거·신뢰도 표시)
```

### 디베이트 5단계 (`src/fdm/agents/debate.py`)

1. **옹호자** — 상품의 적합성 주장 (약관 조항 인용 필수)
2. **페르소나** 1차 반응 (JSON: 발화·가입의향점수·망설임요인)
3. **회의론자/취약고객 대변인** — 숨은 비용·조건 미달·손실 위험 반박 (옹호자 주장 직접 인용)
4. **페르소나** 재반응 (양가감정 허용)
5. **심판** — 금소법 6대 판매원칙 기준 최종 판정 JSON

모든 발화는 근거 인용이 필수이고, 인용이 없으면 1회 재요청한다(`enforce_grounding`).
무근거 발화 수는 신뢰도 점수에서 감점된다.

### 성능 — 느리다면 여기부터

RTX 5070 Laptop 8GB + `qwen3:8b` 실측 기준이다.

| 조치 | 효과 | 방법 |
|---|---|---|
| **사고 모드 끄기** | 디베이트 1건 99초 → **47초** | 기본값(`FDM_THINK=0`). 켜면 느릴 뿐 아니라 토큰 예산을 `<think>`가 다 써서 본문이 빈 응답으로 잘린다 |
| **동시 실행** | 4동시에 **2.04배** 처리량 | `--workers 4` |
| 모델 상주 | 재적재 대기 제거 | `FDM_KEEP_ALIVE=30m` (기본값) |
| 토큰 상한 축소 | 긴 답변 잘림과 맞바꿈 | `FDM_MAX_TOKENS=600` |
| 토론자만 작은 모델 | 토론 턴 4/5가 빨라짐 | `ollama pull qwen3:4b` 후 `FDM_MODEL_SMALL=qwen3:4b` |

디베이트 1건 = LLM 호출 5회(+근거 미인용 시 재요청)이므로,
`세그먼트 수 × 페르소나 수 × 시드 수` 만큼 곱해진다. 처음에는
`--seeds 2 --personas-per-segment 2 --workers 4` 로 시작하라.

`ollama` 백엔드는 OpenAI 호환 엔드포인트 대신 네이티브 `/api/chat` 을 쓴다.
호환 엔드포인트에는 `think` 파라미터가 없어 사고 모드를 끌 수 없기 때문이다.
vLLM 백엔드에서는 `chat_template_kwargs={"enable_thinking": false}` 로 같은 일을 한다.

### 비대칭 모델 배치

| 역할 | 모델 티어 | 환경변수 |
|---|---|---|
| 옹호자·회의론자·페르소나 | 작은 모델 | `FDM_MODEL_SMALL` (기본 `qwen3:8b`) |
| 심판·단발판정 | 큰 모델 | `FDM_MODEL_JUDGE` (Colab에서 EXAONE 4.0 32B) |

### 신뢰도 스코어 (`src/fdm/eval/confidence.py`)

```
confidence = 0.5 × 라벨합의도 + 0.3 × 점수안정성(1 − σ/25) + 0.2 × 심판자기신뢰
             × (1 − 0.04 × 무근거발화수, 최대 20% 감점)
```

`high ≥ 0.80`, `low < 0.55` → low는 리포트에 **"추가 검증 필요"** 로 플래그된다.

### 애블레이션 (`src/fdm/eval/benchmark.py`)

3개 arm을 같은 정답셋으로 대조한다.

| arm | 설명 |
|---|---|
| `single_norag` | 단발 질문, 근거 검색 없음 |
| `single` | 단발 질문 + RAG |
| `debate` | 3진영 디베이트 + RAG |

지표: 적합성 3분류 적중률, 위험탐지(warn·fail 대 pass) 정확도, macro F1, 위반원칙 재현율, LLM 호출 수(비용).
평가 시 해당 사례 문서는 검색에서 제외해 **정답 누출을 막는다**.

## 4. 데이터

| 경로 | 내용 | 상태 |
|---|---|---|
| `data/personas/*.jsonl` | Nemotron-Personas-Korea 캐시 | 없으면 HF 다운로드 → 실패 시 합성 폴백 |
| `data/products/*.json` | 예시 신상품 5종 | 가상 상품 (금융상품 한눈에 API 필드 참고) |
| `data/segments.json` | 타깃 세그먼트 9종 | — |
| `data/rag/laws/*.jsonl` | 금소법 6대 판매원칙 + 감독 가이드 | 조문 요약 (law.go.kr 원문으로 교체 권장) |
| `data/benchmark/dispute_cases.json` | 적합성 정답셋 12건 | **가공 샘플** — 실제 조정결정례로 교체 필요 |
| `data/benchmark/segment_holding_rates.json` | 세그먼트×상품군 보유율 | 근사치 — KOSIS API로 교체 필요 |
| `data/benchmark/kosis_household_finance.json` | 연령대별 소득·자산·부채 분포 | 근사치 |

실제 페르소나 데이터를 쓰려면:

```bash
uv sync --extra personas
```

`load_personas(source="hf")`가 `nvidia/Nemotron-Personas-Korea`를 내려받는다. 내려받은 뒤
`data/personas/nemotron.jsonl`로 저장해두면 이후 오프라인으로 재현된다.

## 5. Colab 전환

`notebooks/colab_vllm.ipynb` — vLLM으로 큰 모델(EXAONE 4.0 32B 등)을 띄우고
`FDM_BACKEND=vllm`으로 같은 코드를 그대로 실행한다.

## 6. 한계 (정직한 고지)

- 합성 페르소나는 개별 변수 분포는 실제와 정합하나 **변수 조합(joint distribution) 정합성은 검증되지 않았다.**
  결과는 **탐색·경보용**이며 확정 근거는 실데이터·조정례를 병용해야 한다.
- 디베이트는 LLM의 내적 충실성을 높이지만 **외적 타당성(실제 시장 일치)을 자동 보장하지 않는다.**
- 본 도구는 세그먼트 설계·검증용이며, 개별 소비자에 대한 **판매권유·투자자문이 아니다.**
- `data/benchmark/dispute_cases.json`의 사례는 실제 조정결정례 원문이 아닌 재구성 샘플이다.
  제출·평가 단계에서는 fss.or.kr 원문으로 교체해야 하며, 그 전까지의 적중률 수치는 잠정치다.

## 7. 구조

```
src/fdm/
  config.py            설정·경로·역할별 모델 배치
  llm.py               OpenAI 호환 클라이언트 (ollama|vllm|mock), JSON 관용 파서
  personas/            스키마 · Nemotron 로더/세그먼트 필터 · KOSIS 재무 프로파일 부여
  products/            상품 정의 스키마 (금융상품 한눈에 필드 참고)
  rag/                 코퍼스(금소법·조정례) · BM25(+선택 bge-m3) 검색기
  agents/              프롬프트 · 3진영 디베이트 오케스트레이터 · 판정 스키마
  eval/                신뢰도 스코어 · 세그먼트 시뮬레이션/민감도 · 벤치마크·애블레이션
  report.py            Markdown 검증 리포트 생성
  cli.py               CLI
ui/app.py              Streamlit 대시보드 (히트맵·신뢰도·근거·애블레이션)
tests/                 파이프라인 스모크 테스트 (mock 백엔드)
```

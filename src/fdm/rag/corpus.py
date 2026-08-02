"""RAG 코퍼스: 금소법 조문 + (가공) 분쟁조정 사례."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

from ..config import BENCHMARK_DIR, RAG_DIR, SETTINGS

LAWS_DIR = RAG_DIR / "laws"
CASES_PATH = BENCHMARK_DIR / "dispute_cases.json"


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
                docs.append(
                    Doc(
                        doc_id=row["doc_id"],
                        title=row["title"],
                        text=row["text"],
                        kind=row.get("kind", "law"),
                        principle=(row["principle"],) if row.get("principle") else (),
                    )
                )

    if CASES_PATH.exists():
        with open(CASES_PATH, encoding="utf-8") as f:
            payload = json.load(f)
        for c in payload["cases"]:
            # 실제 분쟁 사건만 코퍼스에 넣는다.
            #
            # 정답셋의 pass 케이스(origin="constructed")는 분쟁 사건이 아니라 과잉경고를
            # 측정하려고 구성한 정상 상품이다. 이걸 코퍼스에 넣으면 두 가지가 깨진다.
            #   ① 케이스 문서에 `판정 {label}`이 붙어 있고 평가 시 자기 사례만 제외되므로,
            #      pass 케이스를 늘린 만큼 이웃 문서의 라벨 분포가 pass로 쏠린다.
            #      그러면 pass 정확도가 올라도 모델이 좋아진 것인지 RAG 사전확률이
            #      옮겨간 것인지 구분할 수 없다.
            #   ② 정답셋과 코퍼스를 동시에 바꾸면 이전 측정(sev_B)과 비교가 불가능해진다.
            # 정답셋 확대의 효과를 격리하려면 코퍼스는 조정례 12건으로 고정해야 한다.
            if c.get("origin", "dispute") != "dispute":
                continue
            # 라벨을 제목에 박아두면 프롬프트에 그대로 인용된다. 코퍼스의 83.3%가
            # fail/warn이라 이 노출이 곧 과잉경고 방향의 사전확률이 된다.
            # FDM_CASE_HIDE_LABEL=1이면 판단요지 본문은 남기고 라벨만 감춘다.
            title = f"조정사례(가공) {c['title']}"
            if not SETTINGS.case_hide_label:
                title += f" — 판정 {c['label']}"
            docs.append(
                Doc(
                    doc_id=c["case_id"],
                    title=title,
                    text=f"[사실관계] {c['facts']}\n[판단요지] {c['decision_gist']}",
                    kind="case",
                    principle=tuple(c.get("principle", [])),
                )
            )
    return docs

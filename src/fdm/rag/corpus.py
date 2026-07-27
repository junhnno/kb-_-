"""RAG 코퍼스: 금소법 조문 + (가공) 분쟁조정 사례."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

from ..config import BENCHMARK_DIR, RAG_DIR

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
            docs.append(
                Doc(
                    doc_id=c["case_id"],
                    title=f"조정사례(가공) {c['title']} — 판정 {c['label']}",
                    text=f"[사실관계] {c['facts']}\n[판단요지] {c['decision_gist']}",
                    kind="case",
                    principle=tuple(c.get("principle", [])),
                )
            )
    return docs

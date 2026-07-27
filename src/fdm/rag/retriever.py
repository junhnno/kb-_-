"""근거 검색기.

기본은 의존성 없는 BM25(한국어 형태소 분석기 없이 어절 + 문자 2-gram 혼합 토크나이저).
sentence-transformers가 설치되어 있으면 bge-m3 임베딩을 얹어 하이브리드로 동작한다.

`exclude_ids`가 중요하다: 벤치마크에서 특정 조정례를 평가할 때 그 사례 자체를
검색 결과로 주면 정답 라벨이 새어 들어간다(contamination). 평가 코드는 반드시
해당 case_id를 제외해야 한다.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from .corpus import Doc, load_corpus

_WORD = re.compile(r"[0-9A-Za-z가-힣%]+")
_HANGUL = re.compile(r"[가-힣]{2,}")


def tokenize(text: str) -> list[str]:
    toks = [t.lower() for t in _WORD.findall(text)]
    grams: list[str] = []
    for t in toks:
        if _HANGUL.fullmatch(t) and len(t) > 2:
            grams += [t[i : i + 2] for i in range(len(t) - 1)]
    return toks + grams


@dataclass
class Hit:
    doc: Doc
    score: float


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

    def search(self, query: str) -> list[float]:
        q = tokenize(query)
        scores = [0.0] * len(self.docs)
        for term in q:
            idf = self.idf.get(term)
            if not idf:
                continue
            for i, c in enumerate(self.tf):
                f = c.get(term)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg_len)
                scores[i] += idf * f * (self.k1 + 1) / denom
        return scores


class Retriever:
    def __init__(self, docs: list[Doc] | None = None, use_dense: bool = False):
        self.docs = docs if docs is not None else load_corpus()
        self.bm25 = BM25(self.docs)
        self.dense = None
        if use_dense:
            self.dense = _try_load_dense(self.docs)

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

    def grounding_block(self, query: str, k: int = 5, **kw) -> str:
        hits = self.retrieve(query, k=k, **kw)
        if not hits:
            return "(검색된 근거 없음)"
        return "\n".join(f"{i+1}. {h.doc.cite()}" for i, h in enumerate(hits))


def _try_load_dense(docs: list[Doc]):
    try:
        from sentence_transformers import SentenceTransformer  # 선택 의존성

        model = SentenceTransformer("BAAI/bge-m3")
        emb = model.encode([f"{d.title} {d.text}" for d in docs], normalize_embeddings=True)

        def score(query: str) -> list[float]:
            q = model.encode([query], normalize_embeddings=True)[0]
            return (emb @ q).tolist()

        return score
    except Exception:
        return None


@lru_cache(maxsize=2)
def get_retriever(use_dense: bool = False) -> Retriever:
    return Retriever(use_dense=use_dense)

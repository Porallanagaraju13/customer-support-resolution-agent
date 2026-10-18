"""Retrieval over the policy knowledge base.

Default: local TF-IDF index (scikit-learn) - no keys, no downloads.
Optional: OpenAI-compatible embeddings (EMBEDDINGS=openai); on any failure it
falls back to the local index so retrieval never blocks the workflow.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import Settings

KB_DIR = Path(__file__).parent / "kb"
log = logging.getLogger("support_agent.kb")


def load_chunks(kb_dir: Path = KB_DIR) -> list[dict]:
    chunks = []
    for f in sorted(kb_dir.glob("*.md")):
        title, section, buf = f.stem, "Overview", []

        def flush():
            text = " ".join(buf).strip()
            if text:
                chunks.append({"source": f.name, "section": section, "text": text})

        for line in f.read_text().splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
            elif line.startswith("## "):
                flush()
                buf = []
                section = line[3:].strip()
            else:
                buf.append(line.strip())
        flush()
    return chunks


class KnowledgeBase:
    def __init__(self, settings: Settings | None = None, kb_dir: Path = KB_DIR):
        self.settings = settings or Settings()
        self.chunks = load_chunks(kb_dir)
        docs = [f"{c['section']}. {c['text']}" for c in self.chunks]
        self.vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        self.matrix = self.vec.fit_transform(docs)
        self.backend = "tfidf-local"
        self._emb: np.ndarray | None = None
        if self.settings.embeddings == "openai" and self.settings.openai_api_key:
            try:
                self._emb = self._embed(docs)
                self.backend = f"openai:{self.settings.embedding_model}"
            except Exception as exc:  # graceful fallback
                log.warning("embedding_init_failed_fallback_tfidf", extra={"error": str(exc)})

    def _embed(self, texts: list[str]) -> np.ndarray:
        from openai import OpenAI

        client = OpenAI(api_key=self.settings.openai_api_key, base_url=self.settings.base_url or None,
                        timeout=self.settings.llm_timeout_s)
        res = client.embeddings.create(model=self.settings.embedding_model, input=texts)
        m = np.array([d.embedding for d in res.data], dtype=float)
        return m / np.linalg.norm(m, axis=1, keepdims=True)

    def search(self, query: str, k: int = 3) -> list[dict]:
        query = re.sub(r"\s+", " ", query).strip()
        scores = None
        if self._emb is not None:
            try:
                q = self._embed([query])[0]
                scores = self._emb @ q
            except Exception as exc:
                log.warning("embedding_query_failed_fallback_tfidf", extra={"error": str(exc)})
        if scores is None:
            q = self.vec.transform([query])
            scores = (self.matrix @ q.T).toarray().ravel()
        order = np.argsort(-scores)[:k]
        return [{**self.chunks[i], "score": round(float(scores[i]), 3)} for i in order if scores[i] > 0]

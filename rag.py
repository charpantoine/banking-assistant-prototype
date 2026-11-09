"""FAQ retrieval: section chunks, mistral-embed vectors cached on disk, cosine similarity with numpy."""
import hashlib
import os
import re

import numpy as np
from mistralai.client import Mistral

import config


def _split_sections(body):
    """Split a document body on ## headings; text before the first heading joins the first section."""
    parts = [p.strip() for p in re.split(r"(?m)^(?=## )", body) if p.strip()]
    if len(parts) > 1 and not parts[0].startswith("## "):
        parts[:2] = [parts[0] + "\n\n" + parts[1]]
    return parts


def _load_chunks():
    """Read every knowledge document into chunks with text, title and source file."""
    chunks = []
    for path in sorted(config.KNOWLEDGE_DIR.glob("*.md")):
        _, front_matter, body = path.read_text(encoding="utf-8").split("---", 2)
        title = front_matter.split("title:", 1)[1].strip()
        chunks += [{"text": s, "title": title, "source": path.name} for s in _split_sections(body)]
    return chunks


def _embed(texts):
    """Embed texts in batches and return unit-length vectors, so a dot product is a cosine."""
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        raise RuntimeError("MISTRAL_API_KEY missing")
    client = Mistral(api_key=key)
    vectors = []
    for start in range(0, len(texts), config.EMBED_BATCH_SIZE):
        batch = texts[start:start + config.EMBED_BATCH_SIZE]
        vectors += [d.embedding for d in client.embeddings.create(model=config.EMBED_MODEL, inputs=batch).data]
    matrix = np.array(vectors, dtype=float)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def _load_index():
    """Return chunks and their vectors, calling the embedding API only if documents or model changed."""
    chunks = _load_chunks()
    texts = [f"{c['title']}\n\n{c['text']}" for c in chunks]
    digest = hashlib.sha256("\n".join([config.EMBED_MODEL, *texts]).encode("utf-8")).hexdigest()
    if config.INDEX_PATH.exists():
        with np.load(config.INDEX_PATH) as cache:
            if str(cache["digest"]) == digest:
                return chunks, cache["vectors"]
    vectors = _embed(texts)
    np.savez(config.INDEX_PATH, vectors=vectors, digest=digest)
    return chunks, vectors


def search(query, k=config.TOP_K):
    """Return the k closest chunks with their score, or [] if the best score is below the threshold."""
    chunks, vectors = _load_index()
    scores = vectors @ _embed([query])[0]
    best = np.argsort(scores)[::-1][:k]
    if scores[best[0]] < config.RAG_THRESHOLD:
        return []
    return [{**chunks[i], "score": float(scores[i])} for i in best]

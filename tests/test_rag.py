"""Tests of FAQ indexing and retrieval: two against the real embedding API, two without it."""
import numpy as np

import config
import rag


def test_lost_card_retrieves_card_opposition_document():
    """The lost card question has the card opposition document in its top 3 chunks."""
    sources = [c["source"] for c in rag.search("I lost my card, what do I do?")]
    assert "card_opposition.md" in sources


def test_off_topic_query_returns_no_chunk():
    """A question unrelated to banking stays below the threshold and returns nothing."""
    assert rag.search("Who won the World Cup?") == []


def test_chunks_split_by_section_and_keep_intro(tmp_path, monkeypatch):
    """Text before the first ## joins the first chunk; every chunk keeps title and source."""
    (tmp_path / "doc.md").write_text("---\ntitle: My title\n---\nIntro.\n\n## A\nBody a.\n\n## B\nBody b.\n")
    monkeypatch.setattr(config, "KNOWLEDGE_DIR", tmp_path)
    assert rag._load_chunks() == [
        {"text": "Intro.\n\n## A\nBody a.", "title": "My title", "source": "doc.md"},
        {"text": "## B\nBody b.", "title": "My title", "source": "doc.md"},
    ]


def test_index_rebuilt_only_if_documents_or_model_change(tmp_path, monkeypatch):
    """The embedding call happens once, then again only after a document or model change."""
    calls = []

    def fake_embed(texts):
        """Record the call and return one dummy vector per text."""
        calls.append(texts)
        return np.ones((len(texts), 3))

    docs = tmp_path / "docs"
    docs.mkdir()
    document = docs / "doc.md"
    document.write_text("---\ntitle: T\n---\n## A\nFirst.\n")
    monkeypatch.setattr(config, "KNOWLEDGE_DIR", docs)
    monkeypatch.setattr(config, "INDEX_PATH", tmp_path / "index.npz")
    monkeypatch.setattr(rag, "_embed", fake_embed)

    rag._load_index()
    rag._load_index()
    assert len(calls) == 1

    document.write_text("---\ntitle: T\n---\n## A\nChanged.\n")
    rag._load_index()
    assert len(calls) == 2

    monkeypatch.setattr(config, "EMBED_MODEL", "another-model")
    rag._load_index()
    assert len(calls) == 3

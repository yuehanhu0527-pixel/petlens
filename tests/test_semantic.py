
import numpy as np
import pytest

from app.retrieval.semantic import SemanticRetriever


def make_chunk(chunk_id, document_id, category):
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "title": f"Study {document_id}",
        "category": category,
        "authors": "Test Author",
        "url": "https://example.com",
        "text": "Test scientific passage.",
    }


class FakeModel:
    """Avoid downloading or running a real embedding model."""

    def encode(self, text, **kwargs):
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


@pytest.fixture
def retriever(monkeypatch):
    monkeypatch.setattr(
        "app.retrieval.semantic.SentenceTransformer",
        lambda model_name: FakeModel()
    )

    chunks = [
        make_chunk("1", "A", "feeding"),
        make_chunk("2", "B", "behavior"),
        make_chunk("3", "C", "feeding"),
    ]

    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.8, 0.6, 0.0],
    ], dtype=np.float32)

    return SemanticRetriever(chunks, embeddings)


def test_semantic_ranking(retriever):
    results = retriever.search("feeding")

    assert results[0]["document_id"] == "A"
    assert results[1]["document_id"] == "C"
    assert results[2]["document_id"] == "B"


def test_semantic_category_filter(retriever):
    results = retriever.search(
        "feeding",
        category="feeding"
    )

    assert len(results) == 2
    assert all(
        result["category"] == "feeding"
        for result in results
    )


def test_semantic_top_k(retriever):
    results = retriever.search("feeding", top_k=2)

    assert len(results) == 2


def test_semantic_empty_query(retriever):
    assert retriever.search("") == []


def test_semantic_zero_top_k(retriever):
    assert retriever.search("feeding", top_k=0) == []


def test_semantic_document_diversity(monkeypatch):
    monkeypatch.setattr(
        "app.retrieval.semantic.SentenceTransformer",
        lambda model_name: FakeModel()
    )

    chunks = [
        make_chunk("1", "A", "feeding"),
        make_chunk("2", "A", "feeding"),
        make_chunk("3", "B", "behavior"),
    ]

    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.9, 0.4359, 0.0],
        [0.5, 0.8660, 0.0],
    ], dtype=np.float32)

    retriever = SemanticRetriever(chunks, embeddings)
    results = retriever.search("feeding", top_k=3)

    document_ids = [r["document_id"] for r in results]

    assert len(results) == 2
    assert len(document_ids) == len(set(document_ids))


def test_semantic_mismatched_embeddings(monkeypatch):
    monkeypatch.setattr(
        "app.retrieval.semantic.SentenceTransformer",
        lambda model_name: FakeModel()
    )

    chunks = [
        make_chunk("1", "A", "feeding"),
        make_chunk("2", "B", "behavior"),
    ]

    embeddings = np.array([
        [1.0, 0.0, 0.0]
    ], dtype=np.float32)

    with pytest.raises(ValueError):
        SemanticRetriever(chunks, embeddings)


def test_semantic_empty_corpus(monkeypatch):
    monkeypatch.setattr(
        "app.retrieval.semantic.SentenceTransformer",
        lambda model_name: FakeModel()
    )

    with pytest.raises(ValueError):
        SemanticRetriever(
            [],
            np.empty((0, 3), dtype=np.float32)
        )

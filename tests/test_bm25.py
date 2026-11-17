
import pytest

from app.retrieval.bm25 import BM25Retriever, tokenize


def make_chunk(chunk_id, document_id, category, text):
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "title": f"Study {document_id}",
        "category": category,
        "authors": "Test Author",
        "url": "https://example.com",
        "text": text,
    }


def test_tokenize():
    assert tokenize("Cat's LITTER-box!") == [
        "cat", "s", "litter", "box"
    ]


def test_relevant_result_ranked_first():
    chunks = [
        make_chunk(
            "1", "A", "feeding",
            "cat feeding meal frequency daily"
        ),
        make_chunk(
            "2", "B", "behavior",
            "cat aggression stress fighting"
        ),
    ]

    retriever = BM25Retriever(chunks)
    results = retriever.search("feeding meal frequency")

    assert results[0]["document_id"] == "A"


def test_category_filter():
    chunks = [
        make_chunk("1", "A", "feeding", "cat feeding meals"),
        make_chunk("2", "B", "behavior", "cat aggression fighting"),
    ]

    retriever = BM25Retriever(chunks)

    results = retriever.search(
        "cat",
        category="feeding"
    )

    assert len(results) == 1
    assert results[0]["category"] == "feeding"


def test_document_diversity():
    chunks = [
        make_chunk("1", "A", "feeding", "cat feeding meals"),
        make_chunk("2", "A", "feeding", "cat feeding frequency"),
        make_chunk("3", "B", "feeding", "cat feeding schedule"),
    ]

    retriever = BM25Retriever(chunks)
    results = retriever.search("cat feeding", top_k=3)

    document_ids = [
        result["document_id"]
        for result in results
    ]

    assert len(document_ids) == len(set(document_ids))
    assert len(results) == 2


def test_empty_query():
    chunks = [
        make_chunk("1", "A", "feeding", "cat feeding")
    ]

    retriever = BM25Retriever(chunks)

    assert retriever.search("") == []


def test_top_k():
    chunks = [
        make_chunk(
            str(i), str(i), "feeding",
            "cat feeding meal"
        )
        for i in range(10)
    ]

    retriever = BM25Retriever(chunks)

    results = retriever.search("feeding", top_k=3)

    assert len(results) == 3


def test_zero_top_k():
    chunks = [
        make_chunk("1", "A", "feeding", "cat feeding")
    ]

    retriever = BM25Retriever(chunks)

    assert retriever.search("cat", top_k=0) == []


def test_empty_corpus():
    with pytest.raises(ValueError):
        BM25Retriever([])


import pytest

from app.retrieval.hybrid import HybridRetriever


def make_result(document_id, score=1.0):
    return {
        "chunk_id": f"{document_id}_001",
        "document_id": document_id,
        "title": f"Study {document_id}",
        "category": "behavior",
        "authors": "Test Author",
        "url": "https://example.com",
        "text": "Example passage.",
        "score": score,
    }


class FakeRetriever:

    def __init__(self, results):
        self.results = results

    def search(self, query, top_k=5, category=None):
        results = self.results

        if category is not None:
            results = [
                r for r in results
                if r["category"] == category
            ]

        return results[:top_k]


def test_hybrid_combines_results():

    bm25 = FakeRetriever([
        make_result("A"),
        make_result("B"),
    ])

    semantic = FakeRetriever([
        make_result("B"),
        make_result("C"),
    ])

    hybrid = HybridRetriever(bm25, semantic)
    results = hybrid.search("cat behavior")

    document_ids = [r["document_id"] for r in results]

    assert set(document_ids) == {"A", "B", "C"}


def test_hybrid_shared_document_ranks_first():

    bm25 = FakeRetriever([
        make_result("A"),
        make_result("B"),
    ])

    semantic = FakeRetriever([
        make_result("B"),
        make_result("C"),
    ])

    hybrid = HybridRetriever(bm25, semantic)
    results = hybrid.search("cat behavior")

    assert results[0]["document_id"] == "B"


def test_hybrid_no_duplicates():

    bm25 = FakeRetriever([
        make_result("A"),
        make_result("B"),
    ])

    semantic = FakeRetriever([
        make_result("A"),
        make_result("B"),
    ])

    hybrid = HybridRetriever(bm25, semantic)
    results = hybrid.search("cat behavior")

    ids = [r["document_id"] for r in results]

    assert len(ids) == len(set(ids))


def test_hybrid_top_k():

    bm25 = FakeRetriever([
        make_result("A"),
        make_result("B"),
        make_result("C"),
    ])

    semantic = FakeRetriever([
        make_result("C"),
        make_result("B"),
        make_result("A"),
    ])

    hybrid = HybridRetriever(bm25, semantic)

    results = hybrid.search("cat behavior", top_k=2)

    assert len(results) == 2


def test_hybrid_empty_query():

    bm25 = FakeRetriever([])
    semantic = FakeRetriever([])

    hybrid = HybridRetriever(bm25, semantic)

    assert hybrid.search("") == []


def test_hybrid_category_filter():

    feeding_result = make_result("A")
    feeding_result["category"] = "feeding"

    behavior_result = make_result("B")

    bm25 = FakeRetriever([
        feeding_result,
        behavior_result,
    ])

    semantic = FakeRetriever([
        behavior_result,
        feeding_result,
    ])

    hybrid = HybridRetriever(bm25, semantic)

    results = hybrid.search(
        "cat feeding",
        category="feeding"
    )

    assert len(results) == 1
    assert results[0]["category"] == "feeding"


def test_invalid_rrf_k():

    bm25 = FakeRetriever([])
    semantic = FakeRetriever([])

    with pytest.raises(ValueError):
        HybridRetriever(bm25, semantic, rrf_k=-1)


def test_invalid_candidate_k():

    bm25 = FakeRetriever([])
    semantic = FakeRetriever([])

    with pytest.raises(ValueError):
        HybridRetriever(bm25, semantic, candidate_k=0)


class HybridRetriever:
    """
    Combine BM25 and Semantic Search using
    Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        bm25_retriever,
        semantic_retriever,
        rrf_k=60,
        candidate_k=50
    ):
        self.bm25 = bm25_retriever
        self.semantic = semantic_retriever
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k

        if rrf_k < 0:
            raise ValueError("rrf_k cannot be negative.")

        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive.")

    def search(self, query, top_k=5, category=None):

        if not query.strip() or top_k <= 0:
            return []

        # Step 1: Retrieve BM25 candidates
        bm25_results = self.bm25.search(
            query,
            top_k=max(self.candidate_k, top_k),
            category=category
        )

        # Step 2: Retrieve Semantic candidates
        semantic_results = self.semantic.search(
            query,
            top_k=max(self.candidate_k, top_k),
            category=category
        )

        # Step 3: Fuse document rankings
        fused_scores = {}
        document_results = {}

        for source, results in [
            ("bm25", bm25_results),
            ("semantic", semantic_results)
        ]:

            for rank, result in enumerate(results, start=1):

                document_id = result["document_id"]

                if document_id not in fused_scores:
                    fused_scores[document_id] = 0.0

                fused_scores[document_id] += (
                    1.0 / (self.rrf_k + rank)
                )

                # Prefer the best-ranked BM25 passage
                # when both methods retrieve the article.
                if (
                    document_id not in document_results
                    or source == "bm25"
                ):
                    document_results[document_id] = result

        # Step 4: Sort by fused score
        ranked_ids = sorted(
            fused_scores,
            key=lambda doc_id: (
                -fused_scores[doc_id],
                doc_id
            )
        )

        # Step 5: Format top results
        final_results = []

        for document_id in ranked_ids[:top_k]:

            result = document_results[document_id].copy()

            result["score"] = round(
                fused_scores[document_id], 6
            )

            result["retrieval_method"] = "hybrid"

            final_results.append(result)

        return final_results

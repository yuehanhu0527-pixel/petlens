
import numpy as np
from sentence_transformers import CrossEncoder


class RerankedRetriever:
    def __init__(
        self,
        semantic_retriever,
        candidate_k=50,
        model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
        reranker_model=None,
    ):
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive.")

        self.semantic = semantic_retriever
        self.candidate_k = candidate_k

        self.reranker = (
            reranker_model
            if reranker_model is not None
            else CrossEncoder(model_name)
        )

    def search(self, query, top_k=5, category=None):
        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        if not query.strip() or top_k <= 0:
            return []

        # Stage 1: Retrieve candidate passages
        query_embedding = self.semantic.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        scores = self.semantic.embeddings @ query_embedding

        candidates = []

        for index, score in enumerate(scores):
            chunk = self.semantic.chunks[index]

            if category is not None:
                if chunk.get("category") != category:
                    continue

            if not np.isfinite(score):
                continue

            candidates.append((index, float(score)))

        candidates.sort(key=lambda x: (-x[1], x[0]))
        candidates = candidates[:self.candidate_k]

        if not candidates:
            return []

        # Stage 2: Score query-passage pairs
        pairs = []

        for index, _ in candidates:
            chunk = self.semantic.chunks[index]
            passage = chunk["title"] + "\n" + chunk["text"]
            pairs.append((query, passage))

        rerank_scores = np.asarray(
            self.reranker.predict(pairs),
            dtype=float,
        ).reshape(-1)

        if len(rerank_scores) != len(candidates):
            raise ValueError("Unexpected reranker score count.")

        ranked = sorted(
            zip(candidates, rerank_scores),
            key=lambda item: (
                -item[1]
                if np.isfinite(item[1])
                else float("inf")
            ),
        )

        # Deduplicate articles after reranking
        results = []
        seen_documents = set()

        for (index, semantic_score), rerank_score in ranked:
            if not np.isfinite(rerank_score):
                continue

            chunk = self.semantic.chunks[index]

            if chunk["document_id"] in seen_documents:
                continue

            seen_documents.add(chunk["document_id"])

            results.append({
                **chunk,
                "score": float(rerank_score),
                "semantic_score": semantic_score,
            })

            if len(results) >= top_k:
                break

        return results

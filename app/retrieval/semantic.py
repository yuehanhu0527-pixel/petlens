
import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticRetriever:

    def __init__(
        self,
        chunks,
        embeddings,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.chunks = chunks
        self.embeddings = np.asarray(
            embeddings, dtype=np.float32
        )

        if len(chunks) == 0:
            raise ValueError("The search corpus is empty.")

        if self.embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")

        if len(chunks) != len(self.embeddings):
            raise ValueError(
                "Number of chunks and embeddings must match."
            )

        self.model = SentenceTransformer(model_name)

    def search(self, query, top_k=5, category=None):

        if not query.strip() or top_k <= 0:
            return []

        # 1. Convert the user question into a vector
        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        # 2. Compute cosine similarity
        # Both query and stored embeddings are normalized
        scores = self.embeddings @ query_embedding

        # 3. Keep only chunks in the selected category
        candidates = []

        for doc_id, score in enumerate(scores):

            chunk = self.chunks[doc_id]

            if (
                category is not None
                and chunk["category"] != category
            ):
                continue

            candidates.append((doc_id, float(score)))

        # 4. Rank by similarity, highest first
        ranked = sorted(
            candidates,
            key=lambda item: (-item[1], item[0])
        )

        # 5. Return unique articles
        results = []
        seen_documents = set()

        for doc_id, score in ranked:

            chunk = self.chunks[doc_id]
            document_id = chunk["document_id"]

            if document_id in seen_documents:
                continue

            seen_documents.add(document_id)

            results.append({
                "chunk_id": chunk["chunk_id"],
                "document_id": document_id,
                "title": chunk["title"],
                "category": chunk["category"],
                "authors": chunk["authors"],
                "url": chunk["url"],
                "text": chunk["text"],
                "score": round(score, 4)
            })

            if len(results) >= top_k:
                break

        return results

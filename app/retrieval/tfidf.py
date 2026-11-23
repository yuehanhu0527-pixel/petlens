
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDFRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

        # Convert passages into TF-IDF vectors
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True
        )

        texts = [
            chunk["title"] + " " + chunk["text"]
            for chunk in chunks
        ]

        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query, top_k=5, category=None):
        query_vector = self.vectorizer.transform([query])

        # Vectors are normalized, so dot product = cosine similarity
        scores = (self.matrix @ query_vector.T).toarray().ravel()

        results = []

        for i, score in enumerate(scores):
            chunk = self.chunks[i]

            if category and chunk.get("category") != category:
                continue

            results.append({
                **chunk,
                "score": float(score)
            })

        results.sort(key=lambda item: item["score"], reverse=True)

        # Keep only the highest-ranked passage from each article
        unique_results = []
        seen_documents = set()

        for result in results:
            if result["score"] <= 0:
                continue

            document_id = result["document_id"]

            if document_id in seen_documents:
                continue

            seen_documents.add(document_id)
            unique_results.append(result)

            if len(unique_results) >= top_k:
                break

        return unique_results

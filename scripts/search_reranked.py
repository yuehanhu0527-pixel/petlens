
import json
from pathlib import Path

import numpy as np

from app.retrieval.semantic import SemanticRetriever
from app.retrieval.reranker import RerankedRetriever


ROOT = Path(__file__).resolve().parents[1]


def main():
    with open(
        ROOT / "data/processed/chunks.json",
        encoding="utf-8"
    ) as f:
        chunks = json.load(f)

    embeddings = np.load(
        ROOT / "data/processed/embeddings.npy"
    )

    semantic = SemanticRetriever(chunks, embeddings)

    reranked = RerankedRetriever(
        semantic_retriever=semantic,
        candidate_k=50
    )

    print("\nPetLens — Semantic + Cross-Encoder")

    while True:
        query = input(
            "\nAsk a cat-care question (or 'quit'): "
        ).strip()

        if query.lower() in {"quit", "exit"}:
            break

        results = reranked.search(query, top_k=5)

        if not results:
            print("No results found.")
            continue

        for rank, result in enumerate(results, 1):
            print(f"\n{rank}. {result['title']}")
            print(f"   Rerank score: {result['score']:.4f}")
            print(
                f"   Semantic score: "
                f"{result['semantic_score']:.4f}"
            )
            print(f"   URL: {result['url']}")
            print(f"   Passage: {result['text'][:300]}...")


if __name__ == "__main__":
    main()

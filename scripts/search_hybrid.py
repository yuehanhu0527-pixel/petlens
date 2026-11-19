
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.retrieval.bm25 import BM25Retriever
from app.retrieval.semantic import SemanticRetriever
from app.retrieval.hybrid import HybridRetriever


CHUNKS_PATH = ROOT / "data/processed/chunks.json"
EMBEDDINGS_PATH = ROOT / "data/processed/embeddings.npy"

VALID_CATEGORIES = {
    "behavior",
    "feeding",
    "grooming",
    "litter_box",
    "general_care"
}


def main():

    # Load existing corpus
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    embeddings = np.load(
        EMBEDDINGS_PATH,
        allow_pickle=False
    )

    print("Building BM25 index...")
    bm25 = BM25Retriever(chunks)

    print("Loading Semantic Retriever...")
    semantic = SemanticRetriever(chunks, embeddings)

    print("Initializing Hybrid Retriever...")
    hybrid = HybridRetriever(bm25, semantic)

    print()
    print(f"PetLens Hybrid Search - {len(chunks)} passages")
    print("Type 'quit' to exit.")
    print()

    while True:

        query = input("Search: ").strip()

        if query.lower() in ("quit", "exit"):
            break

        if not query:
            continue

        print(
            "Categories: all, behavior, feeding, "
            "grooming, litter_box, general_care"
        )

        category = input(
            "Category [all]: "
        ).strip().lower()

        if category in ("", "all"):
            category = None
        elif category not in VALID_CATEGORIES:
            print("Invalid category.\n")
            continue

        results = hybrid.search(
            query,
            top_k=5,
            category=category
        )

        if not results:
            print("No matching results.\n")
            continue

        for rank, result in enumerate(results, start=1):

            print("-" * 70)
            print(f"Result #{rank}")
            print(f"Title: {result['title']}")
            print(f"Category: {result['category']}")
            print(f"RRF Score: {result['score']:.6f}")
            print(f"Source: {result['url']}")
            print()
            print(result["text"][:500] + "...")
            print()

        print("=" * 70)


if __name__ == "__main__":
    main()

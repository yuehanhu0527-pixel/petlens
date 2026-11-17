
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

from app.retrieval.bm25 import BM25Retriever


DATA_PATH = ROOT / "data/processed/chunks.json"


def main():

    chunks = json.loads(
        DATA_PATH.read_text(encoding="utf-8")
    )

    print("Building BM25 index...")

    retriever = BM25Retriever(chunks)

    print(f"Indexed {retriever.N} passages")
    print(f"Vocabulary size: {len(retriever.idf)}")
    print(f"Average document length: {retriever.avgdl:.1f}")
    print()

    print("PetLens - Feline Care Search Engine")
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

        results = retriever.search(
            query,
            top_k=5,
            category=category
        )

        if not results:
            print("No matching results found.\n")
            continue

        for rank, result in enumerate(results, start=1):

            print("-" * 70)
            print(f"Result #{rank}")
            print(f"Title: {result['title']}")
            print(f"Category: {result['category']}")
            print(f"BM25 Score: {result['score']}")
            print(f"Source: {result['url']}")
            print()
            print(result["text"][:500] + "...")
            print()

        print("=" * 70)


if __name__ == "__main__":
    main()


import json
from pathlib import Path

from app.retrieval.tfidf import TFIDFRetriever


ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "data/processed/chunks.json"

    with open(path, encoding="utf-8") as f:
        chunks = json.load(f)

    retriever = TFIDFRetriever(chunks)

    while True:
        query = input("\nAsk a cat-care question (or 'quit'): ").strip()

        if query.lower() in {"quit", "exit"}:
            break

        if not query:
            continue

        results = retriever.search(query, top_k=5)

        for rank, result in enumerate(results, start=1):
            print(f"\n{rank}. {result['title']}")
            print(f"   Score: {result['score']:.4f}")
            print(f"   URL: {result['url']}")
            print(f"   Passage: {result['text'][:250]}...")


if __name__ == "__main__":
    main()

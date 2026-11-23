
import csv
import json
from pathlib import Path

from app.retrieval.tfidf import TFIDFRetriever


ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = ROOT / "data/processed/chunks.json"
BASELINE_PATH = ROOT / "data/evaluation/baseline_results.csv"
OUTPUT_PATH = ROOT / "data/evaluation/tfidf_results.csv"


def main():
    # Load the same passages used by the other retrievers
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)

    # Reuse the original 10 evaluation queries
    with open(BASELINE_PATH, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        baseline_rows = list(reader)
        fieldnames = reader.fieldnames

    if not fieldnames or not baseline_rows:
        raise ValueError("Baseline evaluation results are empty.")

    queries = {}

    for row in baseline_rows:
        queries[row["query_id"]] = row["query"]

    retriever = TFIDFRetriever(chunks)
    results = []

    for query_id, query in sorted(
        queries.items(), key=lambda item: int(item[0])
    ):
        matches = retriever.search(query, top_k=5)

        for rank, match in enumerate(matches, start=1):
            row = {
                "query_id": query_id,
                "query": query,
                "method": "TF-IDF",
                "rank": rank,
                "document_id": match["document_id"],
                "chunk_id": match["chunk_id"],
                "title": match["title"],
                "url": match.get("url", ""),
                "score": match["score"],
                "snippet": match["text"][:500],
            }

            results.append({
                field: row.get(field, "")
                for field in fieldnames
            })

        print(f"Query {query_id}: {len(matches)} results")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(
        OUTPUT_PATH, "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nTotal TF-IDF results: {len(results)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

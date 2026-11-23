
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/evaluation"

CHUNKS = ROOT / "data/processed/chunks.json"
EMBEDDINGS = ROOT / "data/processed/embeddings.npy"
QUERIES = DATA / "baseline_results.csv"
JUDGMENTS = DATA / "petlens_relevance_judgments_four_methods_en.csv"

OUTPUT = DATA / "corpus_coverage_audit.csv"
INVENTORY = DATA / "corpus_inventory.csv"


def load_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    with open(CHUNKS, encoding="utf-8") as f:
        chunks = json.load(f)

    embeddings = np.load(EMBEDDINGS)

    if len(chunks) != len(embeddings):
        raise ValueError("Chunk and embedding counts do not match.")

    queries = {}
    for row in load_csv(QUERIES):
        queries[row["query_id"]] = row["query"]

    judgments = {}
    if JUDGMENTS.exists():
        for row in load_csv(JUDGMENTS):
            judgments[(row["query_id"], row["chunk_id"])] = row["relevance"]

    # Build an inventory of all articles
    documents = defaultdict(list)

    for index, chunk in enumerate(chunks):
        documents[chunk["document_id"]].append(index)

    inventory = []

    for document_id, indices in sorted(documents.items()):
        chunk = chunks[indices[0]]
        inventory.append({
            "document_id": document_id,
            "title": chunk["title"],
            "category": chunk["category"],
            "passage_count": len(indices),
            "url": chunk.get("url", ""),
        })

    DATA.mkdir(parents=True, exist_ok=True)

    with open(INVENTORY, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(inventory[0].keys())
        )
        writer.writeheader()
        writer.writerows(inventory)

    # Search across the entire corpus
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    rows = []

    for query_id, query in sorted(
        queries.items(), key=lambda item: int(item[0])
    ):
        query_vector = model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        scores = embeddings @ query_vector

        # Find the best passage from each article
        document_candidates = []

        for document_id, indices in documents.items():
            ranked_indices = sorted(
                indices,
                key=lambda i: float(scores[i]),
                reverse=True,
            )

            best_index = ranked_indices[0]

            document_candidates.append(
                (document_id, best_index, ranked_indices)
            )

        document_candidates.sort(
            key=lambda item: float(scores[item[1]]),
            reverse=True,
        )

        # Keep the top 20 articles for manual inspection
        for rank, (document_id, best_index, ranked_indices) in enumerate(
            document_candidates[:20], start=1
        ):
            chunk = chunks[best_index]

            rows.append({
                "query_id": query_id,
                "query": query,
                "document_rank": rank,
                "document_id": document_id,
                "title": chunk["title"],
                "category": chunk["category"],
                "semantic_score": round(
                    float(scores[best_index]), 4
                ),
                "best_chunk_id": chunk["chunk_id"],
                "previous_relevance": judgments.get(
                    (query_id, chunk["chunk_id"]), ""
                ),
                "best_passage": chunk["text"],
                "second_passage": (
                    chunks[ranked_indices[1]]["text"]
                    if len(ranked_indices) > 1 else ""
                ),
                "url": chunk.get("url", ""),
            })

        print(
            f"Query {query_id}: "
            f"reviewing top 20 of {len(documents)} articles"
        )

    fields = [
        "query_id", "query", "document_rank",
        "document_id", "title", "category",
        "semantic_score", "best_chunk_id",
        "previous_relevance", "best_passage",
        "second_passage", "url",
    ]

    with open(OUTPUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("\nCORPUS COVERAGE AUDIT")
    print(f"Articles: {len(documents)}")
    print(f"Passages: {len(chunks)}")
    print(f"Queries: {len(queries)}")
    print(f"Candidate records: {len(rows)}")
    print(f"Saved: {OUTPUT}")
    print(f"Inventory: {INVENTORY}")


if __name__ == "__main__":
    main()

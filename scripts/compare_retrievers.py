
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.retrieval.bm25 import BM25Retriever
from app.retrieval.semantic import SemanticRetriever
from app.retrieval.hybrid import HybridRetriever


CHUNKS_PATH = ROOT / "data/processed/chunks.json"
EMBEDDINGS_PATH = ROOT / "data/processed/embeddings.npy"
OUTPUT_DIR = ROOT / "data/evaluation"

# Fixed evaluation questions
QUERIES = [
    "Why does my cat keep scratching its ears?",
    "How can I introduce a new cat to my resident cat?",
    "How can I prevent hairballs in cats?",
    "Why does my cat pee outside the litter box?",
    "How many litter boxes should I have for two cats?",
    "How often should I feed my cat?",
    "How can I reduce aggression between cats?",
    "How can I encourage my cat to drink more water?",
    "How can I reduce stress in indoor cats?",
    "How can I prevent obesity in indoor cats?",
]

METHODS = ["BM25", "Semantic", "Hybrid"]
TOP_K = 5


def build_retrievers(chunks, embeddings):
    bm25 = BM25Retriever(chunks)
    semantic = SemanticRetriever(chunks, embeddings)
    hybrid = HybridRetriever(bm25, semantic)

    return {
        "BM25": bm25,
        "Semantic": semantic,
        "Hybrid": hybrid
    }


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)

    embeddings = np.load(
        EMBEDDINGS_PATH,
        allow_pickle=False
    )

    print("Initializing retrieval systems...")

    retrievers = build_retrievers(chunks, embeddings)

    print(f"Loaded {len(chunks)} passages")
    print(f"Running {len(QUERIES)} evaluation queries")
    print()

    rows = []
    detailed_results = []

    for query_id, query in enumerate(QUERIES, start=1):

        print("=" * 75)
        print(f"QUERY {query_id}: {query}")
        print("=" * 75)

        query_result = {
            "query_id": query_id,
            "query": query,
            "results": {}
        }

        for method in METHODS:

            results = retrievers[method].search(
                query,
                top_k=TOP_K,
                category=None
            )

            query_result["results"][method] = results

            print(f"\n{method} TOP 3:")

            if not results:
                print("  No results found")

            for rank, result in enumerate(
                results[:3], start=1
            ):
                print(
                    f"  {rank}. {result['title']}"
                )

            for rank, result in enumerate(
                results, start=1
            ):
                rows.append({
                    "query_id": query_id,
                    "query": query,
                    "method": method,
                    "rank": rank,
                    "document_id": result["document_id"],
                    "chunk_id": result["chunk_id"],
                    "title": result["title"],
                    "category": result["category"],
                    "score": result["score"],
                    "url": result["url"],
                    "snippet": result["text"][:500]
                })

        detailed_results.append(query_result)
        print()

    # Save CSV for manual relevance evaluation
    csv_path = OUTPUT_DIR / "baseline_results.csv"

    fieldnames = [
        "query_id",
        "query",
        "method",
        "rank",
        "document_id",
        "chunk_id",
        "title",
        "category",
        "score",
        "url",
        "snippet"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    # Save full results for later experiments
    json_path = OUTPUT_DIR / "baseline_results.json"

    output = {
        "created_at": datetime.now().isoformat(),
        "query_count": len(QUERIES),
        "top_k": TOP_K,
        "methods": METHODS,
        "results": detailed_results
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2
        )

    # Diagnostic: find the ear infection paper
    target = "Prevalence of Otitis Externa"
    query = QUERIES[0]

    total_documents = len({
        chunk["document_id"] for chunk in chunks
    })

    # Use all documents as Hybrid candidates
    diagnostic_hybrid = HybridRetriever(
        retrievers["BM25"],
        retrievers["Semantic"],
        candidate_k=total_documents
    )

    diagnostic_retrievers = {
        "BM25": retrievers["BM25"],
        "Semantic": retrievers["Semantic"],
        "Hybrid": diagnostic_hybrid
    }

    print("=" * 75)
    print("EAR INFECTION PAPER - RANK DIAGNOSTIC")
    print("=" * 75)

    for method, retriever in diagnostic_retrievers.items():

        results = retriever.search(
            query,
            top_k=total_documents
        )

        match_rank = next(
            (
                rank
                for rank, result in enumerate(results, 1)
                if target.lower() in result["title"].lower()
            ),
            None
        )

        print(f"{method}: rank {match_rank}")

    print()
    print("Evaluation completed!")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Total result rows: {len(rows)}")


if __name__ == "__main__":
    main()

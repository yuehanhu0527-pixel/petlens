
import json
from pathlib import Path

import numpy as np

from app.retrieval.semantic import SemanticRetriever


ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = ROOT / "data/processed/chunks.json"
EMBEDDINGS_PATH = ROOT / "data/processed/embeddings.npy"

OUTPUT_PATH = (
    ROOT / "data/evaluation/passage_selection_comparison.txt"
)

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


def passage_level_search(retriever, query, top_k=5):
    query_embedding = retriever.model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    scores = retriever.embeddings @ query_embedding

    ranked_indices = np.argsort(-scores, kind="stable")

    results = []

    for index in ranked_indices[:top_k]:
        chunk = retriever.chunks[int(index)]

        results.append({
            **chunk,
            "score": float(scores[index]),
        })

    return results


def format_results(name, results):
    lines = [f"\n{name}"]

    for rank, result in enumerate(results, 1):
        lines.extend([
            f"\n  Rank {rank}",
            f"  Document: {result['document_id']}",
            f"  Chunk: {result['chunk_id']}",
            f"  Title: {result['title']}",
            f"  Score: {result['score']:.4f}",
            f"  Text: {result['text'][:450]}",
        ])

    return "\n".join(lines)


def main():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)

    embeddings = np.load(
        EMBEDDINGS_PATH,
        allow_pickle=False,
    )

    retriever = SemanticRetriever(
        chunks=chunks,
        embeddings=embeddings,
    )

    report = []

    for query_id, query in enumerate(QUERIES, 1):
        print(f"Comparing query {query_id}/10...")

        original = retriever.search(
            query,
            top_k=5,
        )

        passage_level = passage_level_search(
            retriever,
            query,
            top_k=5,
        )

        report.extend([
            "\n" + "=" * 80,
            f"QUERY {query_id}: {query}",
            format_results("ORIGINAL SEMANTIC", original),
            format_results("PASSAGE-LEVEL SEMANTIC", passage_level),
        ])

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("\nComparison completed.")
    print("Saved to:", OUTPUT_PATH)
    print("Original retrieval code was not modified.")


if __name__ == "__main__":
    main()

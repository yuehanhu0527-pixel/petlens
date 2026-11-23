
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT / "data/evaluation/baseline_results.csv"
OUTPUT_PATH = ROOT / "data/evaluation/relevance_judgments.csv"


def main():

    with open(INPUT_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Pool unique passages per query across all retrievers.
    # Keep the best observed rank for easier review.
    unique = {}

    for row in rows:
        key = (
            row["query_id"],
            row["chunk_id"]
        )

        rank = int(row["rank"])

        if key not in unique:
            unique[key] = {
                "query_id": row["query_id"],
                "query": row["query"],
                "document_id": row["document_id"],
                "chunk_id": row["chunk_id"],
                "title": row["title"],
                "url": row["url"],
                "best_rank": rank,
                "retrieved_by": {row["method"]},
                "snippet": row["snippet"],
                "relevance": "",
            }
        else:
            unique[key]["best_rank"] = min(
                unique[key]["best_rank"], rank
            )

            unique[key]["retrieved_by"].add(
                row["method"]
            )

    output = []

    for row in unique.values():
        row["retrieved_by"] = ", ".join(
            sorted(row["retrieved_by"])
        )
        output.append(row)

    output.sort(
        key=lambda r: (
            int(r["query_id"]),
            r["best_rank"],
            r["title"]
        )
    )

    fields = [
        "query_id",
        "query",
        "document_id",
        "chunk_id",
        "title",
        "url",
        "best_rank",
        "retrieved_by",
        "snippet",
        "relevance",
    ]

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f, fieldnames=fields
        )
        writer.writeheader()
        writer.writerows(output)

    print(f"Original results: {len(rows)}")
    print(f"Unique passages to judge: {len(output)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

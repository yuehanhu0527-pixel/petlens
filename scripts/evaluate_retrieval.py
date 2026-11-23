
import csv
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/evaluation"

BASELINE = DATA_DIR / "baseline_results.csv"
TFIDF = DATA_DIR / "tfidf_results.csv"
JUDGMENTS = DATA_DIR / "petlens_relevance_judgments_four_methods_en.csv"
OUTPUT = DATA_DIR / "evaluation_metrics.csv"

METHODS = ["TF-IDF", "BM25", "Semantic", "Hybrid"]
K = 5


def load_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def precision_at_k(grades, k=5):
    relevant = sum(grade >= 1 for grade in grades[:k])
    return relevant / k


def reciprocal_rank(grades):
    for rank, grade in enumerate(grades, start=1):
        if grade >= 1:
            return 1.0 / rank
    return 0.0


def dcg_at_k(grades, k=5):
    return sum(
        (2 ** grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(grades[:k], start=1)
    )


def ndcg_at_k(grades, ideal_grades, k=5):
    actual = dcg_at_k(grades, k)
    ideal = dcg_at_k(
        sorted(ideal_grades, reverse=True), k
    )

    return actual / ideal if ideal > 0 else 0.0


def main():
    # Load retrieval results from all four methods
    results = load_csv(BASELINE) + load_csv(TFIDF)

    # Load human relevance judgments
    judgments = load_csv(JUDGMENTS)

    relevance = {}
    pooled_grades = defaultdict(list)

    for row in judgments:
        key = (row["query_id"], row["chunk_id"])

        if key in relevance:
            raise ValueError(f"Duplicate judgment: {key}")

        grade = int(row["relevance"])

        if grade not in (0, 1, 2):
            raise ValueError(f"Invalid relevance grade: {key}")

        relevance[key] = grade
        pooled_grades[row["query_id"]].append(grade)

    # Group results by query and retrieval method
    grouped = defaultdict(list)

    for row in results:
        key = (row["query_id"], row["method"])
        grouped[key].append(row)

    query_ids = sorted(
        {row["query_id"] for row in results},
        key=int
    )

    summary = []

    for method in METHODS:
        scores = []

        for query_id in query_ids:
            key = (query_id, method)

            if key not in grouped:
                raise ValueError(f"Missing results: {key}")

            ranked = sorted(
                grouped[key],
                key=lambda row: int(row["rank"])
            )

            if len(ranked) != K:
                raise ValueError(
                    f"Expected {K} results for {key}, "
                    f"found {len(ranked)}"
                )

            grades = []

            for row in ranked:
                judgment_key = (
                    row["query_id"],
                    row["chunk_id"]
                )

                if judgment_key not in relevance:
                    raise ValueError(
                        f"Missing judgment: {judgment_key}"
                    )

                grades.append(relevance[judgment_key])

            p5 = precision_at_k(grades, K)
            mrr = reciprocal_rank(grades)
            ndcg = ndcg_at_k(
                grades,
                pooled_grades[query_id],
                K
            )

            scores.append((p5, mrr, ndcg))

        count = len(scores)

        summary.append({
            "Method": method,
            "Precision@5": sum(s[0] for s in scores) / count,
            "MRR": sum(s[1] for s in scores) / count,
            "nDCG@5": sum(s[2] for s in scores) / count,
        })

    # Display results
    print("\nRETRIEVAL EVALUATION RESULTS\n")

    print(
        f"{'Method':<12}"
        f"{'P@5':>10}"
        f"{'MRR':>10}"
        f"{'nDCG@5':>12}"
    )

    print("-" * 44)

    for row in summary:
        print(
            f"{row['Method']:<12}"
            f"{row['Precision@5']:>10.4f}"
            f"{row['MRR']:>10.4f}"
            f"{row['nDCG@5']:>12.4f}"
        )

    # Save results
    with open(OUTPUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Method", "Precision@5", "MRR", "nDCG@5"]
        )
        writer.writeheader()
        writer.writerows(summary)

    print(f"\nSaved to: {OUTPUT}")


if __name__ == "__main__":
    main()

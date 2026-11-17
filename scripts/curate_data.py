import json
from pathlib import Path
from collections import Counter


ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = ROOT / "data/raw/articles_candidates.json"

REVIEW_FILE = ROOT / "data/curated/review_v1.json"

OUTPUT_FILE = ROOT / "data/curated/articles.json"


def main():

    articles = json.loads(
        RAW_FILE.read_text(encoding="utf-8")
    )

    review = json.loads(
        REVIEW_FILE.read_text(encoding="utf-8")
    )

    accepted_ids = set(review["accept"])

    curated = [
        article
        for article in articles
        if article["document_id"] in accepted_ids
    ]

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_FILE.write_text(
        json.dumps(
            curated,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print("Raw articles:", len(articles))
    print("Accepted articles:", len(curated))
    print("Excluded or pending:", len(articles) - len(curated))

    print("\nCategory distribution:")

    counts = Counter(
        article["category"]
        for article in curated
    )

    for category, count in counts.items():
        print(f"{category}: {count}")

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
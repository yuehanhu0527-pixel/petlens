import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/processed/cleaned_articles.json"
OUTPUT = ROOT / "data/processed/chunks.json"

CHUNK_SIZE = 300
OVERLAP = 50


def chunk_text(text, chunk_size=300, overlap=50):

    if not 0 <= overlap < chunk_size:
        raise ValueError("Invalid chunk size or overlap")

    words = text.split()
    chunks = []

    step = chunk_size - overlap

    for start in range(0, len(words), step):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        if end >= len(words):
            break

    return chunks


def main():

    articles = json.loads(
        INPUT.read_text(encoding="utf-8")
    )

    all_chunks = []

    for article in articles:

        passages = chunk_text(
            article["text"],
            CHUNK_SIZE,
            OVERLAP
        )

        for index, passage in enumerate(passages):

            chunk = {
                "chunk_id": (
                    f'{article["document_id"]}_{index:04d}'
                ),
                "document_id": article["document_id"],
                "chunk_index": index,
                "title": article["title"],
                "category": article["category"],
                "authors": article["authors"],
                "source": article["source"],
                "year": article["year"],
                "url": article["url"],
                "license": article["license"],
                "license_notice": article.get(
                    "license_notice", ""
                ),
                "copyright_notice": article.get(
                    "copyright_notice", ""
                ),
                "text": passage,
                "word_count": len(passage.split())
            }

            all_chunks.append(chunk)

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT.write_text(
        json.dumps(
            all_chunks,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    counts = Counter(
        chunk["category"]
        for chunk in all_chunks
    )

    print("=" * 50)
    print("Passage Chunking Complete")
    print("=" * 50)

    print("Input articles:", len(articles))
    print("Total chunks:", len(all_chunks))
    print("Chunk size:", CHUNK_SIZE)
    print("Overlap:", OVERLAP)

    print("\nCategory distribution:")

    for category, count in counts.items():
        print(f"{category}: {count}")

    print("\nSaved to:", OUTPUT)


if __name__ == "__main__":
    main()

import csv
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/chunks.json"
OUTPUT = ROOT / "data/evaluation/passage_quality_audit.csv"

BOILERPLATE_PATTERNS = [
    r"all claims expressed in this article",
    r"publisher'?s note",
    r"the authors declare that",
    r"conflict of interest statement",
    r"data availability statement",
    r"not used in the creation of this manuscript",
]


def inspect_passage(chunk):
    text = chunk.get("text", "")
    words = re.findall(r"\b\w+\b", text)
    flags = []

    if len(words) < 80:
        flags.append("very_short")

    if len(text) > 0:
        letter_ratio = sum(c.isalpha() for c in text) / len(text)

        if letter_ratio < 0.55:
            flags.append("low_letter_ratio")

    for pattern in BOILERPLATE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            flags.append("possible_boilerplate")
            break

    if re.search(
        r"(?:references|bibliography)\s*\n",
        text,
        re.IGNORECASE
    ):
        flags.append("possible_references")

    return flags, len(words)


def main():
    with open(INPUT, encoding="utf-8") as f:
        chunks = json.load(f)

    flagged = []
    counts = Counter()

    for chunk in chunks:
        flags, word_count = inspect_passage(chunk)

        for flag in flags:
            counts[flag] += 1

        if flags:
            flagged.append({
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "title": chunk["title"],
                "word_count": word_count,
                "flags": "; ".join(flags),
                "snippet": chunk["text"][:500],
            })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with open(
        OUTPUT, "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "chunk_id",
                "document_id",
                "title",
                "word_count",
                "flags",
                "snippet",
            ]
        )
        writer.writeheader()
        writer.writerows(flagged)

    print("\nPASSAGE QUALITY AUDIT")
    print("-" * 35)
    print(f"Total passages: {len(chunks)}")
    print(f"Flagged passages: {len(flagged)}")

    for flag, count in sorted(counts.items()):
        print(f"{flag}: {count}")

    print(f"\nSaved to: {OUTPUT}")
    print("\nNo passages were deleted or modified.")


if __name__ == "__main__":
    main()

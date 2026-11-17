import json
import re
import html
import unicodedata
from pathlib import Path
from collections import Counter


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/curated/articles.json"
OUTPUT = ROOT / "data/processed/cleaned_articles.json"


def clean_text(text):

    # Decode HTML entities
    text = html.unescape(text)

    # Normalize Unicode characters
    text = unicodedata.normalize("NFKC", text)

    # Remove common HTML tags
    text = re.sub(
        r"</?(?:i|b|em|strong|sup|sub|h[1-6]|p)\b[^>]*>",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Remove numeric citation markers
    text = re.sub(
        r"\[\d+(?:\s*[,;–\-]\s*\d+)*\]",
        "",
        text
    )

    # Remove publisher disclaimer paragraphs
    paragraphs = re.split(r"\n\s*\n", text)

    paragraphs = [
        p for p in paragraphs
        if not p.strip().lower().startswith(
            ("disclaimer/publisher", "publisher's note")
        )
    ]

    text = "\n\n".join(paragraphs)

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" +([.,;:!?])", r"\1", text)

    return text.strip()


def main():

    articles = json.loads(
        INPUT.read_text(encoding="utf-8")
    )

    cleaned_articles = []

    words_before = 0
    words_after = 0

    for article in articles:

        original = article["text"]
        cleaned = clean_text(original)

        words_before += len(original.split())
        words_after += len(cleaned.split())

        if not cleaned:
            continue

        new_article = article.copy()

        new_article["text"] = cleaned
        new_article["abstract"] = clean_text(
            article.get("abstract") or ""
        )
        new_article["title"] = clean_text(
            article["title"]
        )

        cleaned_articles.append(new_article)

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT.write_text(
        json.dumps(
            cleaned_articles,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print("=" * 50)
    print("Text Cleaning Complete")
    print("=" * 50)

    print("Input articles:", len(articles))
    print("Cleaned articles:", len(cleaned_articles))
    print("Words before:", words_before)
    print("Words after:", words_after)
    print("Words removed:", words_before - words_after)

    print("\nCategory distribution:")

    counts = Counter(
        article["category"]
        for article in cleaned_articles
    )

    for category, count in counts.items():
        print(f"{category}: {count}")

    print("\nSaved to:", OUTPUT)


if __name__ == "__main__":
    main()
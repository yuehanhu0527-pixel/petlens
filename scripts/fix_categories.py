
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

FILES = [
    ROOT / "data/processed/chunks.json",
    ROOT / "data/curated/articles.json",
]

CORRECTIONS = {
    "PMC11704566": "behavior",
    "PMC7500645": "feeding",
    "PMC7557453": "general_care",
}


def main():
    for path in FILES:
        if not path.exists():
            print(f"Not found, skipped: {path}")
            continue

        with open(path, encoding="utf-8") as f:
            records = json.load(f)

        if not isinstance(records, list):
            raise ValueError(
                f"Expected a list of records: {path}"
            )

        changes = []

        for record in records:
            document_id = record.get("document_id")

            if document_id not in CORRECTIONS:
                continue

            old_category = record.get("category")
            new_category = CORRECTIONS[document_id]

            if old_category != new_category:
                record["category"] = new_category
                changes.append((
                    document_id,
                    old_category,
                    new_category,
                ))

        if not changes:
            print(f"No changes needed: {path.name}")
            continue

        # Preserve the original file before making changes
        backup = path.with_name(
            path.stem + ".before_category_fix.json"
        )

        if not backup.exists():
            shutil.copy2(path, backup)

        temporary = path.with_name(path.name + ".tmp")

        with open(temporary, "w", encoding="utf-8") as f:
            json.dump(
                records,
                f,
                ensure_ascii=False,
                indent=2,
            )

        temporary.replace(path)

        print(f"\nUpdated: {path}")
        print(f"Backup: {backup}")

        for document_id, old, new in changes:
            print(f"  {document_id}: {old} -> {new}")

        print(f"Updated records: {len(changes)}")

    print("\nCategory correction completed.")


if __name__ == "__main__":
    main()

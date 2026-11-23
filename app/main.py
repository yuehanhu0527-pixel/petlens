
"""
PetLens - Evidence-Based Cat Care Search Engine.

Run with:
    python -m app.main
"""

from scripts.search_semantic import main as search_main


def main():
    print("=" * 50)
    print("PetLens")
    print("Evidence-Based Cat Care Search Engine")
    print("=" * 50)
    print()

    search_main()


if __name__ == "__main__":
    main()

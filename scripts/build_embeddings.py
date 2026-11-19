
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = ROOT / "data/processed/chunks.json"
EMBEDDINGS_PATH = ROOT / "data/processed/embeddings.npy"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():

    # 1. Load our existing 1,637 passages
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} passages")

    # 2. Load pretrained embedding model
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    # 3. Prepare text for embedding
    texts = [
        chunk["title"] + ". " + chunk["text"]
        for chunk in chunks
    ]

    # 4. Convert all passages into vectors
    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # 5. Save embeddings to disk
    np.save(EMBEDDINGS_PATH, embeddings)

    print()
    print("Embedding generation completed!")
    print(f"Number of passages: {len(chunks)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Saved to: {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()

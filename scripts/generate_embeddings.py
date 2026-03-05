"""
Generate the benchmark embedding dataset from STS Benchmark.

This script encodes STS-B sentences with all-MiniLM-L6-v2 and saves
them as an NPZ file. The NPZ ships with the library so that the
benchmark has zero runtime dependency on sentence-transformers.

Usage:
    pip install sentence-transformers datasets
    python scripts/generate_embeddings.py

Output:
    data/stsb_embeddings.npz
"""

import numpy as np


def main():
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    print("Loading STS Benchmark dataset...")
    dataset = load_dataset("mteb/stsbenchmark-sts", split="test")

    # Collect all unique sentences
    sentences = set()
    for row in dataset:
        sentences.add(row["sentence1"])
        sentences.add(row["sentence2"])
    sentences = sorted(sentences)
    print(f"  {len(sentences)} unique sentences")

    print("Encoding with all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences, show_progress_bar=True, batch_size=256)
    embeddings = np.array(embeddings, dtype=np.float32)

    print(f"  Shape: {embeddings.shape}")
    print(f"  Dtype: {embeddings.dtype}")
    print(f"  Size: {embeddings.nbytes / 1024 / 1024:.1f} MB")

    # Save
    out_path = "data/stsb_embeddings.npz"
    np.savez_compressed(out_path, embeddings=embeddings, sentences=np.array(sentences))
    print(f"Saved to {out_path}")

    import os
    file_size = os.path.getsize(out_path) / 1024 / 1024
    print(f"  File size: {file_size:.1f} MB")


if __name__ == "__main__":
    main()

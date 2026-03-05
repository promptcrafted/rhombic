"""
Benchmark the FCC vs Cubic lattice index on real or synthetic embeddings.

If data/stsb_embeddings.npz exists, uses real STS-B embeddings.
Otherwise, generates synthetic clustered embeddings (still a valid test).

Usage:
    python scripts/benchmark_index.py
"""

from __future__ import annotations

import os
import time
import numpy as np

from rhombic.index import FCCIndex, CubicIndex, brute_force_knn
from rhombic.context import generate_embeddings


def load_or_generate(max_vectors: int = 2000, dim: int = 384) -> np.ndarray:
    """Load real embeddings or generate synthetic ones.

    Synthetic embeddings use many overlapping clusters to create a
    realistic distribution where lattice topology matters.
    """
    npz_path = os.path.join(os.path.dirname(__file__), "..", "data", "stsb_embeddings.npz")
    npz_path = os.path.normpath(npz_path)

    if os.path.exists(npz_path):
        print(f"Loading real embeddings from {npz_path}...")
        data = np.load(npz_path)
        embeddings = data["embeddings"][:max_vectors]
        print(f"  Loaded {len(embeddings)} vectors, dim={embeddings.shape[1]}")
        return embeddings

    print(f"No NPZ found at {npz_path}")
    print(f"Generating {max_vectors} synthetic embeddings (dim={dim})...")
    # Many overlapping clusters — realistic embedding distribution
    embeddings = generate_embeddings(
        max_vectors, dim, n_clusters=30, seed=42
    )
    return embeddings.astype(np.float32)


def run_benchmark(embeddings: np.ndarray,
                  target_nodes: list[int] = [125, 500, 1000],
                  k: int = 10,
                  max_hops: int = 3,
                  n_queries: int = 100):
    """Run the full index benchmark with matched node counts.

    The comparison is fair: both topologies get the same number of
    lattice nodes. The ONLY variable is the connectivity pattern
    (6 vs 12 neighbors).
    """
    dim = embeddings.shape[1]

    # Split: use first n_queries as queries, rest as corpus
    queries = embeddings[:n_queries]
    corpus = embeddings[n_queries:]

    print(f"\nCorpus: {len(corpus)} vectors, dim={dim}")
    print(f"Queries: {len(queries)}")
    print(f"k={k}, hops=1..{max_hops}")

    # Ground truth
    print("Computing brute-force ground truth...")
    t0 = time.perf_counter()
    gt = brute_force_knn(corpus, queries, k=k)
    bf_time = time.perf_counter() - t0
    print(f"  Brute force: {bf_time:.3f}s")

    # Header
    print(f"\n{'Target':>7} {'Type':>7} {'Nodes':>7} {'Build(s)':>9} "
          f"{'Hops':>5} {'Recall@{k}':>10} {'Query(ms)':>10}")
    print("-" * 70)

    results = []

    for target in target_nodes:
        for IndexClass in [CubicIndex, FCCIndex]:
            idx = IndexClass.from_target_nodes(dim=dim, target_nodes=target)

            # Build
            t0 = time.perf_counter()
            idx.build(corpus)
            build_time = time.perf_counter() - t0

            for hops in range(1, max_hops + 1):
                # Recall
                t0 = time.perf_counter()
                recall = idx.recall_at_k(queries, gt, k=k, hops=hops)
                query_time = (time.perf_counter() - t0) / len(queries) * 1000

                label = "FCC" if "FCC" in idx.name else "Cubic"
                print(f"{target:>7} {label:>7} {idx.node_count:>7} "
                      f"{build_time:>9.3f} {hops:>5} {recall:>10.3f} "
                      f"{query_time:>10.2f}")

                results.append({
                    "target": target,
                    "type": label,
                    "nodes": idx.node_count,
                    "build_s": build_time,
                    "hops": hops,
                    "recall": recall,
                    "query_ms": query_time,
                })

        print()

    # Summary: FCC advantage at 1-hop
    print("\n=== FCC Advantage (1-hop recall@10) ===")
    for target in target_nodes:
        cubic_r = [r for r in results
                   if r["target"] == target and r["type"] == "Cubic" and r["hops"] == 1]
        fcc_r = [r for r in results
                 if r["target"] == target and r["type"] == "FCC" and r["hops"] == 1]
        if cubic_r and fcc_r:
            delta = fcc_r[0]["recall"] - cubic_r[0]["recall"]
            print(f"  ~{target} nodes: FCC {fcc_r[0]['recall']:.3f} "
                  f"(n={fcc_r[0]['nodes']}) vs "
                  f"Cubic {cubic_r[0]['recall']:.3f} "
                  f"(n={cubic_r[0]['nodes']}) "
                  f"delta: {delta:+.3f} ({delta*100:+.1f}pp)")

    return results


def main():
    embeddings = load_or_generate(max_vectors=2000, dim=384)
    run_benchmark(
        embeddings,
        target_nodes=[125, 500, 1000],
        n_queries=200,
    )


if __name__ == "__main__":
    main()

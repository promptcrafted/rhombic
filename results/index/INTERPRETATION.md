# FCC Embedding Index — Interpretation

## What the Index Proves

The FCC lattice index is not a production vector database. It is an
instrument that makes the Rung 4 recall advantage tangible: **build an
index, run a query, measure the gap.**

At matched node counts, the FCC topology captures +6.6 to +19.5
percentage points more of an embedding's true nearest neighbors at 1-hop
than the cubic topology. The only variable is the connectivity pattern:
12 neighbors (FCC) vs 6 neighbors (cubic). Same vectors, same PCA
projection, same assignment algorithm, same cosine ranking.

## Why the Advantage Grows with Scale

At 125 nodes, both topologies have enough vectors per cell that even
cubic's sparser neighborhood captures most neighbors. The delta is
modest (+6.6pp).

At 500 and 1000 nodes, each cell holds fewer vectors. The query's true
neighbors are scattered across more cells. Here, FCC's 12-connected
neighborhood covers a broader lattice volume per hop — the same flood
fill advantage measured in Rung 2 (55% more nodes per hop) now translates
directly into more retrieved neighbors.

This is the Rung 1-4 cascade in a single measurement: shorter paths
(Rung 1) → more flood fill reach (Rung 2) → better spatial coverage
(Rung 3) → higher embedding recall (Rung 4) → a working index that
demonstrates it (this).

## Connection to SYNTHESIS.md Section 2.4

The synthesis argues that FCC topology could improve RAG retrieval by
organizing embedding chunks on a 12-connected lattice instead of flat
top-k. This index is the proof-of-concept: the recall gap exists, it
holds across scales, and it derives from geometry — not tuning.

The cost remains: ~2× edges (12 vs 6 neighbors per node). Whether the
recall improvement justifies the storage and traversal cost depends on
the application. For latency-critical RAG where each hop is expensive,
+17pp recall at 1-hop may justify the denser graph. For applications
that can afford 3+ hops, both topologies converge.

## Design Decisions

**PCA over random projection.** The context.py module uses random
projection (Johnson-Lindenstrauss). The index uses PCA for
determinism — same embeddings always produce same results. PCA also
preserves maximum variance, which means the 3D projection best
separates the clusters that matter for recall.

**Cosine similarity for ranking.** After collecting lattice-neighborhood
candidates, we rank by cosine similarity in the original high-dimensional
space. The lattice provides coarse spatial organization; the final
ranking is exact within the candidate set.

**No quantization, no compression.** This is a proof-of-concept. The
full original vectors are stored. A production system would apply
product quantization, but that would obscure the topology comparison
this index exists to make visible.

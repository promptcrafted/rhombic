# FCC Embedding Index — Results

> Synthetic benchmark: 1,800 corpus vectors (384D, 30 overlapping clusters),
> 200 queries, k=10. Ground truth by brute-force cosine similarity.
> Node counts matched between topologies.

## Recall@10 by Hop Depth

| Target Nodes | Topology | Actual Nodes | 1-hop | 2-hop | 3-hop |
|-------------:|----------|-------------:|------:|------:|------:|
| 125 | Cubic | 125 | 0.379 | 0.788 | 0.999 |
| 125 | **FCC** | 108 | **0.445** | **0.971** | **1.000** |
| 500 | Cubic | 512 | 0.085 | 0.324 | 0.551 |
| 500 | **FCC** | 500 | **0.280** | **0.488** | **0.789** |
| 1000 | Cubic | 1000 | 0.044 | 0.138 | 0.414 |
| 1000 | **FCC** | 864 | **0.216** | **0.349** | **0.616** |

## FCC Advantage at 1-hop

| Target Nodes | FCC Recall | Cubic Recall | Delta |
|-------------:|-----------:|-------------:|------:|
| 125 | 0.445 | 0.379 | **+6.6pp** |
| 500 | 0.280 | 0.085 | **+19.5pp** |
| 1000 | 0.216 | 0.044 | **+17.2pp** |

## Reproduction

```bash
pip install rhombic
python scripts/benchmark_index.py
```

Without the STS-B NPZ, the script generates synthetic clustered embeddings
(30 overlapping clusters in 384D). With the NPZ (`data/stsb_embeddings.npz`),
it uses real sentence embeddings.

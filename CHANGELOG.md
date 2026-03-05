# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-04

### Added

- **Rung 0: Lattice Library** — `CubicLattice` and `FCCLattice` classes with
  matched node counts, interior degree verification, and NetworkX export
- **Rung 1: Graph Theory** — average shortest path, diameter, algebraic
  connectivity, and clustering coefficient benchmarks across three scales
- **Rung 2: Spatial Operations** — flood fill, nearest-neighbor queries,
  range queries, and locality-sensitive hashing benchmarks
- **Rung 3: Signal Processing** — 3D bandlimited signal reconstruction MSE
  and directional isotropy analysis (Petersen-Middleton confirmation)
- **Rung 4: Context Architecture** — embedding neighbor recall, information
  diffusion rate, and consensus convergence benchmarks
- **Rung 5: Synthesis** — "The Shape of the Cell" complete synthesis document
  with cultural genealogy, cybernetic interpretation, and recommendations
- Visualization module with Vadrashinetal color palette
- Banner generation from library code (`scripts/generate_banner.py`)
- CI workflow (Python 3.10, 3.11, 3.12) with benchmark reproduction
- Security scanning (gitleaks)
- Full community health files (CONTRIBUTING, CODE_OF_CONDUCT, SECURITY)

### Key Results

| Metric | FCC vs Cubic |
|--------|-------------|
| Average shortest path | 30% shorter |
| Graph diameter | 40% smaller |
| Algebraic connectivity | 2.4x higher |
| Flood fill reach | 55% more nodes |
| Signal reconstruction MSE | 5-10x lower |
| Embedding neighbor recall | +15-26pp at 1-hop |
| Edge cost | ~2x more edges |

[0.1.0]: https://github.com/promptcrafted/rhombic/releases/tag/v0.1.0

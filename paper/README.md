# arXiv Preprint

**Title:** The Shape of the Cell: Empirical Comparison of Cubic and FCC Lattice
Topologies Across Graph Theory, Spatial Operations, Signal Processing, and
Embedding Retrieval

**Target:** arXiv cs.DS (primary), cs.IR (secondary)

## Building

Requires a LaTeX distribution (TeX Live, MiKTeX, or Overleaf).

```bash
cd paper/
pdflatex rhombic.tex
bibtex rhombic
pdflatex rhombic.tex
pdflatex rhombic.tex
```

Or upload `rhombic.tex` and `rhombic.bib` to [Overleaf](https://overleaf.com).

## Submission

1. Build the PDF and verify all tables render correctly
2. Submit to arXiv at https://arxiv.org/submit
3. Primary category: cs.DS (Data Structures and Algorithms)
4. Cross-list: cs.IR (Information Retrieval)
5. Add comment: "Open-source library: https://github.com/promptcrafted/rhombic"

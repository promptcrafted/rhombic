"""
Lattice-organized approximate nearest-neighbor index.

Proof-of-concept demonstrating that FCC topology captures more true
nearest neighbors per lattice hop than cubic topology. This is "eat
your own cooking" — the Rung 4 recall advantage made into a tool
you can build() and query().

Architecture:
  1. Project high-dim embeddings to 3D via PCA (deterministic)
  2. Quantize to lattice nodes via KDTree nearest-assignment
  3. Store original high-dim vectors at their assigned nodes
  4. Query by projecting query to 3D, finding its node, collecting all
     vectors from k-hop neighborhood, ranking by cosine similarity
  5. Report recall@k against brute-force ground truth

This is a proof-of-concept instrument, not a production vector database.
Clarity over speed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
from scipy.spatial import KDTree

from rhombic.lattice import CubicLattice, FCCLattice
from rhombic.spatial import build_adjacency, flood_fill


class LatticeIndex(ABC):
    """Base class for lattice-organized ANN indexes.

    Parameters
    ----------
    dim : int
        Dimensionality of the embedding vectors.
    n_cells_per_side : int
        Lattice cells per side. Controls granularity.
    """

    def __init__(self, dim: int, n_cells_per_side: int = 5):
        self.dim = dim
        self.n_cells_per_side = n_cells_per_side
        self._lattice = None
        self._adjacency: dict[int, list[int]] = {}
        self._node_vectors: dict[int, list[int]] = defaultdict(list)
        self._embeddings: np.ndarray | None = None
        self._pca_components: np.ndarray | None = None
        self._pca_mean: np.ndarray | None = None
        self._lattice_kdtree: KDTree | None = None
        self._scale_min: np.ndarray | None = None
        self._scale_extent: np.ndarray | None = None
        self._built = False

    @abstractmethod
    def _create_lattice(self, n: int):
        """Create the lattice instance. Subclasses implement this."""
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def node_count(self) -> int:
        return self._lattice.node_count if self._lattice else 0

    @property
    def topology(self) -> str:
        if isinstance(self._lattice, FCCLattice):
            return "FCC (12-connected)"
        elif isinstance(self._lattice, CubicLattice):
            return "Cubic (6-connected)"
        return "Unknown"

    def build(self, embeddings: np.ndarray) -> "LatticeIndex":
        """Build the index from a matrix of embedding vectors.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape (n_vectors, dim). The vectors to index.

        Returns
        -------
        self
            For chaining.
        """
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {embeddings.shape}")
        if embeddings.shape[0] == 0:
            self._built = True
            self._embeddings = embeddings
            return self
        if embeddings.shape[1] != self.dim:
            raise ValueError(
                f"Expected dim={self.dim}, got vectors of dim={embeddings.shape[1]}"
            )

        self._embeddings = embeddings.copy()
        n_vectors = len(embeddings)

        # Step 1: PCA projection to 3D (deterministic)
        self._pca_mean = embeddings.mean(axis=0)
        centered = embeddings - self._pca_mean

        if self.dim <= 3:
            # Already low-dim — pad or use directly
            self._pca_components = np.eye(self.dim, 3)
            projected = centered @ self._pca_components
        else:
            # SVD-based PCA (top 3 components)
            # Use economy SVD for efficiency
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            self._pca_components = Vt[:3].T  # shape (dim, 3)
            projected = centered @ self._pca_components

        # Step 2: Scale to lattice bounding box
        self._lattice = self._create_lattice(self.n_cells_per_side)
        self._adjacency = build_adjacency(
            self._lattice.edges, self._lattice.node_count
        )

        lattice_pos = self._lattice.positions
        lmin = lattice_pos.min(axis=0)
        lmax = lattice_pos.max(axis=0)
        lextent = lmax - lmin
        lextent = np.where(lextent > 0, lextent, 1.0)

        pmin = projected.min(axis=0)
        pmax = projected.max(axis=0)
        pextent = pmax - pmin
        pextent = np.where(pextent > 0, pextent, 1.0)

        self._scale_min = pmin
        self._scale_extent = pextent

        # Map projected points into lattice bounding box with margin
        margin = 0.1
        scaled = lmin + lextent * (
            margin + (projected - pmin) / pextent * (1 - 2 * margin)
        )

        # Step 3: Assign to nearest lattice node
        self._lattice_kdtree = KDTree(lattice_pos)
        _, assignments = self._lattice_kdtree.query(scaled)

        # Step 4: Store vector indices at their assigned nodes
        self._node_vectors = defaultdict(list)
        for vec_idx, node_idx in enumerate(assignments):
            self._node_vectors[node_idx].append(vec_idx)

        self._built = True
        return self

    def query(self, q: np.ndarray, k: int = 10, hops: int = 1) -> np.ndarray:
        """Find approximate k nearest neighbors of query vector.

        Parameters
        ----------
        q : np.ndarray
            Query vector, shape (dim,).
        k : int
            Number of neighbors to return.
        hops : int
            Lattice hops to search (larger = more candidates, slower).

        Returns
        -------
        np.ndarray
            Indices into the original embeddings array, sorted by
            descending cosine similarity. Length <= k.
        """
        if not self._built:
            raise RuntimeError("Index not built. Call build() first.")
        if self._embeddings is None or len(self._embeddings) == 0:
            return np.array([], dtype=int)

        # Project query to 3D
        q_centered = q - self._pca_mean
        q_3d = q_centered @ self._pca_components

        # Scale to lattice space
        lattice_pos = self._lattice.positions
        lmin = lattice_pos.min(axis=0)
        lmax = lattice_pos.max(axis=0)
        lextent = lmax - lmin
        lextent = np.where(lextent > 0, lextent, 1.0)

        margin = 0.1
        q_scaled = lmin + lextent * (
            margin
            + (q_3d - self._scale_min) / self._scale_extent * (1 - 2 * margin)
        )

        # Find nearest lattice node
        _, node_idx = self._lattice_kdtree.query(q_scaled)

        # Collect candidates from k-hop neighborhood
        neighborhood = flood_fill(self._adjacency, int(node_idx), hops)
        candidate_indices = []
        for node in neighborhood:
            candidate_indices.extend(self._node_vectors.get(node, []))

        if not candidate_indices:
            return np.array([], dtype=int)

        # Rank by cosine similarity in original space
        candidates = np.array(candidate_indices)
        candidate_vecs = self._embeddings[candidates]

        q_norm = q / (np.linalg.norm(q) + 1e-10)
        c_norms = candidate_vecs / (
            np.linalg.norm(candidate_vecs, axis=1, keepdims=True) + 1e-10
        )
        similarities = c_norms @ q_norm

        # Top k
        top_k = min(k, len(similarities))
        if top_k >= len(similarities):
            # Fewer candidates than k — just sort all
            top_indices = np.argsort(-similarities)
        else:
            top_indices = np.argpartition(-similarities, top_k)[:top_k]
            top_indices = top_indices[np.argsort(-similarities[top_indices])]

        return candidates[top_indices]

    def recall_at_k(
        self,
        queries: np.ndarray,
        ground_truth: np.ndarray,
        k: int = 10,
        hops: int = 1,
    ) -> float:
        """Measure recall@k against brute-force ground truth.

        Parameters
        ----------
        queries : np.ndarray
            Shape (n_queries, dim). Query vectors.
        ground_truth : np.ndarray
            Shape (n_queries, k). True k-NN indices per query (from
            brute-force search in original space).
        k : int
            Number of neighbors to retrieve.
        hops : int
            Lattice hops to search.

        Returns
        -------
        float
            Mean recall across all queries: fraction of true k-NN
            found in the approximate results.
        """
        if not self._built:
            raise RuntimeError("Index not built. Call build() first.")

        recalls = []
        for i in range(len(queries)):
            retrieved = set(self.query(queries[i], k=k, hops=hops))
            true_set = set(ground_truth[i][:k])
            if len(true_set) == 0:
                continue
            recalls.append(len(retrieved & true_set) / len(true_set))

        return float(np.mean(recalls)) if recalls else 0.0


class FCCIndex(LatticeIndex):
    """FCC (12-connected) lattice index.

    The rhombic dodecahedral topology provides denser local neighborhoods,
    which should capture more true nearest neighbors per hop.
    """

    def _create_lattice(self, n: int):
        return FCCLattice(n)

    @classmethod
    def from_target_nodes(cls, dim: int, target_nodes: int) -> "FCCIndex":
        """Create an FCCIndex sized to have ~target_nodes lattice nodes.

        FCC has ~4n³ nodes, so n ≈ (target/4)^(1/3).
        """
        n = max(2, round((target_nodes / 4) ** (1 / 3)))
        return cls(dim=dim, n_cells_per_side=n)


class CubicIndex(LatticeIndex):
    """Cubic (6-connected) lattice index.

    The standard rectilinear topology. Baseline for comparison.
    """

    def _create_lattice(self, n: int):
        return CubicLattice(n)

    @classmethod
    def from_target_nodes(cls, dim: int, target_nodes: int) -> "CubicIndex":
        """Create a CubicIndex sized to have ~target_nodes lattice nodes.

        Cubic has n³ nodes, so n ≈ target^(1/3).
        """
        n = max(2, round(target_nodes ** (1 / 3)))
        return cls(dim=dim, n_cells_per_side=n)


def brute_force_knn(
    embeddings: np.ndarray, queries: np.ndarray, k: int = 10
) -> np.ndarray:
    """Compute exact k-NN by brute-force cosine similarity.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape (n, dim). The indexed vectors.
    queries : np.ndarray
        Shape (m, dim). The query vectors.
    k : int
        Number of neighbors.

    Returns
    -------
    np.ndarray
        Shape (m, k). Indices of the k nearest neighbors per query.
    """
    # Normalize
    e_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    q_norm = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-10)

    # Cosine similarity matrix
    sim = q_norm @ e_norm.T  # (m, n)

    # Top k per query
    n_candidates = sim.shape[1]
    k = min(k, n_candidates)

    if k >= n_candidates:
        # Fewer candidates than k — sort all
        result = np.argsort(-sim, axis=1)
    else:
        top_k_indices = np.argpartition(-sim, k, axis=1)[:, :k]
        result = np.empty_like(top_k_indices)
        for i in range(len(queries)):
            order = np.argsort(-sim[i, top_k_indices[i]])
            result[i] = top_k_indices[i][order]

    return result

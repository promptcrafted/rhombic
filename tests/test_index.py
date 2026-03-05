"""Tests for the lattice-organized ANN index."""

import numpy as np
import pytest

from rhombic.index import FCCIndex, CubicIndex, LatticeIndex, brute_force_knn


# ── Fixtures ──────────────────────────────────────────────────────────


def _clustered_embeddings(n=200, dim=64, n_clusters=5, seed=42):
    """Generate clustered embeddings that reward spatial organization."""
    rng = np.random.default_rng(seed)
    per_cluster = n // n_clusters
    remainder = n - per_cluster * n_clusters
    embeddings = []
    for i in range(n_clusters):
        center = rng.standard_normal(dim) * 3.0
        spread = rng.uniform(0.3, 1.0)
        count = per_cluster + (1 if i < remainder else 0)
        cluster = center + rng.standard_normal((count, dim)) * spread
        embeddings.append(cluster)
    return np.vstack(embeddings)


@pytest.fixture
def embeddings():
    return _clustered_embeddings(200, 64)


@pytest.fixture
def queries(embeddings):
    """Use a subset of embeddings as queries."""
    return embeddings[:20]


@pytest.fixture
def ground_truth(embeddings, queries):
    """Brute-force k-NN for queries."""
    return brute_force_knn(embeddings, queries, k=10)


# ── Build / Query lifecycle ───────────────────────────────────────────


class TestBuildQuery:
    def test_fcc_build_returns_self(self, embeddings):
        idx = FCCIndex(dim=64, n_cells_per_side=4)
        result = idx.build(embeddings)
        assert result is idx

    def test_cubic_build_returns_self(self, embeddings):
        idx = CubicIndex(dim=64, n_cells_per_side=5)
        result = idx.build(embeddings)
        assert result is idx

    def test_query_returns_indices(self, embeddings):
        idx = FCCIndex(dim=64, n_cells_per_side=4).build(embeddings)
        result = idx.query(embeddings[0], k=5, hops=1)
        assert isinstance(result, np.ndarray)
        assert result.dtype in (np.int32, np.int64, np.intp)
        assert len(result) <= 5
        # All indices should be valid
        for i in result:
            assert 0 <= i < len(embeddings)

    def test_query_before_build_raises(self):
        idx = FCCIndex(dim=64)
        with pytest.raises(RuntimeError, match="not built"):
            idx.query(np.zeros(64))

    def test_recall_before_build_raises(self):
        idx = CubicIndex(dim=64)
        with pytest.raises(RuntimeError, match="not built"):
            idx.recall_at_k(np.zeros((1, 64)), np.zeros((1, 10), dtype=int))

    def test_wrong_dim_raises(self, embeddings):
        idx = FCCIndex(dim=32)  # mismatch
        with pytest.raises(ValueError, match="dim=32"):
            idx.build(embeddings)  # embeddings are dim=64


# ── Recall computation ────────────────────────────────────────────────


class TestRecall:
    def test_recall_between_0_and_1(self, embeddings, queries, ground_truth):
        idx = FCCIndex(dim=64, n_cells_per_side=4).build(embeddings)
        recall = idx.recall_at_k(queries, ground_truth, k=10, hops=2)
        assert 0.0 <= recall <= 1.0

    def test_recall_increases_with_hops(self, embeddings, queries, ground_truth):
        idx = FCCIndex(dim=64, n_cells_per_side=4).build(embeddings)
        r1 = idx.recall_at_k(queries, ground_truth, k=10, hops=1)
        r3 = idx.recall_at_k(queries, ground_truth, k=10, hops=3)
        # More hops should give equal or better recall
        assert r3 >= r1 - 0.01  # small tolerance for edge cases


# ── FCC vs Cubic property test ────────────────────────────────────────


class TestFCCAdvantage:
    def test_fcc_recall_geq_cubic_on_clustered_data(self):
        """FCC should capture >= cubic recall on clustered embeddings.

        This is the core property test: the Rung 4 advantage should
        manifest when the data has cluster structure (which real
        embeddings do).
        """
        embeddings = _clustered_embeddings(300, 64, n_clusters=8, seed=99)
        queries = embeddings[:30]
        gt = brute_force_knn(embeddings, queries, k=10)

        fcc = FCCIndex(dim=64, n_cells_per_side=4).build(embeddings)
        cubic = CubicIndex(dim=64, n_cells_per_side=5).build(embeddings)

        fcc_recall = fcc.recall_at_k(queries, gt, k=10, hops=1)
        cubic_recall = cubic.recall_at_k(queries, gt, k=10, hops=1)

        # FCC should match or beat cubic. Allow small margin for
        # stochastic effects in the PCA projection.
        assert fcc_recall >= cubic_recall - 0.05, (
            f"FCC recall ({fcc_recall:.3f}) significantly below "
            f"cubic ({cubic_recall:.3f})"
        )


# ── Edge cases ────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_embeddings(self):
        idx = FCCIndex(dim=64, n_cells_per_side=3)
        idx.build(np.empty((0, 64)))
        result = idx.query(np.zeros(64), k=5)
        assert len(result) == 0

    def test_single_point(self):
        emb = np.random.default_rng(42).standard_normal((1, 32))
        idx = FCCIndex(dim=32, n_cells_per_side=3).build(emb)
        result = idx.query(emb[0], k=5)
        assert len(result) == 1
        assert result[0] == 0

    def test_duplicate_points(self):
        base = np.random.default_rng(42).standard_normal((1, 32))
        emb = np.tile(base, (10, 1))
        idx = FCCIndex(dim=32, n_cells_per_side=3).build(emb)
        result = idx.query(emb[0], k=5)
        assert len(result) <= 5
        # All results should be valid indices
        for i in result:
            assert 0 <= i < 10

    def test_low_dim_embeddings(self):
        """Test with 3D embeddings (no PCA needed)."""
        emb = np.random.default_rng(42).standard_normal((50, 3))
        idx = FCCIndex(dim=3, n_cells_per_side=3).build(emb)
        result = idx.query(emb[0], k=5)
        assert len(result) <= 5


# ── Brute force helper ────────────────────────────────────────────────


class TestBruteForce:
    def test_self_is_nearest(self):
        emb = np.random.default_rng(42).standard_normal((20, 16))
        gt = brute_force_knn(emb, emb, k=1)
        # Each point should be its own nearest neighbor
        for i in range(20):
            assert gt[i, 0] == i

    def test_shape(self):
        emb = np.random.default_rng(42).standard_normal((50, 16))
        queries = emb[:10]
        gt = brute_force_knn(emb, queries, k=5)
        assert gt.shape == (10, 5)

    def test_k_larger_than_n(self):
        emb = np.random.default_rng(42).standard_normal((3, 8))
        gt = brute_force_knn(emb, emb[:1], k=10)
        assert gt.shape == (1, 3)  # capped at n


# ── Index metadata ────────────────────────────────────────────────────


class TestMetadata:
    def test_name(self):
        assert FCCIndex(dim=64).name == "FCCIndex"
        assert CubicIndex(dim=64).name == "CubicIndex"

    def test_topology(self, embeddings):
        fcc = FCCIndex(dim=64, n_cells_per_side=3).build(embeddings)
        cubic = CubicIndex(dim=64, n_cells_per_side=4).build(embeddings)
        assert "12" in fcc.topology
        assert "6" in cubic.topology

    def test_node_count(self, embeddings):
        idx = FCCIndex(dim=64, n_cells_per_side=3).build(embeddings)
        assert idx.node_count > 0

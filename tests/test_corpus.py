"""Tests for the isopsephy corpus data module."""

import pytest
from rhombic.corpus import (
    TRACKED_PRIMES, CANONICAL_EDGE_ORDER,
    edge_values, trump_values, names_of_power,
    prime_factors, prime_factor_set, shared_factors,
    prime_membership, weight_distributions, corpus_stats,
    corpus_available,
)

# Skip corpus-dependent tests if private data not available
requires_corpus = pytest.mark.skipif(
    not corpus_available(),
    reason="Corpus values are proprietary and not available in this environment"
)


class TestDataIntegrity:
    """Verify the corpus data is complete and correct."""

    @requires_corpus
    def test_24_trump_values(self):
        assert len(trump_values()) == 24

    @requires_corpus
    def test_14_names_of_power(self):
        assert len(names_of_power()) == 14

    def test_8_tracked_primes(self):
        assert len(TRACKED_PRIMES) == 8

    def test_canonical_order_length(self):
        assert len(CANONICAL_EDGE_ORDER) == 24

    @requires_corpus
    def test_edge_values_length(self):
        assert len(edge_values()) == 24

    @requires_corpus
    def test_all_values_positive(self):
        for v in edge_values():
            assert v > 0


class TestKnownValues:
    """Spot-check verified values from the corpus."""

    @requires_corpus
    def test_geareij(self):
        """Card 21: GEAREIJ = 134 = 2 * 67."""
        assert trump_values()[21] == 134

    @requires_corpus
    def test_bral_donace_identity(self):
        """BRAL (Card 1 component) = DONACE (Card 16) = 133 = 7 * 19."""
        assert trump_values()[16] == 133

    @requires_corpus
    def test_maat(self):
        """Card 8: MA∀T standard = 342."""
        assert trump_values()[8] == 342

    @requires_corpus
    def test_dajibaa(self):
        """Card 15: DAJIBA∀ = 29 (prime)."""
        assert trump_values()[15] == 29

    @requires_corpus
    def test_belial(self):
        """Transit: BELIAL = 78 = T(12)."""
        assert trump_values()["T"] == 78

    @requires_corpus
    def test_rirajla(self):
        """Grail: RIRAJLA = 252."""
        assert trump_values()["G"] == 252

    @requires_corpus
    def test_fool_combined(self):
        """Card 0: UAT ASETEDOJ combined = 1296 = 6^4."""
        assert trump_values()[0] == 1296

    @requires_corpus
    def test_samma(self):
        assert names_of_power()["SAMMA"] == 282

    @requires_corpus
    def test_vadusfadahm(self):
        assert names_of_power()["VADUSFADAHM"] == 671

    @requires_corpus
    def test_osiris(self):
        assert names_of_power()["OSIRIS"] == 590


class TestFactorization:
    """Verify prime factorization functions."""

    def test_134_factors(self):
        """134 = 2 * 67."""
        assert prime_factors(134) == [2, 67]

    def test_133_factors(self):
        """133 = 7 * 19."""
        assert prime_factors(133) == [7, 19]

    def test_1296_factors(self):
        """1296 = 2^4 * 3^4 = 6^4."""
        f = prime_factors(1296)
        assert f.count(2) == 4
        assert f.count(3) == 4

    def test_29_is_prime(self):
        assert prime_factors(29) == [29]

    def test_prime_factor_set(self):
        assert prime_factor_set(342) == {2, 3, 19}

    def test_factors_of_1(self):
        assert prime_factors(1) == [1]


class TestSharedFactors:

    def test_shared_134_133(self):
        """134 = 2*67, 133 = 7*19: no shared factors."""
        assert shared_factors(134, 133) == set()

    def test_shared_342_134(self):
        """342 = 2*3^2*19, 134 = 2*67: share {2}."""
        assert shared_factors(342, 134) == {2}

    def test_self_shared(self):
        """A value shares all its own factors."""
        assert shared_factors(342, 342) == {2, 3, 19}


class TestPrimeMembership:

    def test_67_divides_134(self):
        assert prime_membership(134, 67) is True

    def test_67_not_divides_133(self):
        assert prime_membership(133, 67) is False

    def test_23_divides_69(self):
        """ALEBAL = 69 = 3 * 23."""
        assert prime_membership(69, 23) is True


class TestWeightDistributions:

    def test_at_least_three_distributions(self):
        dists = weight_distributions()
        assert {'uniform', 'random', 'power_law'}.issubset(dists.keys())

    @requires_corpus
    def test_four_distributions_with_corpus(self):
        dists = weight_distributions()
        assert 'corpus' in dists

    def test_non_corpus_have_24_values(self):
        dists = weight_distributions()
        for name in ('uniform', 'random', 'power_law'):
            assert len(dists[name]) == 24, f"{name} has {len(dists[name])} values"

    def test_uniform_all_ones(self):
        dists = weight_distributions()
        assert all(v == 1.0 for v in dists['uniform'])

    def test_all_non_negative(self):
        dists = weight_distributions()
        for name, values in dists.items():
            for v in values:
                assert v >= 0.0, f"{name} has negative value {v}"

    def test_deterministic(self):
        d1 = weight_distributions(seed=42)
        d2 = weight_distributions(seed=42)
        assert d1['random'] == d2['random']

    def test_different_seeds_differ(self):
        d1 = weight_distributions(seed=42)
        d2 = weight_distributions(seed=99)
        assert d1['random'] != d2['random']


class TestCorpusStats:

    @requires_corpus
    def test_stats_values(self):
        s = corpus_stats()
        assert s.n_values == 24
        assert s.min_value == 18    # GEJ
        assert s.max_value == 1296  # UAT ASETEDOJ

    @requires_corpus
    def test_tracked_prime_coverage(self):
        s = corpus_stats()
        assert s.tracked_prime_coverage[67] >= 1
        assert s.tracked_prime_coverage[29] >= 1

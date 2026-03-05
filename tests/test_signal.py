"""Tests for Rung 3: Signal Processing benchmarks."""

import numpy as np
import pytest
from rhombic.signal import (
    cubic_samples, fcc_samples, density_matched_samples,
    isotropic_signal, directional_signal,
    reconstruct, compute_mse, compute_psnr,
    run_signal_benchmark,
)


class TestSampleGeneration:
    """Test that lattice point generators produce correct outputs."""

    def test_cubic_count(self):
        pts = cubic_samples(5)
        assert len(pts) == 125  # 5³

    def test_cubic_in_unit_cube(self):
        pts = cubic_samples(8)
        assert pts.min() >= 0.0
        assert pts.max() <= 1.0

    def test_fcc_in_unit_cube(self):
        pts = fcc_samples(4)
        assert pts.min() >= 0.0
        assert pts.max() <= 1.0

    def test_fcc_more_than_cubic_per_cell(self):
        """FCC has ~4 points per unit cell vs cubic's 1."""
        n = 4
        c = cubic_samples(n)
        f = fcc_samples(n)
        # FCC should have substantially more points for same n
        assert len(f) > len(c)

    def test_density_matched_counts(self):
        """Density matching should produce equal or near-equal counts."""
        pts_c, pts_f = density_matched_samples(500)
        assert len(pts_c) == len(pts_f)


class TestSignals:
    """Test signal functions produce valid outputs."""

    def test_isotropic_signal_shape(self):
        pts = np.random.default_rng(0).uniform(0, 1, (100, 3))
        vals = isotropic_signal(pts, freq=1.0)
        assert vals.shape == (100,)

    def test_directional_signal_shape(self):
        pts = np.random.default_rng(0).uniform(0, 1, (100, 3))
        d = np.array([1, 0, 0], dtype=np.float64)
        vals = directional_signal(pts, d, freq=1.0)
        assert vals.shape == (100,)

    def test_isotropic_signal_bounded(self):
        pts = np.random.default_rng(0).uniform(0, 1, (1000, 3))
        vals = isotropic_signal(pts, freq=1.0)
        assert np.all(np.isfinite(vals))


class TestReconstruction:
    """Test reconstruction and metric computation."""

    def test_perfect_reconstruction(self):
        """Reconstructing from the same points should give near-zero MSE."""
        pts = np.random.default_rng(0).uniform(0.1, 0.9, (50, 3))
        vals = isotropic_signal(pts, freq=0.5)
        recon = reconstruct(pts, vals, pts)
        mse = compute_mse(vals, recon)
        assert mse < 1e-6

    def test_psnr_infinite_for_perfect(self):
        true = np.ones(100)
        psnr = compute_psnr(true, true)
        assert psnr == 100.0  # capped

    def test_psnr_positive_for_noisy(self):
        true = np.sin(np.linspace(0, 6, 100))
        noisy = true + 0.1 * np.random.default_rng(0).standard_normal(100)
        psnr = compute_psnr(true, noisy)
        assert 0 < psnr < 100


class TestFCCAdvantage:
    """Core hypothesis tests: FCC should outperform cubic for isotropic signals."""

    def test_fcc_better_psnr_mid_frequency(self):
        """FCC should win at frequencies below Nyquist."""
        r = run_signal_benchmark(target_count=216, n_eval=400, seed=42)
        # Check that FCC wins at the mid-frequency point
        mid = r.sweep[len(r.sweep) // 3]  # ~30% of Nyquist
        assert mid.psnr_diff > 0, (
            f"FCC should have higher PSNR at freq={mid.frequency:.1f}, "
            f"got cubic={mid.cubic_psnr:.2f} fcc={mid.fcc_psnr:.2f}"
        )

    def test_fcc_more_isotropic(self):
        """FCC should have lower directional error variance."""
        r = run_signal_benchmark(target_count=216, n_eval=400, seed=42)
        assert r.fcc_isotropy < r.cubic_isotropy, (
            f"FCC should be more isotropic: cubic={r.cubic_isotropy:.6f} "
            f"fcc={r.fcc_isotropy:.6f}"
        )

    def test_fcc_advantage_increases_near_nyquist(self):
        """FCC advantage should grow as frequency approaches Nyquist."""
        r = run_signal_benchmark(target_count=216, n_eval=400, seed=42)
        # Compare advantage at low freq vs mid freq
        low = r.sweep[0]
        mid = r.sweep[3]  # ~40% Nyquist
        assert mid.psnr_diff > low.psnr_diff, (
            f"FCC advantage should grow with frequency: "
            f"low={low.psnr_diff:+.2f}dB, mid={mid.psnr_diff:+.2f}dB"
        )


class TestBenchmarkRunner:
    """Test the full benchmark pipeline."""

    def test_benchmark_completes(self):
        r = run_signal_benchmark(target_count=125, n_eval=200, seed=42)
        assert r.cubic_count > 0
        assert r.fcc_count > 0
        assert len(r.sweep) == 10
        assert r.elapsed_seconds > 0

    def test_matched_counts(self):
        r = run_signal_benchmark(target_count=216, n_eval=200, seed=42)
        assert r.cubic_count == r.fcc_count

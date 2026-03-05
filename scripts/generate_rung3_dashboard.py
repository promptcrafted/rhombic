"""Generate the Rung 3 signal processing dashboard."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rhombic.signal import run_signal_benchmark

# 8-Law Weave palette
CUBIC_COLOR = '#3D3D6B'
FCC_COLOR = '#B34444'
BG_COLOR = '#0D0D0D'
TEXT_COLOR = '#E8E8E8'
GRID_COLOR = '#2A2A2A'


def main():
    # Run benchmarks at three scales
    scales = [216, 512, 1000]
    results = []
    for count in scales:
        print(f"Running signal benchmark with ~{count} samples...")
        r = run_signal_benchmark(count, n_eval=600)
        results.append(r)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10),
                             facecolor=BG_COLOR)
    fig.suptitle('Rung 3: Signal Processing — FCC vs Cubic Sampling',
                 color=TEXT_COLOR, fontsize=16, fontweight='bold', y=0.98)

    for ax in axes.flat:
        ax.set_facecolor(BG_COLOR)
        ax.tick_params(colors=TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        ax.grid(True, color=GRID_COLOR, alpha=0.3)

    # Panel 1: PSNR vs Frequency (all scales)
    ax = axes[0, 0]
    for i, r in enumerate(results):
        freqs = [s.frequency for s in r.sweep]
        c_psnr = [s.cubic_psnr for s in r.sweep]
        f_psnr = [s.fcc_psnr for s in r.sweep]
        alpha = 0.4 + 0.3 * i
        label_c = f"Cubic ({r.cubic_count})" if i == len(results)-1 else None
        label_f = f"FCC ({r.fcc_count})" if i == len(results)-1 else None
        ax.plot(freqs, c_psnr, color=CUBIC_COLOR, alpha=alpha,
                marker='s', markersize=3, label=label_c)
        ax.plot(freqs, f_psnr, color=FCC_COLOR, alpha=alpha,
                marker='o', markersize=3, label=label_f)
    ax.set_xlabel('Signal Frequency', color=TEXT_COLOR)
    ax.set_ylabel('PSNR (dB)', color=TEXT_COLOR)
    ax.set_title('Reconstruction Quality', color=TEXT_COLOR)
    ax.legend(facecolor=BG_COLOR, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, fontsize=8)

    # Panel 2: PSNR Difference (FCC advantage) vs Frequency
    ax = axes[0, 1]
    for i, r in enumerate(results):
        freqs = [s.frequency for s in r.sweep]
        diffs = [s.psnr_diff for s in r.sweep]
        label = f"~{r.cubic_count} samples"
        ax.plot(freqs, diffs, marker='o', markersize=4,
                label=label, alpha=0.8)
    ax.axhline(0, color=TEXT_COLOR, alpha=0.3, linestyle='--')
    ax.fill_between([0, 10], 0, 15, color=FCC_COLOR, alpha=0.05)
    ax.fill_between([0, 10], -15, 0, color=CUBIC_COLOR, alpha=0.05)
    ax.set_xlabel('Signal Frequency', color=TEXT_COLOR)
    ax.set_ylabel('Δ PSNR (dB)', color=TEXT_COLOR)
    ax.set_title('FCC Advantage (>0 = FCC wins)', color=TEXT_COLOR)
    ax.legend(facecolor=BG_COLOR, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, fontsize=8)

    # Panel 3: MSE Ratio vs Frequency
    ax = axes[0, 2]
    for i, r in enumerate(results):
        freqs = [s.frequency for s in r.sweep]
        ratios = [s.mse_ratio for s in r.sweep]
        label = f"~{r.cubic_count} samples"
        ax.plot(freqs, ratios, marker='o', markersize=4,
                label=label, alpha=0.8)
    ax.axhline(1.0, color=TEXT_COLOR, alpha=0.3, linestyle='--')
    ax.fill_between([0, 10], 0, 1, color=FCC_COLOR, alpha=0.05)
    ax.fill_between([0, 10], 1, 5, color=CUBIC_COLOR, alpha=0.05)
    ax.set_xlabel('Signal Frequency', color=TEXT_COLOR)
    ax.set_ylabel('MSE Ratio (FCC/Cubic)', color=TEXT_COLOR)
    ax.set_title('MSE Ratio (<1 = FCC wins)', color=TEXT_COLOR)
    ax.legend(facecolor=BG_COLOR, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, fontsize=8)

    # Panel 4: Isotropy comparison
    ax = axes[1, 0]
    x = np.arange(len(scales))
    width = 0.35
    c_iso = [r.cubic_isotropy for r in results]
    f_iso = [r.fcc_isotropy for r in results]
    ax.bar(x - width/2, c_iso, width, color=CUBIC_COLOR, label='Cubic')
    ax.bar(x + width/2, f_iso, width, color=FCC_COLOR, label='FCC')
    ax.set_xticks(x)
    ax.set_xticklabels([f"~{r.cubic_count}" for r in results])
    ax.set_xlabel('Sample Count', color=TEXT_COLOR)
    ax.set_ylabel('Directional MSE σ', color=TEXT_COLOR)
    ax.set_title('Isotropy (lower = better)', color=TEXT_COLOR)
    ax.legend(facecolor=BG_COLOR, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, fontsize=8)

    # Panel 5: Peak FCC advantage per scale
    ax = axes[1, 1]
    peak_advantages = []
    for r in results:
        best = max(r.sweep, key=lambda s: s.psnr_diff)
        peak_advantages.append(best.psnr_diff)
    ax.bar(x, peak_advantages, color=FCC_COLOR, width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"~{r.cubic_count}" for r in results])
    ax.set_xlabel('Sample Count', color=TEXT_COLOR)
    ax.set_ylabel('Peak Δ PSNR (dB)', color=TEXT_COLOR)
    ax.set_title('Peak FCC Advantage', color=TEXT_COLOR)
    for i, v in enumerate(peak_advantages):
        ax.text(i, v + 0.2, f"+{v:.1f} dB", ha='center',
                color=TEXT_COLOR, fontsize=9)

    # Panel 6: Summary ratios at optimal frequency
    ax = axes[1, 2]
    labels = ['MSE\n(lower=better)', 'Isotropy\n(lower=better)']
    # At optimal frequency, take the best sweep point
    best_mse_ratios = []
    iso_ratios = []
    for r in results:
        best = min(r.sweep, key=lambda s: s.mse_ratio)
        best_mse_ratios.append(best.mse_ratio)
        iso_ratios.append(r.fcc_isotropy / max(r.cubic_isotropy, 1e-15))

    avg_mse = np.mean(best_mse_ratios)
    avg_iso = np.mean(iso_ratios)
    vals = [avg_mse, avg_iso]
    bars = ax.bar(labels, vals, color=FCC_COLOR, width=0.5)
    ax.axhline(1.0, color=TEXT_COLOR, alpha=0.3, linestyle='--')
    ax.set_ylabel('FCC / Cubic Ratio', color=TEXT_COLOR)
    ax.set_title('Summary: FCC/Cubic (<1 = FCC wins)', color=TEXT_COLOR)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                f"{v:.2f}×", ha='center', color=TEXT_COLOR, fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = 'results/rung-3/dashboard.png'
    plt.savefig(out, dpi=150, facecolor=BG_COLOR,
                bbox_inches='tight')
    print(f"Dashboard saved to {out}")
    plt.close()


if __name__ == "__main__":
    main()

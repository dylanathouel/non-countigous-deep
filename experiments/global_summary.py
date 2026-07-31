"""Concatenates the summary.csv files from 3d/10d/50d and generates the global heatmap.

Usage: python -m experiments.global_summary
Run AFTER run_3d, run_10d, run_50d.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


RESULTS_DIR = "results"
DIMS = [3, 10, 50]
ALGOS = ['backprop', 'ray', 'gwo', 'hybrid']


def main():
    frames = []
    for d in DIMS:
        path = os.path.join(RESULTS_DIR, f"{d}d", "summary.csv")
        if not os.path.exists(path):
            print(f"MISSING: {path} — run run_{d}d.py first.")
            continue
        frames.append(pd.read_csv(path))
    if not frames:
        sys.exit("No summary.csv found. Run the experiments first.")

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(os.path.join(RESULTS_DIR, "global_summary.csv"), index=False)
    print(f"global_summary.csv written ({len(df)} rows).")

    print("\n=== Thesis validation ===")
    rows = []
    for d in df['dim'].unique():
        for f in df[df['dim'] == d]['func'].unique():
            sub = df[(df['dim'] == d) & (df['func'] == f)].set_index('algo')['mse_val']
            best_non_bp = sub.drop('backprop').min()
            ratio = sub['backprop'] / max(best_non_bp, 1e-12)
            hybrid_best = sub['hybrid'] <= min(sub['ray'], sub['gwo']) * 1.001
            rows.append({
                'dim': d, 'func': f,
                'bp_fails (>=5x)': ratio >= 5.0,
                'bp_ratio': round(ratio, 1),
                'hybrid_is_best': hybrid_best,
            })
    val_df = pd.DataFrame(rows)
    print(val_df.to_string(index=False))
    print(f"\nbackprop fails (>=5x) on {val_df['bp_fails (>=5x)'].sum()}/{len(val_df)} combinations.")
    print(f"hybrid is the best on {val_df['hybrid_is_best'].sum()}/{len(val_df)} combinations.")

    if 'r2_val' in df.columns:
        r2_pivot = df.groupby(['dim', 'algo'])['r2_val'].mean().unstack()[ALGOS]
        print("\n=== Mean R²_val per (dim, algo) — 0 = learned nothing beyond the mean ===")
        print(r2_pivot.to_string(float_format=lambda v: f"{v:.3f}"))

    pivot = df.groupby(['dim', 'algo'])['mse_val'].mean().unstack()
    pivot = pivot[ALGOS]
    fig, ax = plt.subplots(figsize=(8, 4))
    data = np.log10(pivot.values + 1e-12)
    im = ax.imshow(data, aspect='auto', cmap='RdYlGn_r')
    ax.set_xticks(range(len(ALGOS)))
    ax.set_xticklabels(ALGOS)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"dim={d}" for d in pivot.index])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.4f}", ha='center', va='center',
                    fontsize=9,
                    color='white' if data[i, j] > np.median(data) else 'black')
    ax.set_title("Mean val MSE per (dim, algo) - color = log10(MSE), green=better")
    plt.colorbar(im, ax=ax, label="log10(MSE val)")
    plt.tight_layout()
    out_png = os.path.join(RESULTS_DIR, "global_comparison.png")
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nHeatmap saved: {out_png}")


if __name__ == "__main__":
    main()

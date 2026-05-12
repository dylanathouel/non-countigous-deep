"""Concatène les summary.csv de 3d/10d/50d et génère la heatmap globale.

Usage : python -m experiments.global_summary
À lancer APRES run_3d, run_10d, run_50d.
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
            print(f"MANQUANT : {path} — lance run_{d}d.py d'abord.")
            continue
        frames.append(pd.read_csv(path))
    if not frames:
        sys.exit("Aucun summary.csv trouvé. Lance les expériences d'abord.")

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(os.path.join(RESULTS_DIR, "global_summary.csv"), index=False)
    print(f"global_summary.csv écrit ({len(df)} lignes).")

    print("\n=== Validation de la thèse ===")
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
    print(f"\nbackprop échoue (>=5x) sur {val_df['bp_fails (>=5x)'].sum()}/{len(val_df)} combinaisons.")
    print(f"hybrid est le meilleur sur {val_df['hybrid_is_best'].sum()}/{len(val_df)} combinaisons.")

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
    ax.set_title("MSE val moyen par (dim, algo) — couleur = log10(MSE), vert=mieux")
    plt.colorbar(im, ax=ax, label="log10(MSE val)")
    plt.tight_layout()
    out_png = os.path.join(RESULTS_DIR, "global_comparison.png")
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nHeatmap sauvée : {out_png}")


if __name__ == "__main__":
    main()

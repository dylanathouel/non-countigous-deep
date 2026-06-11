"""Experiment pipeline: for a given dimension, runs the 4 algorithms on each
of the 5 functions, saves CSV + convergence figures."""
import os
import time
from typing import List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core.network import HeavisideNetwork
from core.functions import FUNCTIONS, generate_dataset
from core.algorithms import (
    train_backprop, train_ray_shooting, train_gwo, train_hybrid_seeded,
)


HIDDEN_SIZES = [16, 12, 8]
ALGO_NAMES = ['backprop', 'ray', 'gwo', 'hybrid']
ALGO_COLORS = {'backprop': 'red', 'ray': 'green', 'gwo': 'blue', 'hybrid': 'purple'}
ALGO_LABELS = {
    'backprop': 'Backprop (expected to fail)',
    'ray': 'Ray Shooting',
    'gwo': 'GWO',
    'hybrid': 'Hybrid (Ray + seeded GWO)',
}


def _train(algo: str, nn, X_tr, y_tr, eval_budget, seed):
    if algo == 'backprop':
        return train_backprop(nn, X_tr, y_tr, eval_budget=eval_budget, lr=0.01)
    if algo == 'ray':
        return train_ray_shooting(nn, X_tr, y_tr, eval_budget=eval_budget, seed=seed)
    if algo == 'gwo':
        return train_gwo(nn, X_tr, y_tr, eval_budget=eval_budget, n_agents=30, seed=seed)
    if algo == 'hybrid':
        return train_hybrid_seeded(nn, X_tr, y_tr, eval_budget=eval_budget,
                                    ray_ratio=0.5, sigma=0.2, n_agents=30, seed=seed)
    raise ValueError(f"Unknown algo: {algo}")


def _plot_convergence(histories: dict, title: str, save_path: str):
    plt.figure(figsize=(10, 6))
    for algo in ALGO_NAMES:
        h = histories[algo]
        if len(h) > 2000:
            idx = np.linspace(0, len(h) - 1, 2000, dtype=int)
            plt.plot(idx, [h[i] for i in idx], color=ALGO_COLORS[algo],
                     label=ALGO_LABELS[algo], linewidth=1.5, alpha=0.85)
        else:
            plt.plot(h, color=ALGO_COLORS[algo], label=ALGO_LABELS[algo],
                     linewidth=1.5, alpha=0.85)
    plt.yscale('log')
    plt.xlabel('Loss function evaluations')
    plt.ylabel('MSE (log scale)')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_benchmark(dim: int, output_dir: str, eval_budget: int = 25000,
                  n_samples: int = 2000, func_names: Optional[List[str]] = None,
                  seed: int = 42) -> pd.DataFrame:
    """Runs the 4 algorithms on each function for a given dimension."""
    if func_names is None:
        func_names = list(FUNCTIONS.keys())
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)

    rows = []
    print(f"\n{'='*72}")
    print(f"BENCHMARK dim={dim} | budget={eval_budget} evals | {n_samples} samples")
    print(f"Functions: {func_names}")
    print(f"{'='*72}")

    for func_name in func_names:
        print(f"\n>>> Function: {func_name}")
        X_tr, y_tr, X_va, y_va = generate_dataset(func_name, n_samples=n_samples,
                                                    dim=dim, seed=seed)
        y_min, y_max = float(y_tr.min()), float(y_tr.max())
        denom = (y_max - y_min) if (y_max - y_min) > 1e-9 else 1.0
        y_tr_n = (y_tr - y_min) / denom
        y_va_n = (y_va - y_min) / denom

        histories = {}
        for algo in ALGO_NAMES:
            print(f"  [{algo:>8}]...", end=' ', flush=True)
            nn = HeavisideNetwork(dim, HIDDEN_SIZES, 1, seed=seed)
            t0 = time.time()
            final_train, history = _train(algo, nn, X_tr, y_tr_n,
                                           eval_budget=eval_budget, seed=seed)
            elapsed = time.time() - t0
            mse_val = float(np.mean((y_va_n - nn.forward(X_va)) ** 2))
            histories[algo] = history
            rows.append({
                'dim': dim, 'func': func_name, 'algo': algo,
                'mse_train': final_train, 'mse_val': mse_val,
                'time_s': elapsed, 'n_evals': len(history),
            })
            print(f"mse_val={mse_val:.5f} (train={final_train:.5f}) [{elapsed:.1f}s]")

        plot_path = os.path.join(output_dir, 'figures', f'{func_name}.png')
        _plot_convergence(histories,
                          title=f"Convergence - {func_name} (dim={dim})",
                          save_path=plot_path)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    pivot = df.pivot(index='func', columns='algo', values='mse_val')
    print("\nMSE val per (func, algo):")
    print(pivot.to_string(float_format=lambda v: f"{v:.5f}"))
    print(f"\nBest algo per function:")
    for func in df['func'].unique():
        sub = df[df['func'] == func].sort_values('mse_val').iloc[0]
        print(f"  {func:<10} -> {sub['algo']:<10} (mse_val={sub['mse_val']:.5f})")

    return df

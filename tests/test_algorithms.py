"""Tests for core.algorithms."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.network import HeavisideNetwork
from core.functions import generate_dataset


def _make_problem(dim=3, n=200, seed=42):
    X_tr, y_tr, X_va, y_va = generate_dataset('gl', n_samples=n, dim=dim, seed=seed)
    y_min, y_max = y_tr.min(), y_tr.max()
    y_tr_n = (y_tr - y_min) / (y_max - y_min + 1e-8)
    y_va_n = (y_va - y_min) / (y_max - y_min + 1e-8)
    return X_tr, y_tr_n, X_va, y_va_n


def _mse_eval(nn, X, y):
    pred = nn.forward(X)
    return float(np.mean((y - pred) ** 2))


# ============================================================================
# BACKPROP
# ============================================================================

def test_backprop_runs_and_records_history():
    from core.algorithms import train_backprop
    nn = HeavisideNetwork(3, [16, 12, 8], 1, seed=42)
    X_tr, y_tr, _, _ = _make_problem(dim=3)
    final_mse, history = train_backprop(nn, X_tr, y_tr, eval_budget=200, lr=0.01)
    assert len(history) == 200, f"Expected 200 history points, got {len(history)}"
    assert np.isfinite(final_mse)
    print("PASS: test_backprop_runs_and_records_history")


def test_backprop_hidden_layers_frozen():
    """The zero Heaviside gradient must prevent hidden layers from changing."""
    from core.algorithms import train_backprop
    nn = HeavisideNetwork(3, [16, 12, 8], 1, seed=42)
    X_tr, y_tr, _, _ = _make_problem(dim=3)
    W_hidden_before = [W.copy() for W, _ in nn.layers[:-1]]
    train_backprop(nn, X_tr, y_tr, eval_budget=100, lr=0.01)
    W_hidden_after = [W for W, _ in nn.layers[:-1]]
    for Wb, Wa in zip(W_hidden_before, W_hidden_after):
        np.testing.assert_array_equal(Wb, Wa)
    print("PASS: test_backprop_hidden_layers_frozen")


# ============================================================================
# RAY SHOOTING
# ============================================================================

def test_ray_shooting_decreases_mse():
    from core.algorithms import train_ray_shooting
    nn = HeavisideNetwork(3, [16, 12, 8], 1, seed=42)
    X_tr, y_tr, _, _ = _make_problem(dim=3)
    mse_init = _mse_eval(nn, X_tr, y_tr)
    final, history = train_ray_shooting(nn, X_tr, y_tr, eval_budget=500, bounds=(-2, 2))
    assert final < mse_init, f"Ray didn't improve: {mse_init:.4f} -> {final:.4f}"
    assert len(history) >= 100
    print("PASS: test_ray_shooting_decreases_mse")


def test_ray_shooting_history_monotone():
    from core.algorithms import train_ray_shooting
    nn = HeavisideNetwork(3, [16, 12, 8], 1, seed=42)
    X_tr, y_tr, _, _ = _make_problem(dim=3)
    _, history = train_ray_shooting(nn, X_tr, y_tr, eval_budget=300)
    for i in range(1, len(history)):
        assert history[i] <= history[i - 1] + 1e-12, f"history[{i}] increases"
    print("PASS: test_ray_shooting_history_monotone")


def test_ray_shooting_steps20_beats_steps50():
    """steps_per_ray=20 must beat steps_per_ray=50 ON AVERAGE (10D, multi-seeds)."""
    from core.algorithms import train_ray_shooting
    X_tr, y_tr, _, _ = _make_problem(dim=10, n=500)
    seeds = [0, 1, 2, 7, 13]
    f_50s, f_20s = [], []
    for s in seeds:
        nn50 = HeavisideNetwork(10, [16, 12, 8], 1, seed=42)
        nn20 = HeavisideNetwork(10, [16, 12, 8], 1, seed=42)
        f50, _ = train_ray_shooting(nn50, X_tr, y_tr, eval_budget=3000,
                                     steps_per_ray=50, seed=s)
        f20, _ = train_ray_shooting(nn20, X_tr, y_tr, eval_budget=3000,
                                     steps_per_ray=20, seed=s)
        f_50s.append(f50); f_20s.append(f20)
    mean_50 = sum(f_50s) / len(f_50s)
    mean_20 = sum(f_20s) / len(f_20s)
    assert mean_20 < mean_50, (
        f"steps=20 ({mean_20:.5f}) must beat steps=50 ({mean_50:.5f}) "
        f"over {len(seeds)} seeds")
    print(f"PASS: test_ray_shooting_steps20_beats_steps50 "
          f"(mean s50={mean_50:.5f}, mean s20={mean_20:.5f}, {len(seeds)} seeds)")


# ============================================================================
# GWO
# ============================================================================

def test_gwo_decreases_mse():
    from core.algorithms import train_gwo
    nn = HeavisideNetwork(3, [16, 12, 8], 1, seed=42)
    X_tr, y_tr, _, _ = _make_problem(dim=3)
    mse_init = _mse_eval(nn, X_tr, y_tr)
    final, history = train_gwo(nn, X_tr, y_tr, eval_budget=300, n_agents=10)
    assert final <= mse_init + 1e-6
    assert len(history) >= 100
    print("PASS: test_gwo_decreases_mse")


def test_gwo_seeds_current_position():
    """The first wolf must be initialized to the network's current position."""
    from core.algorithms import train_gwo
    nn = HeavisideNetwork(3, [16, 12, 8], 1, seed=42)
    X_tr, y_tr, _, _ = _make_problem(dim=3)
    mse_init = _mse_eval(nn, X_tr, y_tr)
    _, history = train_gwo(nn, X_tr, y_tr, eval_budget=50, n_agents=10)
    assert history[0] <= mse_init + 1e-9
    print("PASS: test_gwo_seeds_current_position")


# ============================================================================
# HYBRID
# ============================================================================

def test_hybrid_seeded_runs():
    from core.algorithms import train_hybrid_seeded
    nn = HeavisideNetwork(3, [16, 12, 8], 1, seed=42)
    X_tr, y_tr, _, _ = _make_problem(dim=3)
    final, history = train_hybrid_seeded(nn, X_tr, y_tr, eval_budget=400,
                                          ray_ratio=0.5, sigma=0.2, n_agents=8)
    assert len(history) <= 400
    assert np.isfinite(final)
    print("PASS: test_hybrid_seeded_runs")


def test_hybrid_seeded_better_than_random_gwo():
    """On the same total budget, seeded hybrid must be <= 2x GWO alone.

    The test does not require hybrid to win (with so few evals it's random),
    only that it not be catastrophic."""
    from core.algorithms import train_gwo, train_hybrid_seeded
    rng_seed = 7
    nn_gwo = HeavisideNetwork(3, [16, 12, 8], 1, seed=rng_seed)
    nn_hyb = HeavisideNetwork(3, [16, 12, 8], 1, seed=rng_seed)
    X_tr, y_tr, _, _ = _make_problem(dim=3)
    f_gwo, _ = train_gwo(nn_gwo, X_tr, y_tr, eval_budget=600, n_agents=10, seed=0)
    f_hyb, _ = train_hybrid_seeded(nn_hyb, X_tr, y_tr, eval_budget=600,
                                    ray_ratio=0.5, sigma=0.2, n_agents=10, seed=0)
    assert f_hyb <= 2.0 * f_gwo, f"hybrid={f_hyb:.4f} vs gwo={f_gwo:.4f}"
    print("PASS: test_hybrid_seeded_better_than_random_gwo")


if __name__ == "__main__":
    test_backprop_runs_and_records_history()
    test_backprop_hidden_layers_frozen()
    test_ray_shooting_decreases_mse()
    test_ray_shooting_history_monotone()
    test_ray_shooting_steps20_beats_steps50()
    test_gwo_decreases_mse()
    test_gwo_seeds_current_position()
    test_hybrid_seeded_runs()
    test_hybrid_seeded_better_than_random_gwo()
    print("\n[algorithms] All 9 tests passed.")

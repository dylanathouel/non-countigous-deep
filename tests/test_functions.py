"""Tests for core.functions: 5 n-D discontinuous functions + calibration."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.functions import FUNCTIONS, BOUNDARY_MASK, generate_dataset


def _proportion_inside(func_name, dim, n=10000, seed=42):
    """Proportion of points for which the boundary is 'true'."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(-2, 2, (n, dim))
    mask = BOUNDARY_MASK[func_name](X)
    return float(np.mean(mask))


def test_function_shapes():
    for name, func in FUNCTIONS.items():
        for dim in [3, 10, 50]:
            X = np.random.uniform(-2, 2, (100, dim))
            y = func(X)
            assert y.shape == (100,), f"{name} dim={dim}: shape {y.shape}"
    print("PASS: test_function_shapes")


def test_function_finite_and_bounded():
    for name, func in FUNCTIONS.items():
        for dim in [3, 10, 50]:
            X = np.random.uniform(-2, 2, (500, dim))
            y = func(X)
            assert np.all(np.isfinite(y)), f"{name} dim={dim}: non-finite values"
            assert np.max(np.abs(y)) <= 30, f"{name} dim={dim}: |y|max={np.max(np.abs(y))}"
    print("PASS: test_function_finite_and_bounded")


def test_boundary_proportion_calibrated():
    """The 'inside' proportion must be in [0.20, 0.80] for each (func, dim)."""
    failures = []
    for name in FUNCTIONS.keys():
        for dim in [3, 10, 50]:
            p = _proportion_inside(name, dim)
            if not (0.20 <= p <= 0.80):
                failures.append(f"{name} dim={dim}: p_inside={p:.3f}")
    assert not failures, "Calibration out of range:\n" + "\n".join(failures)
    print("PASS: test_boundary_proportion_calibrated")


def test_function_is_discontinuous():
    """Checks that crossing the boundary makes the function jump (difference > 0.1)."""
    for name in FUNCTIONS.keys():
        for dim in [3, 10]:
            rng = np.random.RandomState(0)
            X = rng.uniform(-1.5, 1.5, (2000, dim))
            y = FUNCTIONS[name](X)
            mask = BOUNDARY_MASK[name](X)
            if mask.any() and (~mask).any():
                gap = np.abs(y[mask].mean() - y[~mask].mean())
                assert gap > 0.1, f"{name} dim={dim}: mean gap too small ({gap:.4f})"
    print("PASS: test_function_is_discontinuous")


def test_generate_dataset_split():
    X_tr, y_tr, X_va, y_va = generate_dataset('gl', n_samples=1000, dim=10, seed=42)
    assert X_tr.shape == (800, 10)
    assert X_va.shape == (200, 10)
    assert y_tr.shape == (800, 1)
    assert y_va.shape == (200, 1)
    print("PASS: test_generate_dataset_split")


def test_generate_dataset_seed_reproducible():
    a = generate_dataset('gs', n_samples=500, dim=10, seed=42)
    b = generate_dataset('gs', n_samples=500, dim=10, seed=42)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    print("PASS: test_generate_dataset_seed_reproducible")


if __name__ == "__main__":
    test_function_shapes()
    test_function_finite_and_bounded()
    test_boundary_proportion_calibrated()
    test_function_is_discontinuous()
    test_generate_dataset_split()
    test_generate_dataset_seed_reproducible()
    print("\n[functions] All 6 tests passed.")

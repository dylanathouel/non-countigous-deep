"""Tests for core.network.HeavisideNetwork."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.network import HeavisideNetwork


def test_forward_shape():
    nn = HeavisideNetwork(input_dim=3, hidden_sizes=[16, 12, 8], output_dim=1, seed=42)
    X = np.random.uniform(-2, 2, (100, 3))
    y = nn.forward(X)
    assert y.shape == (100, 1), f"Expected (100, 1), got {y.shape}"
    print("PASS: test_forward_shape")


def test_forward_50d():
    nn = HeavisideNetwork(input_dim=50, hidden_sizes=[16, 12, 8], output_dim=1, seed=42)
    X = np.random.uniform(-2, 2, (32, 50))
    y = nn.forward(X)
    assert y.shape == (32, 1)
    print("PASS: test_forward_50d")


def test_heaviside_hidden_activations_are_binary():
    nn = HeavisideNetwork(input_dim=3, hidden_sizes=[16], output_dim=1, seed=42)
    X = np.random.uniform(-2, 2, (50, 3))
    W, b = nn.layers[0]
    Z = X @ W + b
    A = nn._heaviside(Z)
    unique_vals = set(np.unique(A).tolist())
    assert unique_vals.issubset({0.0, 1.0}), f"Hidden activations not binary: {unique_vals}"
    print("PASS: test_heaviside_hidden_activations_are_binary")


def test_num_params():
    # 3*16+16 + 16*12+12 + 12*8+8 + 8*1+1 = 48+16+192+12+96+8+8+1 = 381
    nn = HeavisideNetwork(input_dim=3, hidden_sizes=[16, 12, 8], output_dim=1, seed=42)
    assert nn.num_params() == 381, f"Expected 381, got {nn.num_params()}"
    print("PASS: test_num_params")


def test_get_set_params_roundtrip():
    nn = HeavisideNetwork(input_dim=3, hidden_sizes=[16, 12, 8], output_dim=1, seed=42)
    original = nn.get_params()
    new_vec = np.random.randn(nn.num_params())
    nn.set_params(new_vec)
    np.testing.assert_array_equal(nn.get_params(), new_vec)
    nn.set_params(original)
    np.testing.assert_array_equal(nn.get_params(), original)
    print("PASS: test_get_set_params_roundtrip")


def test_seed_reproducibility():
    nn1 = HeavisideNetwork(input_dim=3, hidden_sizes=[16, 12, 8], output_dim=1, seed=42)
    nn2 = HeavisideNetwork(input_dim=3, hidden_sizes=[16, 12, 8], output_dim=1, seed=42)
    np.testing.assert_array_equal(nn1.get_params(), nn2.get_params())
    print("PASS: test_seed_reproducibility")


if __name__ == "__main__":
    test_forward_shape()
    test_forward_50d()
    test_heaviside_hidden_activations_are_binary()
    test_num_params()
    test_get_set_params_roundtrip()
    test_seed_reproducibility()
    print("\n[network] All 6 tests passed.")

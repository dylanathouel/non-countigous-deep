"""Réseau de neurones à activation Heaviside (gradient nul dans les couches cachées).

Le gradient zéro dans les couches cachées est intentionnel : c'est précisément
ce qui fait échouer la rétropropagation, et qui justifie l'usage d'algorithmes
sans gradient (Ray Shooting, GWO, Hybrid).
"""
from typing import List, Tuple
import numpy as np


class HeavisideNetwork:
    """Architecture : input -> Heaviside(hidden_1) -> ... -> Heaviside(hidden_n) -> Linear(output)."""

    def __init__(self, input_dim: int, hidden_sizes: List[int] = [16, 12, 8],
                 output_dim: int = 1, seed: int = None):
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random

        self.input_dim = input_dim
        self.hidden_sizes = list(hidden_sizes)
        self.output_dim = output_dim

        self.layers: List[Tuple[np.ndarray, np.ndarray]] = []
        prev = input_dim
        for h in hidden_sizes:
            limit = np.sqrt(6.0 / (prev + h))
            W = rng.uniform(-limit, limit, (prev, h))
            b = np.zeros((1, h))
            self.layers.append((W, b))
            prev = h
        limit = np.sqrt(6.0 / (prev + output_dim))
        W = rng.uniform(-limit, limit, (prev, output_dim))
        b = np.zeros((1, output_dim))
        self.layers.append((W, b))

    @staticmethod
    def _heaviside(z: np.ndarray) -> np.ndarray:
        return np.where(z >= 0, 1.0, 0.0)

    def forward(self, X: np.ndarray) -> np.ndarray:
        A = X
        for W, b in self.layers[:-1]:
            A = self._heaviside(A @ W + b)
        W, b = self.layers[-1]
        return A @ W + b

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def num_params(self) -> int:
        return sum(W.size + b.size for W, b in self.layers)

    def get_params(self) -> np.ndarray:
        chunks = []
        for W, b in self.layers:
            chunks.append(W.flatten())
            chunks.append(b.flatten())
        return np.concatenate(chunks)

    def set_params(self, vec: np.ndarray) -> None:
        idx = 0
        new_layers = []
        for W, b in self.layers:
            W_size = W.size
            b_size = b.size
            new_W = vec[idx:idx + W_size].reshape(W.shape)
            idx += W_size
            new_b = vec[idx:idx + b_size].reshape(b.shape)
            idx += b_size
            new_layers.append((new_W, new_b))
        self.layers = new_layers

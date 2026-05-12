"""Fonctions de test discontinues généralisées en dimension n (3, 10, 50).

Principe : les seuils géométriques sont calibrés pour que ~50% des points
uniformes sur [-2,2]^D soient "intérieurs" à la frontière, peu importe D.
Ça évite la dégénérescence en haute dimension (concentration de la mesure).

Pour [-2,2]^D et x_i uniforme : E[x_i^2] = 4/3, E[||x||^2] = 4D/3.
Les valeurs renvoyées sont normalisées (division par D ou sqrt(D)) pour
éviter que les sinus/exp explosent en haute dim.
"""
from typing import Callable, Dict, Tuple
import numpy as np


def _norm_sq(x: np.ndarray) -> np.ndarray:
    return np.sum(x ** 2, axis=1)


def _sum_axis1(x: np.ndarray) -> np.ndarray:
    return np.sum(x, axis=1)


# ============================================================================
# Frontières (BOUNDARY_MASK) — renvoient un booléen par sample
# ============================================================================

def _mask_gl(x: np.ndarray) -> np.ndarray:
    """Hyperplan : x_D <= moyenne(x_1..x_{D-1})."""
    D = x.shape[1]
    last = x[:, -1]
    if D == 1:
        return last <= 0
    others_mean = np.mean(x[:, :-1], axis=1)
    return last <= others_mean


def _mask_gs(x: np.ndarray) -> np.ndarray:
    """Hypercube |x_i| <= alpha_gs(D) où alpha = 2 * 0.5^(1/D) -> P(in)=0.5."""
    D = x.shape[1]
    alpha = 2.0 * (0.5 ** (1.0 / D))
    return np.all(np.abs(x) <= alpha, axis=1)


def _mask_geta(x: np.ndarray) -> np.ndarray:
    """Région 'au-dessus' de la surface exponentielle atténuée."""
    D = x.shape[1]
    last = x[:, -1]
    threshold = np.exp(x[:, 0]) / (D ** 0.25)
    return last >= threshold


def _mask_ggamma(x: np.ndarray) -> np.ndarray:
    """Hypersphere ||x||^2 <= 4D/3 (médiane attendue)."""
    D = x.shape[1]
    return _norm_sq(x) <= (4.0 * D / 3.0)


def _mask_complex(x: np.ndarray) -> np.ndarray:
    """Hyperparaboloïde : x_D <= c * (somme x_i^2 (i<D) - (D-1)*4/3) avec c=1/(2*sqrt(D-1))."""
    D = x.shape[1]
    if D == 1:
        return x[:, 0] <= 0
    others_sq = np.sum(x[:, :-1] ** 2, axis=1)
    c = 1.0 / (2.0 * np.sqrt(D - 1))
    threshold = c * (others_sq - (D - 1) * 4.0 / 3.0)
    return x[:, -1] <= threshold


# ============================================================================
# Fonctions cibles — valeurs réelles
# ============================================================================

def gl(x: np.ndarray) -> np.ndarray:
    D = x.shape[1]
    r = np.sqrt(_norm_sq(x) / D)
    mask = _mask_gl(x)
    return np.where(mask,
                    2 * np.sin(1.25 * np.pi * r) + 4,
                    2 * np.sin(0.75 * np.pi * r))


def gs(x: np.ndarray) -> np.ndarray:
    D = x.shape[1]
    norm_sq_avg = _norm_sq(x) / D
    mask = _mask_gs(x)
    return np.where(mask,
                    -2 * norm_sq_avg + 6,
                    4 * np.exp(1 - norm_sq_avg))


def geta(x: np.ndarray) -> np.ndarray:
    D = x.shape[1]
    s_norm = _sum_axis1(x) / np.sqrt(D)
    last = x[:, -1]
    threshold = np.exp(x[:, 0]) / (D ** 0.25)
    above = last >= threshold
    below = last < (threshold - 1.0 / (D ** 0.25))
    between = ~above & ~below
    out = np.zeros_like(last, dtype=np.float64)
    out[above] = np.sin(0.4 * np.pi * s_norm[above])
    out[below] = np.sin(0.7 * np.pi * s_norm[below]) - 4
    out[between] = np.sin(np.pi * s_norm[between]) + 4
    return out


def ggamma(x: np.ndarray) -> np.ndarray:
    D = x.shape[1]
    s_norm = _sum_axis1(x) / np.sqrt(D)
    mask = _mask_ggamma(x)
    return np.where(mask,
                    np.sin(np.pi * s_norm) + 4,
                    np.sin(0.4 * np.pi * s_norm))


def complex_func(x: np.ndarray) -> np.ndarray:
    D = x.shape[1]
    norm_sq_avg = _norm_sq(x) / D
    mask = _mask_complex(x)
    x1 = x[:, 0]
    x2 = x[:, 1] if D >= 2 else x[:, 0]
    xD = x[:, -1]
    triple = x1 * x2 * xD / np.sqrt(D)
    return np.where(mask,
                    norm_sq_avg * np.sin(np.pi * x1) + 3,
                    -triple * np.cos(np.pi * xD) - 2)


FUNCTIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    'gl': gl,
    'gs': gs,
    'geta': geta,
    'ggamma': ggamma,
    'complex': complex_func,
}

BOUNDARY_MASK: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    'gl': _mask_gl,
    'gs': _mask_gs,
    'geta': _mask_geta,
    'ggamma': _mask_ggamma,
    'complex': _mask_complex,
}

FUNCTION_DESCRIPTIONS = {
    'gl': 'Hyperplan (x_D <= moyenne des autres)',
    'gs': 'Hypercube calibré (P(in) ~ 0.5)',
    'geta': 'Surface exponentielle atténuée',
    'ggamma': 'Hypersphère de rayon √(4D/3)',
    'complex': 'Hyperparaboloïde calibré',
}


def generate_dataset(func_name: str, n_samples: int = 2000, dim: int = 3,
                     seed: int = 42, bounds: Tuple[float, float] = (-2.0, 2.0)
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Génère X, y train/val (80/20)."""
    if func_name not in FUNCTIONS:
        raise ValueError(f"Unknown function: {func_name}. Available: {list(FUNCTIONS.keys())}")

    rng = np.random.RandomState(seed)
    X = rng.uniform(bounds[0], bounds[1], (n_samples, dim))
    y = FUNCTIONS[func_name](X).reshape(-1, 1)

    split = int(0.8 * n_samples)
    return X[:split], y[:split], X[split:], y[split:]

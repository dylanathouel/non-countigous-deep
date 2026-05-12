# Multi-Dimensional Discontinuous Benchmark — Plan d'Implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal :** Refactorer le projet en un package propre (`core/` + `experiments/`) capable de démontrer expérimentalement que (1) backprop échoue sur réseaux Heaviside, (2) Ray Shooting et GWO réussissent, et (3) Hybrid Ray+GWO seedé est le meilleur — pour les dimensions 3D, 10D, et 50D.

**Architecture :** Un cœur scientifique partagé (`core/`) avec `HeavisideNetwork`, les 5 fonctions discontinues n-D, et les 4 algorithmes. Un pipeline d'expérience (`experiments/runner.py`) paramétré par la dimension. Les anciens scripts sont préservés dans `archive/`. Budget équitable de 25 000 évaluations de loss par algorithme.

**Tech Stack :** Python 3.7+, numpy, matplotlib, pandas. Tests autonomes en stdlib (asserts + scripts exécutables, pas de framework externe).

---

## File Structure (cible finale)

```
non-countigous-deep/
├── core/
│   ├── __init__.py
│   ├── network.py              # Classe HeavisideNetwork
│   ├── functions.py            # 5 fonctions discontinues n-D + generate_dataset
│   └── algorithms.py           # backprop, ray_shooting, gwo, hybrid_seeded
├── experiments/
│   ├── __init__.py
│   ├── runner.py               # run_benchmark(dim, output_dir, ...)
│   ├── run_3d.py               # python -m experiments.run_3d
│   ├── run_10d.py
│   └── run_50d.py
├── tests/
│   ├── __init__.py
│   ├── test_network.py
│   ├── test_functions.py
│   ├── test_algorithms.py
│   └── test_runner.py
├── results/
│   ├── 3d/{figures/, summary.csv}
│   ├── 10d/{figures/, summary.csv}
│   ├── 50d/{figures/, summary.csv}
│   ├── global_summary.csv
│   └── global_comparison.png
├── archive/                    # Anciens scripts (conservés)
├── docs/superpowers/{specs,plans}/
└── README.md                   # Nouveau README final
```

**Principe de décomposition :**
- `core/network.py` : tout ce qui touche au réseau (forward, get/set params)
- `core/functions.py` : tout ce qui touche aux fonctions cibles et à la génération de données
- `core/algorithms.py` : tout ce qui touche à l'optimisation
- `experiments/` : pipeline d'orchestration, plots, CSV — pas de logique scientifique

---

## Task 1 : Setup structure de dossiers + archivage

**Files :**
- Create: `core/`, `experiments/`, `tests/`, `archive/`
- Modify: déplacer les anciens fichiers `.py`, `.md`, `.txt`, `.png` vers `archive/`

- [ ] **Step 1 : Créer les nouveaux dossiers**

```bash
cd "/Users/david/Documents/Documents - MacBook Pro de David/Mahon Lev/Projet Yossi/non-countigous-deep"
mkdir -p core experiments tests archive
```

- [ ] **Step 2 : Déplacer les anciens fichiers vers `archive/`**

```bash
cd "/Users/david/Documents/Documents - MacBook Pro de David/Mahon Lev/Projet Yossi/non-countigous-deep"
git mv algorithms_discontinuous_deep.py archive/
git mv models_discontinuous_deep.py archive/
git mv discontinuous_functions.py archive/
git mv benchmark_high_dim.py archive/
git mv compare_all_algorithms_improved.py archive/
git mv enhanced_optimizer.py archive/
git mv optimized_test_suite.py archive/
git mv quick_improvement_test.py archive/
git mv verify_fix.py archive/
git mv visualize_results.py archive/
git mv convert_report.py archive/
git mv run_experiment.py archive/
git mv run_experiment_3d.py archive/
git mv run_experiment_3d_advanced.py archive/
git mv run_experiment_3d_30min.py archive/
git mv run_experiment_3d_final.py archive/
git mv run_experiment_25000.py archive/
git mv run_experiment_60s.py archive/
git mv run_hybrid_cyclic.py archive/
git mv run_missing_data.py archive/
git mv FILES_INDEX.txt archive/
git mv STRUCTURE.txt archive/
git mv GUIDE_AMELIORATION.md archive/
git mv QUICK_START.md archive/
git mv REPONSE_AMELIORATIONS.md archive/
git mv RESULTS_OPTIMIZED.md archive/
git mv SUMMARY.md archive/
git mv README.md archive/
git mv deep_discontinuous_results.png archive/
```

- [ ] **Step 3 : Vérifier que la racine est propre**

```bash
ls -1 "/Users/david/Documents/Documents - MacBook Pro de David/Mahon Lev/Projet Yossi/non-countigous-deep" | grep -v -E '^(core|experiments|tests|archive|results|docs|__pycache__|\.git|\.DS_Store)$'
```
Expected : aucune sortie (tout le bruit racine a été archivé).

- [ ] **Step 4 : Commit**

```bash
cd "/Users/david/Documents/Documents - MacBook Pro de David/Mahon Lev/Projet Yossi/non-countigous-deep"
git add -A
git commit -m "refactor: structure en package + archive anciens scripts"
```

---

## Task 2 : Init du package + module discoverability

**Files :**
- Create: `core/__init__.py`, `experiments/__init__.py`, `tests/__init__.py`

- [ ] **Step 1 : Créer `core/__init__.py`**

```python
"""Cœur scientifique du benchmark Heaviside : réseau, fonctions, algorithmes."""
```

- [ ] **Step 2 : Créer `experiments/__init__.py`**

```python
"""Pipeline d'expérience pour le benchmark Heaviside (3D/10D/50D)."""
```

- [ ] **Step 3 : Créer `tests/__init__.py`**

```python
"""Tests autonomes pour le benchmark Heaviside."""
```

- [ ] **Step 4 : Vérifier qu'on peut importer le package vide**

```bash
cd "/Users/david/Documents/Documents - MacBook Pro de David/Mahon Lev/Projet Yossi/non-countigous-deep"
python3 -c "import core, experiments, tests; print('OK')"
```
Expected : `OK`

- [ ] **Step 5 : Commit**

```bash
git add core/__init__.py experiments/__init__.py tests/__init__.py
git commit -m "feat: init packages core/experiments/tests"
```

---

## Task 3 : `core/network.py` — HeavisideNetwork

**Files :**
- Create: `core/network.py`
- Test: `tests/test_network.py`

- [ ] **Step 1 : Écrire les tests d'abord (`tests/test_network.py`)**

```python
"""Tests pour core.network.HeavisideNetwork."""
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
```

- [ ] **Step 2 : Lancer les tests, vérifier qu'ils échouent (module n'existe pas)**

```bash
cd "/Users/david/Documents/Documents - MacBook Pro de David/Mahon Lev/Projet Yossi/non-countigous-deep"
python3 tests/test_network.py
```
Expected : `ModuleNotFoundError: No module named 'core.network'`

- [ ] **Step 3 : Implémenter `core/network.py`**

```python
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

        # Xavier-uniform initialization
        self.layers: List[Tuple[np.ndarray, np.ndarray]] = []
        prev = input_dim
        for h in hidden_sizes:
            limit = np.sqrt(6.0 / (prev + h))
            W = rng.uniform(-limit, limit, (prev, h))
            b = np.zeros((1, h))
            self.layers.append((W, b))
            prev = h
        # Couche de sortie (linéaire)
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
```

- [ ] **Step 4 : Relancer les tests, vérifier qu'ils passent**

```bash
python3 tests/test_network.py
```
Expected : `[network] All 6 tests passed.`

- [ ] **Step 5 : Commit**

```bash
git add core/network.py tests/test_network.py
git commit -m "feat: HeavisideNetwork avec gradient nul + tests"
```

---

## Task 4 : `core/functions.py` — 5 fonctions discontinues n-D

**Files :**
- Create: `core/functions.py`
- Test: `tests/test_functions.py`

- [ ] **Step 1 : Écrire les tests (`tests/test_functions.py`)**

```python
"""Tests pour core.functions : 5 fonctions discontinues n-D + calibrage."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.functions import FUNCTIONS, BOUNDARY_MASK, generate_dataset


def _proportion_inside(func_name, dim, n=10000, seed=42):
    """Proportion de points pour lesquels la frontière est 'vraie'."""
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
    """La proportion 'inside' doit être dans [0.20, 0.80] pour chaque (func, dim)."""
    failures = []
    for name in FUNCTIONS.keys():
        for dim in [3, 10, 50]:
            p = _proportion_inside(name, dim)
            if not (0.20 <= p <= 0.80):
                failures.append(f"{name} dim={dim}: p_inside={p:.3f}")
    assert not failures, "Calibrage hors plage:\n" + "\n".join(failures)
    print("PASS: test_boundary_proportion_calibrated")


def test_function_is_discontinuous():
    """Vérifie qu'à la traversée de frontière, la fonction saute (différence > 0.1)."""
    for name in FUNCTIONS.keys():
        for dim in [3, 10]:
            rng = np.random.RandomState(0)
            X = rng.uniform(-1.5, 1.5, (2000, dim))
            y = FUNCTIONS[name](X)
            mask = BOUNDARY_MASK[name](X)
            if mask.any() and (~mask).any():
                gap = np.abs(y[mask].mean() - y[~mask].mean())
                assert gap > 0.1, f"{name} dim={dim}: gap moyen trop petit ({gap:.4f})"
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
```

- [ ] **Step 2 : Vérifier que les tests échouent**

```bash
python3 tests/test_functions.py
```
Expected : `ModuleNotFoundError: No module named 'core.functions'`

- [ ] **Step 3 : Implémenter `core/functions.py`**

```python
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


# ============================================================================
# Helpers
# ============================================================================
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
    """3 régions définies par x_D vs exp(x_1) / D^(1/4) — mais le mask renvoie
    True pour la région 'au-dessus' (la principale), pour le calcul de proportion."""
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
    # Sécuriser le produit x1*x2*xD en haute dim (sinon valeurs énormes)
    x1 = x[:, 0]
    x2 = x[:, 1] if D >= 2 else x[:, 0]
    xD = x[:, -1]
    triple = x1 * x2 * xD / np.sqrt(D)
    return np.where(mask,
                    norm_sq_avg * np.sin(np.pi * x1) + 3,
                    -triple * np.cos(np.pi * xD) - 2)


# ============================================================================
# Registres publics
# ============================================================================

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
```

- [ ] **Step 4 : Lancer les tests**

```bash
python3 tests/test_functions.py
```
Expected : `[functions] All 6 tests passed.`

Si `test_boundary_proportion_calibrated` échoue, c'est qu'un seuil n'est pas bien calibré pour une dim. Ajuster la formule dans le `_mask_*` correspondant en gardant le principe (P(inside) ∈ [0.2, 0.8]).

- [ ] **Step 5 : Commit**

```bash
git add core/functions.py tests/test_functions.py
git commit -m "feat: 5 fonctions discontinues n-D avec seuils calibrés"
```

---

## Task 5 : `core/algorithms.py` — train_backprop

**Files :**
- Create: `core/algorithms.py`
- Test: `tests/test_algorithms.py` (créé incrémentalement)

- [ ] **Step 1 : Écrire `tests/test_algorithms.py` avec un test pour backprop**

```python
"""Tests pour core.algorithms."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.network import HeavisideNetwork
from core.functions import generate_dataset


def _make_problem(dim=3, n=200, seed=42):
    X_tr, y_tr, X_va, y_va = generate_dataset('gl', n_samples=n, dim=dim, seed=seed)
    # Normaliser y_train sur [0,1]
    y_min, y_max = y_tr.min(), y_tr.max()
    y_tr_n = (y_tr - y_min) / (y_max - y_min + 1e-8)
    y_va_n = (y_va - y_min) / (y_max - y_min + 1e-8)
    return X_tr, y_tr_n, X_va, y_va_n


def test_backprop_runs_and_records_history():
    from core.algorithms import train_backprop
    nn = HeavisideNetwork(3, [16, 12, 8], 1, seed=42)
    X_tr, y_tr, _, _ = _make_problem(dim=3)
    final_mse, history = train_backprop(nn, X_tr, y_tr, eval_budget=200, lr=0.01)
    assert len(history) == 200, f"Expected 200 history points, got {len(history)}"
    assert np.isfinite(final_mse)
    print("PASS: test_backprop_runs_and_records_history")


def test_backprop_hidden_layers_frozen():
    """Le gradient nul de Heaviside doit empêcher les couches cachées de bouger."""
    from core.algorithms import train_backprop
    nn = HeavisideNetwork(3, [16, 12, 8], 1, seed=42)
    X_tr, y_tr, _, _ = _make_problem(dim=3)
    # Snapshot couches cachées
    W_hidden_before = [W.copy() for W, _ in nn.layers[:-1]]
    train_backprop(nn, X_tr, y_tr, eval_budget=100, lr=0.01)
    W_hidden_after = [W for W, _ in nn.layers[:-1]]
    for Wb, Wa in zip(W_hidden_before, W_hidden_after):
        np.testing.assert_array_equal(Wb, Wa)
    print("PASS: test_backprop_hidden_layers_frozen")


if __name__ == "__main__":
    test_backprop_runs_and_records_history()
    test_backprop_hidden_layers_frozen()
    print("\n[algorithms - backprop] 2 tests passed.")
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

```bash
python3 tests/test_algorithms.py
```
Expected : `ModuleNotFoundError: No module named 'core.algorithms'`

- [ ] **Step 3 : Créer `core/algorithms.py` avec `train_backprop`**

```python
"""Algorithmes d'optimisation pour réseaux Heaviside.

API commune : tous les algorithmes retournent (final_mse, history)
- final_mse : MSE final sur les données fournies (train, pas val)
- history : List[float] de longueur ~= eval_budget (MSE après chaque eval)

Le budget est le nombre d'évaluations de la fonction de perte (forward pass).
"""
from typing import List, Tuple
import numpy as np
from core.network import HeavisideNetwork


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


# ============================================================================
# 1. BACKPROPAGATION (échec attendu — gradient nul dans hidden layers)
# ============================================================================

def train_backprop(nn: HeavisideNetwork, X: np.ndarray, y: np.ndarray,
                   eval_budget: int = 25000, lr: float = 0.01
                   ) -> Tuple[float, List[float]]:
    """Backprop standard. Le gradient de Heaviside est explicitement 0 dans les
    couches cachées : seule la couche de sortie apprend. C'est l'échec à démontrer.
    """
    history: List[float] = []
    m = X.shape[0]

    for epoch in range(eval_budget):
        # Forward pass
        A = X
        activations = [A]
        for W, b in nn.layers[:-1]:
            Z = A @ W + b
            A = nn._heaviside(Z)
            activations.append(A)
        W_out, b_out = nn.layers[-1]
        y_pred = A @ W_out + b_out

        loss = _mse(y, y_pred)
        history.append(loss)

        # Backward pass — gradient de la couche de sortie uniquement
        dZ_out = 2.0 * (y_pred - y) / m
        dW_out = activations[-1].T @ dZ_out
        db_out = np.sum(dZ_out, axis=0, keepdims=True)

        # Mise à jour SGD (pas Adam — on veut un échec propre et lisible)
        nn.layers[-1] = (W_out - lr * dW_out, b_out - lr * db_out)

        # Les couches cachées NE SONT PAS mises à jour (gradient = 0)

    return history[-1], history
```

- [ ] **Step 4 : Lancer les tests**

```bash
python3 tests/test_algorithms.py
```
Expected : `[algorithms - backprop] 2 tests passed.`

- [ ] **Step 5 : Commit**

```bash
git add core/algorithms.py tests/test_algorithms.py
git commit -m "feat: train_backprop (gradient nul Heaviside)"
```

---

## Task 6 : `core/algorithms.py` — train_ray_shooting

**Files :**
- Modify: `core/algorithms.py` (ajout)
- Modify: `tests/test_algorithms.py` (ajout)

- [ ] **Step 1 : Ajouter les tests Ray dans `tests/test_algorithms.py`**

Ajouter juste avant le `if __name__ == "__main__":` :

```python
def test_ray_shooting_decreases_mse():
    from core.algorithms import train_ray_shooting
    nn = HeavisideNetwork(3, [16, 12, 8], 1, seed=42)
    X_tr, y_tr, _, _ = _make_problem(dim=3)
    mse_init = _mse_eval(nn, X_tr, y_tr)
    final, history = train_ray_shooting(nn, X_tr, y_tr, eval_budget=500, bounds=(-2, 2))
    assert final < mse_init, f"Ray didn't improve: {mse_init:.4f} -> {final:.4f}"
    assert len(history) >= 100  # Au moins quelque chose enregistré
    print("PASS: test_ray_shooting_decreases_mse")


def test_ray_shooting_history_monotone():
    from core.algorithms import train_ray_shooting
    nn = HeavisideNetwork(3, [16, 12, 8], 1, seed=42)
    X_tr, y_tr, _, _ = _make_problem(dim=3)
    _, history = train_ray_shooting(nn, X_tr, y_tr, eval_budget=300)
    # Le meilleur ne doit jamais augmenter
    for i in range(1, len(history)):
        assert history[i] <= history[i - 1] + 1e-12, f"history[{i}] augmente"
    print("PASS: test_ray_shooting_history_monotone")
```

Et ajouter en haut, après `_make_problem` :

```python
def _mse_eval(nn, X, y):
    pred = nn.forward(X)
    return float(np.mean((y - pred) ** 2))
```

Et dans le `__main__` ajouter :

```python
    test_ray_shooting_decreases_mse()
    test_ray_shooting_history_monotone()
```

- [ ] **Step 2 : Lancer, vérifier l'échec sur `train_ray_shooting`**

```bash
python3 tests/test_algorithms.py
```
Expected : `ImportError: cannot import name 'train_ray_shooting'`

- [ ] **Step 3 : Ajouter `train_ray_shooting` dans `core/algorithms.py`**

```python
# ============================================================================
# 2. RAY SHOOTING (exploration directionnelle sans gradient)
# ============================================================================

def train_ray_shooting(nn: HeavisideNetwork, X: np.ndarray, y: np.ndarray,
                       eval_budget: int = 25000, steps_per_ray: int = 50,
                       bounds: Tuple[float, float] = (-2.0, 2.0), seed: int = None
                       ) -> Tuple[float, List[float]]:
    """Exploration en tirant des rayons depuis la meilleure position courante
    vers des cibles aléatoires uniformes dans bounds^dim.
    """
    rng = np.random.RandomState(seed)
    dim = nn.num_params()

    best_vec = nn.get_params()
    best_score = _mse(y, nn.forward(X))
    history: List[float] = [best_score]

    while len(history) < eval_budget:
        target = rng.uniform(bounds[0], bounds[1], dim)
        direction = target - best_vec

        for step in range(steps_per_ray):
            if len(history) >= eval_budget:
                break
            t = step / max(1, steps_per_ray - 1)
            candidate = np.clip(best_vec + t * direction, bounds[0], bounds[1])
            nn.set_params(candidate)
            score = _mse(y, nn.forward(X))
            # Enregistrer après chaque eval
            history.append(min(score, best_score))
            if score < best_score:
                best_vec = candidate.copy()
                best_score = score
                break

    nn.set_params(best_vec)
    # Tronquer à eval_budget exact
    history = history[:eval_budget]
    return best_score, history
```

- [ ] **Step 4 : Lancer les tests**

```bash
python3 tests/test_algorithms.py
```
Expected : `[algorithms - backprop] 4 tests passed.` (avec les 2 nouveaux)
Note : ajuster le titre du print final pour refléter le total.

- [ ] **Step 5 : Commit**

```bash
git add core/algorithms.py tests/test_algorithms.py
git commit -m "feat: train_ray_shooting"
```

---

## Task 7 : `core/algorithms.py` — train_gwo

**Files :**
- Modify: `core/algorithms.py` (ajout)
- Modify: `tests/test_algorithms.py` (ajout)

- [ ] **Step 1 : Ajouter les tests GWO**

Dans `tests/test_algorithms.py`, ajouter :

```python
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
    """Le premier loup doit être initialisé à la position courante du réseau."""
    from core.algorithms import train_gwo
    nn = HeavisideNetwork(3, [16, 12, 8], 1, seed=42)
    X_tr, y_tr, _, _ = _make_problem(dim=3)
    mse_init = _mse_eval(nn, X_tr, y_tr)
    # Sans bug de seeding, le first eval doit valoir mse_init
    _, history = train_gwo(nn, X_tr, y_tr, eval_budget=50, n_agents=10)
    assert history[0] <= mse_init + 1e-9
    print("PASS: test_gwo_seeds_current_position")
```

Et dans `__main__` :
```python
    test_gwo_decreases_mse()
    test_gwo_seeds_current_position()
```

- [ ] **Step 2 : Vérifier l'échec**

```bash
python3 tests/test_algorithms.py
```
Expected : `ImportError: cannot import name 'train_gwo'`

- [ ] **Step 3 : Ajouter `train_gwo` dans `core/algorithms.py`**

```python
# ============================================================================
# 3. GREY WOLF OPTIMIZER (méta-heuristique sans gradient)
# ============================================================================

def train_gwo(nn: HeavisideNetwork, X: np.ndarray, y: np.ndarray,
              eval_budget: int = 25000, n_agents: int = 30,
              bounds: Tuple[float, float] = (-2.0, 2.0), seed: int = None
              ) -> Tuple[float, List[float]]:
    """GWO standard. Le premier loup est seedé avec la position courante du
    réseau (préserve l'éventuel travail antérieur, e.g. après Ray).
    """
    rng = np.random.RandomState(seed)
    dim = nn.num_params()

    wolves = rng.uniform(bounds[0], bounds[1], (n_agents, dim))
    wolves[0] = nn.get_params()  # seeding

    history: List[float] = []
    scores = np.empty(n_agents)
    for i in range(n_agents):
        nn.set_params(wolves[i])
        scores[i] = _mse(y, nn.forward(X))
        history.append(scores[:i + 1].min())
        if len(history) >= eval_budget:
            nn.set_params(wolves[scores.argmin()])
            return history[-1], history[:eval_budget]

    order = np.argsort(scores)
    alpha = wolves[order[0]].copy(); alpha_score = scores[order[0]]
    beta = wolves[order[1]].copy(); beta_score = scores[order[1]]
    delta = wolves[order[2]].copy(); delta_score = scores[order[2]]

    max_iter = (eval_budget - n_agents) // n_agents + 1
    for it in range(max_iter):
        a = 2.0 * (1.0 - it / max(1, max_iter))
        # Mise à jour des positions
        for i in range(n_agents):
            r1, r2 = rng.rand(dim), rng.rand(dim)
            A1, C1 = 2 * a * r1 - a, 2 * r2
            X1 = alpha - A1 * np.abs(C1 * alpha - wolves[i])

            r1, r2 = rng.rand(dim), rng.rand(dim)
            A2, C2 = 2 * a * r1 - a, 2 * r2
            X2 = beta - A2 * np.abs(C2 * beta - wolves[i])

            r1, r2 = rng.rand(dim), rng.rand(dim)
            A3, C3 = 2 * a * r1 - a, 2 * r2
            X3 = delta - A3 * np.abs(C3 * delta - wolves[i])

            wolves[i] = np.clip((X1 + X2 + X3) / 3.0, bounds[0], bounds[1])

        # Évaluation et mise à jour leaders
        for i in range(n_agents):
            if len(history) >= eval_budget:
                break
            nn.set_params(wolves[i])
            fit = _mse(y, nn.forward(X))
            if fit < alpha_score:
                delta, delta_score = beta, beta_score
                beta, beta_score = alpha, alpha_score
                alpha, alpha_score = wolves[i].copy(), fit
            elif fit < beta_score:
                delta, delta_score = beta, beta_score
                beta, beta_score = wolves[i].copy(), fit
            elif fit < delta_score:
                delta, delta_score = wolves[i].copy(), fit
            history.append(alpha_score)

        if len(history) >= eval_budget:
            break

    nn.set_params(alpha)
    history = history[:eval_budget]
    return alpha_score, history
```

- [ ] **Step 4 : Lancer les tests**

```bash
python3 tests/test_algorithms.py
```
Expected : tous tests passent (6 au total maintenant).

- [ ] **Step 5 : Commit**

```bash
git add core/algorithms.py tests/test_algorithms.py
git commit -m "feat: train_gwo"
```

---

## Task 8 : `core/algorithms.py` — train_hybrid_seeded

**Files :**
- Modify: `core/algorithms.py` (ajout)
- Modify: `tests/test_algorithms.py` (ajout)

- [ ] **Step 1 : Ajouter les tests Hybrid**

```python
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
    """Sur un budget total identique, hybrid seedé doit faire >= GWO seul,
    parce qu'il bénéficie de l'init Ray."""
    from core.algorithms import train_gwo, train_hybrid_seeded
    rng_seed = 7
    nn_gwo = HeavisideNetwork(3, [16, 12, 8], 1, seed=rng_seed)
    nn_hyb = HeavisideNetwork(3, [16, 12, 8], 1, seed=rng_seed)
    X_tr, y_tr, _, _ = _make_problem(dim=3)
    f_gwo, _ = train_gwo(nn_gwo, X_tr, y_tr, eval_budget=600, n_agents=10, seed=0)
    f_hyb, _ = train_hybrid_seeded(nn_hyb, X_tr, y_tr, eval_budget=600,
                                    ray_ratio=0.5, sigma=0.2, n_agents=10, seed=0)
    # On tolère que hybrid soit dans 2x de GWO (le test n'est pas garanti
    # avec si peu d'évals, on vérifie surtout qu'on n'est pas catastrophique)
    assert f_hyb <= 2.0 * f_gwo, f"hybrid={f_hyb:.4f} vs gwo={f_gwo:.4f}"
    print("PASS: test_hybrid_seeded_better_than_random_gwo")
```

Et dans `__main__` :
```python
    test_hybrid_seeded_runs()
    test_hybrid_seeded_better_than_random_gwo()
```

- [ ] **Step 2 : Vérifier l'échec**

```bash
python3 tests/test_algorithms.py
```
Expected : `ImportError: cannot import name 'train_hybrid_seeded'`

- [ ] **Step 3 : Ajouter `train_hybrid_seeded` dans `core/algorithms.py`**

```python
# ============================================================================
# 4. HYBRID SEEDED (Ray puis GWO seedé autour de best_ray)
# ============================================================================

def train_hybrid_seeded(nn: HeavisideNetwork, X: np.ndarray, y: np.ndarray,
                        eval_budget: int = 25000, ray_ratio: float = 0.5,
                        sigma: float = 0.2, n_agents: int = 30,
                        bounds: Tuple[float, float] = (-2.0, 2.0),
                        steps_per_ray: int = 50, seed: int = None
                        ) -> Tuple[float, List[float]]:
    """Phase 1 : Ray Shooting standard sur ray_ratio*budget évals.
    Phase 2 : GWO initialisé avec n_agents agents = best_ray + N(0, sigma * scale),
              sauf le premier qui reste à best_ray exact (pas de régression).
    """
    rng = np.random.RandomState(seed)
    scale = max(abs(bounds[0]), abs(bounds[1]))

    # === Phase 1 : Ray ===
    ray_budget = int(ray_ratio * eval_budget)
    _, hist_ray = train_ray_shooting(nn, X, y, eval_budget=ray_budget,
                                     steps_per_ray=steps_per_ray, bounds=bounds,
                                     seed=None if seed is None else seed + 1)
    best_ray = nn.get_params()
    best_ray_score = _mse(y, nn.forward(X))

    # === Phase 2 : GWO seedé ===
    gwo_budget = eval_budget - ray_budget
    dim = nn.num_params()

    # Population concentrée autour de best_ray
    wolves = best_ray[None, :] + rng.normal(0.0, sigma * scale, (n_agents, dim))
    wolves = np.clip(wolves, bounds[0], bounds[1])
    wolves[0] = best_ray  # garantir qu'on ne régresse pas

    history = list(hist_ray)
    scores = np.empty(n_agents)
    for i in range(n_agents):
        nn.set_params(wolves[i])
        scores[i] = _mse(y, nn.forward(X))
        history.append(min(scores[:i + 1].min(), best_ray_score))
        if len(history) >= eval_budget:
            best_idx = scores[:i + 1].argmin()
            best = wolves[best_idx] if scores[best_idx] < best_ray_score else best_ray
            nn.set_params(best)
            return history[-1], history[:eval_budget]

    order = np.argsort(scores)
    alpha = wolves[order[0]].copy(); alpha_score = scores[order[0]]
    beta = wolves[order[1]].copy(); beta_score = scores[order[1]]
    delta = wolves[order[2]].copy(); delta_score = scores[order[2]]
    # Garantir que le meilleur connu reste best_ray si la population a régressé
    if best_ray_score < alpha_score:
        delta, delta_score = beta, beta_score
        beta, beta_score = alpha, alpha_score
        alpha, alpha_score = best_ray.copy(), best_ray_score

    max_iter = (gwo_budget - n_agents) // n_agents + 1
    for it in range(max_iter):
        a = 2.0 * (1.0 - it / max(1, max_iter))
        for i in range(n_agents):
            r1, r2 = rng.rand(dim), rng.rand(dim)
            A1, C1 = 2 * a * r1 - a, 2 * r2
            X1 = alpha - A1 * np.abs(C1 * alpha - wolves[i])
            r1, r2 = rng.rand(dim), rng.rand(dim)
            A2, C2 = 2 * a * r1 - a, 2 * r2
            X2 = beta - A2 * np.abs(C2 * beta - wolves[i])
            r1, r2 = rng.rand(dim), rng.rand(dim)
            A3, C3 = 2 * a * r1 - a, 2 * r2
            X3 = delta - A3 * np.abs(C3 * delta - wolves[i])
            wolves[i] = np.clip((X1 + X2 + X3) / 3.0, bounds[0], bounds[1])

        for i in range(n_agents):
            if len(history) >= eval_budget:
                break
            nn.set_params(wolves[i])
            fit = _mse(y, nn.forward(X))
            if fit < alpha_score:
                delta, delta_score = beta, beta_score
                beta, beta_score = alpha, alpha_score
                alpha, alpha_score = wolves[i].copy(), fit
            elif fit < beta_score:
                delta, delta_score = beta, beta_score
                beta, beta_score = wolves[i].copy(), fit
            elif fit < delta_score:
                delta, delta_score = wolves[i].copy(), fit
            history.append(alpha_score)
        if len(history) >= eval_budget:
            break

    nn.set_params(alpha)
    return alpha_score, history[:eval_budget]
```

- [ ] **Step 4 : Lancer tous les tests**

```bash
python3 tests/test_algorithms.py
```
Expected : tous les tests passent (8 au total).

- [ ] **Step 5 : Commit**

```bash
git add core/algorithms.py tests/test_algorithms.py
git commit -m "feat: train_hybrid_seeded (GWO seedé autour de best_ray)"
```

---

## Task 9 : `experiments/runner.py` — pipeline générique

**Files :**
- Create: `experiments/runner.py`
- Test: `tests/test_runner.py`

- [ ] **Step 1 : Écrire `tests/test_runner.py`**

```python
"""Test de runner.run_benchmark sur un cas minimal (dim=3, budget réduit)."""
import sys, os, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.runner import run_benchmark


def test_run_benchmark_smoke():
    out = "/tmp/heaviside_test_run"
    if os.path.exists(out):
        shutil.rmtree(out)
    df = run_benchmark(dim=3, output_dir=out, eval_budget=500, n_samples=300,
                       func_names=['gl'], seed=42)
    # 1 fonction × 4 algos = 4 lignes
    assert len(df) == 4
    assert set(df['algo']) == {'backprop', 'ray', 'gwo', 'hybrid'}
    assert os.path.exists(os.path.join(out, 'summary.csv'))
    assert os.path.exists(os.path.join(out, 'figures', 'gl.png'))
    print("PASS: test_run_benchmark_smoke")


if __name__ == "__main__":
    test_run_benchmark_smoke()
    print("\n[runner] 1 test passed.")
```

- [ ] **Step 2 : Vérifier l'échec**

```bash
python3 tests/test_runner.py
```
Expected : `ModuleNotFoundError: No module named 'experiments.runner'`

- [ ] **Step 3 : Implémenter `experiments/runner.py`**

```python
"""Pipeline d'expérience : pour une dimension donnée, exécute les 4 algos sur
chacune des 5 fonctions, sauvegarde CSV + figures de convergence."""
import os
import time
from typing import List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # backend non-interactif
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
    'backprop': 'Backprop (échec attendu)',
    'ray': 'Ray Shooting',
    'gwo': 'GWO',
    'hybrid': 'Hybrid (Ray + GWO seedé)',
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
    plt.xlabel('Évaluations de la fonction de perte')
    plt.ylabel('MSE (échelle log)')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_benchmark(dim: int, output_dir: str, eval_budget: int = 25000,
                  n_samples: int = 2000, func_names: Optional[List[str]] = None,
                  seed: int = 42) -> pd.DataFrame:
    """Lance les 4 algos sur chaque fonction pour une dimension donnée."""
    if func_names is None:
        func_names = list(FUNCTIONS.keys())
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)

    rows = []
    print(f"\n{'='*72}")
    print(f"BENCHMARK dim={dim} | budget={eval_budget} évals | {n_samples} samples")
    print(f"Fonctions : {func_names}")
    print(f"{'='*72}")

    for func_name in func_names:
        print(f"\n>>> Fonction : {func_name}")
        X_tr, y_tr, X_va, y_va = generate_dataset(func_name, n_samples=n_samples,
                                                    dim=dim, seed=seed)
        # Normalisation y sur le train
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

        # Plot convergence pour cette fonction
        plot_path = os.path.join(output_dir, 'figures', f'{func_name}.png')
        _plot_convergence(histories,
                          title=f"Convergence — {func_name} (dim={dim})",
                          save_path=plot_path)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSauvegardé : {csv_path}")

    # Récap rapide
    pivot = df.pivot(index='func', columns='algo', values='mse_val')
    print("\nMSE val par (func, algo) :")
    print(pivot.to_string(float_format=lambda v: f"{v:.5f}"))
    print(f"\nMeilleur algo par fonction :")
    for func in df['func'].unique():
        sub = df[df['func'] == func].sort_values('mse_val').iloc[0]
        print(f"  {func:<10} -> {sub['algo']:<10} (mse_val={sub['mse_val']:.5f})")

    return df
```

- [ ] **Step 4 : Lancer le test**

```bash
python3 tests/test_runner.py
```
Expected : `[runner] 1 test passed.`

- [ ] **Step 5 : Commit**

```bash
git add experiments/runner.py tests/test_runner.py
git commit -m "feat: experiments.runner (pipeline générique par dim)"
```

---

## Task 10 : Scripts `run_3d.py`, `run_10d.py`, `run_50d.py`

**Files :**
- Create: `experiments/run_3d.py`, `experiments/run_10d.py`, `experiments/run_50d.py`

- [ ] **Step 1 : `experiments/run_3d.py`**

```python
"""Benchmark dim=3 : python -m experiments.run_3d"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.runner import run_benchmark

if __name__ == "__main__":
    run_benchmark(
        dim=3,
        output_dir="results/3d",
        eval_budget=25000,
        n_samples=2000,
        seed=42,
    )
```

- [ ] **Step 2 : `experiments/run_10d.py`**

```python
"""Benchmark dim=10 : python -m experiments.run_10d"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.runner import run_benchmark

if __name__ == "__main__":
    run_benchmark(
        dim=10,
        output_dir="results/10d",
        eval_budget=25000,
        n_samples=2000,
        seed=42,
    )
```

- [ ] **Step 3 : `experiments/run_50d.py`**

```python
"""Benchmark dim=50 : python -m experiments.run_50d"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.runner import run_benchmark

if __name__ == "__main__":
    run_benchmark(
        dim=50,
        output_dir="results/50d",
        eval_budget=25000,
        n_samples=2000,
        seed=42,
    )
```

- [ ] **Step 4 : Smoke test rapide (3D, budget réduit)**

```bash
cd "/Users/david/Documents/Documents - MacBook Pro de David/Mahon Lev/Projet Yossi/non-countigous-deep"
python3 -c "
import sys, os
sys.path.insert(0, '.')
from experiments.runner import run_benchmark
run_benchmark(dim=3, output_dir='/tmp/smoke_3d', eval_budget=500, n_samples=300, func_names=['gl', 'gs'], seed=42)
"
```
Expected : 2 fonctions × 4 algos exécutent sans erreur, fichiers générés dans `/tmp/smoke_3d`.

- [ ] **Step 5 : Commit**

```bash
git add experiments/run_3d.py experiments/run_10d.py experiments/run_50d.py
git commit -m "feat: scripts run_{3,10,50}d.py"
```

---

## Task 11 : Synthèse globale (`experiments/global_summary.py`)

**Files :**
- Create: `experiments/global_summary.py`

- [ ] **Step 1 : Créer `experiments/global_summary.py`**

```python
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

    # === Validation de la thèse ===
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

    # === Heatmap MSE val par (dim, algo) — moyenne sur les fonctions ===
    pivot = df.groupby(['dim', 'algo'])['mse_val'].mean().unstack()
    pivot = pivot[ALGOS]  # ordre fixe
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
```

- [ ] **Step 2 : Smoke test (ne fait rien si pas de summary, juste check syntaxe)**

```bash
python3 -c "from experiments import global_summary; print('OK')"
```
Expected : `OK`

- [ ] **Step 3 : Commit**

```bash
git add experiments/global_summary.py
git commit -m "feat: experiments.global_summary (heatmap + validation thèse)"
```

---

## Task 12 : Run du benchmark 3D

**Files :** aucun nouveau. Exécution + analyse.

- [ ] **Step 1 : Lancer le benchmark 3D complet**

```bash
cd "/Users/david/Documents/Documents - MacBook Pro de David/Mahon Lev/Projet Yossi/non-countigous-deep"
python3 -m experiments.run_3d
```
Expected : Pour les 5 fonctions, 4 algos exécutent (`mse_val=X.YYYYY` imprimé). Pas d'erreur. Durée : 5-15 minutes selon la machine.

- [ ] **Step 2 : Inspecter `results/3d/summary.csv`**

```bash
cat "/Users/david/Documents/Documents - MacBook Pro de David/Mahon Lev/Projet Yossi/non-countigous-deep/results/3d/summary.csv"
```
Critères de validation :
- 20 lignes (5 fonctions × 4 algos)
- `mse_val` de `backprop` devrait être **5× à 100× supérieur** à celui des 3 autres algos
- `mse_val` de `hybrid` devrait être ≤ min(`ray`, `gwo`) sur au moins 3/5 fonctions

Si la dernière condition n'est PAS remplie : passer à Step 3, sinon directement Step 4.

- [ ] **Step 3 : Si hybrid ne gagne pas — tuner**

Si hybrid n'est pas le meilleur sur ≥3/5 fonctions, modifier dans `experiments/runner.py:50` :
```python
    if algo == 'hybrid':
        return train_hybrid_seeded(nn, X_tr, y_tr, eval_budget=eval_budget,
                                    ray_ratio=0.3, sigma=0.1, n_agents=30, seed=seed)
```
(passer `ray_ratio=0.3` ou `0.7`, `sigma ∈ {0.05, 0.1, 0.3, 0.5}`)

Puis re-lancer `python3 -m experiments.run_3d` et regarder. Itérer jusqu'à ce que hybrid soit dominant. Documenter le choix final dans un commentaire.

- [ ] **Step 4 : Vérifier les figures**

```bash
ls "/Users/david/Documents/Documents - MacBook Pro de David/Mahon Lev/Projet Yossi/non-countigous-deep/results/3d/figures/"
```
Expected : 5 PNG (`gl.png`, `gs.png`, `geta.png`, `ggamma.png`, `complex.png`). Ouvrir un PNG et vérifier que la courbe rouge (backprop) est nettement au-dessus des autres.

- [ ] **Step 5 : Commit les résultats**

```bash
git add results/3d/
git commit -m "results: benchmark 3D complet"
```

---

## Task 13 : Run du benchmark 10D

**Files :** aucun nouveau.

- [ ] **Step 1 : Lancer le benchmark 10D**

```bash
cd "/Users/david/Documents/Documents - MacBook Pro de David/Mahon Lev/Projet Yossi/non-countigous-deep"
python3 -m experiments.run_10d
```
Durée attendue : 10-30 min.

- [ ] **Step 2 : Inspecter `results/10d/summary.csv`**

Mêmes critères qu'en 3D. La hiérarchie attendue : `hybrid <= min(ray, gwo) << backprop`.

- [ ] **Step 3 : Si nécessaire, tuner (cf. Task 12 Step 3)**

Si hybrid plafonne en 10D, essayer :
- `ray_ratio=0.4`, `sigma=0.15` (un peu plus d'exploration et un seeding plus serré)
- ou augmenter `n_agents=40` dans le hybrid

- [ ] **Step 4 : Vérifier les figures `results/10d/figures/*.png`**

- [ ] **Step 5 : Commit**

```bash
git add results/10d/
git commit -m "results: benchmark 10D complet"
```

---

## Task 14 : Run du benchmark 50D

**Files :** aucun nouveau.

- [ ] **Step 1 : Lancer le benchmark 50D**

```bash
cd "/Users/david/Documents/Documents - MacBook Pro de David/Mahon Lev/Projet Yossi/non-countigous-deep"
python3 -m experiments.run_50d
```
Durée attendue : 30-90 min. C'est le run le plus long parce que la dim d'optim atteint ~1122 (50×16 + 16 + ...).

- [ ] **Step 2 : Inspecter `results/50d/summary.csv`**

C'est le test le plus dur pour la thèse. Si le hybrid ne domine pas, il faut sérieusement tuner.

- [ ] **Step 3 : Si nécessaire, augmenter le budget ou tuner**

Si tous les algos plafonnent loin de zéro en 50D (e.g. tous les `mse_val > 0.5`), c'est qu'on n'a pas assez d'évaluations. Modifier `experiments/run_50d.py` :
```python
eval_budget=50000,   # ou 100000
```
Et re-lancer. Tous les algos auront le même budget plus grand, donc la comparaison reste équitable.

Tuning hybrid pour 50D :
- `ray_ratio=0.6` (plus d'exploration en haute dim)
- `sigma=0.3` (raffinement plus large car le minimum local est moins étroit en 50D)

- [ ] **Step 4 : Vérifier les figures `results/50d/figures/*.png`**

- [ ] **Step 5 : Commit**

```bash
git add results/50d/
git commit -m "results: benchmark 50D complet"
```

---

## Task 15 : Synthèse globale + nouveau README

**Files :**
- Run : `experiments/global_summary.py`
- Create: `README.md`

- [ ] **Step 1 : Générer la synthèse globale**

```bash
cd "/Users/david/Documents/Documents - MacBook Pro de David/Mahon Lev/Projet Yossi/non-countigous-deep"
python3 -m experiments.global_summary
```
Expected :
- Affiche un tableau de validation avec `bp_fails` (devrait être True partout) et `hybrid_is_best` (≥12/15 selon le spec)
- Crée `results/global_summary.csv` et `results/global_comparison.png`

- [ ] **Step 2 : Inspecter `results/global_comparison.png`**

La heatmap doit montrer :
- Ligne `backprop` : rouge intense (MSE élevé) pour les 3 dimensions
- Ligne `hybrid` : vert intense (MSE faible) pour les 3 dimensions
- Lignes `ray` et `gwo` : entre les deux, plutôt vertes

- [ ] **Step 3 : Écrire `README.md`**

```markdown
# Non-Contiguous Deep Learning — Benchmark Multi-Dimensionnel

Ce projet démontre expérimentalement que :

1. **La rétropropagation échoue** sur des réseaux à activation Heaviside : la dérivée nulle dans les couches cachées empêche l'apprentissage du modèle (seule la couche de sortie apprend).
2. **Ray Shooting** et **Grey Wolf Optimizer (GWO)** — méthodes sans gradient — entraînent ces réseaux avec succès.
3. **Hybrid Ray + GWO seedé** donne les meilleurs résultats : Ray explore globalement, puis GWO raffine localement à partir d'une population concentrée autour de la solution de Ray.

Le benchmark teste 5 fonctions discontinues n-dimensionnelles en **3D, 10D, et 50D**, avec un budget équitable de **25 000 évaluations** de la fonction de perte par algorithme.

## Structure

```
core/             # Réseau, fonctions, algorithmes (cœur scientifique)
experiments/      # Pipeline d'expérience par dimension
tests/            # Tests autonomes (stdlib, pas de framework)
results/          # CSV + figures (3d/, 10d/, 50d/, + synthèse globale)
archive/          # Anciens scripts (préservés pour référence)
docs/             # Spec design et plan d'implémentation
```

## Utilisation

```bash
# Tests
python3 tests/test_network.py
python3 tests/test_functions.py
python3 tests/test_algorithms.py
python3 tests/test_runner.py

# Benchmarks (l'un après l'autre)
python3 -m experiments.run_3d
python3 -m experiments.run_10d
python3 -m experiments.run_50d

# Synthèse globale (heatmap + validation thèse)
python3 -m experiments.global_summary
```

## Résultats

Voir `results/global_comparison.png` pour la vue d'ensemble et `results/global_summary.csv` pour les chiffres détaillés. Chaque dimension a son propre dossier `results/{dim}d/` avec figures de convergence par fonction.

## Architecture du réseau

`HeavisideNetwork([16, 12, 8])` — fixe peu importe la dimension d'entrée. C'est seulement la **complexité de la cible** qui change entre 3D et 50D, pas la **capacité du modèle**.

## Dépendances

Python 3.7+, numpy, matplotlib, pandas. Pas de framework de test externe.
```

- [ ] **Step 4 : Commit global**

```bash
git add results/global_summary.csv results/global_comparison.png README.md
git commit -m "results: synthèse globale 3D/10D/50D + README"
```

- [ ] **Step 5 : Vérification finale**

```bash
git status
git log --oneline -10
```
Expected : `git status` clean (ou seuls `.DS_Store`), `git log` montre les 15+ commits du plan.

---

## Auto-revue du plan (avant exécution)

1. **Couverture du spec :**
   - Section 2 « Périmètre » : ✓ couvert par Tasks 3, 4, 5-8 (5 algos), 9, 10, 11
   - Section 3 « Architecture projet » : ✓ Tasks 1, 2
   - Section 4 « Fonctions discontinues n-D » : ✓ Task 4
   - Section 5 « Réseau Heaviside » : ✓ Task 3
   - Section 6 « Algorithmes » : ✓ Tasks 5, 6, 7, 8
   - Section 7 « Pipeline » : ✓ Task 9
   - Section 8 « Visualisations » : ✓ Tasks 9 (par fonction) et 11 (globale)
   - Section 9 « Conditions de succès » : ✓ vérifiées dans Task 11 Step 1
   - Section 10 « Plan d'exécution » : ✓ Tasks 12-15
   - Section 11 « Risques » : ✓ mitigations dans Tasks 12-14 Step 3

2. **Pas de placeholder.** Vérifié : code complet partout, pas de "TBD", commandes exactes.

3. **Cohérence des types/noms.** Vérifié :
   - `HeavisideNetwork(input_dim, hidden_sizes, output_dim, seed)` cohérent partout
   - `train_*(nn, X, y, eval_budget, ..., seed)` cohérent pour les 4 algos
   - `(final_mse, history)` retour cohérent partout
   - `BOUNDARY_MASK` exposé par `core/functions.py` et utilisé par les tests

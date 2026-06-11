# Pointeurs vers le code source

Le code source vit dans `core/`, `experiments/`, `tests/`. Ce fichier sert
d'index pour citer les fichiers et lignes précises dans la thèse.

## Code scientifique — `core/`

| Fichier             | Lignes clés | Contenu |
|---------------------|-------------|---------|
| `core/network.py`   | 11-73       | Classe `HeavisideNetwork` : architecture `[16, 12, 8]`, init Xavier, méthodes `forward`, `get_params`, `set_params`, `num_params`. |
| `core/network.py`   | 38-40       | Définition de la fonction Heaviside : `np.where(z >= 0, 1.0, 0.0)`. |
| `core/functions.py` | 27-66       | Les 5 fonctions `_mask_*` qui définissent les frontières géométriques. |
| `core/functions.py` | 73-125      | Les 5 fonctions cibles `gl`, `gs`, `geta`, `ggamma`, `complex_func`. |
| `core/functions.py` | 153-165     | `generate_dataset` — tirage uniforme + split 80/20. |
| `core/algorithms.py`| 22-51       | `train_backprop` — descente de gradient. Les couches cachées ne sont pas mises à jour (gradient = 0). |
| `core/algorithms.py`| 58-104      | `train_ray_shooting` — exploration directionnelle greedy avec `steps_per_ray=20`. |
| `core/algorithms.py`| 111-176     | `train_gwo` — Grey Wolf Optimizer (30 agents, hiérarchie α/β/δ). |
| `core/algorithms.py`| 183-267     | `train_hybrid_seeded` — Ray (50% budget) + GWO seedé autour de `best_ray`. |

## Pipeline expérimental — `experiments/`

| Fichier                          | Rôle |
|----------------------------------|------|
| `experiments/runner.py`          | `run_benchmark(dim, output_dir, ...)` — exécute les 4 algos × 5 fonctions, sauve CSV + figures de convergence. |
| `experiments/run_3d.py`          | Wrapper : `run_benchmark(dim=3, ..., eval_budget=25000)`. |
| `experiments/run_10d.py`         | Wrapper 10D. |
| `experiments/run_50d.py`         | Wrapper 50D. |
| `experiments/global_summary.py`  | Concatène les 3 CSV, calcule la heatmap `results/global_comparison.png`, imprime la table de validation de thèse. |

## Tests — `tests/`

**22 tests, tous passants.** Cf. `key_evidence.md` pour le détail.

| Fichier                          | Nb | Couvre |
|----------------------------------|----|--------|
| `tests/test_network.py`          | 6  | `HeavisideNetwork` |
| `tests/test_functions.py`        | 6  | 5 fonctions + calibrage P(inside) |
| `tests/test_algorithms.py`       | 9  | 4 algos + le test critique du gel des couches cachées |
| `tests/test_runner.py`           | 1  | Smoke test du pipeline complet |

## Hyperparamètres clés (à citer dans le protocole expérimental)

| Paramètre          | Valeur   | Justification |
|--------------------|----------|---------------|
| Architecture       | `[16, 12, 8]` | Fixe quelle que soit la dim — seule la cible varie, pas la capacité du modèle. |
| Activation cachée  | Heaviside `H(z) = 1 si z ≥ 0 sinon 0` | C'est l'objet d'étude. |
| Activation sortie  | Linéaire | Pour permettre une régression. |
| Init               | Xavier uniforme | `W ~ U(-√(6/(n_in+n_out)), √(6/...))`. |
| Domaine de `x`     | `[-2, 2]^D` | Uniforme. |
| `n_samples`        | 2000     | Split 80/20 → 1600 train / 400 val. |
| `eval_budget`      | 25 000   | Identique pour tous les algos — comparaison équitable. |
| `seed`             | 42       | Reproductibilité. |
| GWO `n_agents`     | 30       | Standard. |
| Ray `steps_per_ray`| 20       | Calibré empiriquement (cf. `test_ray_shooting_steps20_beats_steps50`). |
| Hybrid `ray_ratio` | 0.5      | 12 500 évals Ray + 12 500 évals GWO. |
| Hybrid `sigma`     | 0.2      | Écart-type relatif de la perturbation gaussienne autour de `best_ray`. |
| Backprop `lr`      | 0.01     | Pour la couche de sortie (les autres ne bougent pas). |

## Reproductibilité complète

```bash
git clone <repo>
cd non-countigous-deep

# Benchmarks (≈ 10 min total)
python3 -m experiments.run_3d
python3 -m experiments.run_10d
python3 -m experiments.run_50d
python3 -m experiments.global_summary

# Tests (≈ 30 s)
python3 tests/test_network.py
python3 tests/test_functions.py
python3 tests/test_algorithms.py
python3 tests/test_runner.py
```

Tous les artefacts sont reproductibles à l'identique (seed=42).

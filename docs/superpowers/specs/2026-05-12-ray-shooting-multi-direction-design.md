# Ray Shooting — Best-of-K Directions — Design

**Date** : 2026-05-12
**Auteur** : David (avec assistance Claude)
**Statut** : Approuvé pour implémentation

## Contexte

L'implémentation actuelle de `train_ray_shooting` dans `core/algorithms.py` sous-performe par rapport à GWO sur les benchmarks 3D/10D/50D. Diagnostic empirique sur la fonction `gl` (budget=5000 évaluations) :

| Variante | 3D | 10D | 50D |
|---|---|---|---|
| Actuel (greedy, break) | 0.0340 | 0.0262 | 0.0497 |
| No-break (best sur tout le ray) | 0.0337 | 0.0220 | 0.0512 |
| Fine steps=200 | 0.0378 | 0.1527 (échec) | 0.1038 (échec) |
| **Best-of-K=5 directions** | 0.0356 | **0.0187** | **0.0489** |

Le `break` actuel coupe trop vite l'exploration d'un ray dès qu'une amélioration est trouvée. Augmenter `steps_per_ray` est contre-productif (les steps sont utilisés sur une seule direction au lieu de plusieurs). La meilleure stratégie est de **multiplier les directions essayées** par round, et de garder le meilleur point trouvé.

## Modification

### `core/algorithms.py::train_ray_shooting`

Nouvelle signature :
```python
def train_ray_shooting(nn, X, y, eval_budget=25000,
                       steps_per_ray=20, k_directions=5,
                       bounds=(-2.0, 2.0), seed=None):
```

Algorithme :
```
best = nn.get_params(); best_score = MSE
while budget > 0:
    round_best = None; round_best_score = inf
    for j in range(k_directions):
        target = uniform(bounds, dim)
        direction = target - best
        for step in range(steps_per_ray):
            t = step / (steps_per_ray - 1)
            cand = clip(best + t * direction, bounds)
            score = MSE(cand)
            budget -= 1
            if score < round_best_score:
                round_best, round_best_score = cand, score
            history.append(min(best_score, round_best_score))
            if budget <= 0: break
        if budget <= 0: break
    if round_best_score < best_score:
        best, best_score = round_best, round_best_score
return best_score, history
```

Changements par rapport à l'existant :
- **Pas de `break` interne** : on parcourt tous les `steps_per_ray` points de chaque direction
- **Round = K directions** : on garde le meilleur point trouvé sur les K × steps évaluations
- `steps_per_ray` par défaut passe de **50 à 20** (compensé par `k_directions=5` → 100 evals/round comme avant)
- `k_directions=1` restaure quasi le comportement actuel (mais sans le break — version no-break)

### `core/algorithms.py::train_hybrid_seeded`

Aucune modification de code requise : la phase Ray appelle `train_ray_shooting` avec les nouveaux defaults, donc profite de l'amélioration automatiquement.

## Tests

### Nouveau test dans `tests/test_algorithms.py`
```python
def test_ray_shooting_multi_direction_beats_single():
    """Best-of-K=5 bat K=1 sur un problème 10D."""
    nn_k1 = HeavisideNetwork(10, [16,12,8], 1, seed=42)
    nn_k5 = HeavisideNetwork(10, [16,12,8], 1, seed=42)
    X_tr, y_tr, _, _ = _make_problem(dim=10, n=500)
    f_k1, _ = train_ray_shooting(nn_k1, X_tr, y_tr, eval_budget=2000, k_directions=1, seed=0)
    f_k5, _ = train_ray_shooting(nn_k5, X_tr, y_tr, eval_budget=2000, k_directions=5, seed=0)
    assert f_k5 <= f_k1, f"k=5 ({f_k5:.4f}) doit battre k=1 ({f_k1:.4f})"
```

### Les 8 tests existants doivent passer
Le test `test_ray_shooting_decreases_mse` (vérifie que Ray converge) doit toujours passer.
Le test `test_ray_shooting_history_monotone` (history non-croissant) doit toujours passer.

## Re-exécution

Une fois l'algorithme amélioré, re-lancer :
- `python3 -m experiments.run_3d`
- `python3 -m experiments.run_10d`
- `python3 -m experiments.run_50d`
- `python3 -m experiments.global_summary`

Et mettre à jour le tableau de résultats dans `README.md`.

## Critères de succès

- 8 tests existants + 1 nouveau test passent
- Sur les 15 combinaisons (dim × func), Ray améliore sur **au moins 10/15** vs l'ancien Ray
- Hybrid (qui utilise Ray en phase 1) doit aussi s'améliorer ou rester comparable
- La hiérarchie globale Hybrid > GWO > Ray > Backprop doit être préservée ou améliorée

## Hors-scope

- Pattern search adaptatif avec restart (option c écartée — trop de complexité pour ce projet)
- Step size adaptatif (bisection, doublement) — peut-être en addendum si le résultat actuel ne suffit pas

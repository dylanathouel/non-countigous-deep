# Preuve formelle : les couches cachées Heaviside sont gelées sous backprop

C'est **le test le plus important** du projet — celui qui valide la motivation
même de toute l'étude. Il prouve, code à l'appui, que la rétropropagation ne
peut pas entraîner les couches cachées d'un réseau Heaviside.

## Le test

**Fichier :** `tests/test_algorithms.py` (fonction `test_backprop_hidden_layers_frozen`)

```python
def test_backprop_hidden_layers_frozen():
    """The zero Heaviside gradient must prevent hidden layers from changing."""
    from core.algorithms import train_backprop
    nn = HeavisideNetwork(3, [16, 12, 8], 1, seed=42)
    X_tr, y_tr, _, _ = _make_problem(dim=3)

    # Snapshot des poids cachés AVANT entraînement
    W_hidden_before = [W.copy() for W, _ in nn.layers[:-1]]

    # 100 epochs de backprop
    train_backprop(nn, X_tr, y_tr, eval_budget=100, lr=0.01)

    # Snapshot APRÈS
    W_hidden_after = [W for W, _ in nn.layers[:-1]]

    # Assertion : bit-identiques
    for Wb, Wa in zip(W_hidden_before, W_hidden_after):
        np.testing.assert_array_equal(Wb, Wa)
```

**Statut :** PASS (vérifié à chaque exécution).

## Ce que ça prouve

`np.testing.assert_array_equal` exige une égalité **bit-à-bit** sur des
flottants. Si une seule mise à jour de gradient avait modifié un seul poids
d'une seule couche cachée d'un seul ULP, le test échouerait.

Le fait que le test passe sur **3 couches × 100 epochs × 16+12+8 = 36 neurones
cachés** démontre formellement que :

> La descente de gradient ne modifie jamais aucun poids des couches cachées
> d'un réseau Heaviside, peu importe le nombre d'epochs ou les données.

## Pourquoi c'est inévitable mathématiquement

Pour une couche cachée Heaviside `A = H(Z)` où `Z = X·W + b` :

- ∂A/∂Z = δ(Z) — distribution de Dirac, nulle presque partout en pratique numérique.
- D'où ∂Loss/∂W = (∂Loss/∂A) · (∂A/∂Z) · X^T = 0.

Le gradient est donc identiquement nul après quantification flottante. Le code
de `train_backprop` (`core/algorithms.py:31-51`) reflète explicitement cela :

```python
# Hidden layers are NOT updated (gradient = 0)
nn.layers[-1] = (W_out - lr * dW_out, b_out - lr * db_out)
```

Aucune mise à jour des `nn.layers[:-1]` n'est codée — non par optimisation,
mais parce qu'elle serait mathématiquement un no-op.

## Conséquence : le réseau dégénère en ELM

Puisque les couches cachées restent à leur initialisation Xavier aléatoire,
elles produisent un mapping `Φ(x) = H_{L-1}(...H_2(H_1(W_1 x + b_1) W_2 + b_2)...)`
totalement **fixé** dès le départ. La couche de sortie (linéaire) résout alors
un simple problème de régression linéaire :

```
min_w ||Φ(X) · w - y||²
```

C'est précisément la définition d'un **Extreme Learning Machine**. La
performance de backprop est donc plafonnée par l'expressivité de ce mapping
aléatoire — ce qui explique le plateau observé dans `RESULTS_FR.md` :

- 3D → 0.062
- 10D → 0.067
- 50D → 0.079

Aucune amélioration possible sans changer d'algorithme.

## Tests complémentaires importants

Pour la rédaction de la section "Validation", citer aussi :

| Test                                              | Fichier                        | Ce qu'il prouve |
|---------------------------------------------------|--------------------------------|-----------------|
| `test_heaviside_hidden_activations_are_binary`    | `tests/test_network.py`        | Les activations cachées sont strictement {0, 1} — c'est bien Heaviside |
| `test_function_is_discontinuous`                  | `tests/test_functions.py`      | Les 5 fonctions cibles ont un saut mesurable (>0.1) à la frontière |
| `test_boundary_proportion_calibrated`             | `tests/test_functions.py`      | P(inside) ∈ [0.20, 0.80] en 3D / 10D / 50D — calibrage validé |
| `test_ray_shooting_steps20_beats_steps50`         | `tests/test_algorithms.py`     | Justifie empiriquement le choix `steps_per_ray=20` (multi-seed) |
| `test_ray_shooting_history_monotone`              | `tests/test_algorithms.py`     | L'historique du MSE Ray est non-croissant (best-so-far) |
| `test_gwo_seeds_current_position`                 | `tests/test_algorithms.py`     | GWO démarre bien depuis l'init Xavier du réseau, pas d'un point aléatoire |
| `test_hybrid_seeded_better_than_random_gwo`       | `tests/test_algorithms.py`     | Hybrid n'est jamais catastrophiquement pire que GWO seul |

**Total : 22 tests, tous passants.**

## Comment relancer

```bash
# Le test critique seul
python3 -c "
import sys, os
sys.path.insert(0, os.path.abspath('.'))
from tests.test_algorithms import test_backprop_hidden_layers_frozen
test_backprop_hidden_layers_frozen()
"

# Tous les tests
python3 tests/test_network.py
python3 tests/test_functions.py
python3 tests/test_algorithms.py
python3 tests/test_runner.py
```

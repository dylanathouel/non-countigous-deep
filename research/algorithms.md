# Algorithmes étudiés

Tous les algorithmes ont la même **API** :

```python
def train_X(nn, X, y, eval_budget=25000, ...) -> (final_mse, history)
```

Le **budget équitable** est de 25 000 évaluations de la fonction de perte
(forward pass) pour chaque algorithme et chaque (fonction × dimension). C'est
ce qui rend la comparaison équitable malgré des coûts par itération différents.

Implémentation : `core/algorithms.py`.

---

## 1. Backpropagation — l'échec à démontrer

**Principe.** Descente de gradient standard sur le MSE. À chaque epoch :
forward pass → calcul du gradient par rétropropagation → mise à jour des poids.

**Le problème spécifique à Heaviside.** La dérivée de la fonction Heaviside
H(z) = 1 si z ≥ 0 sinon 0 est nulle presque partout (sauf en z = 0 où elle
n'est pas définie). En pratique :

```python
@staticmethod
def _heaviside(z):
    return np.where(z >= 0, 1.0, 0.0)
```

Le gradient back-propagé dans les couches cachées est donc **identiquement
nul**. Seule la couche de sortie (qui est linéaire) reçoit un gradient utile.

**Conséquence.** Le réseau dégénère en un Extreme Learning Machine (ELM) :
les couches cachées agissent comme un mapping aléatoire figé, et la couche
de sortie résout une simple régression linéaire sur ces features. La
performance est plafonnée par l'expressivité de ce mapping aléatoire.

**Cf. `key_evidence.md`** pour la preuve formelle (test unitaire) que les poids
cachés sont bit-identiques avant et après training.

---

## 2. Ray Shooting — exploration directionnelle sans gradient

**Principe.** À chaque rayon :

1. On part du **meilleur point connu** `best_vec` (mémoire greedy).
2. On tire une cible aléatoire uniforme dans `[-2, 2]^D`.
3. On définit la direction `direction = target - best_vec`.
4. On parcourt `steps_per_ray` points le long du segment `best_vec + t * direction`.
5. Dès qu'un point est meilleur, on met à jour `best_vec` et on tire un nouveau rayon.

**Pseudocode** (`core/algorithms.py:58-104`) :

```python
best_vec = nn.get_params()
best_score = mse(...)
while budget restant:
    target = rng.uniform(-2, 2, dim)
    direction = target - best_vec
    for t in linspace(0, 1, steps_per_ray):
        candidate = best_vec + t * direction
        if mse(candidate) < best_score:
            best_vec, best_score = candidate, score
            break  # direction suivante
```

**Choix calibré : `steps_per_ray = 20`** (et non 50). Étude empirique multi-seeds
en 10D : essayer **plus de directions plus courtement** (20 pas) bat **moins
de directions explorées plus profondément** (50 pas). Gain ~8-10 % en 3D/10D,
neutre en 50D. Validé par `tests/test_algorithms.py::test_ray_shooting_steps20_beats_steps50`.

**Limite fondamentale.** La cible est **uniforme aléatoire** dans le cube. En
haute dimension, la direction `target - best_vec` devient quasi-isotrope
(concentration de la mesure). Pas de mémoire des directions qui ont
historiquement bien marché. C'est pourquoi Ray seul reste proche de Backprop
sur la moyenne.

---

## 3. Grey Wolf Optimizer (GWO) — méta-heuristique populationnelle

**Principe.** Une population de `n_agents=30` "loups" évolue dans l'espace
des paramètres. À chaque itération, les 3 meilleurs (alpha, beta, delta)
guident les positions des autres :

```
X1 = alpha - A1 * |C1 * alpha - X_i|     # tirage vers alpha
X2 = beta  - A2 * |C2 * beta  - X_i|     # tirage vers beta
X3 = delta - A3 * |C3 * delta - X_i|     # tirage vers delta
X_i_new = (X1 + X2 + X3) / 3
```

Le coefficient `a = 2 * (1 - it/max_iter)` décroît linéairement : exploration
au début (sauts longs), exploitation à la fin (sauts courts autour des leaders).

**Init.** Le premier loup est seedé avec la position courante du réseau (les
poids initiaux Xavier), les autres sont uniformes dans `[-2, 2]^D`.

**Pourquoi ça marche mieux que Ray.** GWO a une **mémoire collective**
(alpha/beta/delta) qui maintient en permanence les 3 meilleures solutions
trouvées. Les nouvelles positions sont attirées vers cette mémoire, pas vers
des cibles aléatoires.

---

## 4. Hybrid (Ray + GWO seedé) — l'optimum proposé

**Principe : exploration globale + raffinement local.**

```
Phase 1 — Ray Shooting (12 500 évals, ray_ratio=0.5)
         → produit best_ray
Phase 2 — GWO seedé (12 500 évals)
         wolves[0]    = best_ray              ← garantie de non-régression
         wolves[1:30] = best_ray + N(0, σ·scale)  avec σ = 0.2
```

**La clé.** Différence avec un Hybrid naïf (Ray puis GWO standard) : on **ne
perd pas** la connaissance acquise par Ray quand GWO démarre. La population
GWO commence concentrée autour de la meilleure solution trouvée par Ray, donc
l'exploitation se fait dans la bonne zone dès l'itération 1.

**Garde-fou anti-régression.** Si la perturbation gaussienne crée une
population dont aucun élément n'est aussi bon que `best_ray`, on injecte
explicitement `best_ray` comme alpha (`core/algorithms.py:227-231`).

**Résultat.** Hybrid bat tous les autres algos en moyenne sur les 3 dimensions
(cf. `results/RESULTS_FR.md`). Sur les 6 cas où il ne gagne pas individuellement,
il reste à moins de 5 % du gagnant.

---

## Tableau comparatif

| Algorithme    | Type            | Mémoire   | Exploration | Exploitation |
|---------------|-----------------|-----------|-------------|--------------|
| Backprop      | Gradient        | Locale    | Nulle (hidden gelés) | Limitée à output |
| Ray Shooting  | Greedy random   | best_vec  | Directionnelle aléatoire | Courte (steps=20) |
| GWO           | Pop. meta-heur. | α/β/δ     | a décroissant | Convergence vers leaders |
| **Hybrid**    | Ray → GWO seedé | best_ray + α/β/δ | Phase 1 globale | Phase 2 locale |

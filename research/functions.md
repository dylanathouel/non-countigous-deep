# Fonctions cibles discontinues

Les 5 fonctions cibles utilisées dans le benchmark sont **généralisées en
dimension n** (testées en D = 3, 10, 50) et présentent toutes une
**discontinuité réelle** à la traversée d'une frontière géométrique.

Implémentation : `core/functions.py`.

## Principe du calibrage

En haute dimension, deux problèmes surgissent naturellement :

1. **Concentration de la mesure** — la quasi-totalité des points uniformes
   d'un hypercube ont une norme proche de la médiane attendue. Si les seuils
   sont fixés "à la main", la proportion `P(inside)` tend rapidement vers 0
   ou 1 et la frontière disparaît.

2. **Explosion numérique** — les fonctions de type sin / exp appliquées à
   une somme/norme non normalisée saturent ou divergent quand D grandit.

**Solutions adoptées :**

- Les seuils géométriques sont **calibrés analytiquement** pour que
  `P(inside) ∈ [0.20, 0.80]` quelle que soit D ∈ {3, 10, 50}. Validation :
  `tests/test_functions.py::test_boundary_proportion_calibrated`.
- Les valeurs scalaires alimentant les sin/exp sont **normalisées** :
  division par D, sqrt(D), ou D^(1/4) selon le cas.

Pour x ~ U(-2, 2)^D :
- E[x_i^2] = 4/3
- E[||x||^2] = 4D/3 (utilisé comme seuil pour `ggamma`)

---

## Les 5 fonctions

### 1. `gl` — Hyperplan

**Frontière :** `x_D ≤ moyenne(x_1, ..., x_{D-1})`

**Valeurs :**
- inside : `2 sin(1.25π · ||x||/√D) + 4`
- outside : `2 sin(0.75π · ||x||/√D)`

Saut d'amplitude ≈ 4 à la frontière.

### 2. `gs` — Hypercube calibré

**Frontière :** `|x_i| ≤ α_D` pour tout i, avec `α_D = 2 · 0.5^(1/D)`
(calibré pour P(inside) = 0.5).

**Valeurs :**
- inside : `-2 · (||x||²/D) + 6`
- outside : `4 · exp(1 - ||x||²/D)`

### 3. `geta` — Surface exponentielle atténuée

**Frontière (3 régions !) :** comparée à `threshold = exp(x_1) / D^(1/4)`
- above : `x_D ≥ threshold`
- below : `x_D < threshold - 1/D^(1/4)`
- between : zone intermédiaire de largeur `1/D^(1/4)`

**Valeurs :**
- above : `sin(0.4π · (Σx_i)/√D)`
- below : `sin(0.7π · (Σx_i)/√D) - 4`
- between : `sin(π · (Σx_i)/√D) + 4`

C'est la fonction la plus complexe — **trois plateaux** séparés par deux frontières.

### 4. `ggamma` — Hypersphère

**Frontière :** `||x||² ≤ 4D/3` (médiane attendue, donc P(inside) ≈ 0.5).

**Valeurs :**
- inside : `sin(π · (Σx_i)/√D) + 4`
- outside : `sin(0.4π · (Σx_i)/√D)`

**Cas pathologique en 50D :** tous les algorithmes plafonnent à ~0.13. La
raison : en 50D, presque tous les points uniformes ont une norme très proche
de √(4·50/3), donc la frontière sphérique est *quasi-tangente* à la
distribution. La fonction devient quasi-aléatoire, **irréductible**.

### 5. `complex` — Hyperparaboloïde

**Frontière :** `x_D ≤ c · (Σ_{i<D} x_i² - (D-1) · 4/3)` avec `c = 1/(2√(D-1))`
(calibrée pour P(inside) ≈ 0.5).

**Valeurs :**
- inside : `(||x||²/D) · sin(π · x_1) + 3`
- outside : `-(x_1 · x_2 · x_D / √D) · cos(π · x_D) - 2`

Frontière quadratique + valeurs qui dépendent de produits → la fonction la
plus "riche" en variation locale.

---

## Tableau récapitulatif

| Fonction  | Géométrie de la frontière | Saut typique | Difficulté |
|-----------|---------------------------|---------------|------------|
| `gl`      | Hyperplan                 | ≈ 4           | Facile     |
| `gs`      | Hypercube                 | varie         | Moyen      |
| `geta`    | Surface exp. (3 régions)  | ≈ 4-8         | Difficile  |
| `ggamma`  | Hypersphère               | ≈ 4           | Facile / dégénère en 50D |
| `complex` | Hyperparaboloïde          | varie         | Difficile  |

## Génération du dataset

```python
from core.functions import generate_dataset
X_train, y_train, X_val, y_val = generate_dataset(
    func_name='gl', n_samples=2000, dim=10, seed=42)
```

- Tirage uniforme dans `[-2, 2]^D`.
- Split 80/20 train/val.
- Toutes les expériences utilisent **`seed=42`** pour reproductibilité.
- Normalisation MSE : `y` est rescalé sur `[0, 1]` via `(y - y_min)/(y_max - y_min)`
  côté runner — c'est pourquoi les MSE rapportés sont des nombres petits
  (typiquement 0.01 – 0.13).

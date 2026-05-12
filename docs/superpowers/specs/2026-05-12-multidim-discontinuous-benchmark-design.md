# Multi-Dimensional Discontinuous Functions Benchmark — Design

**Date** : 2026-05-12
**Auteur** : David (avec assistance Claude)
**Statut** : Design approuvé, prêt pour planification d'implémentation

---

## 1. Contexte et thèse à démontrer

Ce projet est un devoir de recherche académique. La thèse scientifique à démontrer expérimentalement est :

1. **La rétropropagation (backpropagation) échoue** à entraîner des réseaux de neurones dont l'activation est Heaviside (`H(z) = 1 si z ≥ 0, sinon 0`). Raison : la dérivée de Heaviside est nulle presque partout, donc le gradient s'annule dans toutes les couches cachées. Seule la couche de sortie peut apprendre.
2. **Ray Shooting** (exploration directionnelle sans gradient) **et Grey Wolf Optimizer (GWO)** (méta-heuristique sociale) **réussissent** à entraîner ces réseaux car ils ne dépendent pas du gradient.
3. **Le Hybrid Ray+GWO seedé** donne les **meilleurs résultats** : Ray explore l'espace globalement, puis GWO raffine localement à partir d'une population concentrée autour de la meilleure solution de Ray.
4. L'utilisation de réseaux Heaviside est justifiée par le fait que les **fonctions cibles sont discontinues** : un réseau à activation discontinue représente naturellement une cible discontinue.

L'objectif technique est d'étendre cette démonstration aux **dimensions 3D, 10D et 50D**.

## 2. Périmètre

**Inclus** :
- 5 fonctions de test discontinues, généralisées en dimension n (3, 10, 50)
- 4 algorithmes d'optimisation : Backprop, Ray Shooting, GWO, Hybrid Seeded
- Pipeline d'expérience reproductible avec budget équitable (25 000 évaluations de loss par algo)
- Visualisations (courbes de convergence) et tableaux récapitulatifs par dimension
- Architecture de réseau fixe `[16, 12, 8]` pour les trois dimensions
- Conservation des anciens scripts dans `archive/` (rien n'est perdu)

**Exclus** :
- Implémentation GPU / parallélisation
- Apprentissage par mini-batches (le projet utilise du full-batch)
- Variantes hybrides exotiques (cyclique imbriqué, adaptive switch) — gardées en réserve en cas d'échec du hybrid seedé
- Datasets réels (MNIST, CIFAR, etc.) — hors scope

## 3. Architecture du projet

```
non-countigous-deep/
├── core/                       # Cœur scientifique partagé
│   ├── __init__.py
│   ├── network.py              # Classe HeavisideNetwork
│   ├── functions.py            # 5 fonctions discontinues n-D
│   └── algorithms.py           # 4 algorithmes d'optimisation
├── experiments/                # Scripts d'expérience par dimension
│   ├── __init__.py
│   ├── runner.py               # Pipeline générique paramétré par dim
│   ├── run_3d.py
│   ├── run_10d.py
│   └── run_50d.py
├── results/
│   ├── 3d/{figures/, summary.csv}
│   ├── 10d/{figures/, summary.csv}
│   ├── 50d/{figures/, summary.csv}
│   ├── global_summary.csv
│   └── global_comparison.png
├── archive/                    # Anciens scripts (conservés)
│   └── ... (tous les fichiers actuels y sont déplacés)
├── docs/superpowers/specs/
│   └── 2026-05-12-multidim-discontinuous-benchmark-design.md (ce fichier)
└── README.md                   # Nouveau README expliquant la nouvelle structure
```

**Justification** : un unique cœur `core/` paramétré par la dimension. Chaque script `run_*d.py` se contente de fixer `dim`, le budget, et invoque `runner.run(dim=...)`. Pas de duplication, scaling propre.

## 4. Fonctions discontinues n-D (`core/functions.py`)

Toutes les fonctions acceptent un array `x: np.ndarray` de shape `(N, D)` et retournent un vecteur de shape `(N,)`.

**Principe de scaling** : pour éviter la dégénérescence en haute dimension (concentration de la mesure), les **seuils géométriques sont calibrés par dimension** de sorte que la proportion de points "intérieurs" reste comparable à la version 3D originale (~30-70% inside selon la fonction).

### Domaine d'échantillonnage

`bounds = (-2, 2)^D` pour toutes les dimensions (cohérent avec l'existant 3D).

### Spécification mathématique

Soit `D` la dimension d'entrée. On note `r² = Σ_i x_i²` (norme euclidienne au carré), `s = Σ_i x_i` (somme). Pour un échantillonnage uniforme sur `[-2, 2]^D` : `E[x_i²] = 4/3`, donc `E[r²] = 4D/3`.

#### Seuils calibrés par dimension

| Fonction | Frontière (paramétrée par `D`) | Valeur du seuil |
|---|---|---|
| `gl` | `x_D ≤ (1/(D-1)) · Σ_{i<D} x_i` | seuil = 0 (pas de scaling, l'hyperplan reste équilibré) |
| `gs` | tous les `|x_i| ≤ α_gs(D)` | `α_gs(D) = 2 · 0.5^(1/D)` (cible P(inside) = 0.5) → 3D: 1.59 · 10D: 1.87 · 50D: 1.97 |
| `geta` | régions définies par `x_D` vs `exp(x_1)/D^(1/4)` (atténuation pour stabiliser) | seuil exponentiel atténué |
| `ggamma` | `r² ≤ θ_g(D)` | `θ_g(D) = 4D/3` (médiane attendue de `r²` sur `U(-2,2)^D`) → 3D: 4 · 10D: 13.33 · 50D: 66.67 |
| `complex` | `x_D ≤ c_cplx(D) · (Σ_{i<D} x_i² − (D−1)·4/3)` | `c_cplx(D) = 1/(2·√(D-1))` (atténuation pour rester ~50/50) |

#### Valeurs renvoyées (inchangées conceptuellement par rapport à la 3D)

| Fonction | Valeur si frontière vraie | Valeur si frontière fausse |
|---|---|---|
| `gl` | `2·sin(1.25π·√(r²/D)) + 4` | `2·sin(0.75π·√(r²/D))` |
| `gs` | `-2·(r²/D) + 6` (intérieur) | `4·exp((1 - r²/D))` (extérieur) |
| `geta` | 3 régions : `sin(0.4π·s/√D)`, `sin(0.7π·s/√D) − 4`, `sin(π·s/√D) + 4` |
| `ggamma` | `sin(π·s/√D) + 4` | `sin(0.4π·s/√D)` |
| `complex` | `(r²/D)·sin(π·x_1) + 3` | `−x_1·x_2·x_D·cos(π·x_D) − 2` |

**Note sur la normalisation** : on divise par `D` (ou `√D`) les arguments des sinus et de l'exp pour éviter que les valeurs explosent en haute dimension. Sinon `r² ≈ 4D/3` ferait osciller `sin(1.25π·√r²)` à très haute fréquence en 50D, rendant la cible quasi-aléatoire — incapprenable même pour Ray/GWO.

#### Vérification empirique requise (test de sanité)

Au moment de l'implémentation, pour chaque (`func`, `D`) :
1. Générer 10 000 points uniformes sur `[-2,2]^D`
2. Calculer la proportion `p_in` de points dans la région "vraie" de la frontière
3. **Asserter** `0.20 ≤ p_in ≤ 0.80` — si non, ajuster le seuil pour cette fonction/dim
4. Vérifier que `f(x)` reste dans une plage finie et raisonnable (`|f| ≤ 20`)

### API publique

```python
from core.functions import FUNCTIONS, generate_dataset

# FUNCTIONS = {'gl': gl, 'gs': gs, 'geta': geta, 'ggamma': ggamma, 'complex': complex_func}
X_train, y_train, X_val, y_val = generate_dataset(
    func_name='gl', n_samples=2000, dim=10, seed=42, bounds=(-2, 2)
)
```

## 5. Réseau de neurones (`core/network.py`)

```python
class HeavisideNetwork:
    def __init__(self, input_dim, hidden_sizes=[16, 12, 8], output_dim=1, seed=None):
        # Xavier init par défaut, biais initialisés à 0
        ...
    def forward(self, X): ...           # Heaviside en hidden, linéaire en output
    def predict(self, X): ...           # Alias de forward
    def get_params(self) -> np.ndarray: # Vecteur plat des paramètres
    def set_params(self, vec: np.ndarray): # Reconstruit les couches depuis un vecteur
    def num_params(self) -> int:        # Total scalaires entraînables
```

**Architecture fixe** : `[16, 12, 8]` peu importe la dimension d'input. Conséquences :
- 3D : ~370 params
- 10D : ~482 params
- 50D : ~1122 params

C'est gérable par Ray/GWO en 25 000 évals (dimension d'optim raisonnable). Le facteur expérimental qui varie entre les trois benchmarks est uniquement la **difficulté de la cible**, pas la **capacité du modèle** — comparaison plus propre.

## 6. Algorithmes (`core/algorithms.py`)

Tous reçoivent une `network: HeavisideNetwork`, des données `X, y`, et un `eval_budget` (nombre d'évaluations de loss). Tous retournent `(final_mse, history: List[float])` où `history` a au moins `eval_budget` points (ou plus, on tronque à la fin).

### 6.1 `train_backprop(nn, X, y, eval_budget=25000, lr=0.01)`
Une époch = un forward pass complet = une eval. On exécute `eval_budget` epochs. Gradient calculé en respectant la dérivée nulle de Heaviside dans les couches cachées (`dZ_hidden = dA · 0`). Seule la dernière couche se met à jour. Historique = MSE par epoch.

### 6.2 `train_ray_shooting(nn, X, y, eval_budget=25000, steps_per_ray=50, bounds=(-2,2))`
- `max_rays = eval_budget // steps_per_ray` (par défaut 500)
- À chaque ray : tirer une cible aléatoire `z ∈ bounds^dim`, direction `z − best`, échantillonner `steps_per_ray` points le long du ray, garder le premier amélioré
- Historique : `best_score` après chaque step (un point par eval)

### 6.3 `train_gwo(nn, X, y, eval_budget=25000, n_agents=30, bounds=(-2,2))`
- `max_iter = eval_budget // n_agents` (par défaut ≈ 833)
- Init : population uniforme dans `bounds^dim`, sauf le premier loup = position courante du réseau (seeding minimal)
- Boucle GWO classique : `a` décroît linéairement de 2 à 0, mise à jour par alpha/beta/delta
- Historique : `alpha_score` après chaque évaluation d'agent

### 6.4 `train_hybrid_seeded(nn, X, y, eval_budget=25000, ray_ratio=0.5, sigma=0.2, bounds=(-2,2))`

**Le cœur de la thèse — c'est ce qui doit gagner**.

- **Phase 1 — Ray Shooting** : budget = `int(ray_ratio · eval_budget)` (par défaut 12 500), exécution standard de `train_ray_shooting`. Récupère `best_ray` à la fin.
- **Phase 2 — GWO seedé** : budget = restant (~12 500). **Différence cruciale avec GWO standard** : la population initiale n'est PAS uniforme aléatoire. Tous les `n_agents` agents sont initialisés à `best_ray + N(0, sigma · scale)` où `scale = max(|bounds|)`. Le premier agent reste à `best_ray` exact (pas de bruit) pour ne jamais régresser.
- Historique = concat des deux historiques.

**Hyperparamètres tunables si la thèse ne tient pas** :
- `ray_ratio ∈ {0.3, 0.5, 0.7}` (combien Ray vs GWO)
- `sigma ∈ {0.05, 0.1, 0.2, 0.5}` (amplitude du bruit autour de `best_ray`)

## 7. Pipeline d'expérience (`experiments/runner.py`)

```python
def run_benchmark(dim: int, output_dir: str,
                  func_names=None, eval_budget=25000,
                  n_samples=2000, seed=42):
    """
    Pour chaque fonction discontinue:
      1. Générer dataset (n_samples, dim) + split 80/20
      2. Normaliser y_train sur [0,1], appliquer la même norm sur y_val
      3. Pour chaque algo:
         a. nouveau HeavisideNetwork(input_dim=dim, hidden=[16,12,8])
         b. train avec budget=eval_budget
         c. record final_mse_val, history, time
      4. Plot convergence (log scale) -> output_dir/figures/{func}.png
      5. Ligne CSV : {dim, func, algo, mse_val, mse_train, time_s, n_evals_recorded}
    Écrit summary.csv dans output_dir.
    Retourne le DataFrame agrégé.
    """
```

`run_3d.py`, `run_10d.py`, `run_50d.py` ne font qu'appeler `run_benchmark(dim=3, ...)`, etc., et imprimer un résumé.

**Métriques enregistrées par expérience** :
- `mse_val` : MSE final sur le set de validation (la métrique-clé pour la thèse)
- `mse_train` : pour détecter overfitting (improbable ici, mais documenté)
- `time_s` : temps wall-clock (informationnel, pas un critère)
- `ratio_bp_vs_best` : `mse_val_backprop / min(mse_val_{ray, gwo, hybrid})`

**Seed** : `seed=42` partout pour la reproductibilité initiale. À documenter qu'une étude de variance (multi-seed) est out-of-scope mais souhaitable.

## 8. Visualisations

### 8.1 Par expérience (15 fichiers)
`{output_dir}/figures/{func}.png` : 4 courbes (Backprop rouge, Ray verte, GWO bleue, Hybrid violette) sur le même axe, ordonnée en log-scale MSE, abscisse = numéro d'évaluation.

### 8.2 Synthèses
- `results/global_summary.csv` : table longue `{dim, func, algo, mse_val, ratio_bp}`
- `results/global_comparison.png` : heatmap MSE moyenne par `(algo, dim)`, échelle log, palette divergente. Permet de voir d'un coup d'œil que backprop est uniformément mauvais et que hybrid gagne.

## 9. Conditions de succès

La thèse est validée si **pour les 15 (dim × func) combinaisons** :

| Condition | Vérification |
|---|---|
| Backprop échoue | `mse_val_backprop ≥ 5 × min(mse_val_{ray, gwo, hybrid})` |
| Ray et GWO réussissent | `mse_val_{ray, gwo} ≤ 0.2 · mse_val_backprop` |
| Hybrid est le meilleur | `mse_val_hybrid ≤ mse_val_{ray, gwo}` sur **au moins 12/15** combinaisons |

Si la dernière condition échoue, on entre en cycle de tuning de l'hybrid : ajuster `ray_ratio` et `sigma`, et si besoin essayer la variante adaptive switch (option (c) du brainstorm) comme plan B.

## 10. Plan d'exécution global

1. Refactor : créer `core/` et `experiments/`, déplacer l'existant utile dedans, déplacer le reste dans `archive/`
2. Implémenter `core/network.py` (réutiliser l'existant nettoyé)
3. Implémenter `core/functions.py` (généralisation n-D des 5 fonctions)
4. Implémenter `core/algorithms.py` (les 4 algos avec API uniforme `eval_budget`)
5. Implémenter `experiments/runner.py` + `run_3d.py`
6. Run 3D — valider que la thèse tient → ajuster si besoin
7. Run 10D — valider que la thèse tient encore
8. Run 50D — valider que la thèse tient encore (point le plus risqué)
9. Générer les synthèses globales (CSV + heatmap)
10. Mettre à jour `README.md` racine

## 11. Risques identifiés

| Risque | Impact | Mitigation |
|---|---|---|
| Hybrid seedé ne gagne pas en 50D | Thèse cassée | Tuner `sigma` et `ray_ratio`, fallback variante adaptive switch |
| Concentration de la mesure en 50D fait que la frontière disparaît | Cible triviale → tous les algos convergent vite vers la valeur moyenne | Le scaling √D atténue mais ne supprime pas le problème ; documenter l'effet et choisir des bornes plus serrées si besoin |
| 25 000 évals insuffisants en 50D (1122 params) | Tous les algos plateau loin de zéro | Augmenter `eval_budget` localement à 50 000 ou 100 000 pour la 50D si nécessaire — budget reste équitable entre algos |
| GWO `O(n_agents × dim)` par itération devient lent en 50D | Temps wall-clock élevé | Acceptable (le budget est en évaluations, pas en temps) ; documenter |

## 12. Hors-scope explicite

- Multi-seed (calcul d'intervalles de confiance) — souhaitable mais pas requis pour la première itération
- Comparaison avec d'autres méta-heuristiques (PSO, Genetic Algorithm) — éventuellement en addendum
- Optimisation des hyperparamètres par grid search — on utilise des valeurs raisonnables fixées
- Approximations lisses de Heaviside (sigmoid, hardtanh) — explicitement exclus, on veut LE vrai Heaviside pour montrer l'échec du backprop

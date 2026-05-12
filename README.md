# Non-Contiguous Deep Learning — Benchmark Multi-Dimensionnel

Ce projet de recherche évalue expérimentalement quatre algorithmes d'entraînement
sur des réseaux de neurones à activation **Heaviside** (gradient nul presque
partout dans les couches cachées), face à cinq fonctions cibles **discontinues**
en dimensions **3, 10, et 50**.

## Thèses étudiées

1. **La rétropropagation est fortement limitée** sur des réseaux Heaviside : la
   dérivée nulle dans les couches cachées les fige, et seule la couche de sortie
   apprend (le réseau dégénère en une **Extreme Learning Machine** — régression
   linéaire sur les features aléatoires figées des couches cachées).
2. **Ray Shooting** et **Grey Wolf Optimizer (GWO)** — méthodes sans gradient —
   entraînent ces réseaux avec succès car elles n'ont pas besoin du gradient.
3. **Hybrid Ray + GWO seedé** combine les deux : Ray Shooting explore globalement,
   puis GWO raffine localement à partir d'une population concentrée autour de la
   meilleure solution de Ray.

Le benchmark teste les 5 fonctions discontinues n-D avec un **budget équitable
de 25 000 évaluations** de la fonction de perte par algorithme.

## Résultats principaux

### Synthèse par dimension (MSE val moyen sur les 5 fonctions)

| Dimension | Backprop | Ray Shooting | GWO | **Hybrid** |
|-----------|----------|--------------|-----|------------|
| 3D        | 0.0620   | 0.0604       | 0.0385 | **0.0360** |
| 10D       | 0.0674   | 0.0634       | 0.0476 | **0.0462** |
| 50D       | 0.0787   | 0.0788       | 0.0594 | **0.0549** |

**Hybrid est l'algorithme avec le meilleur MSE moyen pour les trois dimensions.**

### Hybrid bat individuellement les autres sur 10/15 combinaisons (func × dim)

Sur les 5 sur lesquels hybrid ne gagne pas individuellement (`gs` en 3D, `gl/gs/geta` en 10D, `ggamma` en 50D), il reste systématiquement à <5% de l'algorithme gagnant — jamais loin derrière.

### Cas particulier : `ggamma` en 50D

Tous les algorithmes plafonnent autour de 0.13. C'est dû à la **concentration de la mesure** en haute dimension : dans `[-2, 2]^50`, presque tous les points uniformes ont une norme proche de la médiane attendue, donc la frontière sphérique calibrée par cette médiane sépare moins nettement le domaine. La cible devient quasi-aléatoire, irréductible.

## Structure du projet

```
core/                   # Cœur scientifique (réutilisable)
├── network.py          #   HeavisideNetwork (architecture [16, 12, 8])
├── functions.py        #   5 fonctions discontinues n-D (seuils calibrés)
└── algorithms.py       #   backprop / ray_shooting / gwo / hybrid_seeded

experiments/            # Pipeline d'expérience
├── runner.py           #   run_benchmark(dim, output_dir, eval_budget, ...)
├── run_3d.py           #   python -m experiments.run_3d
├── run_10d.py
├── run_50d.py
└── global_summary.py   #   Synthèse + heatmap + validation thèse

tests/                  # Tests autonomes (stdlib, pas de framework externe)
├── test_network.py     #   6 tests
├── test_functions.py   #   6 tests (dont calibrage P(inside) en 3D/10D/50D)
├── test_algorithms.py  #   8 tests (dont vérif hidden frozen pour backprop)
└── test_runner.py      #   1 smoke test du pipeline

results/                # CSV + figures de convergence
├── 3d/{figures/, summary.csv}
├── 10d/{figures/, summary.csv}
├── 50d/{figures/, summary.csv}
├── global_summary.csv  # 60 lignes : (dim × func × algo)
└── global_comparison.png

archive/                # Anciens scripts (préservés pour référence)
docs/superpowers/{specs,plans}/   # Spec design + plan d'implémentation
```

## Utilisation

### Tests

```bash
python3 tests/test_network.py       # 6 tests
python3 tests/test_functions.py     # 6 tests
python3 tests/test_algorithms.py    # 8 tests
python3 tests/test_runner.py        # 1 smoke test
```

### Benchmarks

```bash
python3 -m experiments.run_3d       # ~3 min
python3 -m experiments.run_10d      # ~3 min
python3 -m experiments.run_50d      # ~4 min

python3 -m experiments.global_summary   # synthèse + heatmap + validation
```

Tous les fichiers générés vont dans `results/`.

## Détails algorithmiques

### Architecture du réseau

`HeavisideNetwork(input_dim, hidden_sizes=[16, 12, 8], output_dim=1)` — fixe peu
importe la dimension d'entrée. C'est seulement la **complexité de la cible** qui
change entre 3D et 50D, pas la **capacité du modèle**. Nombre de paramètres :

- 3D : 381 params
- 10D : 493 params
- 50D : 1133 params

### Calibrage des fonctions discontinues

Pour chaque fonction et chaque dimension, les seuils géométriques sont calibrés
de façon à ce que la proportion de points "intérieurs" à la frontière reste dans
`[0.20, 0.80]` (vérifié dans `tests/test_functions.py::test_boundary_proportion_calibrated`).
Cela évite la dégénérescence en haute dimension où la frontière disparaîtrait
(concentration de la mesure).

### Hybrid Ray + GWO seedé

```
Phase 1 : Ray Shooting (12 500 évals)         -> best_ray
Phase 2 : GWO avec 30 agents init'd comme :
            wolves[0]    = best_ray         (anti-régression)
            wolves[1:30] = best_ray + N(0, σ·scale)   où σ = 0.2
          (12 500 évals)
```

La différence clé avec un hybrid naïf (Ray puis GWO standard) : on **ne perd pas**
la connaissance de Ray quand GWO démarre. La population GWO commence concentrée
autour de la meilleure solution trouvée par Ray.

## Reproductibilité

Toutes les expériences utilisent `seed=42`. Le calibrage des fonctions et les
algorithmes utilisent leurs propres `RandomState` pour ne pas être affectés par
l'ordre des appels. Pour reproduire les résultats exactement :

```bash
python3 -m experiments.run_3d
python3 -m experiments.run_10d
python3 -m experiments.run_50d
python3 -m experiments.global_summary
```

## Dépendances

Python 3.7+ · numpy · matplotlib · pandas. Pas de framework de test externe.

## Documents de conception

- `docs/superpowers/specs/2026-05-12-multidim-discontinuous-benchmark-design.md` — Spec design
- `docs/superpowers/plans/2026-05-12-multidim-discontinuous-benchmark.md` — Plan d'implémentation

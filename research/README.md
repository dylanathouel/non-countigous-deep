# Research — Reading Guide for the Thesis

Ce dossier rassemble tout ce qu'il faut pour rédiger l'article/la thèse à
partir de ce projet. Le **code source** reste dans `core/`, `experiments/`,
`tests/`. Ce dossier `research/` est une **couche d'écriture** : explications,
résultats, et pointeurs.

## Ordre de lecture conseillé

1. **`results/RESULTS_FR.md`** (ou `RESULTS_EN.md`) — synthèse des chiffres et
   conclusion par algorithme. Le **document central** à partir duquel structurer
   la section "Résultats" de la thèse.

2. **`algorithms.md`** — description littéraire des 4 algorithmes
   (Backprop, Ray Shooting, GWO, Hybrid). Pour la section "Méthodes".

3. **`functions.md`** — description des 5 fonctions cibles discontinues et du
   principe de calibrage. Pour la section "Protocole expérimental".

4. **`key_evidence.md`** — le test critique qui prouve que les couches cachées
   Heaviside sont gelées sous backprop. C'est la **preuve formelle de la thèse**
   sur laquelle s'appuyer dans la section "Validation".

5. **`code_pointers.md`** — index des fichiers source à citer si le jury veut
   inspecter le code.

## Structure

```
research/
├── README.md           # ce fichier
├── algorithms.md       # 4 algorithmes (Backprop / Ray / GWO / Hybrid)
├── functions.md        # 5 fonctions cibles + calibrage
├── key_evidence.md     # preuve du gel des couches cachées
├── code_pointers.md    # où trouver le code
└── results/
    ├── RESULTS_FR.md       # synthèse en français
    ├── RESULTS_EN.md       # synthèse en anglais
    ├── global_comparison.png  → heatmap MSE par (dim, algo)
    └── figures_per_dim/
        ├── 3d/    → courbes de convergence par fonction (3D)
        ├── 10d/   → 10D
        └── 50d/   → 50D
```

## Thèse en une phrase

> Sur des réseaux à activation discontinue (Heaviside), les méthodes
> sans gradient (GWO, Hybrid Ray+GWO seedé) sont **strictement supérieures**
> à la rétropropagation, et le combo Hybrid est l'optimum robuste sur les
> dimensions 3D, 10D, 50D.

## Citation rapide des chiffres clés

- MSE val moyen 3D / 10D / 50D — Hybrid : **0.036 / 0.046 / 0.055**.
- Backprop atteint le plafond ELM dès 3D (~0.062) et ne s'améliore jamais.
- Hybrid bat individuellement les autres sur **9/15** combinaisons (func × dim).
- Sur le cas le plus spectaculaire (`gl` 50D) : Hybrid 0.013 vs Backprop 0.050
  (≈ 3.7× mieux).

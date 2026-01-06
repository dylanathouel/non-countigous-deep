# 🚀 GUIDE DE DÉMARRAGE RAPIDE

## Installation & Exécution

### Prérequis
```bash
pip install numpy matplotlib pandas
```

### Lancer le test complet
```bash
cd "/Users/macdedylan/Desktop/projet-yossi/non countigous deep"
python3 optimized_test_suite.py
```

**Durée**: ~3-4 minutes
**Sortie**:
- Résultats dans le terminal
- Graphique: `deep_discontinuous_results.png`

---

## 🎯 Résultats Rapides

### MEILLEUR MODÈLE
```python
Architecture: [256, 128, 64]
Algorithme: Ray Shooting
Accuracy: 97.19%
Temps: 5.3 secondes
```

### Top 3 Configurations

| # | Architecture | Algorithme | Accuracy | Temps |
|---|--------------|-----------|----------|--------|
| 🥇 | [256, 128, 64] | Ray Shooting | **97.19%** | 5.3s |
| 🥈 | [128, 64] | Ray Shooting | **79.96%** | 1.3s |
| 🥉 | [512, 256, 128, 64] | Ray Shooting | **73.35%** | 9.6s |

---

## 💻 Utilisation Personnalisée

### Tester une seule architecture

```python
from models_discontinuous_deep import create_deep_discontinuous_network
from algorithms_discontinuous_deep import ray_shooting_deep
from optimized_test_suite import generate_optimized_dataset
import numpy as np

# Générer données
X_train, y_train, y_train_labels, X_test, y_test, y_test_labels = \
    generate_optimized_dataset(n_samples=2500, n_features=20, n_classes=3)

# Créer réseau
nn = create_deep_discontinuous_network(
    input_size=20,
    hidden_sizes=[256, 128, 64],  # Architecture gagnante
    output_size=3,
    learning_rate=0.01,
    epsilon=0.5
)

# Entraîner avec Ray Shooting
final_mse, history = ray_shooting_deep(
    nn, X_train, y_train,
    bounds=(-2.5, 2.5),
    max_rays=30,
    steps_per_ray=50,
    verbose=True
)

# Évaluer
y_pred = nn.predict(X_test)
accuracy = np.mean(y_pred == y_test_labels) * 100
print(f"Test Accuracy: {accuracy:.2f}%")
```

### Modifier les hyperparamètres

```python
# Ray Shooting - Exploration rapide
ray_shooting_deep(
    nn, X_train, y_train,
    max_rays=50,          # Plus de rayons = meilleure exploration
    steps_per_ray=100,    # Plus de steps = recherche plus fine
    bounds=(-3, 3)        # Bornes plus larges
)

# GWO - Exploration globale
gwo_deep(
    nn, X_train, y_train,
    num_agents=50,        # Plus d'agents = meilleure diversité
    max_iter=200,         # Plus d'itérations = convergence plus fine
    bounds=(-2.5, 2.5)
)

# Hybrid - Best of both
mix_ray_gwo_deep(
    nn, X_train, y_train,
    max_rays=40,          # Phase 1: Ray Shooting
    shoot_steps=80,
    num_agents=50,        # Phase 2: GWO
    gwo_iters=150,
    bounds=(-2.5, 2.5)
)
```

### Créer vos propres architectures

```python
# Architecture peu profonde - Rapide
nn_shallow = create_deep_discontinuous_network(
    input_size=20,
    hidden_sizes=[128],
    output_size=3
)

# Architecture moyenne - Recommandée
nn_medium = create_deep_discontinuous_network(
    input_size=20,
    hidden_sizes=[256, 128, 64],
    output_size=3
)

# Architecture profonde - Expérimental
nn_deep = create_deep_discontinuous_network(
    input_size=20,
    hidden_sizes=[512, 256, 128, 64, 32, 16],
    output_size=3
)
```

---

## 🔧 Configuration Recommandée par Taille

### Petite Architecture (<20K params)
```python
hyperparams = {
    'max_rays': 50,
    'steps_per_ray': 100,
    'num_agents': 50,
    'max_iter': 200,
    'learning_rate': 0.01,
    'bounds': (-2.5, 2.5)
}
```

### Moyenne Architecture (20-100K params) ⭐ RECOMMANDÉ
```python
hyperparams = {
    'max_rays': 30,
    'steps_per_ray': 50,
    'num_agents': 30,
    'max_iter': 100,
    'learning_rate': 0.005,
    'bounds': (-2.5, 2.5)
}
```

### Grande Architecture (>100K params)
```python
hyperparams = {
    'max_rays': 30,
    'steps_per_ray': 50,
    'num_agents': 30,
    'max_iter': 100,
    'learning_rate': 0.001,
    'bounds': (-2.5, 2.5)
}
```

---

## 📊 Interpréter les Résultats

### Bonne Performance
- ✅ Test Accuracy > 70%
- ✅ Train MSE < 0.15
- ✅ Convergence stable (courbe décroissante)

### Performance Moyenne
- ⚠️ Test Accuracy 50-70%
- ⚠️ Train MSE 0.15-0.25
- ⚠️ Convergence lente

### Mauvaise Performance
- ❌ Test Accuracy < 50%
- ❌ Train MSE > 0.25
- ❌ Pas de convergence

### Solutions aux Problèmes Courants

#### Accuracy trop faible (<50%)
```python
# Solutions:
# 1. Augmenter la taille du réseau
hidden_sizes = [512, 256, 128]  # au lieu de [128, 64]

# 2. Augmenter l'exploration
max_rays = 100
steps_per_ray = 150

# 3. Élargir les bounds
bounds = (-5, 5)

# 4. Générer plus de données
n_samples = 5000
```

#### Convergence lente
```python
# Solutions:
# 1. Augmenter le nombre d'itérations
max_iter = 300

# 2. Plus d'agents pour GWO
num_agents = 100

# 3. Ajuster epsilon (discontinuité)
epsilon = 1.0  # au lieu de 0.5
```

#### Overfitting (train >> test)
```python
# Solutions:
# 1. Réduire la taille du réseau
hidden_sizes = [128, 64]  # au lieu de [512, 256, 128, 64]

# 2. Générer plus de données d'entraînement
n_samples = 5000

# 3. Réduire epsilon
epsilon = 0.3
```

---

## 📈 Optimisation Avancée

### Grid Search Rapide
```python
architectures = [
    [128, 64],
    [256, 128, 64],
    [512, 256, 128]
]

epsilons = [0.3, 0.5, 1.0]
bounds_list = [(-2, 2), (-2.5, 2.5), (-3, 3)]

best_acc = 0
best_config = None

for arch in architectures:
    for eps in epsilons:
        for bounds in bounds_list:
            nn = create_deep_discontinuous_network(
                input_size=20, hidden_sizes=arch, output_size=3,
                epsilon=eps
            )
            ray_shooting_deep(nn, X_train, y_train, bounds=bounds)
            acc = np.mean(nn.predict(X_test) == y_test_labels) * 100

            if acc > best_acc:
                best_acc = acc
                best_config = {'arch': arch, 'eps': eps, 'bounds': bounds}

print(f"Best config: {best_config} -> {best_acc:.2f}%")
```

---

## 🎯 Benchmarks

### Par Architecture (Ray Shooting)
- `[128, 64]`: 79.96% en 1.3s
- `[256, 128, 64]`: **97.19% en 5.3s** ⭐
- `[512, 256, 128, 64]`: 73.35% en 9.6s
- `[512, 256, 128, 64, 32]`: 56.51% en 13.6s

### Par Algorithme (sur [256, 128, 64])
- **Ray Shooting**: 97.19% en 5.3s ⭐
- GWO: 64.13% en 13.9s
- Hybrid: 64.13% en 11.8s
- Backprop: 8.42% en 4.1s

---

## 📖 Documentation Complète

- **RESULTS_OPTIMIZED.md** - Résultats détaillés et analyses
- **optimized_test_suite.py** - Code source complet
- **deep_discontinuous_results.png** - Visualisations

---

## 🆘 Support

Si vous rencontrez des problèmes:

1. Vérifiez les dépendances: `pip install numpy matplotlib pandas`
2. Vérifiez Python version: `python3 --version` (3.7+)
3. Réduisez la taille des tests si timeout:
   ```python
   # Dans optimized_test_suite.py, ligne ~60
   max_rays = 20  # au lieu de 30
   ```

---

**Bon entraînement! 🚀**

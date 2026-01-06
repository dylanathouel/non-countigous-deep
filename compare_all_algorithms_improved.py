"""
COMPARAISON COMPLÈTE DES 4 ALGORITHMES AVEC AMÉLIORATIONS
==========================================================

Teste tous les algorithmes avec:
- Dataset amélioré (5000 échantillons)
- Plus d'itérations
- Architecture championne [256, 128, 64]

Durée: ~5-7 minutes
"""

import numpy as np
import time
from models_discontinuous_deep import create_deep_discontinuous_network
from algorithms_discontinuous_deep import ray_shooting_deep, gwo_deep, mix_ray_gwo_deep, backprop_train_deep

def generate_enhanced_dataset_quick(n_samples=5000, n_features=3, n_classes=3, random_state=42):
    """Dataset amélioré simplifié (3D)"""
    np.random.seed(random_state)
    X_list, y_list = [], []
    samples_per_class = n_samples // n_classes

    for cls in range(n_classes):
        n_clusters = 3
        samples_per_cluster = samples_per_class // n_clusters

        for cluster in range(n_clusters):
            center = np.random.randn(n_features) * 2 + cls * 8 + cluster * 2
            base_samples = np.random.randn(samples_per_cluster, n_features) * 0.5

            # Non-linéarités
            nonlinear1 = np.sin(base_samples[:, :n_features//3] * np.pi / 2)
            base_samples[:, :n_features//3] += nonlinear1 * 0.3

            cluster_samples = base_samples + center
            X_list.append(cluster_samples)
            y_list.append(np.full(samples_per_cluster, cls))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    X = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0) + 1e-8)

    y_onehot = np.zeros((len(y), n_classes))
    y_onehot[np.arange(len(y)), y.astype(int)] = 1

    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y_onehot[:split_idx], y_onehot[split_idx:]
    y_train_labels, y_test_labels = y[:split_idx], y[split_idx:]

    return X_train, y_train, y_train_labels, X_test, y_test, y_test_labels


print("="*80)
print("COMPARAISON COMPLÈTE: 4 ALGORITHMES AVEC AMÉLIORATIONS")
print("="*80)
print("Architecture: [256, 128, 64, 32]")
print("Dataset: 5,000 échantillons (3D)")
print("Itérations: Augmentées pour chaque algorithme")
print("="*80)

# Générer dataset amélioré
print("\n📊 Génération dataset amélioré...")
X_train, y_train, y_train_labels, X_test, y_test, y_test_labels = \
    generate_enhanced_dataset_quick(n_samples=5000, n_features=3, n_classes=3)

print(f"Training: {X_train.shape}, Test: {X_test.shape}")

# Configuration des tests
tests = [
    {
        'name': 'Ray Shooting (BASELINE)',
        'algo': 'ray',
        'params': {'max_rays': 30, 'steps_per_ray': 50, 'bounds': (-2.5, 2.5)}
    },
    {
        'name': 'Ray Shooting (5× itérations)',
        'algo': 'ray',
        'params': {'max_rays': 100, 'steps_per_ray': 150, 'bounds': (-2.5, 2.5)}
    },
    {
        'name': 'GWO (BASELINE)',
        'algo': 'gwo',
        'params': {'num_agents': 30, 'max_iter': 100, 'bounds': (-2.5, 2.5)}
    },
    {
        'name': 'GWO (5× itérations)',
        'algo': 'gwo',
        'params': {'num_agents': 80, 'max_iter': 400, 'bounds': (-2.5, 2.5)}
    },
    {
        'name': 'Hybrid Ray+GWO (BASELINE)',
        'algo': 'hybrid',
        'params': {'max_rays': 20, 'shoot_steps': 30, 'num_agents': 25, 'gwo_iters': 80, 'bounds': (-2.5, 2.5)}
    },
    {
        'name': 'Hybrid Ray+GWO (5× itérations)',
        'algo': 'hybrid',
        'params': {'max_rays': 60, 'shoot_steps': 80, 'num_agents': 60, 'gwo_iters': 250, 'bounds': (-2.5, 2.5)}
    },
    {
        'name': 'Backpropagation (BASELINE)',
        'algo': 'backprop',
        'params': {'epochs': 500, 'learning_rate': 0.005}
    },
    {
        'name': 'Backpropagation (5× itérations)',
        'algo': 'backprop',
        'params': {'epochs': 2500, 'learning_rate': 0.005}
    },
]

results = []

print(f"\n{'='*80}")
print("TESTS EN COURS")
print(f"{'='*80}\n")

for test_num, test in enumerate(tests, 1):
    print(f"Test {test_num}/{len(tests)}: {test['name']}")
    print("-" * 80)

    # Créer réseau frais
    learning_rate = test['params'].get('learning_rate', 0.005)
    nn = create_deep_discontinuous_network(
        input_size=3,
        hidden_sizes=[256, 128, 64, 32],
        output_size=3,
        learning_rate=learning_rate,
        epsilon=0.5
    )

    # Entraîner selon l'algorithme
    start = time.time()

    if test['algo'] == 'ray':
        final_mse, history = ray_shooting_deep(
            nn, X_train, y_train,
            bounds=test['params']['bounds'],
            max_rays=test['params']['max_rays'],
            steps_per_ray=test['params']['steps_per_ray'],
            verbose=False
        )

    elif test['algo'] == 'gwo':
        final_mse, history = gwo_deep(
            nn, X_train, y_train,
            bounds=test['params']['bounds'],
            num_agents=test['params']['num_agents'],
            max_iter=test['params']['max_iter'],
            verbose=False
        )

    elif test['algo'] == 'hybrid':
        final_mse, history = mix_ray_gwo_deep(
            nn, X_train, y_train,
            bounds=test['params']['bounds'],
            max_rays=test['params']['max_rays'],
            shoot_steps=test['params']['shoot_steps'],
            num_agents=test['params']['num_agents'],
            gwo_iters=test['params']['gwo_iters'],
            verbose=False
        )

    elif test['algo'] == 'backprop':
        final_mse, history = backprop_train_deep(
            nn, X_train, y_train,
            epochs=test['params']['epochs'],
            desired_mse=None,
            verbose=False
        )

    elapsed = time.time() - start

    # Évaluer
    y_pred = nn.predict(X_test)
    accuracy = np.mean(y_pred == y_test_labels) * 100

    # Afficher
    print(f"  ✓ Accuracy: {accuracy:.2f}%")
    print(f"  ✓ Temps: {elapsed:.1f}s")
    print(f"  ✓ MSE: {final_mse:.4f}")
    print()

    results.append({
        'name': test['name'],
        'algo': test['algo'],
        'accuracy': accuracy,
        'time': elapsed,
        'mse': final_mse
    })

# Résumé par algorithme
print(f"{'='*80}")
print("📊 RÉSUMÉ PAR ALGORITHME")
print(f"{'='*80}\n")

# Regrouper par algorithme
algos = ['ray', 'gwo', 'hybrid', 'backprop']
algo_names = {
    'ray': 'Ray Shooting',
    'gwo': 'GWO',
    'hybrid': 'Hybrid Ray+GWO',
    'backprop': 'Backpropagation'
}

for algo in algos:
    algo_results = [r for r in results if r['algo'] == algo]

    if len(algo_results) >= 2:
        baseline = algo_results[0]
        improved = algo_results[1]

        gain = improved['accuracy'] - baseline['accuracy']
        time_factor = improved['time'] / baseline['time']

        print(f"📌 {algo_names[algo]}")
        print(f"   Baseline:  {baseline['accuracy']:6.2f}%  ({baseline['time']:5.1f}s)")
        print(f"   Amélioré:  {improved['accuracy']:6.2f}%  ({improved['time']:5.1f}s)")
        print(f"   Gain:      {gain:+6.2f}%  (temps: {time_factor:.1f}×)")
        print()

# Tableau comparatif final
print(f"{'='*80}")
print("📊 TABLEAU COMPARATIF COMPLET")
print(f"{'='*80}\n")

print(f"{'Configuration':<40} {'Accuracy':>10} {'Temps':>10} {'MSE':>10}")
print("-" * 80)

for r in results:
    print(f"{r['name']:<40} {r['accuracy']:>9.2f}% {r['time']:>9.1f}s {r['mse']:>10.4f}")

# Meilleur résultat global
print(f"\n{'='*80}")
best = max(results, key=lambda x: x['accuracy'])
print(f"🏆 MEILLEUR RÉSULTAT GLOBAL")
print(f"{'='*80}")
print(f"Configuration: {best['name']}")
print(f"Accuracy:      {best['accuracy']:.2f}%")
print(f"Temps:         {best['time']:.1f}s")
print(f"MSE:           {best['mse']:.4f}")
print(f"{'='*80}")

# Comparaison avec ancien record
if best['accuracy'] > 97.19:
    print(f"\n🎉 NOUVEAU RECORD!")
    print(f"   Ancien record: 97.19%")
    print(f"   Nouveau:       {best['accuracy']:.2f}%")
    print(f"   Amélioration:  +{best['accuracy'] - 97.19:.2f}%")
else:
    print(f"\n   Record actuel reste: 97.19%")

# Meilleur par algorithme
print(f"\n{'='*80}")
print("🥇 MEILLEUR PAR ALGORITHME (avec 5× itérations)")
print(f"{'='*80}")

for algo in algos:
    algo_results = [r for r in results if r['algo'] == algo and '5×' in r['name']]
    if algo_results:
        best_algo = max(algo_results, key=lambda x: x['accuracy'])
        print(f"{algo_names[algo]:<20} → {best_algo['accuracy']:6.2f}%  ({best_algo['time']:5.1f}s)")

print(f"{'='*80}\n")

print("✅ Test complet terminé!")

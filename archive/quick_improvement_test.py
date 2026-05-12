"""
TEST RAPIDE D'AMÉLIORATION
Compare baseline vs améliorations sur l'architecture championne
Durée: ~2-3 minutes
"""

import numpy as np
import time
from models_discontinuous_deep import create_deep_discontinuous_network
from algorithms_discontinuous_deep import ray_shooting_deep

def generate_enhanced_dataset_quick(n_samples=5000, n_features=20, n_classes=3, random_state=42):
    """Dataset amélioré simplifié"""
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
print("TEST RAPIDE D'AMÉLIORATION - Architecture Championne [256, 128, 64]")
print("="*80)

# Générer dataset amélioré
print("\n📊 Génération dataset amélioré (5000 échantillons)...")
X_train, y_train, y_train_labels, X_test, y_test, y_test_labels = \
    generate_enhanced_dataset_quick(n_samples=5000, n_features=20, n_classes=3)

print(f"Training: {X_train.shape}, Test: {X_test.shape}")

# Tests comparatifs
results = []

configs = [
    {"name": "BASELINE (actuel)", "rays": 30, "steps": 50},
    {"name": "2× itérations", "rays": 50, "steps": 100},
    {"name": "5× itérations", "rays": 100, "steps": 150},
]

print(f"\n{'='*80}")
print("TESTS COMPARATIFS")
print(f"{'='*80}\n")

for config in configs:
    print(f"{config['name']:25} (rays={config['rays']:3}, steps={config['steps']:3})")
    print("-" * 80)

    # Créer réseau frais
    nn = create_deep_discontinuous_network(
        input_size=20,
        hidden_sizes=[256, 128, 64],
        output_size=3,
        learning_rate=0.005,
        epsilon=0.5
    )

    # Entraîner
    start = time.time()
    final_mse, history = ray_shooting_deep(
        nn, X_train, y_train,
        bounds=(-2.5, 2.5),
        max_rays=config['rays'],
        steps_per_ray=config['steps'],
        verbose=False
    )
    elapsed = time.time() - start

    # Évaluer
    y_pred = nn.predict(X_test)
    accuracy = np.mean(y_pred == y_test_labels) * 100

    # Afficher
    gain = accuracy - 97.19 if config['name'] != "BASELINE (actuel)" else 0
    gain_str = f" (+{gain:.2f}%)" if gain > 0 else ""

    print(f"  ✓ Accuracy: {accuracy:.2f}%{gain_str}")
    print(f"  ✓ Temps: {elapsed:.1f}s")
    print(f"  ✓ MSE: {final_mse:.4f}")
    print()

    results.append({
        'config': config['name'],
        'accuracy': accuracy,
        'time': elapsed,
        'mse': final_mse
    })

# Résumé
print(f"{'='*80}")
print("📊 RÉSUMÉ DES GAINS")
print(f"{'='*80}\n")

baseline_acc = results[0]['accuracy']

for i, r in enumerate(results):
    gain = r['accuracy'] - baseline_acc
    gain_str = f"+{gain:.2f}%" if gain > 0 else "baseline"
    factor = r['time'] / results[0]['time']

    print(f"{r['config']:25} → {r['accuracy']:6.2f}%  (gain: {gain_str:8})  [temps: {r['time']:.1f}s = {factor:.1f}×]")

print(f"\n{'='*80}")
best_result = max(results, key=lambda x: x['accuracy'])
print(f"🏆 MEILLEUR RÉSULTAT: {best_result['config']}")
print(f"   Accuracy: {best_result['accuracy']:.2f}%")
print(f"   Gain vs baseline: +{best_result['accuracy'] - baseline_acc:.2f}%")
print(f"   Temps: {best_result['time']:.1f}s")
print(f"{'='*80}")

if best_result['accuracy'] > 97.19:
    print(f"\n🎉 NOUVEAU RECORD! (ancien: 97.19%)")
    print(f"   Amélioration: +{best_result['accuracy'] - 97.19:.2f}%")

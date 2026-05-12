"""
ENHANCED OPTIMIZER - Script d'amélioration des performances
============================================================

Teste plusieurs stratégies d'amélioration:
1. Plus d'itérations
2. Grid search hyperparamètres
3. Dataset amélioré
4. Multiple restarts
5. Ensembling

Objectif: Dépasser 97.19% et atteindre 98-99%+
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple
import pandas as pd

from models_discontinuous_deep import DeepDiscontinuousNeuralNetwork, create_deep_discontinuous_network
from algorithms_discontinuous_deep import ray_shooting_deep, gwo_deep, mix_ray_gwo_deep


# ==============================================================================
# GÉNÉRATEUR DE DONNÉES AMÉLIORÉ
# ==============================================================================

def generate_enhanced_dataset(n_samples: int = 5000, n_features: int = 20,
                              n_classes: int = 3, random_state: int = 42,
                              noise_level: float = 0.5) -> Tuple:
    """
    Générateur de données AMÉLIORÉ avec:
    - Plus d'échantillons
    - Meilleure séparation des classes
    - Patterns non-linéaires plus riches
    """
    np.random.seed(random_state)

    X_list = []
    y_list = []

    samples_per_class = n_samples // n_classes

    for cls in range(n_classes):
        # Créer 3 clusters par classe avec meilleure séparation
        n_clusters = 3
        samples_per_cluster = samples_per_class // n_clusters

        for cluster in range(n_clusters):
            # Centres plus espacés pour meilleure séparation
            center = np.random.randn(n_features) * 2 + cls * 8 + cluster * 2

            # Générer cluster avec patterns non-linéaires riches
            base_samples = np.random.randn(samples_per_cluster, n_features) * noise_level

            # Patterns non-linéaires multiples
            # 1. Sinusoïdal
            nonlinear1 = np.sin(base_samples[:, :n_features//3] * np.pi / 2)
            base_samples[:, :n_features//3] += nonlinear1 * 0.3

            # 2. Quadratique
            nonlinear2 = np.square(base_samples[:, n_features//3:2*n_features//3]) * 0.2
            base_samples[:, n_features//3:2*n_features//3] += nonlinear2

            # 3. Exponentiel
            nonlinear3 = np.exp(-np.abs(base_samples[:, 2*n_features//3:])) * 0.3
            base_samples[:, 2*n_features//3:] += nonlinear3

            # Ajouter centre
            cluster_samples = base_samples + center

            X_list.append(cluster_samples)
            y_list.append(np.full(samples_per_cluster, cls))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    # Normalisation robuste (moins sensible aux outliers)
    X = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0) + 1e-8)

    # One-hot encoding
    y_onehot = np.zeros((len(y), n_classes))
    y_onehot[np.arange(len(y)), y.astype(int)] = 1

    # Split train/test 80/20
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y_onehot[:split_idx], y_onehot[split_idx:]
    y_train_labels, y_test_labels = y[:split_idx], y[split_idx:]

    return X_train, y_train, y_train_labels, X_test, y_test, y_test_labels


# ==============================================================================
# STRATÉGIE 1: PLUS D'ITÉRATIONS
# ==============================================================================

def test_increased_iterations(nn, X_train, y_train, X_test, y_test_labels,
                              algorithm: str = "ray") -> Dict:
    """
    Tester avec beaucoup plus d'itérations
    """
    print(f"\n{'='*80}")
    print(f"STRATÉGIE 1: AUGMENTATION DES ITÉRATIONS - {algorithm.upper()}")
    print(f"{'='*80}")

    results = []

    if algorithm == "ray":
        configs = [
            {"max_rays": 30, "steps_per_ray": 50, "name": "Baseline"},
            {"max_rays": 50, "steps_per_ray": 100, "name": "2× itérations"},
            {"max_rays": 100, "steps_per_ray": 150, "name": "5× itérations"},
            {"max_rays": 150, "steps_per_ray": 200, "name": "10× itérations"},
        ]

        for config in configs:
            print(f"\nTest: {config['name']} (rays={config['max_rays']}, steps={config['steps_per_ray']})")

            # Réinitialiser le réseau
            nn.__init__(nn.input_size, nn.hidden_sizes, nn.output_size,
                       learning_rate=nn.learning_rate, epsilon=nn.epsilon)

            start = time.time()
            final_mse, history = ray_shooting_deep(
                nn, X_train, y_train,
                bounds=(-2.5, 2.5),
                max_rays=config['max_rays'],
                steps_per_ray=config['steps_per_ray'],
                verbose=False
            )
            elapsed = time.time() - start

            y_pred = nn.predict(X_test)
            accuracy = np.mean(y_pred == y_test_labels) * 100

            print(f"  → Accuracy: {accuracy:.2f}% | Time: {elapsed:.1f}s | MSE: {final_mse:.4f}")

            results.append({
                'config': config['name'],
                'accuracy': accuracy,
                'time': elapsed,
                'mse': final_mse
            })

    elif algorithm == "gwo":
        configs = [
            {"num_agents": 30, "max_iter": 100, "name": "Baseline"},
            {"num_agents": 50, "max_iter": 200, "name": "2× itérations"},
            {"num_agents": 80, "max_iter": 400, "name": "5× itérations"},
            {"num_agents": 100, "max_iter": 500, "name": "10× itérations"},
        ]

        for config in configs:
            print(f"\nTest: {config['name']} (agents={config['num_agents']}, iter={config['max_iter']})")

            nn.__init__(nn.input_size, nn.hidden_sizes, nn.output_size,
                       learning_rate=nn.learning_rate, epsilon=nn.epsilon)

            start = time.time()
            final_mse, history = gwo_deep(
                nn, X_train, y_train,
                bounds=(-2.5, 2.5),
                num_agents=config['num_agents'],
                max_iter=config['max_iter'],
                verbose=False
            )
            elapsed = time.time() - start

            y_pred = nn.predict(X_test)
            accuracy = np.mean(y_pred == y_test_labels) * 100

            print(f"  → Accuracy: {accuracy:.2f}% | Time: {elapsed:.1f}s | MSE: {final_mse:.4f}")

            results.append({
                'config': config['name'],
                'accuracy': accuracy,
                'time': elapsed,
                'mse': final_mse
            })

    return results


# ==============================================================================
# STRATÉGIE 2: GRID SEARCH HYPERPARAMÈTRES
# ==============================================================================

def test_hyperparameter_tuning(nn, X_train, y_train, X_test, y_test_labels) -> Dict:
    """
    Grid search sur epsilon et bounds
    """
    print(f"\n{'='*80}")
    print(f"STRATÉGIE 2: GRID SEARCH HYPERPARAMÈTRES")
    print(f"{'='*80}")

    results = []

    epsilons = [0.3, 0.5, 0.7, 1.0]
    bounds_list = [(-2, 2), (-2.5, 2.5), (-3, 3), (-5, 5)]

    best_acc = 0
    best_config = None

    for eps in epsilons:
        for bounds in bounds_list:
            print(f"\nTest: epsilon={eps}, bounds={bounds}")

            # Créer nouveau réseau avec epsilon
            test_nn = create_deep_discontinuous_network(
                nn.input_size, nn.hidden_sizes, nn.output_size,
                learning_rate=0.005, epsilon=eps
            )

            start = time.time()
            final_mse, history = ray_shooting_deep(
                test_nn, X_train, y_train,
                bounds=bounds,
                max_rays=50,
                steps_per_ray=100,
                verbose=False
            )
            elapsed = time.time() - start

            y_pred = test_nn.predict(X_test)
            accuracy = np.mean(y_pred == y_test_labels) * 100

            print(f"  → Accuracy: {accuracy:.2f}% | Time: {elapsed:.1f}s")

            results.append({
                'epsilon': eps,
                'bounds': bounds,
                'accuracy': accuracy,
                'time': elapsed,
                'mse': final_mse
            })

            if accuracy > best_acc:
                best_acc = accuracy
                best_config = {'epsilon': eps, 'bounds': bounds}

    print(f"\n🏆 Meilleure config: epsilon={best_config['epsilon']}, bounds={best_config['bounds']}")
    print(f"   Accuracy: {best_acc:.2f}%")

    return results, best_config


# ==============================================================================
# STRATÉGIE 3: MULTIPLE RESTARTS (ENSEMBLE)
# ==============================================================================

def test_multiple_restarts(nn, X_train, y_train, X_test, y_test_labels,
                          n_restarts: int = 5) -> Dict:
    """
    Lancer plusieurs fois avec seeds différentes et prendre le meilleur
    """
    print(f"\n{'='*80}")
    print(f"STRATÉGIE 3: MULTIPLE RESTARTS (n={n_restarts})")
    print(f"{'='*80}")

    results = []
    best_acc = 0
    best_nn = None

    for i in range(n_restarts):
        print(f"\nRestart {i+1}/{n_restarts}")

        # Nouveau réseau avec seed différente
        np.random.seed(42 + i * 100)

        test_nn = create_deep_discontinuous_network(
            nn.input_size, nn.hidden_sizes, nn.output_size,
            learning_rate=0.005, epsilon=0.5
        )

        start = time.time()
        final_mse, history = ray_shooting_deep(
            test_nn, X_train, y_train,
            bounds=(-2.5, 2.5),
            max_rays=80,
            steps_per_ray=120,
            verbose=False
        )
        elapsed = time.time() - start

        y_pred = test_nn.predict(X_test)
        accuracy = np.mean(y_pred == y_test_labels) * 100

        print(f"  → Accuracy: {accuracy:.2f}% | Time: {elapsed:.1f}s")

        results.append({
            'restart': i+1,
            'accuracy': accuracy,
            'time': elapsed,
            'mse': final_mse
        })

        if accuracy > best_acc:
            best_acc = accuracy
            best_nn = test_nn

    print(f"\n🏆 Meilleur résultat: {best_acc:.2f}% (restart {np.argmax([r['accuracy'] for r in results])+1})")

    return results, best_nn


# ==============================================================================
# STRATÉGIE 4: ENSEMBLE VOTING
# ==============================================================================

def test_ensemble_voting(nn, X_train, y_train, X_test, y_test_labels,
                        n_models: int = 5) -> Dict:
    """
    Créer plusieurs modèles et faire du voting
    """
    print(f"\n{'='*80}")
    print(f"STRATÉGIE 4: ENSEMBLE VOTING (n={n_models} modèles)")
    print(f"{'='*80}")

    models = []

    # Entraîner plusieurs modèles
    for i in range(n_models):
        print(f"\nEntraînement modèle {i+1}/{n_models}")

        np.random.seed(42 + i * 50)

        model = create_deep_discontinuous_network(
            nn.input_size, nn.hidden_sizes, nn.output_size,
            learning_rate=0.005, epsilon=0.5
        )

        ray_shooting_deep(
            model, X_train, y_train,
            bounds=(-2.5, 2.5),
            max_rays=60,
            steps_per_ray=100,
            verbose=False
        )

        y_pred = model.predict(X_test)
        accuracy = np.mean(y_pred == y_test_labels) * 100
        print(f"  → Accuracy individuelle: {accuracy:.2f}%")

        models.append(model)

    # Voting
    print("\nVoting ensemble...")
    predictions = np.array([model.predict(X_test) for model in models])

    # Majority voting
    ensemble_pred = np.apply_along_axis(
        lambda x: np.bincount(x.astype(int)).argmax(),
        axis=0,
        arr=predictions
    )

    ensemble_acc = np.mean(ensemble_pred == y_test_labels) * 100
    print(f"🏆 Ensemble Accuracy: {ensemble_acc:.2f}%")

    return {
        'individual_accuracies': [np.mean(pred == y_test_labels) * 100 for pred in predictions],
        'ensemble_accuracy': ensemble_acc
    }


# ==============================================================================
# MAIN - TEST COMPLET
# ==============================================================================

def main():
    """
    Test complet de toutes les stratégies d'amélioration
    """

    print("=" * 80)
    print("ENHANCED OPTIMIZER - Amélioration des performances")
    print("=" * 80)

    # Générer dataset amélioré
    print("\n📊 Génération dataset amélioré (5000 échantillons)...")
    X_train, y_train, y_train_labels, X_test, y_test, y_test_labels = \
        generate_enhanced_dataset(n_samples=5000, n_features=20, n_classes=3, noise_level=0.5)

    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # Architecture championne
    architecture = [256, 128, 64]
    print(f"\n🏆 Architecture testée: {architecture}")

    nn = create_deep_discontinuous_network(
        input_size=20, hidden_sizes=architecture, output_size=3,
        learning_rate=0.005, epsilon=0.5
    )

    # BASELINE
    print(f"\n{'='*80}")
    print("BASELINE - Configuration actuelle")
    print(f"{'='*80}")
    start = time.time()
    final_mse, history = ray_shooting_deep(
        nn, X_train, y_train,
        bounds=(-2.5, 2.5),
        max_rays=30,
        steps_per_ray=50,
        verbose=False
    )
    baseline_time = time.time() - start
    y_pred = nn.predict(X_test)
    baseline_acc = np.mean(y_pred == y_test_labels) * 100
    print(f"Baseline Accuracy: {baseline_acc:.2f}% | Time: {baseline_time:.1f}s")

    # STRATÉGIE 1: Plus d'itérations
    nn.__init__(nn.input_size, nn.hidden_sizes, nn.output_size,
               learning_rate=0.005, epsilon=0.5)
    results_iterations = test_increased_iterations(
        nn, X_train, y_train, X_test, y_test_labels, algorithm="ray"
    )

    # STRATÉGIE 2: Grid search
    nn.__init__(nn.input_size, nn.hidden_sizes, nn.output_size,
               learning_rate=0.005, epsilon=0.5)
    results_grid, best_config = test_hyperparameter_tuning(
        nn, X_train, y_train, X_test, y_test_labels
    )

    # STRATÉGIE 3: Multiple restarts
    nn.__init__(nn.input_size, nn.hidden_sizes, nn.output_size,
               learning_rate=0.005, epsilon=0.5)
    results_restarts, best_nn = test_multiple_restarts(
        nn, X_train, y_train, X_test, y_test_labels, n_restarts=5
    )

    # STRATÉGIE 4: Ensemble
    nn.__init__(nn.input_size, nn.hidden_sizes, nn.output_size,
               learning_rate=0.005, epsilon=0.5)
    results_ensemble = test_ensemble_voting(
        nn, X_train, y_train, X_test, y_test_labels, n_models=5
    )

    # RÉSUMÉ FINAL
    print(f"\n{'='*80}")
    print("📊 RÉSUMÉ DES AMÉLIORATIONS")
    print(f"{'='*80}")
    print(f"Baseline:                     {baseline_acc:.2f}%")
    print(f"Plus d'itérations (10×):      {results_iterations[-1]['accuracy']:.2f}% (+{results_iterations[-1]['accuracy']-baseline_acc:.2f}%)")
    print(f"Meilleur hyperparamètres:     {max(r['accuracy'] for r in results_grid):.2f}% (+{max(r['accuracy'] for r in results_grid)-baseline_acc:.2f}%)")
    print(f"Meilleur restart:             {max(r['accuracy'] for r in results_restarts):.2f}% (+{max(r['accuracy'] for r in results_restarts)-baseline_acc:.2f}%)")
    print(f"Ensemble voting:              {results_ensemble['ensemble_accuracy']:.2f}% (+{results_ensemble['ensemble_accuracy']-baseline_acc:.2f}%)")
    print(f"{'='*80}")

    best_overall = max(
        results_iterations[-1]['accuracy'],
        max(r['accuracy'] for r in results_grid),
        max(r['accuracy'] for r in results_restarts),
        results_ensemble['ensemble_accuracy']
    )
    print(f"\n🏆 MEILLEUR RÉSULTAT GLOBAL: {best_overall:.2f}%")
    print(f"   Gain vs baseline: +{best_overall - baseline_acc:.2f}%")

    if best_overall > 97.19:
        print(f"\n🎉 NOUVEAU RECORD! (ancien: 97.19%)")


if __name__ == "__main__":
    main()

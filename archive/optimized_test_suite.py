"""
OPTIMIZED TEST SUITE FOR DEEP DISCONTINUOUS NEURAL NETWORKS
==============================================================

Tests 4 architectures with 4 algorithms using optimized hyperparameters
Target: >70% accuracy in <10 minutes per architecture

Architectures:
- [128, 64] (~11K params)
- [256, 128, 64] (~46K params)
- [512, 256, 128, 64] (~185K params)
- [512, 256, 128, 64, 32] (~190K params)

Algorithms:
- Ray Shooting
- Grey Wolf Optimizer (GWO)
- Hybrid Ray+GWO
- Backpropagation
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple
import pandas as pd

from models_discontinuous_deep import DeepDiscontinuousNeuralNetwork, create_deep_discontinuous_network
from algorithms_discontinuous_deep import ray_shooting_deep, gwo_deep, mix_ray_gwo_deep, backprop_train_deep


# ==============================================================================
# OPTIMIZED DATA GENERATOR
# ==============================================================================

def generate_optimized_dataset(n_samples: int = 2500, n_features: int = 20,
                              n_classes: int = 3, random_state: int = 42) -> Tuple:
    """
    Generate challenging non-linear dataset with clear class boundaries

    Strategy:
    - Non-linear patterns with multiple clusters per class
    - Controlled overlap for realistic difficulty
    - Scaled features for better optimization
    """
    np.random.seed(random_state)

    X_list = []
    y_list = []

    samples_per_class = n_samples // n_classes

    for cls in range(n_classes):
        # Create 2-3 clusters per class for complexity
        n_clusters = np.random.randint(2, 4)
        samples_per_cluster = samples_per_class // n_clusters

        for cluster in range(n_clusters):
            # Non-linear transformation
            center = np.random.randn(n_features) * 3 + cls * 5

            # Generate cluster with non-linear patterns
            base_samples = np.random.randn(samples_per_cluster, n_features) * 1.5

            # Add non-linearity: quadratic and sinusoidal patterns
            nonlinear = np.sin(base_samples[:, :n_features//2] * np.pi / 2)
            base_samples[:, :n_features//2] += nonlinear * 0.5

            # Add center offset
            cluster_samples = base_samples + center

            X_list.append(cluster_samples)
            y_list.append(np.full(samples_per_cluster, cls))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    # Normalize features
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # Convert labels to one-hot
    y_onehot = np.zeros((len(y), n_classes))
    y_onehot[np.arange(len(y)), y.astype(int)] = 1

    # Split train/test
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y_onehot[:split_idx], y_onehot[split_idx:]
    y_train_labels, y_test_labels = y[:split_idx], y[split_idx:]

    return X_train, y_train, y_train_labels, X_test, y_test, y_test_labels


# ==============================================================================
# OPTIMIZED HYPERPARAMETERS
# ==============================================================================

def get_optimized_hyperparameters(architecture: List[int], total_params: int) -> Dict:
    """
    Return optimized hyperparameters based on network size

    Strategy:
    - Smaller networks: More exploration (more rays/agents/epochs)
    - Larger networks: Faster convergence with fewer but smarter iterations
    - Adaptive learning rates and bounds
    """

    # Base hyperparameters that scale inversely with network size
    complexity_factor = np.log10(total_params + 1) / 5.0  # Normalized complexity

    hyperparams = {
        # Ray Shooting parameters
        'ray': {
            'max_rays': max(30, int(100 * (1 - complexity_factor))),
            'steps_per_ray': max(50, int(120 * (1 - complexity_factor))),
            'bounds': (-2.5, 2.5)
        },

        # GWO parameters
        'gwo': {
            'num_agents': max(30, int(60 * (1 - complexity_factor))),
            'max_iter': max(100, int(250 * (1 - complexity_factor))),
            'bounds': (-2.5, 2.5)
        },

        # Hybrid parameters (balanced)
        'hybrid': {
            'max_rays': max(20, int(40 * (1 - complexity_factor))),
            'shoot_steps': max(30, int(60 * (1 - complexity_factor))),
            'num_agents': max(25, int(50 * (1 - complexity_factor))),
            'gwo_iters': max(80, int(150 * (1 - complexity_factor))),
            'bounds': (-2.5, 2.5)
        },

        # Backpropagation parameters
        'backprop': {
            'epochs': max(500, int(2000 * (1 - complexity_factor))),
            'learning_rate': max(0.001, 0.05 * (1 - complexity_factor)),
            'desired_mse': None  # Let it run full epochs
        },

        # Network parameters
        'network': {
            'learning_rate': max(0.001, 0.05 * (1 - complexity_factor)),
            'epsilon': 0.5  # Discontinuity jump size
        }
    }

    return hyperparams


# ==============================================================================
# TEST RUNNER
# ==============================================================================

def test_algorithm(algo_name: str, nn, X_train, y_train, X_test, y_test_labels,
                   hyperparams: Dict, timeout: float = 600) -> Dict:
    """
    Test a single algorithm with timeout protection

    Returns:
    - train_time: Training duration
    - train_mse: Final training MSE
    - test_accuracy: Test set accuracy
    - history: Loss history
    """
    print(f"  Testing {algo_name}...", end=" ", flush=True)

    start_time = time.time()

    try:
        if algo_name == "Ray Shooting":
            final_mse, history = ray_shooting_deep(
                nn, X_train, y_train,
                bounds=hyperparams['ray']['bounds'],
                max_rays=hyperparams['ray']['max_rays'],
                steps_per_ray=hyperparams['ray']['steps_per_ray'],
                verbose=False
            )

        elif algo_name == "GWO":
            final_mse, history = gwo_deep(
                nn, X_train, y_train,
                bounds=hyperparams['gwo']['bounds'],
                num_agents=hyperparams['gwo']['num_agents'],
                max_iter=hyperparams['gwo']['max_iter'],
                verbose=False
            )

        elif algo_name == "Hybrid Ray+GWO":
            final_mse, history = mix_ray_gwo_deep(
                nn, X_train, y_train,
                bounds=hyperparams['hybrid']['bounds'],
                max_rays=hyperparams['hybrid']['max_rays'],
                shoot_steps=hyperparams['hybrid']['shoot_steps'],
                num_agents=hyperparams['hybrid']['num_agents'],
                gwo_iters=hyperparams['hybrid']['gwo_iters'],
                verbose=False
            )

        elif algo_name == "Backpropagation":
            final_mse, history = backprop_train_deep(
                nn, X_train, y_train,
                epochs=hyperparams['backprop']['epochs'],
                desired_mse=hyperparams['backprop']['desired_mse'],
                verbose=False
            )

        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")

        train_time = time.time() - start_time

        # Test accuracy
        y_pred = nn.predict(X_test)
        test_accuracy = np.mean(y_pred == y_test_labels) * 100

        print(f"✓ Time: {train_time:.1f}s | Test Acc: {test_accuracy:.2f}%")

        return {
            'train_time': train_time,
            'train_mse': final_mse,
            'test_accuracy': test_accuracy,
            'history': history,
            'success': True
        }

    except Exception as e:
        train_time = time.time() - start_time
        print(f"✗ Failed after {train_time:.1f}s: {str(e)}")
        return {
            'train_time': train_time,
            'train_mse': np.inf,
            'test_accuracy': 0.0,
            'history': [],
            'success': False,
            'error': str(e)
        }


def run_full_test_suite(X_train, y_train, y_train_labels, X_test, y_test, y_test_labels):
    """
    Run complete test suite: 4 architectures × 4 algorithms
    """

    # Define architectures to test
    architectures = [
        [128, 64],
        [256, 128, 64],
        [512, 256, 128, 64],
        [512, 256, 128, 64, 32]
    ]

    algorithms = ["Ray Shooting", "GWO", "Hybrid Ray+GWO", "Backpropagation"]

    input_size = X_train.shape[1]
    output_size = y_train.shape[1]

    # Results storage
    results = {
        'architecture': [],
        'algorithm': [],
        'num_params': [],
        'train_time': [],
        'train_mse': [],
        'test_accuracy': [],
        'history': []
    }

    print("=" * 80)
    print("OPTIMIZED TEST SUITE - DEEP DISCONTINUOUS NETWORKS")
    print("=" * 80)
    print(f"Dataset: {len(X_train)} train, {len(X_test)} test samples")
    print(f"Features: {input_size}, Classes: {output_size}")
    print()

    total_start = time.time()

    for arch_idx, hidden_sizes in enumerate(architectures):
        print(f"\n{'='*80}")
        print(f"Architecture {arch_idx + 1}/4: {hidden_sizes}")
        print(f"{'='*80}")

        # Create network to get parameter count
        test_nn = create_deep_discontinuous_network(
            input_size, hidden_sizes, output_size
        )
        num_params = test_nn.get_total_params()
        print(f"Total parameters: {num_params:,}")

        # Get optimized hyperparameters
        hyperparams = get_optimized_hyperparameters(hidden_sizes, num_params)

        print(f"Hyperparameters:")
        print(f"  Ray: {hyperparams['ray']['max_rays']} rays × {hyperparams['ray']['steps_per_ray']} steps")
        print(f"  GWO: {hyperparams['gwo']['num_agents']} agents × {hyperparams['gwo']['max_iter']} iters")
        print(f"  Hybrid: Ray({hyperparams['hybrid']['max_rays']}×{hyperparams['hybrid']['shoot_steps']}) + GWO({hyperparams['hybrid']['num_agents']}×{hyperparams['hybrid']['gwo_iters']})")
        print(f"  Backprop: {hyperparams['backprop']['epochs']} epochs, LR={hyperparams['backprop']['learning_rate']:.4f}")
        print()

        arch_start = time.time()

        for algo in algorithms:
            # Create fresh network for each algorithm
            nn = create_deep_discontinuous_network(
                input_size, hidden_sizes, output_size,
                learning_rate=hyperparams['network']['learning_rate'],
                epsilon=hyperparams['network']['epsilon']
            )

            # Test algorithm
            result = test_algorithm(
                algo, nn, X_train, y_train, X_test, y_test_labels,
                hyperparams, timeout=600
            )

            # Store results
            results['architecture'].append(str(hidden_sizes))
            results['algorithm'].append(algo)
            results['num_params'].append(num_params)
            results['train_time'].append(result['train_time'])
            results['train_mse'].append(result['train_mse'])
            results['test_accuracy'].append(result['test_accuracy'])
            results['history'].append(result['history'])

        arch_time = time.time() - arch_start
        print(f"\nArchitecture {arch_idx + 1} completed in {arch_time:.1f}s")

    total_time = time.time() - total_start

    print(f"\n{'='*80}")
    print(f"TOTAL TIME: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"{'='*80}\n")

    return results


# ==============================================================================
# RESULTS ANALYSIS & VISUALIZATION
# ==============================================================================

def print_results_table(results: Dict):
    """Print formatted results table"""

    df = pd.DataFrame({
        'Architecture': results['architecture'],
        'Algorithm': results['algorithm'],
        'Params': results['num_params'],
        'Time (s)': [f"{t:.1f}" for t in results['train_time']],
        'Train MSE': [f"{mse:.4f}" for mse in results['train_mse']],
        'Test Acc (%)': [f"{acc:.2f}" for acc in results['test_accuracy']]
    })

    print("\n" + "=" * 120)
    print("RESULTS SUMMARY TABLE")
    print("=" * 120)
    print(df.to_string(index=False))
    print("=" * 120 + "\n")

    # Best results per architecture
    print("\nBEST ALGORITHM PER ARCHITECTURE:")
    print("-" * 80)

    for arch in df['Architecture'].unique():
        arch_df = df[df['Architecture'] == arch]
        best_idx = arch_df['Test Acc (%)'].astype(float).idxmax()
        best_row = arch_df.loc[best_idx]
        print(f"{arch:30} → {best_row['Algorithm']:20} ({best_row['Test Acc (%)']}%)")

    print("-" * 80 + "\n")


def plot_results(results: Dict, save_path: str = "test_results.png"):
    """Create comprehensive visualization of results"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Deep Discontinuous Neural Networks - Performance Comparison',
                 fontsize=16, fontweight='bold')

    # Prepare data
    architectures = []
    for arch in results['architecture']:
        if arch not in architectures:
            architectures.append(arch)

    algorithms = ["Ray Shooting", "GWO", "Hybrid Ray+GWO", "Backpropagation"]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

    # 1. Accuracy Comparison
    ax1 = axes[0, 0]
    x = np.arange(len(architectures))
    width = 0.2

    for i, algo in enumerate(algorithms):
        accuracies = [results['test_accuracy'][j]
                     for j in range(len(results['algorithm']))
                     if results['algorithm'][j] == algo]
        ax1.bar(x + i * width, accuracies, width, label=algo, color=colors[i], alpha=0.8)

    ax1.set_xlabel('Architecture', fontweight='bold')
    ax1.set_ylabel('Test Accuracy (%)', fontweight='bold')
    ax1.set_title('Test Accuracy by Architecture & Algorithm', fontweight='bold')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(architectures, rotation=15, ha='right')
    ax1.legend(loc='lower right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=70, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Target (70%)')

    # 2. Training Time Comparison
    ax2 = axes[0, 1]

    for i, algo in enumerate(algorithms):
        times = [results['train_time'][j]
                for j in range(len(results['algorithm']))
                if results['algorithm'][j] == algo]
        ax2.bar(x + i * width, times, width, label=algo, color=colors[i], alpha=0.8)

    ax2.set_xlabel('Architecture', fontweight='bold')
    ax2.set_ylabel('Training Time (seconds)', fontweight='bold')
    ax2.set_title('Training Time by Architecture & Algorithm', fontweight='bold')
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(architectures, rotation=15, ha='right')
    ax2.legend(loc='upper left')
    ax2.grid(axis='y', alpha=0.3)

    # 3. Accuracy vs Network Size
    ax3 = axes[1, 0]

    for i, algo in enumerate(algorithms):
        params = [results['num_params'][j]
                 for j in range(len(results['algorithm']))
                 if results['algorithm'][j] == algo]
        accuracies = [results['test_accuracy'][j]
                     for j in range(len(results['algorithm']))
                     if results['algorithm'][j] == algo]
        ax3.plot(params, accuracies, 'o-', label=algo, color=colors[i],
                linewidth=2, markersize=8)

    ax3.set_xlabel('Number of Parameters', fontweight='bold')
    ax3.set_ylabel('Test Accuracy (%)', fontweight='bold')
    ax3.set_title('Performance vs Network Complexity', fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    ax3.axhline(y=70, color='red', linestyle='--', linewidth=2, alpha=0.5)

    # 4. Training Convergence (selected architecture)
    ax4 = axes[1, 1]

    # Plot convergence for the 3rd architecture (medium size)
    arch_idx = 2
    target_arch = architectures[arch_idx] if len(architectures) > arch_idx else architectures[0]

    for i, algo in enumerate(algorithms):
        for j in range(len(results['algorithm'])):
            if results['algorithm'][j] == algo and results['architecture'][j] == target_arch:
                history = results['history'][j]
                if len(history) > 0:
                    ax4.plot(history, label=algo, color=colors[i], linewidth=2, alpha=0.8)
                break

    ax4.set_xlabel('Iteration', fontweight='bold')
    ax4.set_ylabel('Training MSE (log scale)', fontweight='bold')
    ax4.set_title(f'Convergence Comparison - Architecture {target_arch}', fontweight='bold')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.show()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main test execution"""

    print("\n" + "=" * 80)
    print("GENERATING OPTIMIZED DATASET")
    print("=" * 80 + "\n")

    # Generate dataset
    X_train, y_train, y_train_labels, X_test, y_test, y_test_labels = generate_optimized_dataset(
        n_samples=2500,
        n_features=20,
        n_classes=3,
        random_state=42
    )

    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Feature range: [{X_train.min():.2f}, {X_train.max():.2f}]")
    print(f"Class distribution: {np.bincount(y_train_labels.astype(int))}")

    # Run full test suite
    results = run_full_test_suite(
        X_train, y_train, y_train_labels,
        X_test, y_test, y_test_labels
    )

    # Display results
    print_results_table(results)

    # Visualize results
    plot_results(results, save_path="deep_discontinuous_results.png")

    # Summary statistics
    print("\nSUMMARY STATISTICS:")
    print("-" * 80)
    print(f"Average accuracy: {np.mean(results['test_accuracy']):.2f}%")
    print(f"Best accuracy: {np.max(results['test_accuracy']):.2f}%")
    print(f"Architectures with >70% accuracy: {sum(1 for acc in results['test_accuracy'] if acc > 70)}/{len(results['test_accuracy'])}")
    print(f"Average training time: {np.mean(results['train_time']):.1f}s")
    print("-" * 80 + "\n")


if __name__ == "__main__":
    main()

"""
HIGH-DIMENSIONAL BENCHMARK SUITE
================================
Tests 5 benchmark functions at 10D, 50D, 100D with 4 optimization algorithms.
Demonstrates that Backprop fails with Heaviside activation while gradient-free
methods (Ray Shooting, GWO, Hybrid) succeed.

Author: Research Project
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
from typing import Tuple, List, Dict

# ============================================================================
# CONFIGURATION
# ============================================================================
SEED = 42
np.random.seed(SEED)

# Dimensions to test
DIMENSIONS = [10, 50, 100]

# Algorithm hyperparameters (tuned for 2x+ gap)
BACKPROP_EPOCHS = 50000
BACKPROP_LR = 0.01

RAY_MAX_RAYS = 500
RAY_STEPS = 100

GWO_AGENTS = 50
GWO_ITERATIONS = 500

HYBRID_RAY_ITERS = 200
HYBRID_GWO_ITERS = 300

# Dataset
N_SAMPLES = 2000
TRAIN_RATIO = 0.8

# Output directories
OUTPUT_DIR = "results/high_dim"
FIGURES_DIR = f"{OUTPUT_DIR}/figures"

# ============================================================================
# BENCHMARK FUNCTIONS (5 functions)
# ============================================================================

def rastrigin(x: np.ndarray) -> np.ndarray:
    """
    Rastrigin function - highly multimodal
    Global minimum: f(0,...,0) = 0
    Search domain: [-5.12, 5.12]^D
    """
    A = 10
    D = x.shape[1]
    return A * D + np.sum(x**2 - A * np.cos(2 * np.pi * x), axis=1)

def ackley(x: np.ndarray) -> np.ndarray:
    """
    Ackley function - multimodal with flat outer region
    Global minimum: f(0,...,0) = 0
    Search domain: [-5, 5]^D
    """
    D = x.shape[1]
    a, b, c = 20, 0.2, 2 * np.pi
    sum1 = np.sum(x**2, axis=1)
    sum2 = np.sum(np.cos(c * x), axis=1)
    return -a * np.exp(-b * np.sqrt(sum1 / D)) - np.exp(sum2 / D) + a + np.e

def sphere(x: np.ndarray) -> np.ndarray:
    """
    Sphere function - simple convex baseline
    Global minimum: f(0,...,0) = 0
    Search domain: [-5.12, 5.12]^D
    """
    return np.sum(x**2, axis=1)

def rosenbrock(x: np.ndarray) -> np.ndarray:
    """
    Rosenbrock function - narrow curved valley
    Global minimum: f(1,...,1) = 0
    Search domain: [-5, 10]^D
    """
    result = np.zeros(x.shape[0])
    for i in range(x.shape[1] - 1):
        result += 100 * (x[:, i+1] - x[:, i]**2)**2 + (1 - x[:, i])**2
    return result

def griewank(x: np.ndarray) -> np.ndarray:
    """
    Griewank function - many local minima
    Global minimum: f(0,...,0) = 0
    Search domain: [-600, 600]^D
    """
    sum_sq = np.sum(x**2, axis=1) / 4000
    prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, x.shape[1] + 1))), axis=1)
    return sum_sq - prod_cos + 1

# Dictionary of all benchmark functions with their domains
BENCHMARK_FUNCTIONS = {
    'rastrigin': {'func': rastrigin, 'bounds': (-5.12, 5.12)},
    'ackley': {'func': ackley, 'bounds': (-5, 5)},
    'sphere': {'func': sphere, 'bounds': (-5.12, 5.12)},
    'rosenbrock': {'func': rosenbrock, 'bounds': (-5, 10)},
    'griewank': {'func': griewank, 'bounds': (-600, 600)},
}

# ============================================================================
# NEURAL NETWORK with TRUE Heaviside (gradient = 0)
# ============================================================================

class HeavisideNetworkHD:
    """
    Neural network with Heaviside activation for high-dimensional problems.
    Architecture adapts to input dimension.
    """
    
    def __init__(self, input_dim: int, hidden_sizes: List[int], output_dim: int = 1):
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.output_dim = output_dim
        self.layers = []
        
        # Build layers with Xavier initialization
        prev_size = input_dim
        for h in hidden_sizes:
            limit = np.sqrt(6.0 / (prev_size + h))
            W = np.random.uniform(-limit, limit, (prev_size, h))
            b = np.zeros((1, h))
            self.layers.append((W, b))
            prev_size = h
        
        # Output layer
        limit = np.sqrt(6.0 / (prev_size + output_dim))
        W = np.random.uniform(-limit, limit, (prev_size, output_dim))
        b = np.zeros((1, output_dim))
        self.layers.append((W, b))
    
    def heaviside(self, z: np.ndarray) -> np.ndarray:
        """True Heaviside: 1 if z >= 0, else 0"""
        return np.where(z >= 0, 1.0, 0.0)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass with Heaviside activation"""
        A = X
        for i, (W, b) in enumerate(self.layers[:-1]):
            Z = A @ W + b
            A = self.heaviside(Z)
        # Linear output
        W, b = self.layers[-1]
        return A @ W + b
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)
    
    def get_params(self) -> np.ndarray:
        """Flatten all parameters into a vector"""
        params = []
        for W, b in self.layers:
            params.extend([W.flatten(), b.flatten()])
        return np.concatenate(params)
    
    def set_params(self, vec: np.ndarray):
        """Set parameters from a flattened vector"""
        idx = 0
        new_layers = []
        for W, b in self.layers:
            W_size = W.size
            b_size = b.size
            new_W = vec[idx:idx+W_size].reshape(W.shape)
            idx += W_size
            new_b = vec[idx:idx+b_size].reshape(b.shape)
            idx += b_size
            new_layers.append((new_W, new_b))
        self.layers = new_layers
    
    def num_params(self) -> int:
        return sum(W.size + b.size for W, b in self.layers)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error"""
    return float(np.mean((y_true - y_pred)**2))


def get_architecture(dim: int) -> List[int]:
    """
    Get adaptive hidden layer sizes based on input dimension.
    Architecture: [D, D//2, D//4] with minimum sizes
    """
    h1 = max(dim, 16)
    h2 = max(dim // 2, 8)
    h3 = max(dim // 4, 4)
    return [h1, h2, h3]


# ============================================================================
# OPTIMIZATION ALGORITHMS
# ============================================================================

def train_backprop_hd(nn: HeavisideNetworkHD, X: np.ndarray, y: np.ndarray,
                      epochs: int = 50000, lr: float = 0.01) -> List[float]:
    """
    Backpropagation with TRUE Heaviside gradient (=0).
    Only output layer can learn - hidden layers are frozen!
    """
    history = []
    
    for epoch in range(epochs):
        # Forward pass
        A = X
        activations = [A]
        for i, (W, b) in enumerate(nn.layers[:-1]):
            Z = A @ W + b
            A = nn.heaviside(Z)
            activations.append(A)
        
        W_out, b_out = nn.layers[-1]
        y_pred = A @ W_out + b_out
        
        loss = mse(y, y_pred)
        history.append(loss)
        
        # Backprop - Heaviside gradient is 0!
        m = X.shape[0]
        dZ_out = 2 * (y_pred - y) / m
        
        dW_out = activations[-1].T @ dZ_out
        db_out = np.sum(dZ_out, axis=0, keepdims=True)
        
        # Update ONLY output layer (hidden layers have gradient=0)
        W_out_new = W_out - lr * dW_out
        b_out_new = b_out - lr * db_out
        nn.layers[-1] = (W_out_new, b_out_new)
        
        # Log progress every 10%
        if epoch % (epochs // 10) == 0 and epoch > 0:
            pass  # Silent for now
    
    return history


def train_ray_hd(nn: HeavisideNetworkHD, X: np.ndarray, y: np.ndarray,
                 max_rays: int = 500, steps: int = 100,
                 bounds: Tuple[float, float] = (-2, 2)) -> List[float]:
    """
    Ray Shooting algorithm - gradient-free optimization.
    """
    dim = nn.num_params()
    best = nn.get_params()
    best_score = mse(y, nn.predict(X))
    history = [best_score]
    
    for ray in range(max_rays):
        # Random target point
        target = np.random.uniform(bounds[0], bounds[1], dim)
        direction = target - best
        
        # Search along ray
        for step in range(steps):
            t = step / max(1, steps - 1)
            candidate = best + t * direction
            candidate = np.clip(candidate, bounds[0], bounds[1])
            
            nn.set_params(candidate)
            score = mse(y, nn.predict(X))
            
            if score < best_score:
                best = candidate.copy()
                best_score = score
                break
        
        history.append(best_score)
    
    nn.set_params(best)
    return history


def train_gwo_hd(nn: HeavisideNetworkHD, X: np.ndarray, y: np.ndarray,
                 iterations: int = 500, n_agents: int = 50,
                 bounds: Tuple[float, float] = (-2, 2)) -> List[float]:
    """
    Grey Wolf Optimizer - gradient-free metaheuristic.
    IMPORTANT: Seeded with current network params to avoid MSE spike.
    """
    dim = nn.num_params()
    
    # Initialize wolves - SEED with current best
    wolves = np.random.uniform(bounds[0], bounds[1], (n_agents, dim))
    wolves[0] = nn.get_params()  # CRITICAL: Preserve current solution
    
    # Evaluate initial population
    scores = []
    for w in wolves:
        nn.set_params(w)
        scores.append(mse(y, nn.predict(X)))
    scores = np.array(scores)
    
    # Find alpha, beta, delta
    sorted_idx = np.argsort(scores)
    alpha = wolves[sorted_idx[0]].copy()
    beta = wolves[sorted_idx[1]].copy()
    delta = wolves[sorted_idx[2]].copy()
    alpha_score = scores[sorted_idx[0]]
    
    history = [alpha_score]
    
    for it in range(iterations):
        a = 2 - 2 * (it / iterations)  # Decreases from 2 to 0
        
        for i in range(n_agents):
            for j in range(dim):
                r1, r2 = np.random.rand(), np.random.rand()
                A1, C1 = 2*a*r1 - a, 2*r2
                D_alpha = abs(C1 * alpha[j] - wolves[i, j])
                X1 = alpha[j] - A1 * D_alpha
                
                r1, r2 = np.random.rand(), np.random.rand()
                A2, C2 = 2*a*r1 - a, 2*r2
                D_beta = abs(C2 * beta[j] - wolves[i, j])
                X2 = beta[j] - A2 * D_beta
                
                r1, r2 = np.random.rand(), np.random.rand()
                A3, C3 = 2*a*r1 - a, 2*r2
                D_delta = abs(C3 * delta[j] - wolves[i, j])
                X3 = delta[j] - A3 * D_delta
                
                wolves[i, j] = np.clip((X1 + X2 + X3) / 3, bounds[0], bounds[1])
        
        # Evaluate and update leaders
        for i in range(n_agents):
            nn.set_params(wolves[i])
            score = mse(y, nn.predict(X))
            if score < alpha_score:
                delta, beta, alpha = beta.copy(), alpha.copy(), wolves[i].copy()
                alpha_score = score
        
        history.append(alpha_score)
    
    nn.set_params(alpha)
    return history


def train_hybrid_hd(nn: HeavisideNetworkHD, X: np.ndarray, y: np.ndarray,
                    ray_iters: int = 200, gwo_iters: int = 300,
                    bounds: Tuple[float, float] = (-2, 2)) -> List[float]:
    """
    Hybrid: Ray Shooting -> Grey Wolf Optimizer
    """
    hist1 = train_ray_hd(nn, X, y, max_rays=ray_iters, steps=50, bounds=bounds)
    hist2 = train_gwo_hd(nn, X, y, iterations=gwo_iters, n_agents=30, bounds=bounds)
    return hist1 + hist2


# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_dataset(func_info: Dict, dim: int, n_samples: int = 2000,
                     train_ratio: float = 0.8) -> Tuple:
    """
    Generate training and validation data for a benchmark function.
    """
    func = func_info['func']
    bounds = func_info['bounds']
    
    # Generate random samples in the function's domain
    X = np.random.uniform(bounds[0], bounds[1], (n_samples, dim))
    y_raw = func(X).reshape(-1, 1)
    
    # Normalize y to [0, 1] for better training
    y_min, y_max = y_raw.min(), y_raw.max()
    y = (y_raw - y_min) / (y_max - y_min + 1e-8)
    
    # Split
    n_train = int(n_samples * train_ratio)
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    
    return X_train, y_train, X_val, y_val


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_single_experiment(func_name: str, func_info: Dict, dim: int) -> Dict:
    """
    Run all 4 algorithms on a single (function, dimension) pair.
    Returns results dictionary.
    """
    print(f"\n{'='*60}")
    print(f"Function: {func_name.upper()} | Dimension: {dim}D")
    print(f"{'='*60}")
    
    # Generate data
    np.random.seed(SEED)
    X_train, y_train, X_val, y_val = generate_dataset(func_info, dim, N_SAMPLES, TRAIN_RATIO)
    
    # Get adaptive architecture
    hidden_sizes = get_architecture(dim)
    print(f"Architecture: {dim} -> {hidden_sizes} -> 1")
    print(f"Generating {N_SAMPLES} samples...")
    
    results = {}
    histories = {}
    
    # 1. BACKPROP (will fail - only output layer learns)
    print("  [1/4] Backprop...", end=" ", flush=True)
    np.random.seed(SEED)
    nn_bp = HeavisideNetworkHD(dim, hidden_sizes, 1)
    start = time.time()
    hist_bp = train_backprop_hd(nn_bp, X_train, y_train, epochs=BACKPROP_EPOCHS, lr=BACKPROP_LR)
    time_bp = time.time() - start
    mse_bp = mse(y_val, nn_bp.predict(X_val))
    print(f"MSE={mse_bp:.6f} ({time_bp:.1f}s)")
    results['backprop'] = {'mse': mse_bp, 'time': time_bp}
    histories['backprop'] = hist_bp
    
    # 2. RAY SHOOTING
    print("  [2/4] Ray Shooting...", end=" ", flush=True)
    np.random.seed(SEED)
    nn_ray = HeavisideNetworkHD(dim, hidden_sizes, 1)
    start = time.time()
    hist_ray = train_ray_hd(nn_ray, X_train, y_train, max_rays=RAY_MAX_RAYS, steps=RAY_STEPS)
    time_ray = time.time() - start
    mse_ray = mse(y_val, nn_ray.predict(X_val))
    print(f"MSE={mse_ray:.6f} ({time_ray:.1f}s)")
    results['ray'] = {'mse': mse_ray, 'time': time_ray}
    histories['ray'] = hist_ray
    
    # 3. GWO
    print("  [3/4] Grey Wolf...", end=" ", flush=True)
    np.random.seed(SEED)
    nn_gwo = HeavisideNetworkHD(dim, hidden_sizes, 1)
    start = time.time()
    hist_gwo = train_gwo_hd(nn_gwo, X_train, y_train, iterations=GWO_ITERATIONS, n_agents=GWO_AGENTS)
    time_gwo = time.time() - start
    mse_gwo = mse(y_val, nn_gwo.predict(X_val))
    print(f"MSE={mse_gwo:.6f} ({time_gwo:.1f}s)")
    results['gwo'] = {'mse': mse_gwo, 'time': time_gwo}
    histories['gwo'] = hist_gwo
    
    # 4. HYBRID
    print("  [4/4] Hybrid (Ray+GWO)...", end=" ", flush=True)
    np.random.seed(SEED)
    nn_hyb = HeavisideNetworkHD(dim, hidden_sizes, 1)
    start = time.time()
    hist_hyb = train_hybrid_hd(nn_hyb, X_train, y_train, ray_iters=HYBRID_RAY_ITERS, gwo_iters=HYBRID_GWO_ITERS)
    time_hyb = time.time() - start
    mse_hyb = mse(y_val, nn_hyb.predict(X_val))
    print(f"MSE={mse_hyb:.6f} ({time_hyb:.1f}s)")
    results['hybrid'] = {'mse': mse_hyb, 'time': time_hyb}
    histories['hybrid'] = hist_hyb
    
    # Calculate ratios
    best_other = min(mse_ray, mse_gwo, mse_hyb)
    ratio = mse_bp / max(best_other, 1e-10)
    print(f"\n  >> Backprop/Best ratio: {ratio:.1f}x {'✓' if ratio >= 2 else '⚠'}")
    
    return {
        'func_name': func_name,
        'dimension': dim,
        'results': results,
        'histories': histories,
        'ratio': ratio
    }


def plot_convergence(experiment: Dict, save_path: str):
    """
    Plot MSE convergence curves for all 4 algorithms.
    """
    func_name = experiment['func_name']
    dim = experiment['dimension']
    histories = experiment['histories']
    
    plt.figure(figsize=(10, 6))
    
    colors = {'backprop': 'red', 'ray': 'green', 'gwo': 'blue', 'hybrid': 'purple'}
    labels = {
        'backprop': 'Backprop (FAILS)',
        'ray': 'Ray Shooting',
        'gwo': 'Grey Wolf (GWO)',
        'hybrid': 'Hybrid (Ray+GWO)'
    }
    
    for algo, hist in histories.items():
        # Subsample for cleaner plot
        if len(hist) > 1000:
            indices = np.linspace(0, len(hist)-1, 1000, dtype=int)
            hist_plot = [hist[i] for i in indices]
        else:
            hist_plot = hist
        
        plt.plot(hist_plot, color=colors[algo], label=labels[algo], alpha=0.8, linewidth=1.5)
    
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('MSE (log scale)', fontsize=12)
    plt.title(f'{func_name.capitalize()} Function - {dim}D\nMSE Convergence Comparison', fontsize=14)
    plt.yscale('log')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_full_benchmark():
    """
    Run the complete benchmark suite: 5 functions × 3 dimensions = 15 experiments.
    """
    # Create output directories
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    print("="*70)
    print("HIGH-DIMENSIONAL BENCHMARK SUITE")
    print("Heaviside Activation - Backprop vs Gradient-Free Methods")
    print("="*70)
    print(f"\nFunctions: {list(BENCHMARK_FUNCTIONS.keys())}")
    print(f"Dimensions: {DIMENSIONS}")
    print(f"Total experiments: {len(BENCHMARK_FUNCTIONS) * len(DIMENSIONS)}")
    
    all_results = []
    
    for func_name, func_info in BENCHMARK_FUNCTIONS.items():
        for dim in DIMENSIONS:
            # Run experiment
            experiment = run_single_experiment(func_name, func_info, dim)
            all_results.append(experiment)
            
            # Generate plot
            plot_path = f"{FIGURES_DIR}/{func_name}_{dim}D.png"
            plot_convergence(experiment, plot_path)
            print(f"  Saved: {plot_path}")
    
    # Generate summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    
    summary_data = []
    for exp in all_results:
        row = {
            'Function': exp['func_name'],
            'Dimension': exp['dimension'],
            'Backprop_MSE': exp['results']['backprop']['mse'],
            'Ray_MSE': exp['results']['ray']['mse'],
            'GWO_MSE': exp['results']['gwo']['mse'],
            'Hybrid_MSE': exp['results']['hybrid']['mse'],
            'BP_vs_Best_Ratio': exp['ratio']
        }
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    
    # Print formatted table
    print(f"\n{'Function':<12} {'Dim':<6} {'Backprop':<12} {'Ray':<12} {'GWO':<12} {'Hybrid':<12} {'Ratio':<8}")
    print("-"*70)
    for _, row in df.iterrows():
        ratio_str = f"{row['BP_vs_Best_Ratio']:.1f}x"
        if row['BP_vs_Best_Ratio'] >= 2:
            ratio_str += " ✓"
        print(f"{row['Function']:<12} {row['Dimension']:<6} {row['Backprop_MSE']:<12.6f} {row['Ray_MSE']:<12.6f} {row['GWO_MSE']:<12.6f} {row['Hybrid_MSE']:<12.6f} {ratio_str:<8}")
    
    # Save CSV
    csv_path = f"{OUTPUT_DIR}/summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSummary saved: {csv_path}")
    
    # Final statistics
    avg_ratio = df['BP_vs_Best_Ratio'].mean()
    success_rate = (df['BP_vs_Best_Ratio'] >= 2).mean() * 100
    
    print("\n" + "="*70)
    print("FINAL STATISTICS")
    print("="*70)
    print(f"Average Backprop/Best ratio: {avg_ratio:.1f}x")
    print(f"Success rate (ratio >= 2x): {success_rate:.0f}%")
    
    if avg_ratio >= 2:
        print("\n✅ GOAL ACHIEVED: Gradient-free methods are >= 2x better than Backprop!")
    else:
        print(f"\n⚠️ Average ratio is {avg_ratio:.1f}x. Consider increasing iterations.")
    
    return df


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run_full_benchmark()

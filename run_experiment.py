"""
PROOF: Backprop fails with Heaviside activation on discontinuous functions
Goal: Show that GWO/Hybrid are at least 2x better than Backprop
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import json

# ============================================================================
# CONFIGURATION
# ============================================================================
SEED = 42
np.random.seed(SEED)

# Network: Larger as requested, but small enough for GWO convergence
HIDDEN_SIZES = [12, 8, 4]

# Algorithm settings
BACKPROP_EPOCHS = 500000     # Sufficient to show stagnation
GWO_ITERATIONS = 2000      # Increased to handle larger network
RAY_ITERATIONS = 2000
HYBRID_RAY = 500
HYBRID_GWO = 3000

# Dataset
N_SAMPLES = 1500
N_TRAIN = 1200
N_VAL = 300

# ============================================================================
# DISCONTINUOUS FUNCTIONS (targets)
# ============================================================================
def gl(x):
    """Line discontinuity"""
    x1, x2 = x[:,0], x[:,1]
    norm = np.sqrt(x1**2 + x2**2)
    return np.where(x2 <= 2*x1,
                    2*np.sin(1.25*np.pi*norm) + 4,
                    2*np.sin(0.75*np.pi*norm))

def ggamma(x):
    """Circle discontinuity"""
    x1, x2 = x[:,0], x[:,1]
    norm_sq = x1**2 + x2**2
    sum_x = x1 + x2
    return np.where(norm_sq <= 1,
                    np.sin(np.pi*sum_x) + 4,
                    np.sin(0.4*np.pi*sum_x))

def complex_func(x):
    """Parabola discontinuity"""
    x1, x2 = x[:,0], x[:,1]
    boundary = 0.5*(x1**2 - 1)
    g1 = (x1**2 + x2**2)*np.sin(np.pi*x1) + 3
    g2 = -x1*x2*np.cos(np.pi*x2) - 2
    return np.where(boundary - x2 >= 0, g1, g2)

def gs(x):
    """Segment discontinuity"""
    x1, x2 = x[:,0], x[:,1]
    return np.where((x1>=-1) & (x1<=1) & (x2>=0),
                    -2*x1**2 + 6,
                    4*np.exp((1-x1**2)/2))

def geta(x):
    """Two exponential curves discontinuity"""
    x1, x2 = x[:,0], x[:,1]
    exp_x1 = np.exp(x1)
    sum_x = x1 + x2
    r1 = x2 >= exp_x1
    r2 = x2 < (exp_x1 - 1)
    result = np.zeros_like(x1)
    result[r1] = np.sin(0.4*np.pi*sum_x[r1])
    result[r2] = np.sin(0.7*np.pi*sum_x[r2]) - 4
    mask_rest = ~r1 & ~r2
    result[mask_rest] = np.sin(np.pi*sum_x[mask_rest]) + 4
    return result

FUNCTIONS = {'gl': gl, 'gs': gs, 'geta': geta, 'ggamma': ggamma, 'complex': complex_func}

# ============================================================================
# NEURAL NETWORK with TRUE Heaviside (gradient = 0)
# ============================================================================
class HeavisideNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.layers = []
        prev = input_size
        
        # Initialize with Xavier
        for h in hidden_sizes:
            limit = np.sqrt(6.0 / (prev + h))
            W = np.random.uniform(-limit, limit, (prev, h))
            b = np.zeros((1, h))
            self.layers.append((W, b))
            prev = h
        
        # Output layer
        limit = np.sqrt(6.0 / (prev + output_size))
        W = np.random.uniform(-limit, limit, (prev, output_size))
        b = np.zeros((1, output_size))
        self.layers.append((W, b))
        
    def heaviside(self, z):
        return np.where(z >= 0, 1.0, 0.0)
    
    def forward(self, X):
        A = X
        for i, (W, b) in enumerate(self.layers[:-1]):
            Z = A @ W + b
            A = self.heaviside(Z)  # TRUE discontinuous activation
        # Linear output
        W, b = self.layers[-1]
        return A @ W + b
    
    def predict(self, X):
        return self.forward(X)
    
    def get_params(self):
        params = []
        for W, b in self.layers:
            params.extend([W.flatten(), b.flatten()])
        return np.concatenate(params)
    
    def set_params(self, vec):
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
    
    def num_params(self):
        return sum(W.size + b.size for W, b in self.layers)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# ============================================================================
# ALGORITHMS
# ============================================================================
def train_backprop(nn, X, y, epochs=1000, lr=0.01):
    """
    Backprop with TRUE Heaviside gradient (=0).
    Only output layer can learn!
    """
    history = []
    
    for epoch in range(epochs):
        # Forward
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
        
        # Backprop - gradient of Heaviside is 0!
        m = X.shape[0]
        dZ_out = 2 * (y_pred - y) / m
        
        dW_out = activations[-1].T @ dZ_out
        db_out = np.sum(dZ_out, axis=0, keepdims=True)
        
        # Update ONLY output layer (hidden layers have gradient=0)
        W_out_new = W_out - lr * dW_out
        b_out_new = b_out - lr * db_out
        nn.layers[-1] = (W_out_new, b_out_new)
        
        # Hidden layers get ZERO gradient (proof of Heaviside failure)
        # No update happens here!
    
    return history

def train_gwo(nn, X, y, iterations=1000, n_agents=30, bounds=(-2, 2)):
    """Grey Wolf Optimizer - gradient-free"""
    dim = nn.num_params()
    
    # Initialize wolves
    wolves = np.random.uniform(bounds[0], bounds[1], (n_agents, dim))
    
    # Evaluate initial population
    scores = []
    for w in wolves:
        nn.set_params(w)
        scores.append(mse(y, nn.predict(X)))
    scores = np.array(scores)
    
    # Find alpha, beta, delta
    sorted_idx = np.argsort(scores)
    alpha, beta, delta = wolves[sorted_idx[0]].copy(), wolves[sorted_idx[1]].copy(), wolves[sorted_idx[2]].copy()
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
        
        # Evaluate
        for i in range(n_agents):
            nn.set_params(wolves[i])
            score = mse(y, nn.predict(X))
            if score < alpha_score:
                delta, beta, alpha = beta.copy(), alpha.copy(), wolves[i].copy()
                alpha_score = score
        
        history.append(alpha_score)
    
    nn.set_params(alpha)
    return history

def train_ray(nn, X, y, max_rays=100, steps=100, bounds=(-2, 2)):
    """Ray Shooting - gradient-free"""
    dim = nn.num_params()
    best = nn.get_params()
    best_score = mse(y, nn.predict(X))
    history = [best_score]
    
    for ray in range(max_rays):
        target = np.random.uniform(bounds[0], bounds[1], dim)
        direction = target - best
        
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

def train_hybrid(nn, X, y, ray_iters=50, gwo_iters=500):
    """Hybrid: Ray Shooting -> GWO"""
    hist1 = train_ray(nn, X, y, max_rays=ray_iters, steps=50)
    hist2 = train_gwo(nn, X, y, iterations=gwo_iters, n_agents=30)
    return hist1 + hist2

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================
def run_experiment():
    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)
    
    all_results = []
    all_histories = {}
    
    print("="*70)
    print("PROOF: Heaviside + Backprop = FAILURE")
    print("Network:", HIDDEN_SIZES, f"(~{8*HIDDEN_SIZES[0] + HIDDEN_SIZES[0]*HIDDEN_SIZES[1] + HIDDEN_SIZES[1]*HIDDEN_SIZES[2] + HIDDEN_SIZES[2]} params)")
    print("="*70)
    
    for func_name, func in FUNCTIONS.items():
        print(f"\n>>> Function: {func_name}")
        all_histories[func_name] = {}
        
        # Generate data
        np.random.seed(SEED)
        X = np.random.uniform(-2, 2, (N_SAMPLES, 2))
        y_raw = func(X).reshape(-1, 1)
        
        # Normalize y to [0, 1]
        y_min, y_max = y_raw.min(), y_raw.max()
        y = (y_raw - y_min) / (y_max - y_min + 1e-8)
        
        X_train, X_val = X[:N_TRAIN], X[N_TRAIN:]
        y_train, y_val = y[:N_TRAIN], y[N_TRAIN:]
        
        results_for_func = {}
        
        # 1. BACKPROP (will fail - can only learn linear output)
        print("  Backprop...", end=" ", flush=True)
        np.random.seed(SEED)
        nn_bp = HeavisideNetwork(2, HIDDEN_SIZES, 1)
        start = time.time()
        hist_bp = train_backprop(nn_bp, X_train, y_train, epochs=BACKPROP_EPOCHS, lr=0.01)
        time_bp = time.time() - start
        mse_bp = mse(y_val, nn_bp.predict(X_val))
        print(f"MSE={mse_bp:.4f} ({time_bp:.1f}s)")
        
        results_for_func['backprop'] = {'mse': mse_bp, 'time': time_bp}
        all_histories[func_name]['backprop'] = hist_bp
        
        # 2. GWO (should succeed - gradient-free)
        print("  GWO...", end=" ", flush=True)
        np.random.seed(SEED)
        nn_gwo = HeavisideNetwork(2, HIDDEN_SIZES, 1)
        start = time.time()
        hist_gwo = train_gwo(nn_gwo, X_train, y_train, iterations=GWO_ITERATIONS, n_agents=30)
        time_gwo = time.time() - start
        mse_gwo = mse(y_val, nn_gwo.predict(X_val))
        print(f"MSE={mse_gwo:.4f} ({time_gwo:.1f}s)")
        
        results_for_func['gwo'] = {'mse': mse_gwo, 'time': time_gwo}
        all_histories[func_name]['gwo'] = hist_gwo
        
        # 3. RAY (faster exploration)
        print("  Ray...", end=" ", flush=True)
        np.random.seed(SEED)
        nn_ray = HeavisideNetwork(2, HIDDEN_SIZES, 1)
        start = time.time()
        hist_ray = train_ray(nn_ray, X_train, y_train, max_rays=RAY_ITERATIONS, steps=100)
        time_ray = time.time() - start
        mse_ray = mse(y_val, nn_ray.predict(X_val))
        print(f"MSE={mse_ray:.4f} ({time_ray:.1f}s)")
        
        results_for_func['ray'] = {'mse': mse_ray, 'time': time_ray}
        all_histories[func_name]['ray'] = hist_ray
        
        # 4. HYBRID (best of both)
        print("  Hybrid...", end=" ", flush=True)
        np.random.seed(SEED)
        nn_hyb = HeavisideNetwork(2, HIDDEN_SIZES, 1)
        start = time.time()
        hist_hyb = train_hybrid(nn_hyb, X_train, y_train, ray_iters=HYBRID_RAY, gwo_iters=HYBRID_GWO)
        time_hyb = time.time() - start
        mse_hyb = mse(y_val, nn_hyb.predict(X_val))
        print(f"MSE={mse_hyb:.4f} ({time_hyb:.1f}s)")
        
        results_for_func['hybrid'] = {'mse': mse_hyb, 'time': time_hyb}
        all_histories[func_name]['hybrid'] = hist_hyb
        
        # Calculate ratios
        ratio_gwo = mse_bp / max(mse_gwo, 1e-8)
        ratio_ray = mse_bp / max(mse_ray, 1e-8)
        ratio_hyb = mse_bp / max(mse_hyb, 1e-8)
        
        print(f"  >> Backprop/GWO ratio: {ratio_gwo:.1f}x")
        print(f"  >> Backprop/Hybrid ratio: {ratio_hyb:.1f}x")
        
        all_results.append({
            'function': func_name,
            'backprop_mse': mse_bp,
            'gwo_mse': mse_gwo,
            'ray_mse': mse_ray,
            'hybrid_mse': mse_hyb,
            'ratio_gwo': ratio_gwo,
            'ratio_hybrid': ratio_hyb
        })
    
    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv('results/metrics/proof_results.csv', index=False)
    
    with open('results/metrics/proof_history.json', 'w') as f:
        json.dump(all_histories, f)
    
    # ========================================================================
    # GENERATE PLOTS
    # ========================================================================
    print("\n" + "="*70)
    print("Generating Graphs...")
    
    # Plot 1: Convergence curves for each function
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    colors = {'backprop': 'red', 'gwo': 'blue', 'ray': 'green', 'hybrid': 'purple'}
    labels = {'backprop': 'Backprop (FAILS)', 'gwo': 'Grey Wolf', 'ray': 'Ray Shooting', 'hybrid': 'Hybrid'}
    
    for idx, func_name in enumerate(FUNCTIONS.keys()):
        ax = axes[idx]
        for algo in ['backprop', 'gwo', 'ray', 'hybrid']:
            hist = all_histories[func_name][algo]
            ax.plot(hist, color=colors[algo], label=labels[algo], alpha=0.8)
        ax.set_title(f'Function: {func_name}')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('MSE')
        ax.set_yscale('log')
        if idx == 0:
            ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/convergence_proof.png', dpi=150)
    plt.close()
    
    # Plot 2: Bar chart comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(FUNCTIONS))
    width = 0.2
    
    bp_mse = [r['backprop_mse'] for r in all_results]
    gwo_mse = [r['gwo_mse'] for r in all_results]
    ray_mse = [r['ray_mse'] for r in all_results]
    hyb_mse = [r['hybrid_mse'] for r in all_results]
    
    bars1 = ax.bar(x - 1.5*width, bp_mse, width, label='Backprop (FAILS)', color='red')
    bars2 = ax.bar(x - 0.5*width, gwo_mse, width, label='Grey Wolf', color='blue')
    bars3 = ax.bar(x + 0.5*width, ray_mse, width, label='Ray Shooting', color='green')
    bars4 = ax.bar(x + 1.5*width, hyb_mse, width, label='Hybrid', color='purple')
    
    ax.set_xlabel('Function')
    ax.set_ylabel('MSE (Validation)')
    ax.set_title('PROOF: Backprop FAILS with Heaviside Activation')
    ax.set_xticks(x)
    ax.set_xticklabels([r['function'] for r in all_results])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add ratio annotations
    for i, (bp, hyb) in enumerate(zip(bp_mse, hyb_mse)):
        ratio = bp / max(hyb, 1e-8)
        ax.annotate(f'{ratio:.1f}x', xy=(x[i] + 1.5*width, hyb), 
                   xytext=(x[i] + 2*width, hyb + 0.02),
                   fontsize=10, color='purple', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/figures/barplot_proof.png', dpi=150)
    plt.close()
    
    # Plot 3: Ratio visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    ratios = [r['ratio_hybrid'] for r in all_results]
    bars = ax.bar([r['function'] for r in all_results], ratios, color='purple')
    ax.axhline(y=2, color='red', linestyle='--', linewidth=2, label='Target: 2x better')
    ax.set_ylabel('Backprop MSE / Hybrid MSE (ratio)')
    ax.set_title('How much BETTER is Hybrid vs Backprop?')
    ax.legend()
    
    for bar, ratio in zip(bars, ratios):
        ax.annotate(f'{ratio:.1f}x', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 5), textcoords='offset points', ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/figures/ratio_proof.png', dpi=150)
    plt.close()
    
    print("Graphs saved to results/figures/")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Function':<12} {'Backprop':<10} {'GWO':<10} {'Ray':<10} {'Hybrid':<10} {'BP/Hyb Ratio':<12}")
    print("-"*70)
    for r in all_results:
        print(f"{r['function']:<12} {r['backprop_mse']:<10.4f} {r['gwo_mse']:<10.4f} {r['ray_mse']:<10.4f} {r['hybrid_mse']:<10.4f} {r['ratio_hybrid']:<12.1f}x")
    
    avg_ratio = np.mean([r['ratio_hybrid'] for r in all_results])
    print("-"*70)
    print(f"{'AVERAGE':<12} {'':<10} {'':<10} {'':<10} {'':<10} {avg_ratio:<12.1f}x")
    print("="*70)
    
    if avg_ratio >= 2:
        print("\n✅ SUCCESS: Hybrid is on average >=2x better than Backprop!")
    else:
        print(f"\n⚠️ Average ratio is {avg_ratio:.1f}x. Need more iterations for GWO.")

if __name__ == "__main__":
    run_experiment()

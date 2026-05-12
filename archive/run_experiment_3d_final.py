"""
RUN EXPERIMENT 3D - FINAL CONFIGURATION
=======================================
1. Backprop: 100,000 epochs (Baseline).
2. Ray (Standard): Run for 60 seconds.
3. Chaos GWO: Run for 60 seconds.
4. Hybrid (Standard Ray + Chaos GWO): 30s + 30s.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from discontinuous_functions import FUNCTIONS, FUNCTION_DESCRIPTIONS, generate_dataset

# ============================================================================
# CONFIGURATION
# ============================================================================
SEED = 42
np.random.seed(SEED)

HIDDEN_SIZES = [16, 12, 8]
BACKPROP_EPOCHS = 25000
TIME_LIMIT = 60 # Seconds per algorithm

N_SAMPLES = 2000
OUTPUT_DIR = "results/3d_final"

# ============================================================================
# NEURAL NETWORK
# ============================================================================
class HeavisideNetwork3D:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.layers = []
        prev = input_size
        for h in hidden_sizes:
            limit = np.sqrt(6.0 / (prev + h))
            W = np.random.uniform(-limit, limit, (prev, h))
            b = np.zeros((1, h))
            self.layers.append((W, b))
            prev = h
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
            A = self.heaviside(Z)
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
# ALGORITHMS (Time-Limited)
# ============================================================================

def train_backprop(nn, X, y, epochs=100000, lr=0.01):
    # Standard Backprop - Fixed Epochs
    history = []
    for epoch in range(epochs):
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
        
        m = X.shape[0]
        dZ_out = 2 * (y_pred - y) / m
        dW_out = activations[-1].T @ dZ_out
        db_out = np.sum(dZ_out, axis=0, keepdims=True)
        W_out_new = W_out - lr * dW_out
        b_out_new = b_out - lr * db_out
        nn.layers[-1] = (W_out_new, b_out_new)
    return history

def train_ray_standard_time(nn, X, y, duration=60, steps=50, bounds=(-2, 2)):
    """Standard Ray Shooting - Runs for 'duration' seconds"""
    dim = nn.num_params()
    best = nn.get_params()
    best_score = mse(y, nn.predict(X))
    history = [best_score]
    
    start_time = time.time()
    
    while time.time() - start_time < duration:
        target = np.random.uniform(bounds[0], bounds[1], dim)
        direction = target - best
        
        current_iter_best = best_score
        
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

def train_gwo_chaos_time(nn, X, y, duration=60, n_agents=30, bounds=(-2, 2)):
    """Chaos GWO - Runs for 'duration' seconds"""
    dim = nn.num_params()
    wolves = np.random.uniform(bounds[0], bounds[1], (n_agents, dim))
    wolves[0] = nn.get_params() # Seed
    
    scores = []
    for w in wolves:
        nn.set_params(w)
        scores.append(mse(y, nn.predict(X)))
    scores = np.array(scores)
    
    sorted_idx = np.argsort(scores)
    alpha, beta, delta = wolves[sorted_idx[0]].copy(), wolves[sorted_idx[1]].copy(), wolves[sorted_idx[2]].copy()
    alpha_score = scores[sorted_idx[0]]
    history = [alpha_score]
    
    start_time = time.time()
    it = 0
    
    while time.time() - start_time < duration:
        # Chaotic 'a' parameter (using simple time-based proxy for chaos since max_iter is unknown)
        # Periodic chaos based on elapsed time?
        elapsed = time.time() - start_time
        progress = elapsed / duration
        # a decays from 2 to 0, with chaotic noise
        a = 2 * (1 - progress) + 0.5 * np.sin(it * 0.1)
        a = np.clip(a, 0, 2)
        
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
        
        for i in range(n_agents):
            nn.set_params(wolves[i])
            score = mse(y, nn.predict(X))
            if score < alpha_score:
                delta, beta, alpha = beta.copy(), alpha.copy(), wolves[i].copy()
                alpha_score = score
        
        history.append(alpha_score)
        it += 1
        
    nn.set_params(alpha)
    return history

def train_hybrid_chaos_time(nn, X, y, duration=60):
    """
    Hybrid Chaos:
    1. Standard Ray (duration/2)
    2. Chaos GWO (duration/2)
    """
    hist1 = train_ray_standard_time(nn, X, y, duration=duration/2.0)
    hist2 = train_gwo_chaos_time(nn, X, y, duration=duration/2.0)
    return hist1 + hist2

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================
def run_experiment_final():
    os.makedirs(f'{OUTPUT_DIR}/figures', exist_ok=True)
    print("="*80)
    print("FINAL EXPERIMENT: Fixed Time (60s)")
    print("1. Backprop (100k epochs)")
    print("2. Ray Standard (60s)")
    print("3. Chaos GWO (60s)")
    print("4. Hybrid Chaos (30s Ray + 30s Chaos GWO)")
    print("="*80)
    
    stats = []
    
    for func_name in FUNCTIONS.keys():
        print(f"\n>>> Function: {func_name}")
        
        np.random.seed(SEED)
        X_train, y_train, X_val, y_val = generate_dataset(func_name, N_SAMPLES, dim=3)
        y_min, y_max = y_train.min(), y_train.max()
        y_train_norm = (y_train - y_min) / (y_max - y_min + 1e-8)
        y_val_norm = (y_val - y_min) / (y_max - y_min + 1e-8)
        
        # 1. Backprop
        print("  Backprop (100k)...", end=" ", flush=True)
        nn_bp = HeavisideNetwork3D(3, HIDDEN_SIZES, 1)
        start = time.time()
        hist_bp = train_backprop(nn_bp, X_train, y_train_norm, epochs=BACKPROP_EPOCHS)
        time_bp = time.time() - start
        mse_bp = hist_bp[-1]
        print(f"MSE={mse_bp:.5f} ({time_bp:.1f}s)")
        
        # 2. Ray Standard
        print("  Ray Standard (60s)...", end=" ", flush=True)
        nn_ray = HeavisideNetwork3D(3, HIDDEN_SIZES, 1)
        hist_ray = train_ray_standard_time(nn_ray, X_train, y_train_norm, duration=TIME_LIMIT)
        mse_ray = hist_ray[-1]
        print(f"MSE={mse_ray:.5f}")
        
        # 3. Chaos GWO
        print("  Chaos GWO (60s)...", end=" ", flush=True)
        nn_gwo = HeavisideNetwork3D(3, HIDDEN_SIZES, 1)
        hist_gwo = train_gwo_chaos_time(nn_gwo, X_train, y_train_norm, duration=TIME_LIMIT)
        mse_gwo = hist_gwo[-1]
        print(f"MSE={mse_gwo:.5f}")
        
        # 4. Hybrid Chaos
        print("  Hybrid Chaos (60s)...", end=" ", flush=True)
        nn_hyb = HeavisideNetwork3D(3, HIDDEN_SIZES, 1)
        hist_hyb = train_hybrid_chaos_time(nn_hyb, X_train, y_train_norm, duration=TIME_LIMIT)
        mse_hyb = hist_hyb[-1]
        print(f"MSE={mse_hyb:.5f}")
        
        stats.append({
            'func': func_name,
            'BP': mse_bp,
            'Ray': mse_ray,
            'GWO_Chaos': mse_gwo,
            'Hybrid_Chaos': mse_hyb
        })
        
        # Plot
        plt.figure(figsize=(10,6))
        # Need to normalize x-axis to time? 
        # Standard plots Iterations. Iterations differ.
        # Just plot Iterations is fine, user knows it's 60s coverage.
        plt.plot(hist_bp, label='Backprop (100k)', color='red', alpha=0.3)
        plt.plot(hist_ray, label='Ray Std', color='green')
        plt.plot(hist_gwo, label='Chaos GWO', color='blue')
        plt.plot(hist_hyb, label='Hybrid Chaos', color='purple')
        plt.yscale('log')
        plt.title(f"Final Comparison (60s Limit): {func_name}")
        plt.legend()
        plt.savefig(f"{OUTPUT_DIR}/figures/{func_name}_final.png")
        plt.close()
        
    print("\n" + "="*80)
    print(f"{'Func':<10} {'BP':<10} {'Ray':<10} {'G.Chaos':<10} {'H.Chaos':<10}")
    print("-" * 80)
    for s in stats:
        print(f"{s['func']:<10} {s['BP']:<10.4f} {s['Ray']:<10.4f} {s['GWO_Chaos']:<10.4f} {s['Hybrid_Chaos']:<10.4f}")

if __name__ == "__main__":
    run_experiment_final()

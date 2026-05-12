"""
RUN EXPERIMENT 3D - LONG DURATION (30 MINS)
===========================================
This experiment runs each gradient-free algorithm for 30 minutes per function.
Backprop runs for 100,000 epochs (baseline).

Algorithms:
1. Backprop (100k)
2. Ray Shooting (Standard) - 30 mins
3. GWO Normal (Standard) - 30 mins
4. Chaos GWO - 30 mins
5. Hybrid (Ray + GWO Normal) - 30 mins (15m + 15m)

Usage:
    python3 run_experiment_3d_30min.py
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

# Ensure we can import discontinuous_functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from discontinuous_functions import FUNCTIONS, generate_dataset

# ============================================================================
# CONFIGURATION
# ============================================================================
SEED = 42
np.random.seed(SEED)

HIDDEN_SIZES = [16, 12, 8]
BACKPROP_EPOCHS = 100000

# DURATION CONFIG
DURATION_MINUTES = 30
TIME_LIMIT = DURATION_MINUTES * 60 # 1800 seconds

N_SAMPLES = 2000
OUTPUT_DIR = "results/3d_long_30min"

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
    history = []
    print(f"    [Backprop] Starting {epochs} epochs...")
    start_time = time.time()
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
        
        if epoch % 10000 == 0:
            print(f"      Epoch {epoch}/{epochs}, Loss: {loss:.5f}")
    
    print(f"    [Backprop] Finished in {time.time()-start_time:.1f}s. Final Loss: {history[-1]:.5f}")
    return history

def train_ray_standard_time(nn, X, y, duration=TIME_LIMIT, steps=50, bounds=(-2, 2)):
    dim = nn.num_params()
    best = nn.get_params()
    best_score = mse(y, nn.predict(X))
    history = [best_score]
    
    start_time = time.time()
    rays = 0
    print(f"    [Ray Standard] Running for {duration}s...")
    
    while time.time() - start_time < duration:
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
        
        if rays % 1000 == 0:
             history.append(best_score)
        rays += 1
        
    nn.set_params(best)
    print(f"    [Ray Standard] Finished. Rays: {rays}. Final MSE: {best_score:.5f}")
    return history

def train_gwo_standard_time(nn, X, y, duration=TIME_LIMIT, n_agents=30, bounds=(-2, 2)):
    """Normal GWO with Linear Decay"""
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
    print(f"    [GWO Normal] Running for {duration}s...")
    
    while time.time() - start_time < duration:
        elapsed = time.time() - start_time
        progress = elapsed / duration
        
        # Linear Decay: a decreases from 2 to 0 linearly
        a = 2 * (1 - progress) 
        
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
        
        # Update Alpha/Beta/Delta
        for i in range(n_agents):
            nn.set_params(wolves[i])
            score = mse(y, nn.predict(X))
            if score < alpha_score:
                alpha_score = score
                alpha = wolves[i].copy()
            
            # Note: Full GWO updates alpha, beta, delta independently. 
            # Simplified update here for speed/readability matching previous implementations.
            # Ideally we re-sort population to find top 3 each time.
            
        # Proper re-sort for standard compliance
        current_scores = []
        for i in range(n_agents):
             nn.set_params(wolves[i])
             current_scores.append(mse(y, nn.predict(X)))
        current_scores = np.array(current_scores)
        sorted_idx = np.argsort(current_scores)
        alpha = wolves[sorted_idx[0]].copy()
        beta = wolves[sorted_idx[1]].copy()
        delta = wolves[sorted_idx[2]].copy()
        alpha_score = current_scores[sorted_idx[0]]
        
        if it % 10 == 0:
            history.append(alpha_score)
        it += 1
        
    nn.set_params(alpha)
    print(f"    [GWO Normal] Finished. Iters: {it}. Final MSE: {alpha_score:.5f}")
    return history

def train_gwo_chaos_time(nn, X, y, duration=TIME_LIMIT, n_agents=30, bounds=(-2, 2)):
    """Chaos GWO with Chaotic Decay"""
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
    print(f"    [Chaos GWO] Running for {duration}s...")
    
    while time.time() - start_time < duration:
        elapsed = time.time() - start_time
        progress = elapsed / duration
        
        # Chaotic Decay
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
        
        # Re-sort for update
        current_scores = []
        for i in range(n_agents):
             nn.set_params(wolves[i])
             current_scores.append(mse(y, nn.predict(X)))
        current_scores = np.array(current_scores)
        sorted_idx = np.argsort(current_scores)
        alpha = wolves[sorted_idx[0]].copy()
        beta = wolves[sorted_idx[1]].copy()
        delta = wolves[sorted_idx[2]].copy()
        alpha_score = current_scores[sorted_idx[0]]
        
        if it % 10 == 0:
            history.append(alpha_score)
        it += 1
        
    nn.set_params(alpha)
    print(f"    [Chaos GWO] Finished. Iters: {it}. Final MSE: {alpha_score:.5f}")
    return history

def train_hybrid_standard_time(nn, X, y, duration=TIME_LIMIT):
    """
    Hybrid (Mix):
    1. Standard Ray (duration/2)
    2. Standard GWO (duration/2)
    """
    print(f"    [Hybrid] Starting... Total Duration: {duration}s")
    hist1 = train_ray_standard_time(nn, X, y, duration=duration/2.0)
    
    # Pass the best from Ray to GWO is handled by object state (nn params are updated)
    # But GWO init needs to seed properly. 
    # train_gwo_standard_time uses 'wolves[0] = nn.get_params()', so it IS seeded. Correct.
    
    hist2 = train_gwo_standard_time(nn, X, y, duration=duration/2.0)
    return hist1 + hist2

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================
def run_experiment_long():
    os.makedirs(f'{OUTPUT_DIR}/figures', exist_ok=True)
    print("="*80)
    print(f"LONG DURATION EXPERIMENT ({DURATION_MINUTES} MINS PER ALGO)")
    print("1. Backprop (100k epochs)")
    print("2. Ray Shooting (Standard)")
    print("3. GWO Normal (Standard)")
    print("4. Chaos GWO")
    print("5. Hybrid (Standard Ray + Standard GWO)")
    print("="*80)
    
    stats = []
    
    for func_name in FUNCTIONS.keys():
        print(f"\n========================================")
        print(f"PROCESSING FUNCTION: {func_name}")
        print(f"========================================")
        
        np.random.seed(SEED)
        X_train, y_train, X_val, y_val = generate_dataset(func_name, N_SAMPLES, dim=3)
        y_min, y_max = y_train.min(), y_train.max()
        y_train_norm = (y_train - y_min) / (y_max - y_min + 1e-8)
        
        # 1. Backprop
        nn_bp = HeavisideNetwork3D(3, HIDDEN_SIZES, 1)
        hist_bp = train_backprop(nn_bp, X_train, y_train_norm, epochs=BACKPROP_EPOCHS)
        mse_bp = hist_bp[-1]
        
        # 2. Ray Standard
        nn_ray = HeavisideNetwork3D(3, HIDDEN_SIZES, 1)
        hist_ray = train_ray_standard_time(nn_ray, X_train, y_train_norm, duration=TIME_LIMIT)
        mse_ray = hist_ray[-1]
        
        # 3. GWO Normal
        nn_gwo_std = HeavisideNetwork3D(3, HIDDEN_SIZES, 1)
        hist_gwo_std = train_gwo_standard_time(nn_gwo_std, X_train, y_train_norm, duration=TIME_LIMIT)
        mse_gwo_std = hist_gwo_std[-1]
        
        # 4. Chaos GWO
        nn_gwo_chaos = HeavisideNetwork3D(3, HIDDEN_SIZES, 1)
        hist_gwo_chaos = train_gwo_chaos_time(nn_gwo_chaos, X_train, y_train_norm, duration=TIME_LIMIT)
        mse_gwo_chaos = hist_gwo_chaos[-1]
        
        # 5. Hybrid (Ray + GWO Normal)
        nn_hyb = HeavisideNetwork3D(3, HIDDEN_SIZES, 1)
        hist_hyb = train_hybrid_standard_time(nn_hyb, X_train, y_train_norm, duration=TIME_LIMIT)
        mse_hyb = hist_hyb[-1]
        
        stats.append({
            'func': func_name,
            'BP': mse_bp,
            'Ray': mse_ray,
            'GWO_Std': mse_gwo_std,
            'GWO_Chaos': mse_gwo_chaos,
            'Hybrid': mse_hyb
        })
        
        # Plot
        plt.figure(figsize=(12,8))
        plt.plot(hist_bp, label='Backprop (100k)', color='red', alpha=0.3)
        plt.plot(hist_ray, label='Ray Std (30m)', color='green')
        plt.plot(hist_gwo_std, label='GWO Std (30m)', color='orange')
        plt.plot(hist_gwo_chaos, label='Chaos GWO (30m)', color='blue')
        plt.plot(hist_hyb, label='Hybrid (30m)', color='purple')
        plt.yscale('log')
        plt.title(f"30-Minute Run: {func_name}")
        plt.xlabel("Iterations (Approx)")
        plt.ylabel("Training MSE (Log Scale)")
        plt.legend()
        plt.savefig(f"{OUTPUT_DIR}/figures/{func_name}_30min.png")
        plt.close()
        
    print("\n" + "="*80)
    print(f"{'Func':<10} {'BP':<10} {'Ray':<10} {'GWO_Std':<10} {'GWO_Chaos':<10} {'Hybrid':<10}")
    print("-" * 80)
    for s in stats:
        print(f"{s['func']:<10} {s['BP']:<10.4f} {s['Ray']:<10.4f} {s['GWO_Std']:<10.4f} {s['GWO_Chaos']:<10.4f} {s['Hybrid']:<10.4f}")

if __name__ == "__main__":
    run_experiment_long()

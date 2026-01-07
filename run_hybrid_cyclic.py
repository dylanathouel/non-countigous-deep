"""
RUN EXPERIMENT - CYCLIC HYBRID (Optimized)
==========================================
Implements "Interleaved Hybrid":
Alternates between Ray Shooting (60s) and GWO Standard (60s) repeatedly.
This forces GWO to "sprint" (resetting decay) multiple times, avoiding dilution.
"""
import numpy as np
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
TOTAL_DURATION = 60 # 30 minutes
CYCLE_DURATION = 15 # 60 seconds per algo per cycle

N_SAMPLES = 2000
OUTPUT_DIR = "results/3d_cyclic"

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
# CYCLIC ALGO
# ============================================================================

def train_ray_burst(nn, X, y, duration=60, steps=50, bounds=(-2, 2)):
    """Run Ray for a short burst"""
    dim = nn.num_params()
    best = nn.get_params()
    best_score = mse(y, nn.predict(X))
    
    start_time = time.time()
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
                
    nn.set_params(best)
    return best_score

def train_gwo_burst(nn, X, y, duration=60, n_agents=30, bounds=(-2, 2)):
    """Run Standard GWO for a short burst (Resetting Decay)"""
    dim = nn.num_params()
    wolves = np.random.uniform(bounds[0], bounds[1], (n_agents, dim))
    wolves[0] = nn.get_params() # Seed with current best (Crucial)
    
    # Init Pack
    scores = []
    for w in wolves:
        nn.set_params(w)
        scores.append(mse(y, nn.predict(X)))
    scores = np.array(scores)
    
    sorted_idx = np.argsort(scores)
    alpha, beta, delta = wolves[sorted_idx[0]].copy(), wolves[sorted_idx[1]].copy(), wolves[sorted_idx[2]].copy()
    alpha_score = scores[sorted_idx[0]]
    
    start_time = time.time()
    
    while time.time() - start_time < duration:
        elapsed = time.time() - start_time
        progress = elapsed / duration
        # Fast decay sequence: 2 -> 0 in 'duration' seconds
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
        
        # Update Alpha
        for i in range(n_agents):
            nn.set_params(wolves[i])
            score = mse(y, nn.predict(X))
            if score < alpha_score:
                alpha_score = score
                alpha = wolves[i].copy()
                beta = wolves[sorted_idx[0]].copy() # Shift roles roughly? 
                # Strict GWO checks all 3. Simplified here for burst speed.
                
    nn.set_params(alpha)
    return alpha_score

def train_cyclic_hybrid(nn, X, y, total_duration, cycle_duration=60):
    start_total = time.time()
    best_score = mse(y, nn.predict(X))
    history = [best_score]
    
    cycle = 1
    while time.time() - start_total < total_duration:
        remaining = total_duration - (time.time() - start_total)
        if remaining <= 0: break
        
        # Phase 1: Ray Shooting
        t_ray = min(cycle_duration, remaining)
        print(f"  [Cycle {cycle}] Ray Burst ({t_ray:.0f}s)...", end=" ", flush=True)
        score_ray = train_ray_burst(nn, X, y, duration=t_ray)
        print(f"MSE: {score_ray:.5f}")
        history.append(score_ray)
        
        remaining = total_duration - (time.time() - start_total)
        if remaining <= 0: break
        
        # Phase 2: GWO Standard
        t_gwo = min(cycle_duration, remaining)
        print(f"  [Cycle {cycle}] GWO Burst ({t_gwo:.0f}s)...", end=" ", flush=True)
        score_gwo = train_gwo_burst(nn, X, y, duration=t_gwo)
        print(f"MSE: {score_gwo:.5f}")
        history.append(score_gwo)
        
        cycle += 1
        
    return history

def run_experiment_cyclic():
    print("="*80)
    print(f"CYCLIC HYBRID EXPERIMENT")
    print(f"Total Time: {TOTAL_DURATION}s ({TOTAL_DURATION/60} mins)")
    print(f"Strategy: Ray ({CYCLE_DURATION}s) <-> GWO ({CYCLE_DURATION}s)")
    print("="*80)
    
    stats = []
    
    for func_name in FUNCTIONS.keys():
        print(f"\n>>> Function: {func_name}")
        np.random.seed(SEED)
        X_train, y_train, X_val, y_val = generate_dataset(func_name, N_SAMPLES, dim=3)
        y_min, y_max = y_train.min(), y_train.max()
        y_train_norm = (y_train - y_min) / (y_max - y_min + 1e-8)
        
        nn = HeavisideNetwork3D(3, HIDDEN_SIZES, 1)
        history = train_cyclic_hybrid(nn, X_train, y_train_norm, total_duration=TOTAL_DURATION, cycle_duration=CYCLE_DURATION)
        
        stats.append((func_name, history[-1]))
        
    print("\n" + "="*80)
    print(f"{'Func':<10} {'Final MSE':<15}")
    print("-" * 80)
    for func, mse_val in stats:
        print(f"{func:<10} {mse_val:<15.4f}")

if __name__ == "__main__":
    run_experiment_cyclic()

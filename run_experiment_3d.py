"""
RUN EXPERIMENT 3D - TARGET BASED STOPPING (TRAIN MSE)
=====================================================
1. Run Backprop for 100,000 epochs (establish baseline).
2. Calculate Target = Backprop_Train_MSE / 2.
3. Run Ray/GWO/Hybrid UNTIL Train_MSE <= Target.
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
BACKPROP_EPOCHS = 100000
BACKPROP_LR = 0.01

# Safety limits (to avoid true infinite loops if target is impossible)
SAFETY_MAX_RAYS = 50000
SAFETY_MAX_GWO_ITER = 10000
SAFETY_MAX_HYBRID_RAY = 20000
SAFETY_MAX_HYBRID_GWO = 5000

# Dataset
N_SAMPLES = 2000
OUTPUT_DIR = "results/3d_target_based"

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
# ALGORITHMS (Modified for Target Stopping)
# ============================================================================
def train_backprop(nn, X, y, epochs=100000, lr=0.01):
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

def train_ray_target(nn, X, y, target_mse, max_rays=SAFETY_MAX_RAYS, steps=50, bounds=(-2, 2)):
    """Run Ray Shooting UNTIL Train MSE <= target_mse"""
    dim = nn.num_params()
    best = nn.get_params()
    best_score = mse(y, nn.predict(X))
    history = [best_score]
    
    for ray in range(max_rays):
        if best_score <= target_mse:
            break
            
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
                break # New best found
        history.append(best_score)
        
    nn.set_params(best)
    return history

def train_gwo_target(nn, X, y, target_mse, max_iter=SAFETY_MAX_GWO_ITER, n_agents=30, bounds=(-2, 2)):
    """Run GWO UNTIL Train MSE <= target_mse"""
    dim = nn.num_params()
    wolves = np.random.uniform(bounds[0], bounds[1], (n_agents, dim))
    wolves[0] = nn.get_params() # Expect to be seeded
    
    scores = []
    for w in wolves:
        nn.set_params(w)
        scores.append(mse(y, nn.predict(X)))
    scores = np.array(scores)
    
    sorted_idx = np.argsort(scores)
    alpha, beta, delta = wolves[sorted_idx[0]].copy(), wolves[sorted_idx[1]].copy(), wolves[sorted_idx[2]].copy()
    alpha_score = scores[sorted_idx[0]]
    history = [alpha_score]
    
    for it in range(max_iter):
        if alpha_score <= target_mse:
            break
            
        a = 2 - 2 * (it / max_iter) 
        
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
        
    nn.set_params(alpha)
    return history

def train_hybrid_target(nn, X, y, target_mse):
    """Hybrid: Ray until target; if not, GWO until target"""
    hist1 = train_ray_target(nn, X, y, target_mse, max_rays=SAFETY_MAX_HYBRID_RAY, steps=50)
    if hist1[-1] <= target_mse:
        return hist1
    
    hist2 = train_gwo_target(nn, X, y, target_mse, max_iter=SAFETY_MAX_HYBRID_GWO, n_agents=30)
    return hist1 + hist2

# ============================================================================
# MAIN LOOP
# ============================================================================
def run_experiment():
    os.makedirs(f'{OUTPUT_DIR}/figures', exist_ok=True)
    print("="*80)
    print("TARGET-BASED STOPPING EXPERIMENT (Goal: Train MSE <= BP_Train / 2)")
    print(f"1. Run Backprop ({BACKPROP_EPOCHS} epochs)")
    print("2. Run others until MSE <= Target")
    print("="*80)
    
    stats = []
    
    for func_name in FUNCTIONS.keys():
        print(f"\n>>> Function: {func_name}")
        
        # Prepare Data
        np.random.seed(SEED)
        X_train, y_train, X_val, y_val = generate_dataset(func_name, N_SAMPLES, dim=3)
        y_min, y_max = y_train.min(), y_train.max()
        y_train_norm = (y_train - y_min) / (y_max - y_min + 1e-8)
        y_val_norm = (y_val - y_min) / (y_max - y_min + 1e-8)
        
        histories = {}
        
        # 1. Backprop
        print("  Running Backprop (100k)...", end=" ", flush=True)
        nn_bp = HeavisideNetwork3D(3, HIDDEN_SIZES, 1)
        start = time.time()
        hist_bp = train_backprop(nn_bp, X_train, y_train_norm, epochs=BACKPROP_EPOCHS)
        time_bp = time.time() - start
        
        mse_bp_train = mse(y_train_norm, nn_bp.predict(X_train))
        mse_bp_val = mse(y_val_norm, nn_bp.predict(X_val))
        
        print(f"MSE_Train={mse_bp_train:.5f}, MSE_Val={mse_bp_val:.5f} ({time_bp:.1f}s)")
        histories['backprop'] = hist_bp
        
        # TARGET
        target_mse = mse_bp_train / 2.0
        print(f"  [TARGET TRAIN MSE: {target_mse:.5f}]")
        
        # 2. GWO
        print("  Running GWO until target...", end=" ", flush=True)
        nn_gwo = HeavisideNetwork3D(3, HIDDEN_SIZES, 1)
        start = time.time()
        hist_gwo = train_gwo_target(nn_gwo, X_train, y_train_norm, target_mse)
        time_gwo = time.time() - start
        
        mse_gwo_train = hist_gwo[-1]
        iters_gwo = len(hist_gwo)
        success_gwo = "✓" if mse_gwo_train <= target_mse else "FAIL (limit)"
        print(f"MSE_Train={mse_gwo_train:.5f} ({iters_gwo} iters, {time_gwo:.1f}s) {success_gwo}")
        histories['gwo'] = hist_gwo
        
        # 3. Ray
        print("  Running Ray until target...", end=" ", flush=True)
        nn_ray = HeavisideNetwork3D(3, HIDDEN_SIZES, 1)
        start = time.time()
        hist_ray = train_ray_target(nn_ray, X_train, y_train_norm, target_mse)
        time_ray = time.time() - start
        
        mse_ray_train = hist_ray[-1]
        iters_ray = len(hist_ray)
        success_ray = "✓" if mse_ray_train <= target_mse else "FAIL (limit)"
        print(f"MSE_Train={mse_ray_train:.5f} ({iters_ray} iters, {time_ray:.1f}s) {success_ray}")
        histories['ray'] = hist_ray
        
        # 4. Hybrid
        print("  Running Hybrid until target...", end=" ", flush=True)
        nn_hyb = HeavisideNetwork3D(3, HIDDEN_SIZES, 1)
        start = time.time()
        hist_hyb = train_hybrid_target(nn_hyb, X_train, y_train_norm, target_mse)
        time_hyb = time.time() - start
        
        mse_hyb_train = hist_hyb[-1]
        iters_hyb = len(hist_hyb)
        success_hyb = "✓" if mse_hyb_train <= target_mse else "FAIL (limit)"
        print(f"MSE_Train={mse_hyb_train:.5f} ({iters_hyb} iters, {time_hyb:.1f}s) {success_hyb}")
        histories['hybrid'] = hist_hyb
        
        stats.append({
            'func': func_name, 
            'bp': mse_bp_train, 
            'target': target_mse,
            'gwo': mse_gwo_train, 
            'ray': mse_ray_train, 
            'hyb': mse_hyb_train
        })
        
        # Plot
        plt.figure(figsize=(10,6))
        plt.plot(hist_bp, color='red', label='Backprop (100k)', alpha=0.5)
        plt.plot(hist_gwo, color='blue', label='GWO (Target)', linewidth=2)
        plt.plot(hist_ray, color='green', label='Ray (Target)', linewidth=2)
        plt.plot(hist_hyb, color='purple', label='Hybrid (Target)', linewidth=2)
        plt.axhline(y=target_mse, color='black', linestyle='--', label='Target')
        plt.title(f"{func_name}: Target Based Stopping (Training MSE)")
        plt.yscale('log')
        plt.ylabel('Training MSE')
        plt.xlabel('Iterations')
        plt.legend()
        plt.savefig(f"{OUTPUT_DIR}/figures/{func_name}_target.png")
        plt.close()
        
    print("\n" + "="*80)
    print(f"{'Func':<10} {'BP_Train':<10} {'Target':<10} {'GWO':<10} {'Ray':<10} {'Hybrid':<10} {'Status':<10}")
    print("-" * 80)
    for s in stats:
        success = "PASS" if (s['gwo'] <= s['target'] + 1e-6 and s['ray'] <= s['target'] + 1e-6) else "SOME FAIL"
        print(f"{s['func']:<10} {s['bp']:<10.4f} {s['target']:<10.4f} {s['gwo']:<10.4f} {s['ray']:<10.4f} {s['hyb']:<10.4f} {success:<10}")

if __name__ == "__main__":
    run_experiment()

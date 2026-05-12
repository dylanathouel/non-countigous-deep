"""
RUN EXPERIMENT 3D - ADVANCED ALGORITHMS (Ray++, GWO++, Hybrid++)
================================================================
Implements vectorized optimization and chaotic dynamics for speed/convergence.
Target: Beat Backprop Train MSE / 2.
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

# Safety limits
SAFETY_TICKS = 10000 # Generic "ticks" for loops
TIME_LIMIT_PER_ALGO = 300 # 5 minutes max per algo

N_SAMPLES = 2000
OUTPUT_DIR = "results/3d_advanced"

# ============================================================================
# NEURAL NETWORK (Vectorized for Batch Evaluation)
# ============================================================================
class HeavisideNetwork3D:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.shapes = []
        for i in range(len(self.layer_sizes)-1):
            n_in, n_out = self.layer_sizes[i], self.layer_sizes[i+1]
            self.shapes.append((n_in, n_out))
        
        # Initialize one set of weights
        self.layers = []
        for n_in, n_out in self.shapes:
            limit = np.sqrt(6.0 / (n_in + n_out))
            W = np.random.uniform(-limit, limit, (n_in, n_out))
            b = np.zeros((1, n_out))
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
        for (n_in, n_out) in self.shapes:
            w_size = n_in * n_out
            b_size = n_out
            W = vec[idx:idx+w_size].reshape((n_in, n_out))
            idx += w_size
            b = vec[idx:idx+b_size].reshape((1, n_out))
            idx += b_size
            new_layers.append((W, b))
        self.layers = new_layers
    
    def num_params(self):
        return sum(n_in*n_out + n_out for n_in, n_out in self.shapes)

    def evaluate_batch(self, X, param_batch):
        """
        Evaluate K networks in parallel.
        X: (N, 3)
        param_batch: (K, num_params)
        Returns: (K,) MSE scores
        """
        K = param_batch.shape[0]
        N = X.shape[0]
        
        # We need to perform batch matmuls: (K, N, In) @ (K, In, Out) -> (K, N, Out)
        # First, restructure params into batch of weights
        idx = 0
        
        # X is shared, broadcast it to (K, N, 3)
        # Actually easier: A = X (N, 3). 
        # Layer 1 W: (K, 3, H1). 
        # A @ W[0]... No.
        # Efficient way: Use Einstein summation or loop if K is small.
        # Given complexity, simple loop over K with manually setting params is slow in Python.
        # Optimized: 
        # A: (K, N, In) initial = tile X
        A = np.tile(X, (K, 1, 1)) # (K, N, 3)
        
        for i, (n_in, n_out) in enumerate(self.shapes):
            w_size = n_in * n_out
            b_size = n_out
            
            # Extract W and b for all K
            W_K = param_batch[:, idx:idx+w_size].reshape(K, n_in, n_out)
            idx += w_size
            b_K = param_batch[:, idx:idx+b_size].reshape(K, 1, n_out)
            idx += b_size
            
            # Batch Matmul: (K, N, In) @ (K, In, Out) -> (K, N, Out)
            Z = np.matmul(A, W_K) + b_K
            
            # Activation
            if i < len(self.shapes) - 1:
                A = np.where(Z >= 0, 1.0, 0.0) # Heaviside
            else:
                A = Z # Linear output
        
        # A is now (K, N, 1). 
        # Calculate MSE against y (N, 1)
        # Need to supply y from somewhere. 
        # Pass y to this function? For now this returns predictions?
        # Let's adjust signature
        return A # (K, N, 1)


def batch_mse(preds, y):
    # preds: (K, N, 1), y: (N, 1)
    diff = preds - y # (K, N, 1) via broadcast
    return np.mean(diff**2, axis=(1, 2)) # Average over N and output_dim -> (K,)

# ============================================================================
# ALGORITHMS ++
# ============================================================================

def train_backprop(nn, X, y, epochs=100000, lr=0.01):
    # Standard Backprop (unchanged)
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
        loss = np.mean((y - y_pred)**2)
        history.append(loss)
        
        m = X.shape[0]
        dZ_out = 2 * (y_pred - y) / m
        dW_out = activations[-1].T @ dZ_out
        db_out = np.sum(dZ_out, axis=0, keepdims=True)
        W_out_new = W_out - lr * dW_out
        b_out_new = b_out - lr * db_out
        nn.layers[-1] = (W_out_new, b_out_new)
    return history

def train_ray_multi(nn, X, y, target_mse, max_batches=SAFETY_TICKS, batch_size=100, bounds=(-2, 2)):
    """
    Ray++: Vectorized Ray Shooting. 
    Evaluates 'batch_size' rays in parallel in one Python step.
    """
    dim = nn.num_params()
    best_params = nn.get_params()
    # Initial score
    preds = nn.evaluate_batch(X, best_params.reshape(1, -1))
    best_score = batch_mse(preds, y)[0]
    history = [best_score]
    
    start_time = time.time()
    
    for batch in range(max_batches):
        if best_score <= target_mse:
            break
        if time.time() - start_time > TIME_LIMIT_PER_ALGO:
            break
            
        # Generate Batch of Targets
        # (B, dim)
        targets = np.random.uniform(bounds[0], bounds[1], (batch_size, dim))
        
        # Directions: target - best
        # best is (dim,). targets (B, dim).
        directions = targets - best_params
        
        # To simplify, we only check one step per ray? 
        # Or vectorized checks of X steps?
        # Let's check the endpoint (t=1) and maybe midpoint (t=0.5)? 
        # Standard Ray checks 50 steps.
        # Vectorized 50 steps * 100 rays = 5000 evals. Heavy.
        # Let's check random t in [0, 1] for each ray?
        # Or just check the targets directly? (Random Search).
        # Ray Shooting logic: search along line.
        # Let's do: Generate 100 random directions. For each, evaluate 10 steps.
        # Total 1000 candidates.
        
        steps = 10
        candidates = []
        for s in range(steps):
             t = (s + 1) / steps
             # Shape (B, dim)
             c = best_params + t * directions
             c = np.clip(c, bounds[0], bounds[1])
             candidates.append(c)
        
        # Stack all candidates: (B*steps, dim)
        all_candidates = np.vstack(candidates) 
        
        # Evaluate all
        preds_batch = nn.evaluate_batch(X, all_candidates)
        scores = batch_mse(preds_batch, y)
        
        min_idx = np.argmin(scores)
        min_score = scores[min_idx]
        
        if min_score < best_score:
            best_score = min_score
            best_params = all_candidates[min_idx]
            
        history.append(best_score)
        
    nn.set_params(best_params)
    return history

def train_gwo_chaos(nn, X, y, target_mse, max_iter=SAFETY_TICKS, n_agents=30, bounds=(-2, 2)):
    """
    GWO++: Chaotic Decay.
    Uses non-linear parameter modulation to escape local optima.
    """
    dim = nn.num_params()
    wolves = np.random.uniform(bounds[0], bounds[1], (n_agents, dim))
    wolves[0] = nn.get_params() # Seed
    
    # Init scores
    # We can use evaluate_batch here too!
    preds = nn.evaluate_batch(X, wolves)
    scores = batch_mse(preds, y)
    
    sorted_idx = np.argsort(scores)
    alpha, beta, delta = wolves[sorted_idx[0]], wolves[sorted_idx[1]], wolves[sorted_idx[2]]
    alpha_score = scores[sorted_idx[0]]
    history = [alpha_score]
    
    start_time = time.time()
    
    for it in range(max_iter):
        if alpha_score <= target_mse:
            break
        if time.time() - start_time > TIME_LIMIT_PER_ALGO:
            break
            
        # Chaotic 'a' parameter
        # Normal GWO: a = 2 - 2 * (it/max_iter)
        # Chaos: Non-linear with sine wave modulation
        progress = it / max_iter
        a = 2 * (1 - progress**2) + 0.5 * np.sin(it * 0.1)
        a = np.clip(a, 0, 2)
        
        # Update wolves (Vectorized-ish?)
        # GWO update equation is per-dimension. Numpy handles this.
        # A1, C1... for all agents/dims?
        # Lets loop agents for clarity, logic is complex.
        
        # Vectorized Update:
        # r1, r2 (N, dim)
        r1 = np.random.rand(n_agents, dim)
        r2 = np.random.rand(n_agents, dim)
        A1 = 2*a*r1 - a
        C1 = 2*r2
        D_alpha = np.abs(C1 * alpha - wolves)
        X1 = alpha - A1 * D_alpha
        
        r1 = np.random.rand(n_agents, dim)
        r2 = np.random.rand(n_agents, dim)
        A2 = 2*a*r1 - a
        C2 = 2*r2
        D_beta = np.abs(C2 * beta - wolves)
        X2 = beta - A2 * D_beta
        
        r1 = np.random.rand(n_agents, dim)
        r2 = np.random.rand(n_agents, dim)
        A3 = 2*a*r1 - a
        C3 = 2*r2
        D_delta = np.abs(C3 * delta - wolves)
        X3 = delta - A3 * D_delta
        
        wolves = (X1 + X2 + X3) / 3.0
        wolves = np.clip(wolves, bounds[0], bounds[1])
        
        # Evaluate batch
        preds = nn.evaluate_batch(X, wolves)
        scores = batch_mse(preds, y)
        
        # Update Alphas
        # Simple Logic: Re-sort all
        current_best_idx = np.argmin(scores)
        current_best = scores[current_best_idx]
        
        if current_best < alpha_score:
            # We have a new global best
            alpha_score = current_best
            alpha = wolves[current_best_idx].copy()
            
        # Correct GWO logic: we need top 3 of population
        # But we also keep Alpha/Beta/Delta from previous separate?
        # Actually GWO usually keeps Alpha/Beta/Delta as distinct entities that guide the pack.
        # If a wolf becomes better than Alpha, it becomes Alpha.
        # Let's resort the population + old_alpha + old_beta + old_delta?
        # Simplification: Just take best 3 of current wolves + old leaders
        candidates = np.vstack([wolves, alpha, beta, delta])
        cand_preds = nn.evaluate_batch(X, candidates)
        cand_scores = batch_mse(cand_preds, y)
        
        sorted_full = np.argsort(cand_scores)
        alpha = candidates[sorted_full[0]].copy()
        beta = candidates[sorted_full[1]].copy()
        delta = candidates[sorted_full[2]].copy()
        alpha_score = cand_scores[sorted_full[0]]
        
        history.append(alpha_score)
        
    nn.set_params(alpha)
    return history

def train_hybrid_plus(nn, X, y, target_mse):
    """
    Hybrid++: Interleaved.
    Ray++ (fast explore) <-> GWO++ (fine tune).
    """
    best_score = np.inf
    history = []
    
    start_time = time.time()
    
    while best_score > target_mse:
        if time.time() - start_time > TIME_LIMIT_PER_ALGO:
            break
            
        # 1. Burst of Ray++
        h1 = train_ray_multi(nn, X, y, target_mse, max_batches=50, batch_size=50)
        history.extend(h1)
        if h1[-1] <= target_mse: break
        
        # 2. Burst of GWO++
        h2 = train_gwo_chaos(nn, X, y, target_mse, max_iter=50, n_agents=20)
        history.extend(h2)
        best_score = h2[-1]
        
    return history


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================
def run_experiment_advanced():
    os.makedirs(f'{OUTPUT_DIR}/figures', exist_ok=True)
    print("="*80)
    print("ADVANCED ALGORITHMS: Ray++, GWO++, Hybrid++")
    print("Goal: Train MSE <= Backprop Train / 2")
    print("="*80)
    
    stats = []
    
    for func_name in FUNCTIONS.keys():
        print(f"\n>>> Function: {func_name}")
        
        np.random.seed(SEED)
        X_train, y_train, X_val, y_val = generate_dataset(func_name, N_SAMPLES, dim=3)
        y_min, y_max = y_train.min(), y_train.max()
        y_train_norm = (y_train - y_min) / (y_max - y_min + 1e-8)
        y_val_norm = (y_val - y_min) / (y_max - y_min + 1e-8)
        
        # 1. Backprop (Baseline)
        print("  Running Backprop (100k)...", end=" ", flush=True)
        nn_bp = HeavisideNetwork3D(3, HIDDEN_SIZES, 1)
        start = time.time()
        hist_bp = train_backprop(nn_bp, X_train, y_train_norm, epochs=BACKPROP_EPOCHS)
        time_bp = time.time() - start
        
        mse_bp_train = hist_bp[-1]
        target_mse = mse_bp_train / 2.0
        print(f"MSE={mse_bp_train:.5f} ({time_bp:.1f}s) | Target={target_mse:.5f}")
        
        # 2. Ray++
        print("  Ray++ (Multi)...", end=" ", flush=True)
        nn_ray = HeavisideNetwork3D(3, HIDDEN_SIZES, 1)
        start = time.time()
        hist_ray = train_ray_multi(nn_ray, X_train, y_train_norm, target_mse)
        time_ray = time.time() - start
        mse_ray = hist_ray[-1]
        print(f"MSE={mse_ray:.5f} ({time_ray:.1f}s) {'✓' if mse_ray <= target_mse else 'FAIL'}")
        
        # 3. GWO++
        print("  GWO++ (Chaos)...", end=" ", flush=True)
        nn_gwo = HeavisideNetwork3D(3, HIDDEN_SIZES, 1)
        start = time.time()
        hist_gwo = train_gwo_chaos(nn_gwo, X_train, y_train_norm, target_mse)
        time_gwo = time.time() - start
        mse_gwo = hist_gwo[-1]
        print(f"MSE={mse_gwo:.5f} ({time_gwo:.1f}s) {'✓' if mse_gwo <= target_mse else 'FAIL'}")
        
        # 4. Hybrid++
        print("  Hybrid++ (Interleaved)...", end=" ", flush=True)
        nn_hyb = HeavisideNetwork3D(3, HIDDEN_SIZES, 1)
        start = time.time()
        hist_hyb = train_hybrid_plus(nn_hyb, X_train, y_train_norm, target_mse)
        time_hyb = time.time() - start
        mse_hyb = hist_hyb[-1]
        print(f"MSE={mse_hyb:.5f} ({time_hyb:.1f}s) {'✓' if mse_hyb <= target_mse else 'FAIL'}")
        
        stats.append({
            'func': func_name,
            'Ray++': mse_ray,
            'GWO++': mse_gwo,
            'Hyb++': mse_hyb,
            'Target': target_mse
        })
        
        # Plot
        plt.figure(figsize=(10,6))
        plt.plot(hist_bp, label='Backprop', color='red', alpha=0.3)
        plt.plot(hist_ray, label='Ray++', color='green')
        plt.plot(hist_gwo, label='GWO++', color='blue')
        plt.plot(hist_hyb, label='Hybrid++', color='purple')
        plt.axhline(target_mse, color='black', linestyle='--')
        plt.yscale('log')
        plt.title(f"Advanced Algorithms: {func_name}")
        plt.legend()
        plt.savefig(f"{OUTPUT_DIR}/figures/{func_name}_advanced.png")
        plt.close()

if __name__ == "__main__":
    run_experiment_advanced()

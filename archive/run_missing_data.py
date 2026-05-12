"""
RUN EXPERIMENT - MISSING DATA (Standard GWO 60s)
================================================
Runs Backprop and Standard GWO (60s) to fill the comparison table.
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
BACKPROP_EPOCHS = 100000
TIME_LIMIT = 60 # 60 seconds

N_SAMPLES = 2000

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
# ALGORITHMS
# ============================================================================
def train_backprop(nn, X, y, epochs=100000, lr=0.01):
    history = []
    print(f"    [Backprop] Starting {epochs} epochs...", end=" ", flush=True)
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
    
    print(f"Done in {time.time()-start_time:.1f}s. Final: {history[-1]:.5f}")
    return history

def train_gwo_standard_time(nn, X, y, duration=60, n_agents=30, bounds=(-2, 2)):
    """Normal GWO with Linear Decay for 60s"""
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
    print(f"    [GWO Standard] Running for {duration}s...", end=" ", flush=True)
    
    while time.time() - start_time < duration:
        elapsed = time.time() - start_time
        progress = elapsed / duration
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
        
        current_scores = []
        for i in range(n_agents):
             nn.set_params(wolves[i])
             current_scores.append(mse(y, nn.predict(X)))
        current_scores = np.array(current_scores)
        sorted_idx = np.argsort(current_scores)
        if current_scores[sorted_idx[0]] < alpha_score:
            alpha = wolves[sorted_idx[0]].copy()
            beta = wolves[sorted_idx[1]].copy()
            delta = wolves[sorted_idx[2]].copy()
            alpha_score = current_scores[sorted_idx[0]]
        
        history.append(alpha_score)
        it += 1
        
    nn.set_params(alpha)
    print(f"Done. Final: {alpha_score:.5f}")
    return history

def run_missing_data():
    print("="*60)
    print("FILLING MISSING DATA: GWO Standard (60s) & Backprop")
    print("="*60)
    
    stats = []
    
    for func_name in FUNCTIONS.keys():
        print(f"\n>>> Function: {func_name}")
        np.random.seed(SEED)
        X_train, y_train, X_val, y_val = generate_dataset(func_name, N_SAMPLES, dim=3)
        y_min, y_max = y_train.min(), y_train.max()
        y_train_norm = (y_train - y_min) / (y_max - y_min + 1e-8)
        
        # 1. Backprop
        nn_bp = HeavisideNetwork3D(3, HIDDEN_SIZES, 1)
        hist_bp = train_backprop(nn_bp, X_train, y_train_norm)
        mse_bp = hist_bp[-1]
        
        # 2. GWO Standard (60s)
        nn_gwo = HeavisideNetwork3D(3, HIDDEN_SIZES, 1)
        hist_gwo = train_gwo_standard_time(nn_gwo, X_train, y_train_norm, duration=TIME_LIMIT)
        mse_gwo = hist_gwo[-1]
        
        stats.append((func_name, mse_bp, mse_gwo))
        
    print("\n" + "="*60)
    print(f"{'Func':<10} {'BP_Observed':<15} {'GWO_Std_60s':<15}")
    print("-" * 60)
    for func, bp, gwo in stats:
        print(f"{func:<10} {bp:<15.4f} {gwo:<15.4f}")

if __name__ == "__main__":
    run_missing_data()

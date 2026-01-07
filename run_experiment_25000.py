"""
RUN EXPERIMENT 3D - 25000 ITERATIONS
=====================================
Tous les algorithmes exécutent exactement 25000 itérations:
1. Backpropagation: 25,000 epochs
2. Ray Shooting: 25,000 itérations (500 rays × 50 steps)
3. GWO Standard: 25,000 itérations (50 agents × 500 iterations)
4. Hybrid Cyclic: 25,000 itérations (5 cycles de Ray+GWO)
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
TOTAL_ITERATIONS = 25000

# Paramètres pour chaque algorithme (tous = ~25000 évaluations)
BACKPROP_EPOCHS = 25000  # 25000 points

# Ray Shooting: Enregistre après chaque step
# 500 rays × 50 steps = 25000 points
RAY_MAX_RAYS = 500
RAY_STEPS_PER_RAY = 50

# GWO: Enregistre après chaque agent × iteration
# 833 iterations × 30 agents = 24990 points ≈ 25000
GWO_AGENTS = 30
GWO_MAX_ITER = 833

# Hybrid Cyclic: 10 cycles
# Chaque cycle: (125 rays × 10 steps) + (25 agents × 50 iter) = 1250 + 1250 = 2500 points
# 10 cycles × 2500 = 25000 points
HYBRID_CYCLES = 10
HYBRID_RAY_PER_CYCLE = 125
HYBRID_RAY_STEPS = 10
HYBRID_GWO_AGENTS = 25
HYBRID_GWO_ITER = 50

N_SAMPLES = 2000
OUTPUT_DIR = "results/3d_25000"

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
# ALGORITHMS (Iteration-Based)
# ============================================================================

def train_backprop(nn, X, y, epochs=25000, lr=0.01):
    """Standard Backpropagation - Fixed Epochs"""
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

def train_ray_shooting(nn, X, y, max_rays=500, steps_per_ray=50, bounds=(-2, 2)):
    """Standard Ray Shooting - Fixed Iterations"""
    dim = nn.num_params()
    best = nn.get_params()
    best_score = mse(y, nn.predict(X))
    history = [best_score]

    for ray in range(max_rays):
        target = np.random.uniform(bounds[0], bounds[1], dim)
        direction = target - best

        for step in range(steps_per_ray):
            t = step / max(1, steps_per_ray - 1)
            candidate = best + t * direction
            candidate = np.clip(candidate, bounds[0], bounds[1])

            nn.set_params(candidate)
            score = mse(y, nn.predict(X))

            # Enregistrer après CHAQUE step pour avoir rays × steps points
            history.append(best_score if score >= best_score else score)

            if score < best_score:
                best = candidate.copy()
                best_score = score
                break

    nn.set_params(best)
    return history

def train_gwo_standard(nn, X, y, n_agents=50, max_iter=500, bounds=(-2, 2)):
    """Standard GWO - Fixed Iterations"""
    dim = nn.num_params()
    wolves = np.random.uniform(bounds[0], bounds[1], (n_agents, dim))
    wolves[0] = nn.get_params()  # Seed with current weights

    scores = []
    for w in wolves:
        nn.set_params(w)
        scores.append(mse(y, nn.predict(X)))
    scores = np.array(scores)

    sorted_idx = np.argsort(scores)
    alpha = wolves[sorted_idx[0]].copy()
    beta = wolves[sorted_idx[1]].copy()
    delta = wolves[sorted_idx[2]].copy()
    alpha_score = scores[sorted_idx[0]]
    history = [alpha_score]

    for it in range(max_iter):
        # Linear decay from 2 to 0
        a = 2 * (1 - it / max_iter)

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

        # Évaluer chaque agent et enregistrer après CHAQUE évaluation
        for i in range(n_agents):
            nn.set_params(wolves[i])
            score = mse(y, nn.predict(X))
            if score < alpha_score:
                delta, beta, alpha = beta.copy(), alpha.copy(), wolves[i].copy()
                alpha_score = score
            elif score < scores[sorted_idx[1]]:
                delta, beta = beta.copy(), wolves[i].copy()
            elif score < scores[sorted_idx[2]]:
                delta = wolves[i].copy()

            # Enregistrer après CHAQUE agent pour avoir max_iter × n_agents points
            history.append(alpha_score)

    nn.set_params(alpha)
    return history

def train_ray_burst_iter(nn, X, y, max_rays=50, steps=50, bounds=(-2, 2)):
    """Ray Shooting burst (for hybrid) - Returns history"""
    dim = nn.num_params()
    best = nn.get_params()
    best_score = mse(y, nn.predict(X))
    history = []

    for ray in range(max_rays):
        target = np.random.uniform(bounds[0], bounds[1], dim)
        direction = target - best

        for step in range(steps):
            t = step / max(1, steps - 1)
            candidate = best + t * direction
            candidate = np.clip(candidate, bounds[0], bounds[1])

            nn.set_params(candidate)
            score = mse(y, nn.predict(X))

            # Enregistrer après CHAQUE step
            history.append(best_score if score >= best_score else score)

            if score < best_score:
                best = candidate.copy()
                best_score = score
                break

    nn.set_params(best)
    return history

def train_gwo_burst_iter(nn, X, y, n_agents=50, max_iter=50, bounds=(-2, 2)):
    """GWO burst (for hybrid) - Returns history"""
    dim = nn.num_params()
    wolves = np.random.uniform(bounds[0], bounds[1], (n_agents, dim))
    wolves[0] = nn.get_params()  # Seed with current best

    # Initialize pack
    scores = []
    for w in wolves:
        nn.set_params(w)
        scores.append(mse(y, nn.predict(X)))
    scores = np.array(scores)

    sorted_idx = np.argsort(scores)
    alpha = wolves[sorted_idx[0]].copy()
    beta = wolves[sorted_idx[1]].copy()
    delta = wolves[sorted_idx[2]].copy()
    alpha_score = scores[sorted_idx[0]]
    history = []

    for it in range(max_iter):
        # Fast decay: 2 -> 0
        a = 2 * (1 - it / max_iter)

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

        # Update alpha, beta, delta and record after EACH agent
        for i in range(n_agents):
            nn.set_params(wolves[i])
            score = mse(y, nn.predict(X))
            if score < alpha_score:
                alpha_score = score
                alpha = wolves[i].copy()

            # Enregistrer après CHAQUE agent
            history.append(alpha_score)

    nn.set_params(alpha)
    return history

def train_hybrid_cyclic(nn, X, y, num_cycles=5,
                       ray_per_cycle=50, ray_steps=50,
                       gwo_agents=50, gwo_iter=50, bounds=(-2, 2)):
    """
    Hybrid Cyclic: Alternates Ray Shooting and GWO in cycles
    Each cycle: Ray burst + GWO burst (resetting decay each time)
    Total iterations: num_cycles × (ray_per_cycle × ray_steps + gwo_agents × gwo_iter)
    """
    history = []
    best_score = mse(y, nn.predict(X))
    history.append(best_score)

    for cycle in range(num_cycles):
        # Phase 1: Ray Shooting burst - retourne un historique complet
        hist_ray = train_ray_burst_iter(nn, X, y,
                                        max_rays=ray_per_cycle,
                                        steps=ray_steps,
                                        bounds=bounds)
        history.extend(hist_ray)  # Concaténer tous les points

        # Phase 2: GWO burst (restarts with fresh decay) - retourne un historique complet
        hist_gwo = train_gwo_burst_iter(nn, X, y,
                                        n_agents=gwo_agents,
                                        max_iter=gwo_iter,
                                        bounds=bounds)
        history.extend(hist_gwo)  # Concaténer tous les points

    return history

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================
def run_experiment_25000():
    os.makedirs(f'{OUTPUT_DIR}/figures', exist_ok=True)
    print("="*80)
    print("EXPERIMENT: ~25000 EVALUATIONS FOR ALL ALGORITHMS")
    print("="*80)
    print(f"1. Backpropagation: {BACKPROP_EPOCHS} epochs = {BACKPROP_EPOCHS} points")
    print(f"2. Ray Shooting: {RAY_MAX_RAYS} rays × {RAY_STEPS_PER_RAY} steps = {RAY_MAX_RAYS * RAY_STEPS_PER_RAY} points")
    print(f"3. GWO Standard: {GWO_MAX_ITER} iter × {GWO_AGENTS} agents = {GWO_MAX_ITER * GWO_AGENTS} points")
    print(f"4. Hybrid Cyclic: {HYBRID_CYCLES} cycles")
    print(f"   Per cycle: {HYBRID_RAY_PER_CYCLE}×{HYBRID_RAY_STEPS} Ray + {HYBRID_GWO_ITER}×{HYBRID_GWO_AGENTS} GWO = {HYBRID_RAY_PER_CYCLE*HYBRID_RAY_STEPS + HYBRID_GWO_ITER*HYBRID_GWO_AGENTS} points")
    print(f"   Total: {HYBRID_CYCLES * (HYBRID_RAY_PER_CYCLE*HYBRID_RAY_STEPS + HYBRID_GWO_ITER*HYBRID_GWO_AGENTS)} points")
    print("="*80)

    stats = []

    for func_name in FUNCTIONS.keys():
        print(f"\n>>> Function: {func_name}")

        np.random.seed(SEED)
        X_train, y_train, X_val, y_val = generate_dataset(func_name, N_SAMPLES, dim=3)
        y_min, y_max = y_train.min(), y_train.max()
        y_train_norm = (y_train - y_min) / (y_max - y_min + 1e-8)
        y_val_norm = (y_val - y_min) / (y_max - y_min + 1e-8)

        # 1. Backpropagation
        print(f"  Backprop ({BACKPROP_EPOCHS} epochs)...", end=" ", flush=True)
        nn_bp = HeavisideNetwork3D(3, HIDDEN_SIZES, 1)
        start = time.time()
        hist_bp = train_backprop(nn_bp, X_train, y_train_norm, epochs=BACKPROP_EPOCHS)
        time_bp = time.time() - start
        mse_bp = hist_bp[-1]
        print(f"MSE={mse_bp:.5f} ({time_bp:.1f}s)")

        # 2. Ray Shooting
        print(f"  Ray Shooting ({RAY_MAX_RAYS} rays × {RAY_STEPS_PER_RAY} steps)...", end=" ", flush=True)
        nn_ray = HeavisideNetwork3D(3, HIDDEN_SIZES, 1)
        start = time.time()
        hist_ray = train_ray_shooting(nn_ray, X_train, y_train_norm,
                                      max_rays=RAY_MAX_RAYS,
                                      steps_per_ray=RAY_STEPS_PER_RAY)
        time_ray = time.time() - start
        mse_ray = hist_ray[-1]
        print(f"MSE={mse_ray:.5f} ({time_ray:.1f}s)")

        # 3. GWO Standard
        print(f"  GWO Standard ({GWO_MAX_ITER} iter × {GWO_AGENTS} agents)...", end=" ", flush=True)
        nn_gwo = HeavisideNetwork3D(3, HIDDEN_SIZES, 1)
        start = time.time()
        hist_gwo = train_gwo_standard(nn_gwo, X_train, y_train_norm,
                                     n_agents=GWO_AGENTS,
                                     max_iter=GWO_MAX_ITER)
        time_gwo = time.time() - start
        mse_gwo = hist_gwo[-1]
        print(f"MSE={mse_gwo:.5f} ({time_gwo:.1f}s)")

        # 4. Hybrid Cyclic
        print(f"  Hybrid Cyclic ({HYBRID_CYCLES} cycles)...", end=" ", flush=True)
        nn_hyb = HeavisideNetwork3D(3, HIDDEN_SIZES, 1)
        start = time.time()
        hist_hyb = train_hybrid_cyclic(nn_hyb, X_train, y_train_norm,
                                      num_cycles=HYBRID_CYCLES,
                                      ray_per_cycle=HYBRID_RAY_PER_CYCLE,
                                      ray_steps=HYBRID_RAY_STEPS,
                                      gwo_agents=HYBRID_GWO_AGENTS,
                                      gwo_iter=HYBRID_GWO_ITER)
        time_hyb = time.time() - start
        mse_hyb = hist_hyb[-1]
        print(f"MSE={mse_hyb:.5f} ({time_hyb:.1f}s)")

        stats.append({
            'func': func_name,
            'BP': mse_bp,
            'Ray': mse_ray,
            'GWO': mse_gwo,
            'Hybrid_Cyclic': mse_hyb,
            'Time_BP': time_bp,
            'Time_Ray': time_ray,
            'Time_GWO': time_gwo,
            'Time_Hyb': time_hyb
        })

        # Plot convergence curves
        plt.figure(figsize=(12,7))
        plt.plot(hist_bp, label=f'Backprop (25000 epochs)', color='red', alpha=0.7, linewidth=2)
        plt.plot(hist_ray, label=f'Ray Shooting (500×50 = 25000)', color='green', linewidth=2)
        plt.plot(hist_gwo, label=f'GWO (833×30 ≈ 25000)', color='blue', linewidth=2)
        plt.plot(hist_hyb, label=f'Hybrid Cyclic (10 cycles = 25000)', color='purple', linewidth=2)
        plt.yscale('log')
        plt.xlabel('Evaluations')
        plt.ylabel('MSE (log scale)')
        plt.title(f"Comparison (~25000 evaluations each): {func_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{OUTPUT_DIR}/figures/{func_name}_25000.png", dpi=150, bbox_inches='tight')
        plt.close()

    # Summary table
    print("\n" + "="*100)
    print(f"{'Function':<12} {'BP MSE':<12} {'Ray MSE':<12} {'GWO MSE':<12} {'Hyb MSE':<12} {'Best':<12}")
    print("-" * 100)
    for s in stats:
        results = [s['BP'], s['Ray'], s['GWO'], s['Hybrid_Cyclic']]
        best_algo = ['BP', 'Ray', 'GWO', 'Hybrid'][np.argmin(results)]
        print(f"{s['func']:<12} {s['BP']:<12.4f} {s['Ray']:<12.4f} {s['GWO']:<12.4f} {s['Hybrid_Cyclic']:<12.4f} {best_algo:<12}")

    # Timing summary
    print("\n" + "="*100)
    print("TIMING SUMMARY")
    print("-" * 100)
    print(f"{'Function':<12} {'BP Time':<12} {'Ray Time':<12} {'GWO Time':<12} {'Hyb Time':<12}")
    print("-" * 100)
    for s in stats:
        print(f"{s['func']:<12} {s['Time_BP']:<12.1f} {s['Time_Ray']:<12.1f} {s['Time_GWO']:<12.1f} {s['Time_Hyb']:<12.1f}")

    # Overall statistics
    print("\n" + "="*100)
    print("OVERALL STATISTICS (Average across all functions)")
    print("-" * 100)
    avg_bp = np.mean([s['BP'] for s in stats])
    avg_ray = np.mean([s['Ray'] for s in stats])
    avg_gwo = np.mean([s['GWO'] for s in stats])
    avg_hyb = np.mean([s['Hybrid_Cyclic'] for s in stats])

    print(f"Backpropagation:  {avg_bp:.4f}")
    print(f"Ray Shooting:     {avg_ray:.4f}")
    print(f"GWO Standard:     {avg_gwo:.4f}")
    print(f"Hybrid Cyclic:    {avg_hyb:.4f}")

    best_overall = min(avg_bp, avg_ray, avg_gwo, avg_hyb)
    best_name = ['Backpropagation', 'Ray Shooting', 'GWO Standard', 'Hybrid Cyclic'][[avg_bp, avg_ray, avg_gwo, avg_hyb].index(best_overall)]
    print(f"\nBest overall: {best_name} with average MSE = {best_overall:.4f}")
    print("="*100)

if __name__ == "__main__":
    run_experiment_25000()

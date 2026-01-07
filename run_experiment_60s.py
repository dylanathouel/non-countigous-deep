"""
RUN EXPERIMENT 3D - 60 SECONDS TIME LIMIT
==========================================
Tous les algorithmes s'exécutent pendant exactement 60 secondes:
1. Backpropagation: 60 secondes
2. Ray Shooting: 60 secondes
3. GWO Standard: 60 secondes
4. Hybrid Cyclic: 60 secondes (alternance Ray/GWO)

Chaque évaluation est enregistrée pour un graphique détaillé.
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
TIME_LIMIT = 60  # Secondes par algorithme

# Paramètres pour les algorithmes
RAY_STEPS_PER_RAY = 50
GWO_AGENTS = 30
HYBRID_CYCLE_DURATION = 15  # Secondes par cycle (Ray + GWO)

N_SAMPLES = 2000
OUTPUT_DIR = "results/3d_60s"

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

def train_backprop_time(nn, X, y, duration=60, lr=0.01):
    """Backpropagation with time limit"""
    history = []
    start_time = time.time()
    epoch = 0

    while time.time() - start_time < duration:
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
        epoch += 1

    return history, epoch

def train_ray_shooting_time(nn, X, y, duration=60, steps_per_ray=50, bounds=(-2, 2)):
    """Ray Shooting with time limit - enregistre chaque step"""
    dim = nn.num_params()
    best = nn.get_params()
    best_score = mse(y, nn.predict(X))
    history = [best_score]

    start_time = time.time()
    ray_count = 0

    while time.time() - start_time < duration:
        target = np.random.uniform(bounds[0], bounds[1], dim)
        direction = target - best

        for step in range(steps_per_ray):
            if time.time() - start_time >= duration:
                break

            t = step / max(1, steps_per_ray - 1)
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

        ray_count += 1

    nn.set_params(best)
    return history, ray_count

def train_gwo_standard_time(nn, X, y, duration=60, n_agents=30, bounds=(-2, 2)):
    """GWO with time limit - enregistre chaque évaluation d'agent"""
    dim = nn.num_params()
    wolves = np.random.uniform(bounds[0], bounds[1], (n_agents, dim))
    wolves[0] = nn.get_params()

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

    start_time = time.time()
    iteration = 0

    while time.time() - start_time < duration:
        elapsed = time.time() - start_time
        progress = min(elapsed / duration, 1.0)
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

        # Évaluer chaque agent et enregistrer après CHAQUE évaluation
        for i in range(n_agents):
            if time.time() - start_time >= duration:
                break

            nn.set_params(wolves[i])
            score = mse(y, nn.predict(X))
            if score < alpha_score:
                delta, beta, alpha = beta.copy(), alpha.copy(), wolves[i].copy()
                alpha_score = score
            elif score < scores[sorted_idx[1]]:
                delta, beta = beta.copy(), wolves[i].copy()
            elif score < scores[sorted_idx[2]]:
                delta = wolves[i].copy()

            # Enregistrer après CHAQUE agent
            history.append(alpha_score)

        iteration += 1

        if time.time() - start_time >= duration:
            break

    nn.set_params(alpha)
    return history, iteration

def train_ray_burst_time(nn, X, y, duration=10, steps=50, bounds=(-2, 2)):
    """Ray Shooting burst (for hybrid) - time limited"""
    dim = nn.num_params()
    best = nn.get_params()
    best_score = mse(y, nn.predict(X))
    history = []

    start_time = time.time()

    while time.time() - start_time < duration:
        target = np.random.uniform(bounds[0], bounds[1], dim)
        direction = target - best

        for step in range(steps):
            if time.time() - start_time >= duration:
                break

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

def train_gwo_burst_time(nn, X, y, duration=10, n_agents=30, bounds=(-2, 2)):
    """GWO burst (for hybrid) - time limited"""
    dim = nn.num_params()
    wolves = np.random.uniform(bounds[0], bounds[1], (n_agents, dim))
    wolves[0] = nn.get_params()

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

    start_time = time.time()

    while time.time() - start_time < duration:
        elapsed = time.time() - start_time
        progress = min(elapsed / duration, 1.0)
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

        # Update alpha, beta, delta and record after EACH agent
        for i in range(n_agents):
            if time.time() - start_time >= duration:
                break

            nn.set_params(wolves[i])
            score = mse(y, nn.predict(X))
            if score < alpha_score:
                alpha_score = score
                alpha = wolves[i].copy()

            # Enregistrer après CHAQUE agent
            history.append(alpha_score)

        if time.time() - start_time >= duration:
            break

    nn.set_params(alpha)
    return history

def train_hybrid_cyclic_time(nn, X, y, duration=60, cycle_duration=10, bounds=(-2, 2)):
    """
    Hybrid Cyclic with time limit: Alternates Ray Shooting and GWO
    Each cycle: Ray burst (cycle_duration/2) + GWO burst (cycle_duration/2)
    """
    history = []
    best_score = mse(y, nn.predict(X))
    history.append(best_score)

    start_time = time.time()
    cycle = 0

    while time.time() - start_time < duration:
        remaining = duration - (time.time() - start_time)
        if remaining <= 0:
            break

        # Phase 1: Ray Shooting burst
        ray_duration = min(cycle_duration / 2, remaining)
        hist_ray = train_ray_burst_time(nn, X, y, duration=ray_duration, bounds=bounds)
        history.extend(hist_ray)

        remaining = duration - (time.time() - start_time)
        if remaining <= 0:
            break

        # Phase 2: GWO burst (restarts with fresh decay)
        gwo_duration = min(cycle_duration / 2, remaining)
        hist_gwo = train_gwo_burst_time(nn, X, y, duration=gwo_duration, bounds=bounds)
        history.extend(hist_gwo)

        cycle += 1

    return history, cycle

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================
def run_experiment_60s():
    os.makedirs(f'{OUTPUT_DIR}/figures', exist_ok=True)
    print("="*80)
    print("EXPERIMENT: 60 SECONDS TIME LIMIT FOR ALL ALGORITHMS")
    print("="*80)
    print(f"Each algorithm runs for {TIME_LIMIT} seconds")
    print(f"All evaluations are recorded for detailed convergence curves")
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
        print(f"  Backprop ({TIME_LIMIT}s)...", end=" ", flush=True)
        nn_bp = HeavisideNetwork3D(3, HIDDEN_SIZES, 1)
        start = time.time()
        hist_bp, epochs_bp = train_backprop_time(nn_bp, X_train, y_train_norm, duration=TIME_LIMIT)
        time_bp = time.time() - start
        mse_bp = hist_bp[-1]
        print(f"MSE={mse_bp:.5f} ({epochs_bp} epochs, {len(hist_bp)} points)")

        # 2. Ray Shooting
        print(f"  Ray Shooting ({TIME_LIMIT}s)...", end=" ", flush=True)
        nn_ray = HeavisideNetwork3D(3, HIDDEN_SIZES, 1)
        start = time.time()
        hist_ray, rays_count = train_ray_shooting_time(nn_ray, X_train, y_train_norm, duration=TIME_LIMIT)
        time_ray = time.time() - start
        mse_ray = hist_ray[-1]
        print(f"MSE={mse_ray:.5f} ({rays_count} rays, {len(hist_ray)} points)")

        # 3. GWO Standard
        print(f"  GWO Standard ({TIME_LIMIT}s)...", end=" ", flush=True)
        nn_gwo = HeavisideNetwork3D(3, HIDDEN_SIZES, 1)
        start = time.time()
        hist_gwo, iter_gwo = train_gwo_standard_time(nn_gwo, X_train, y_train_norm, duration=TIME_LIMIT)
        time_gwo = time.time() - start
        mse_gwo = hist_gwo[-1]
        print(f"MSE={mse_gwo:.5f} ({iter_gwo} iter, {len(hist_gwo)} points)")

        # 4. Hybrid Cyclic
        print(f"  Hybrid Cyclic ({TIME_LIMIT}s)...", end=" ", flush=True)
        nn_hyb = HeavisideNetwork3D(3, HIDDEN_SIZES, 1)
        start = time.time()
        hist_hyb, cycles_hyb = train_hybrid_cyclic_time(nn_hyb, X_train, y_train_norm,
                                                        duration=TIME_LIMIT,
                                                        cycle_duration=HYBRID_CYCLE_DURATION)
        time_hyb = time.time() - start
        mse_hyb = hist_hyb[-1]
        print(f"MSE={mse_hyb:.5f} ({cycles_hyb} cycles, {len(hist_hyb)} points)")

        stats.append({
            'func': func_name,
            'BP': mse_bp,
            'Ray': mse_ray,
            'GWO': mse_gwo,
            'Hybrid_Cyclic': mse_hyb,
            'Time_BP': time_bp,
            'Time_Ray': time_ray,
            'Time_GWO': time_gwo,
            'Time_Hyb': time_hyb,
            'Points_BP': len(hist_bp),
            'Points_Ray': len(hist_ray),
            'Points_GWO': len(hist_gwo),
            'Points_Hyb': len(hist_hyb)
        })

        # Plot convergence curves
        plt.figure(figsize=(12,7))
        plt.plot(hist_bp, label=f'Backprop ({len(hist_bp)} evals)', color='red', alpha=0.7, linewidth=2)
        plt.plot(hist_ray, label=f'Ray Shooting ({len(hist_ray)} evals)', color='green', linewidth=2)
        plt.plot(hist_gwo, label=f'GWO ({len(hist_gwo)} evals)', color='blue', linewidth=2)
        plt.plot(hist_hyb, label=f'Hybrid Cyclic ({len(hist_hyb)} evals)', color='purple', linewidth=2)
        plt.yscale('log')
        plt.xlabel('Evaluations')
        plt.ylabel('MSE (log scale)')
        plt.title(f"Comparison (60s each): {func_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{OUTPUT_DIR}/figures/{func_name}_60s.png", dpi=150, bbox_inches='tight')
        plt.close()

    # Summary table
    print("\n" + "="*100)
    print(f"{'Function':<12} {'BP MSE':<12} {'Ray MSE':<12} {'GWO MSE':<12} {'Hyb MSE':<12} {'Best':<12}")
    print("-" * 100)
    for s in stats:
        results = [s['BP'], s['Ray'], s['GWO'], s['Hybrid_Cyclic']]
        best_algo = ['BP', 'Ray', 'GWO', 'Hybrid'][np.argmin(results)]
        print(f"{s['func']:<12} {s['BP']:<12.4f} {s['Ray']:<12.4f} {s['GWO']:<12.4f} {s['Hybrid_Cyclic']:<12.4f} {best_algo:<12}")

    # Evaluation counts
    print("\n" + "="*100)
    print("EVALUATION COUNTS (Number of points in convergence curves)")
    print("-" * 100)
    print(f"{'Function':<12} {'BP Evals':<12} {'Ray Evals':<12} {'GWO Evals':<12} {'Hyb Evals':<12}")
    print("-" * 100)
    for s in stats:
        print(f"{s['func']:<12} {s['Points_BP']:<12} {s['Points_Ray']:<12} {s['Points_GWO']:<12} {s['Points_Hyb']:<12}")

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

    # Average evaluation counts
    avg_evals_bp = np.mean([s['Points_BP'] for s in stats])
    avg_evals_ray = np.mean([s['Points_Ray'] for s in stats])
    avg_evals_gwo = np.mean([s['Points_GWO'] for s in stats])
    avg_evals_hyb = np.mean([s['Points_Hyb'] for s in stats])

    print(f"\nAverage evaluations in 60s:")
    print(f"  Backpropagation:  {avg_evals_bp:.0f}")
    print(f"  Ray Shooting:     {avg_evals_ray:.0f}")
    print(f"  GWO Standard:     {avg_evals_gwo:.0f}")
    print(f"  Hybrid Cyclic:    {avg_evals_hyb:.0f}")
    print("="*100)

if __name__ == "__main__":
    run_experiment_60s()

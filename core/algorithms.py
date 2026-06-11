"""Optimization algorithms for Heaviside networks.

Common API: all return (final_mse, history)
- final_mse: final MSE on the provided data (train, not val)
- history: List[float] of length eval_budget (MSE after each eval)

The budget is the number of loss-function evaluations (forward pass).
"""
from typing import List, Tuple
import numpy as np
from core.network import HeavisideNetwork


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


# ============================================================================
# 1. BACKPROPAGATION (expected to fail — zero gradient in hidden layers)
# ============================================================================

def train_backprop(nn: HeavisideNetwork, X: np.ndarray, y: np.ndarray,
                   eval_budget: int = 25000, lr: float = 0.01
                   ) -> Tuple[float, List[float]]:
    """Standard backprop. The Heaviside gradient is explicitly 0 in the
    hidden layers: only the output layer learns. This is the failure to demonstrate.
    """
    history: List[float] = []
    m = X.shape[0]

    for epoch in range(eval_budget):
        A = X
        activations = [A]
        for W, b in nn.layers[:-1]:
            Z = A @ W + b
            A = nn._heaviside(Z)
            activations.append(A)
        W_out, b_out = nn.layers[-1]
        y_pred = A @ W_out + b_out

        loss = _mse(y, y_pred)
        history.append(loss)

        dZ_out = 2.0 * (y_pred - y) / m
        dW_out = activations[-1].T @ dZ_out
        db_out = np.sum(dZ_out, axis=0, keepdims=True)

        nn.layers[-1] = (W_out - lr * dW_out, b_out - lr * db_out)
        # Hidden layers are NOT updated (gradient = 0)

    return history[-1], history


# ============================================================================
# 2. RAY SHOOTING (gradient-free directional exploration)
# ============================================================================

def train_ray_shooting(nn: HeavisideNetwork, X: np.ndarray, y: np.ndarray,
                       eval_budget: int = 25000, steps_per_ray: int = 20,
                       k_directions: int = 1,
                       bounds: Tuple[float, float] = (-2.0, 2.0), seed: int = None
                       ) -> Tuple[float, List[float]]:
    """Greedy Ray Shooting with early break.

    For each ray: draws a uniform random target within bounds, walks
    steps_per_ray points along the ray, breaks as soon as a point better than
    best is found. Empirical key: steps_per_ray=20 (default) explores more
    directions per budget than steps_per_ray=50 (~8-10% gain in 3D/10D,
    neutral in 50D).

    k_directions structures exploration into bursts of K consecutive directions.
    With greedy break, this is algorithmically equivalent to K rounds of K=1,
    but the parameter is kept for future extensions (best-of-K without break,
    pattern search, etc.)."""
    rng = np.random.RandomState(seed)
    dim = nn.num_params()

    best_vec = nn.get_params()
    best_score = _mse(y, nn.forward(X))
    history: List[float] = [best_score]

    while len(history) < eval_budget:
        for _ in range(k_directions):
            if len(history) >= eval_budget:
                break
            target = rng.uniform(bounds[0], bounds[1], dim)
            direction = target - best_vec

            for step in range(steps_per_ray):
                if len(history) >= eval_budget:
                    break
                t = step / max(1, steps_per_ray - 1)
                candidate = np.clip(best_vec + t * direction, bounds[0], bounds[1])
                nn.set_params(candidate)
                score = _mse(y, nn.forward(X))
                history.append(min(score, best_score))
                if score < best_score:
                    best_vec = candidate.copy()
                    best_score = score
                    break  # move on to next direction

    nn.set_params(best_vec)
    history = history[:eval_budget]
    return best_score, history


# ============================================================================
# 3. GREY WOLF OPTIMIZER (gradient-free meta-heuristic)
# ============================================================================

def train_gwo(nn: HeavisideNetwork, X: np.ndarray, y: np.ndarray,
              eval_budget: int = 25000, n_agents: int = 30,
              bounds: Tuple[float, float] = (-2.0, 2.0), seed: int = None
              ) -> Tuple[float, List[float]]:
    """Standard GWO. The first wolf is seeded with the network's current position."""
    rng = np.random.RandomState(seed)
    dim = nn.num_params()

    wolves = rng.uniform(bounds[0], bounds[1], (n_agents, dim))
    wolves[0] = nn.get_params()

    history: List[float] = []
    scores = np.empty(n_agents)
    for i in range(n_agents):
        nn.set_params(wolves[i])
        scores[i] = _mse(y, nn.forward(X))
        history.append(scores[:i + 1].min())
        if len(history) >= eval_budget:
            nn.set_params(wolves[scores.argmin()])
            return history[-1], history[:eval_budget]

    order = np.argsort(scores)
    alpha = wolves[order[0]].copy(); alpha_score = scores[order[0]]
    beta = wolves[order[1]].copy(); beta_score = scores[order[1]]
    delta = wolves[order[2]].copy(); delta_score = scores[order[2]]

    max_iter = (eval_budget - n_agents) // n_agents + 1
    for it in range(max_iter):
        a = 2.0 * (1.0 - it / max(1, max_iter))
        for i in range(n_agents):
            r1, r2 = rng.rand(dim), rng.rand(dim)
            A1, C1 = 2 * a * r1 - a, 2 * r2
            X1 = alpha - A1 * np.abs(C1 * alpha - wolves[i])

            r1, r2 = rng.rand(dim), rng.rand(dim)
            A2, C2 = 2 * a * r1 - a, 2 * r2
            X2 = beta - A2 * np.abs(C2 * beta - wolves[i])

            r1, r2 = rng.rand(dim), rng.rand(dim)
            A3, C3 = 2 * a * r1 - a, 2 * r2
            X3 = delta - A3 * np.abs(C3 * delta - wolves[i])

            wolves[i] = np.clip((X1 + X2 + X3) / 3.0, bounds[0], bounds[1])

        for i in range(n_agents):
            if len(history) >= eval_budget:
                break
            nn.set_params(wolves[i])
            fit = _mse(y, nn.forward(X))
            if fit < alpha_score:
                delta, delta_score = beta, beta_score
                beta, beta_score = alpha, alpha_score
                alpha, alpha_score = wolves[i].copy(), fit
            elif fit < beta_score:
                delta, delta_score = beta, beta_score
                beta, beta_score = wolves[i].copy(), fit
            elif fit < delta_score:
                delta, delta_score = wolves[i].copy(), fit
            history.append(alpha_score)

        if len(history) >= eval_budget:
            break

    nn.set_params(alpha)
    history = history[:eval_budget]
    return alpha_score, history


# ============================================================================
# 4. HYBRID SEEDED (Ray then GWO seeded around best_ray)
# ============================================================================

def train_hybrid_seeded(nn: HeavisideNetwork, X: np.ndarray, y: np.ndarray,
                        eval_budget: int = 25000, ray_ratio: float = 0.5,
                        sigma: float = 0.2, n_agents: int = 30,
                        bounds: Tuple[float, float] = (-2.0, 2.0),
                        steps_per_ray: int = 50, seed: int = None
                        ) -> Tuple[float, List[float]]:
    """Phase 1: Ray Shooting (ray_ratio*budget evals).
    Phase 2: GWO with n_agents agents all initialized to best_ray + N(0, sigma*scale),
             except the first which stays at exact best_ray (no-regression)."""
    rng = np.random.RandomState(seed)
    scale = max(abs(bounds[0]), abs(bounds[1]))

    # === Phase 1: Ray ===
    ray_budget = int(ray_ratio * eval_budget)
    _, hist_ray = train_ray_shooting(nn, X, y, eval_budget=ray_budget,
                                     steps_per_ray=steps_per_ray, bounds=bounds,
                                     seed=None if seed is None else seed + 1)
    best_ray = nn.get_params()
    best_ray_score = _mse(y, nn.forward(X))

    # === Phase 2: seeded GWO ===
    gwo_budget = eval_budget - ray_budget
    dim = nn.num_params()

    wolves = best_ray[None, :] + rng.normal(0.0, sigma * scale, (n_agents, dim))
    wolves = np.clip(wolves, bounds[0], bounds[1])
    wolves[0] = best_ray  # ensure no regression

    history = list(hist_ray)
    scores = np.empty(n_agents)
    for i in range(n_agents):
        nn.set_params(wolves[i])
        scores[i] = _mse(y, nn.forward(X))
        history.append(min(scores[:i + 1].min(), best_ray_score))
        if len(history) >= eval_budget:
            best_idx = scores[:i + 1].argmin()
            best = wolves[best_idx] if scores[best_idx] < best_ray_score else best_ray
            nn.set_params(best)
            return history[-1], history[:eval_budget]

    order = np.argsort(scores)
    alpha = wolves[order[0]].copy(); alpha_score = scores[order[0]]
    beta = wolves[order[1]].copy(); beta_score = scores[order[1]]
    delta = wolves[order[2]].copy(); delta_score = scores[order[2]]
    # Ensure the known best stays at best_ray if the population regressed
    if best_ray_score < alpha_score:
        delta, delta_score = beta, beta_score
        beta, beta_score = alpha, alpha_score
        alpha, alpha_score = best_ray.copy(), best_ray_score

    max_iter = (gwo_budget - n_agents) // n_agents + 1
    for it in range(max_iter):
        a = 2.0 * (1.0 - it / max(1, max_iter))
        for i in range(n_agents):
            r1, r2 = rng.rand(dim), rng.rand(dim)
            A1, C1 = 2 * a * r1 - a, 2 * r2
            X1 = alpha - A1 * np.abs(C1 * alpha - wolves[i])
            r1, r2 = rng.rand(dim), rng.rand(dim)
            A2, C2 = 2 * a * r1 - a, 2 * r2
            X2 = beta - A2 * np.abs(C2 * beta - wolves[i])
            r1, r2 = rng.rand(dim), rng.rand(dim)
            A3, C3 = 2 * a * r1 - a, 2 * r2
            X3 = delta - A3 * np.abs(C3 * delta - wolves[i])
            wolves[i] = np.clip((X1 + X2 + X3) / 3.0, bounds[0], bounds[1])

        for i in range(n_agents):
            if len(history) >= eval_budget:
                break
            nn.set_params(wolves[i])
            fit = _mse(y, nn.forward(X))
            if fit < alpha_score:
                delta, delta_score = beta, beta_score
                beta, beta_score = alpha, alpha_score
                alpha, alpha_score = wolves[i].copy(), fit
            elif fit < beta_score:
                delta, delta_score = beta, beta_score
                beta, beta_score = wolves[i].copy(), fit
            elif fit < delta_score:
                delta, delta_score = wolves[i].copy(), fit
            history.append(alpha_score)
        if len(history) >= eval_budget:
            break

    nn.set_params(alpha)
    return alpha_score, history[:eval_budget]

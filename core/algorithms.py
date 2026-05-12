"""Algorithmes d'optimisation pour réseaux Heaviside.

API commune : tous retournent (final_mse, history)
- final_mse : MSE final sur les données fournies (train, pas val)
- history : List[float] de longueur eval_budget (MSE après chaque eval)

Le budget est le nombre d'évaluations de la fonction de perte (forward pass).
"""
from typing import List, Tuple
import numpy as np
from core.network import HeavisideNetwork


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


# ============================================================================
# 1. BACKPROPAGATION (échec attendu — gradient nul dans hidden layers)
# ============================================================================

def train_backprop(nn: HeavisideNetwork, X: np.ndarray, y: np.ndarray,
                   eval_budget: int = 25000, lr: float = 0.01
                   ) -> Tuple[float, List[float]]:
    """Backprop standard. Le gradient de Heaviside est explicitement 0 dans les
    couches cachées : seule la couche de sortie apprend. C'est l'échec à démontrer.
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
        # Les couches cachées NE SONT PAS mises à jour (gradient = 0)

    return history[-1], history


# ============================================================================
# 2. RAY SHOOTING (exploration directionnelle sans gradient)
# ============================================================================

def train_ray_shooting(nn: HeavisideNetwork, X: np.ndarray, y: np.ndarray,
                       eval_budget: int = 25000, steps_per_ray: int = 20,
                       k_directions: int = 5,
                       bounds: Tuple[float, float] = (-2.0, 2.0), seed: int = None
                       ) -> Tuple[float, List[float]]:
    """Ray Shooting "best-of-K directions".

    À chaque round, essaie k_directions rayons (cibles uniformes dans bounds).
    Pour chaque rayon, parcourt steps_per_ray points sans break.
    Garde le meilleur point trouvé sur l'ensemble du round et met à jour best.

    k_directions=1 + steps_per_ray=50 reproduit (presque) l'ancien comportement
    (sans le break interne)."""
    rng = np.random.RandomState(seed)
    dim = nn.num_params()

    best_vec = nn.get_params()
    best_score = _mse(y, nn.forward(X))
    history: List[float] = [best_score]

    while len(history) < eval_budget:
        round_best_vec = None
        round_best_score = float('inf')

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
                if score < round_best_score:
                    round_best_vec = candidate.copy()
                    round_best_score = score
                history.append(min(best_score, round_best_score))

        if round_best_score < best_score:
            best_vec = round_best_vec
            best_score = round_best_score

    nn.set_params(best_vec)
    history = history[:eval_budget]
    return best_score, history


# ============================================================================
# 3. GREY WOLF OPTIMIZER (méta-heuristique sans gradient)
# ============================================================================

def train_gwo(nn: HeavisideNetwork, X: np.ndarray, y: np.ndarray,
              eval_budget: int = 25000, n_agents: int = 30,
              bounds: Tuple[float, float] = (-2.0, 2.0), seed: int = None
              ) -> Tuple[float, List[float]]:
    """GWO standard. Le premier loup est seedé avec la position courante du réseau."""
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
# 4. HYBRID SEEDED (Ray puis GWO seedé autour de best_ray)
# ============================================================================

def train_hybrid_seeded(nn: HeavisideNetwork, X: np.ndarray, y: np.ndarray,
                        eval_budget: int = 25000, ray_ratio: float = 0.5,
                        sigma: float = 0.2, n_agents: int = 30,
                        bounds: Tuple[float, float] = (-2.0, 2.0),
                        steps_per_ray: int = 50, seed: int = None
                        ) -> Tuple[float, List[float]]:
    """Phase 1 : Ray Shooting (ray_ratio*budget évals).
    Phase 2 : GWO avec n_agents agents tous initialisés à best_ray + N(0, sigma*scale),
              sauf le premier qui reste à best_ray exact (anti-régression)."""
    rng = np.random.RandomState(seed)
    scale = max(abs(bounds[0]), abs(bounds[1]))

    # === Phase 1 : Ray ===
    ray_budget = int(ray_ratio * eval_budget)
    _, hist_ray = train_ray_shooting(nn, X, y, eval_budget=ray_budget,
                                     steps_per_ray=steps_per_ray, bounds=bounds,
                                     seed=None if seed is None else seed + 1)
    best_ray = nn.get_params()
    best_ray_score = _mse(y, nn.forward(X))

    # === Phase 2 : GWO seedé ===
    gwo_budget = eval_budget - ray_budget
    dim = nn.num_params()

    wolves = best_ray[None, :] + rng.normal(0.0, sigma * scale, (n_agents, dim))
    wolves = np.clip(wolves, bounds[0], bounds[1])
    wolves[0] = best_ray  # garantir qu'on ne régresse pas

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
    # Garantir que le meilleur connu reste best_ray si la population a régressé
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

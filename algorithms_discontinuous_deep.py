"""
Optimization algorithms for DEEP discontinuous networks
Adapted to work with multi-layer architectures
"""
from typing import Tuple, Optional, Any
import numpy as np


def ray_shooting_deep(
    nn, X: np.ndarray, y: np.ndarray, bounds: tuple = (-2, 2),
    max_rays: int = 100, steps_per_ray: int = 100,
    verbose: bool = False
) -> Tuple[float, list]:
    """
    RAY SHOOTING for deep networks
    Works with networks of any depth
    """
    # Get total parameter dimension
    dim = nn.get_total_params()
    
    # Initialize best solution
    best_vec = nn.get_weights_as_vector()
    best_loss = nn.compute_mse(y, nn.forward(X))
    
    history = [best_loss]
    
    if verbose:
        print(f"Deep Network: {len(nn.hidden_sizes)} hidden layers")
        print(f"Total parameters: {dim:,}")
        print(f"Initial MSE: {best_loss:.6f}")
    
    # Main ray shooting loop
    for ray in range(max_rays):
        # Sample random point z
        if np.random.rand() < 0.95:
            z = np.random.uniform(bounds[0], bounds[1], dim)
        else:
            z = np.random.uniform(bounds[0], bounds[1], dim)
        
        # Ray direction: from best to z
        direction = z - best_vec
        
        # Sample along the ray
        improved = False
        
        for j in range(steps_per_ray):
            # Parameter t: from 0 (best) to 1 (z)
            t = j / (steps_per_ray - 1.0) if steps_per_ray > 1 else 0.0
            
            # Parametric equation of ray
            candidate = best_vec + t * direction
            
            # Check bounds
            if not np.all((candidate >= bounds[0]) & (candidate <= bounds[1])):
                continue
            
            # Set network weights
            nn.set_weights_from_vector(candidate)
            
            # Evaluate
            loss = nn.compute_mse(y, nn.forward(X))
            
            # Update if better
            if loss < best_loss:
                best_vec = candidate.copy()
                best_loss = loss
                improved = True
                break
        
        history.append(best_loss)
        
        # Progress
        if verbose and (ray + 1) % 20 == 0:
            print(f"Ray {ray+1}/{max_rays} | MSE: {best_loss:.6f}")
    
    # Set final best weights
    nn.set_weights_from_vector(best_vec)
    
    if verbose:
        print(f"\nFinal MSE: {best_loss:.6f}")
    
    return best_loss, history


def gwo_deep(
    nn, X_train: np.ndarray, y_train: np.ndarray, bounds: tuple = (-2, 2),
    num_agents: int = 50, max_iter: int = 200, verbose: bool = False
) -> Tuple[float, list]:
    """
    Grey Wolf Optimizer for deep networks
    """
    dim = nn.get_total_params()
    
    if verbose:
        print(f"GWO with {num_agents} agents, {dim:,} parameters")
    
    # Initialize wolves
    wolves = np.random.uniform(bounds[0], bounds[1], (num_agents, dim))
    alpha = beta = delta = None
    alpha_score = beta_score = delta_score = float("inf")
    history = []
    
    # Initial evaluation
    for i in range(num_agents):
        nn.set_weights_from_vector(wolves[i])
        fit = nn.compute_mse(y_train, nn.forward(X_train))
        
        if fit < alpha_score:
            delta_score, beta_score, alpha_score = beta_score, alpha_score, fit
            delta, beta, alpha = beta, alpha, wolves[i].copy()
        elif fit < beta_score:
            delta_score, beta_score = beta_score, fit
            delta, beta = beta, wolves[i].copy()
        elif fit < delta_score:
            delta_score = fit
            delta = wolves[i].copy()

    for it in range(max_iter):
        
        # Update a (decreases from 2 to 0)
        a = 2 - it * (2 / max_iter)
        
        # Update wolf positions
        for i in range(num_agents):
            r1, r2 = np.random.rand(), np.random.rand()
            A1, C1 = 2 * a * r1 - a, 2 * r2
            D_alpha = np.abs(C1 * alpha - wolves[i])
            X1 = alpha - A1 * D_alpha
            
            r1, r2 = np.random.rand(), np.random.rand()
            A2, C2 = 2 * a * r1 - a, 2 * r2
            D_beta = np.abs(C2 * beta - wolves[i])
            X2 = beta - A2 * D_beta
            
            r1, r2 = np.random.rand(), np.random.rand()
            A3, C3 = 2 * a * r1 - a, 2 * r2
            D_delta = np.abs(C3 * delta - wolves[i])
            X3 = delta - A3 * D_delta
            
            wolves[i] = np.clip((X1 + X2 + X3) / 3, bounds[0], bounds[1])
            
        # Evaluate updated wolves
        for i in range(num_agents):
            nn.set_weights_from_vector(wolves[i])
            fit = nn.compute_mse(y_train, nn.forward(X_train))
            
            if fit < alpha_score:
                delta_score, beta_score, alpha_score = beta_score, alpha_score, fit
                delta, beta, alpha = beta, alpha, wolves[i].copy()
            elif fit < beta_score:
                delta_score, beta_score = beta_score, fit
                delta, beta = beta, wolves[i].copy()
            elif fit < delta_score:
                delta_score = fit
                delta = wolves[i].copy()
        
        history.append(alpha_score)
        
        if verbose and (it + 1) % 40 == 0:
            print(f"Iteration {it+1}/{max_iter} | Best MSE: {alpha_score:.6f}")
    
    # Set best weights
    nn.set_weights_from_vector(alpha)
    
    if verbose:
        print(f"\nFinal MSE: {alpha_score:.6f}")
    
    return alpha_score, history


def backprop_train_deep(
    nn, X: np.ndarray, y: np.ndarray, epochs: int = 1000,
    desired_mse: Optional[float] = 1e-3, verbose: bool = False,
    learning_rate: float = 0.001
) -> Tuple[float, Any]:
    """
    Train deep network using backpropagation
    Returns final MSE and full history dict
    """
    if verbose:
        print(f"Training with backprop for {epochs} epochs...")
    
    # Pass explicit learning rate
    history = nn.train(X, y, epochs, learning_rate=learning_rate)
    
    # Check if history has loss, otherwise inf
    final_mse = history['loss'][-1] if history['loss'] else float('inf')
    
    if verbose:
        print(f"Final MSE: {final_mse:.6f}")
    
    return final_mse, history


def mix_ray_gwo_deep(
    nn, X: np.ndarray, y: np.ndarray, bounds: tuple = (-2, 2),
    max_rays: int = 30, shoot_steps: int = 50,
    num_agents: int = 40, gwo_iters: int = 150, verbose: bool = False
) -> Tuple[float, list]:
    """
    Hybrid: Ray Shooting -> GWO for deep networks
    """
    if verbose:
        print("Phase 1: Ray Shooting")
    
    _, hist1 = ray_shooting_deep(
        nn, X, y, bounds=bounds, max_rays=max_rays,
        steps_per_ray=shoot_steps, verbose=verbose
    )
    
    if verbose:
        print("\nPhase 2: Grey Wolf Optimizer")
    
    # GWO continues from current weights (implicit in nn state)
    final_mse, hist2 = gwo_deep(
        nn, X, y, bounds=bounds,
        num_agents=num_agents,
        max_iter=gwo_iters, verbose=verbose
    )
    
    return final_mse, hist1 + hist2

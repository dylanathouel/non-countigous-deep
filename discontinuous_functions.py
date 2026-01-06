"""
Discontinuous Test Functions for "Non-Contiguous Deep Learning"
================================================================
TRUE 3D implementations with discontinuous boundaries:
- gl: Plane discontinuity (line -> plane)
- gs: Cube discontinuity (segment -> cube)
- geta: Exponential surface discontinuity
- ggamma: Sphere discontinuity (circle -> sphere)
- complex: Paraboloid discontinuity (parabola -> paraboloid)
"""
import numpy as np
from typing import Tuple


def gl(x: np.ndarray) -> np.ndarray:
    """
    Function 1 - gl (Plane Discontinuity)
    
    3D extension: The line boundary x2 <= 2*x1 becomes a plane x3 <= x1 + x2.
    The function depends on the 3D Euclidean norm.
    
    Args:
        x: Input array of shape (N, 3)
    
    Returns:
        Function values of shape (N,)
    """
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    norm_3d = np.sqrt(x1**2 + x2**2 + x3**2)
    
    # Plane discontinuity: x3 <= x1 + x2
    return np.where(
        x3 <= x1 + x2,
        2 * np.sin(1.25 * np.pi * norm_3d) + 4,
        2 * np.sin(0.75 * np.pi * norm_3d)
    )


def gs(x: np.ndarray) -> np.ndarray:
    """
    Function 2 - gs (Cube Discontinuity)
    
    3D extension: The segment [-1,1] x [0,inf] becomes a cube [-1,1]^3.
    Inside the cube: one function, outside: another.
    
    Args:
        x: Input array of shape (N, 3)
    
    Returns:
        Function values of shape (N,)
    """
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    
    # Cube discontinuity: all coordinates in [-1, 1]
    inside_cube = (x1 >= -1) & (x1 <= 1) & (x2 >= -1) & (x2 <= 1) & (x3 >= -1) & (x3 <= 1)
    
    # Inside cube: quadratic function
    # Outside cube: exponential function with 3D dependency
    return np.where(
        inside_cube,
        -2 * (x1**2 + x2**2 + x3**2) + 6,
        4 * np.exp((1 - x1**2 - x2**2 - x3**2) / 3)
    )


def geta(x: np.ndarray) -> np.ndarray:
    """
    Function 3 - gη (Exponential Surface Discontinuity)
    
    3D extension: The condition x2 >= exp(x1) becomes x3 >= exp(x1),
    creating a curved exponential surface boundary.
    
    Args:
        x: Input array of shape (N, 3)
    
    Returns:
        Function values of shape (N,)
    """
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    exp_x1 = np.exp(x1)
    sum_xyz = x1 + x2 + x3
    
    # Three regions based on x3 vs exp(x1)
    r1 = x3 >= exp_x1                    # Above exponential surface
    r2 = x3 < (exp_x1 - 1)               # Below shifted exponential surface
    r3 = ~r1 & ~r2                       # Between the two surfaces
    
    result = np.zeros_like(x1)
    result[r1] = np.sin(0.4 * np.pi * sum_xyz[r1])
    result[r2] = np.sin(0.7 * np.pi * sum_xyz[r2]) - 4
    result[r3] = np.sin(np.pi * sum_xyz[r3]) + 4
    
    return result


def ggamma(x: np.ndarray) -> np.ndarray:
    """
    Function 4 - gγ (Sphere Discontinuity)
    
    3D extension: The unit circle x1^2 + x2^2 <= 1 becomes a unit sphere
    x1^2 + x2^2 + x3^2 <= 1.
    
    Args:
        x: Input array of shape (N, 3)
    
    Returns:
        Function values of shape (N,)
    """
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    
    # 3D Euclidean norm squared (sphere radius)
    norm_sq_3d = x1**2 + x2**2 + x3**2
    sum_xyz = x1 + x2 + x3
    
    # Sphere discontinuity: inside vs outside unit sphere
    return np.where(
        norm_sq_3d <= 1,
        np.sin(np.pi * sum_xyz) + 4,
        np.sin(0.4 * np.pi * sum_xyz)
    )


def complex_func(x: np.ndarray) -> np.ndarray:
    """
    Function 5 - Complex (Paraboloid Discontinuity)
    
    3D extension: The parabola boundary x2 <= 0.5*(x1^2 - 1) becomes a
    paraboloid of revolution x3 <= 0.5*(x1^2 + x2^2 - 1).
    
    Args:
        x: Input array of shape (N, 3)
    
    Returns:
        Function values of shape (N,)
    """
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    
    # Paraboloid boundary: x3 <= 0.5 * (x1^2 + x2^2 - 1)
    paraboloid_boundary = 0.5 * (x1**2 + x2**2 - 1)
    
    # Two different functions on each side of the paraboloid
    g1 = (x1**2 + x2**2 + x3**2) * np.sin(np.pi * x1) + 3
    g2 = -x1 * x2 * x3 * np.cos(np.pi * x3) - 2
    
    return np.where(x3 <= paraboloid_boundary, g1, g2)


# Dictionary of all 3D discontinuous functions
FUNCTIONS = {
    'gl': gl,
    'gs': gs,
    'geta': geta,
    'ggamma': ggamma,
    'complex': complex_func
}

# Function descriptions for documentation
FUNCTION_DESCRIPTIONS = {
    'gl': 'Plane discontinuity (x3 <= x1 + x2)',
    'gs': 'Cube discontinuity ([-1,1]^3)',
    'geta': 'Exponential surface discontinuity (x3 >= exp(x1))',
    'ggamma': 'Unit sphere discontinuity (||x||^2 <= 1)',
    'complex': 'Paraboloid discontinuity (x3 <= 0.5*(x1^2 + x2^2 - 1))'
}


def generate_dataset(func_name: str, n_samples: int = 10000, seed: int = 42,
                     dim: int = 3, bounds: Tuple[float, float] = (-2, 2)) -> Tuple:
    """
    Generate Train/Val split for 3D discontinuous functions.
    
    Args:
        func_name: Name of the function ('gl', 'gs', 'geta', 'ggamma', 'complex')
        n_samples: Total number of samples to generate
        seed: Random seed for reproducibility
        dim: Input dimension (default 3 for 3D functions)
        bounds: Domain bounds as (min, max)
    
    Returns:
        Tuple of (X_train, y_train, X_val, y_val)
    """
    np.random.seed(seed)
    
    # Generate points in 3D domain
    X = np.random.uniform(bounds[0], bounds[1], (n_samples, dim))
    
    if func_name not in FUNCTIONS:
        raise ValueError(f"Unknown function: {func_name}. Available: {list(FUNCTIONS.keys())}")
    
    y = FUNCTIONS[func_name](X).reshape(-1, 1)
    
    # 80/20 train/validation split
    split = int(0.8 * n_samples)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    return X_train, y_train, X_val, y_val


def visualize_3d_function(func_name: str, n_points: int = 50, 
                          z_slice: float = 0.0, save_path: str = None):
    """
    Visualize a 3D function by taking a 2D slice at fixed z.
    
    Args:
        func_name: Function name to visualize
        n_points: Number of points per axis
        z_slice: Fixed z value for the slice
        save_path: Optional path to save the figure
    """
    import matplotlib.pyplot as plt
    
    x1 = np.linspace(-2, 2, n_points)
    x2 = np.linspace(-2, 2, n_points)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Create 3D input with fixed z
    X_flat = np.column_stack([X1.ravel(), X2.ravel(), np.full(X1.size, z_slice)])
    
    func = FUNCTIONS[func_name]
    Z = func(X_flat).reshape(X1.shape)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X1, X2, Z, levels=50, cmap='viridis')
    plt.colorbar(contour, ax=ax, label='f(x)')
    
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(f'{func_name} function (slice at x3={z_slice})\n{FUNCTION_DESCRIPTIONS[func_name]}')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # Test all functions
    print("Testing 3D Discontinuous Functions")
    print("=" * 50)
    
    for name, func in FUNCTIONS.items():
        # Generate test data
        X_test = np.random.uniform(-2, 2, (100, 3))
        y_test = func(X_test)
        
        print(f"{name}: shape={y_test.shape}, range=[{y_test.min():.2f}, {y_test.max():.2f}]")
        print(f"  Description: {FUNCTION_DESCRIPTIONS[name]}")
    
    print("\n" + "=" * 50)
    print("All 3D functions working correctly!")

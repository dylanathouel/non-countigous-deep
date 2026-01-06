"""
Discontinuous Test Functions for "Non-Contiguous Deep Learning"
"""
import numpy as np

def gl(x):
    """Fonction 1 - gl (ligne)"""
    x1, x2 = x[:,0], x[:,1]
    norm = np.sqrt(x1**2 + x2**2)
    return np.where(x2 <= 2*x1,
                    2*np.sin(1.25*np.pi*norm) + 4,
                    2*np.sin(0.75*np.pi*norm))

def gs(x):
    """Fonction 2 - gs (segment)"""
    x1, x2 = x[:,0], x[:,1]
    return np.where((x1>=-1) & (x1<=1) & (x2>=0),
                    -2*x1**2 + 6,
                    4*np.exp((1-x1**2)/2))

def geta(x):
    """Fonction 3 - gη (2 courbes exp)"""
    x1, x2 = x[:,0], x[:,1]
    exp_x1 = np.exp(x1)
    sum_x = x1 + x2
    r1 = x2 >= exp_x1
    r2 = x2 < (exp_x1 - 1)
    result = np.zeros_like(x1)
    result[r1] = np.sin(0.4*np.pi*sum_x[r1])
    result[r2] = np.sin(0.7*np.pi*sum_x[r2]) - 4
    # Condition ~r1 & ~r2
    mask_rest = ~r1 & ~r2
    result[mask_rest] = np.sin(np.pi*sum_x[mask_rest]) + 4
    return result

def ggamma(x):
    """Fonction 4 - gγ (cercle)"""
    x1, x2 = x[:,0], x[:,1]
    norm_sq = x1**2 + x2**2
    sum_x = x1 + x2
    return np.where(norm_sq <= 1,
                    np.sin(np.pi*sum_x) + 4,
                    np.sin(0.4*np.pi*sum_x))

def complex_func(x):
    """Fonction 5 - Complex (parabole)"""
    x1, x2 = x[:,0], x[:,1]
    boundary = 0.5*(x1**2 - 1)
    g1 = (x1**2 + x2**2)*np.sin(np.pi*x1) + 3
    g2 = -x1*x2*np.cos(np.pi*x2) - 2
    return np.where(boundary - x2 >= 0, g1, g2)

FUNCTIONS = {
    'gl': gl,
    'gs': gs,
    'geta': geta,
    'ggamma': ggamma,
    'complex': complex_func
}

def generate_dataset(func_name, n_samples=10000, seed=42):
    """Generate Train/Val split"""
    np.random.seed(seed)
    
    # Domain [-2, 2] x [-2, 2]
    X = np.random.uniform(-2, 2, (n_samples, 2))
    
    if func_name not in FUNCTIONS:
        raise ValueError(f"Unknown function: {func_name}")
        
    y = FUNCTIONS[func_name](X).reshape(-1, 1)
    
    # Split 80/20
    split = int(0.8 * n_samples)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    return X_train, y_train, X_val, y_val

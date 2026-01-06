import numpy as np
from run_experiment import HeavisideNetwork, train_ray, train_gwo, mse

def verify_fix():
    print("Verifying fix for MSE spikes...")
    
    # Setup
    input_size = 2
    hidden_sizes = [4, 4]
    output_size = 1
    nn = HeavisideNetwork(input_size, hidden_sizes, output_size)
    
    # Mock data
    X = np.random.uniform(-1, 1, (10, 2))
    y = np.sum(X, axis=1, keepdims=True)
    
    # 1. Run Ray Shooting
    print("Running Ray Shooting...")
    hist_ray = train_ray(nn, X, y, max_rays=5, steps=10)
    ray_final_mse = hist_ray[-1]
    print(f"Ray Final MSE: {ray_final_mse:.6f}")
    
    # 2. Run GWO (should be seeded with best ray solution)
    print("Running GWO...")
    hist_gwo = train_gwo(nn, X, y, iterations=5, n_agents=10)
    gwo_start_mse = hist_gwo[0]
    print(f"GWO Start MSE: {gwo_start_mse:.6f}")
    
    # 3. Verify
    if gwo_start_mse <= ray_final_mse + 1e-9:
        print("✅ SUCCESS: GWO started at least as good as Ray Shooting ended.")
        print(f"   Diff: {gwo_start_mse - ray_final_mse:.9f}")
    else:
        print("❌ FAILURE: GWO spiked!")
        print(f"   Ray End: {ray_final_mse:.6f}")
        print(f"   GWO Start: {gwo_start_mse:.6f}")
        print(f"   Spike: {gwo_start_mse - ray_final_mse:.6f}")
        exit(1)

if __name__ == "__main__":
    verify_fix()

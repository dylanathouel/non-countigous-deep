"""
Visualization Script for "Non-Contiguous Deep Learning"
(Matplotlib only version)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from discontinuous_functions import FUNCTIONS

# Setup
plt.style.use('ggplot')
os.makedirs('results/figures', exist_ok=True)

# Load Data
try:
    df_metrics = pd.read_csv('results/metrics/all_results.csv')
    with open('results/metrics/history.json', 'r') as f:
        histories = json.load(f)
except FileNotFoundError:
    print("Results not found. Run experiment first.")
    exit()

FUNCS = list(FUNCTIONS.keys())
ALGOS = ['backprop', 'gwo', 'ray', 'hybrid']
ALGO_COLORS = {'backprop': 'red', 'gwo': 'blue', 'ray': 'green', 'hybrid': 'purple'}
ALGO_LABELS = {
    'backprop': 'Backpropagation', 
    'gwo': 'Grey Wolf', 
    'ray': 'Ray Shooting', 
    'hybrid': 'Hybrid (Ray+GWO)'
}

# 1. 5 Functions Plot (Top View)
def plot_functions():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    x = np.linspace(-2, 2, 200)
    grid_x, grid_y = np.meshgrid(x, x)
    grid_input = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    
    for i, func_name in enumerate(FUNCS):
        ax = axes[i]
        z = FUNCTIONS[func_name](grid_input).reshape(200, 200)
        im = ax.imshow(z, extent=[-2, 2, -2, 2], origin='lower', cmap='viridis')
        ax.set_title(func_name)
        fig.colorbar(im, ax=ax)
    
    axes[-1].axis('off')
    plt.tight_layout()
    plt.savefig('results/figures/1_functions.png')
    plt.close()

# 2. Convergence
def plot_convergence():
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, func_name in enumerate(FUNCS):
        ax = axes[i]
        
        for algo in ALGOS:
            if algo in histories[func_name] and '42' in histories[func_name][algo]:
                loss_hist = histories[func_name][algo]['42']
                ax.plot(loss_hist, label=ALGO_LABELS[algo], color=ALGO_COLORS[algo], alpha=0.7)
        
        ax.set_title(f'Convergence: {func_name}')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('MSE')
        ax.set_yscale('log')
        ax.legend()
        
    axes[-1].axis('off')
    plt.tight_layout()
    plt.savefig('results/figures/2_convergence.png')
    plt.close()

# 3. Heatmap
def plot_heatmap():
    pivot = df_metrics.pivot_table(index='function', columns='algorithm', values='mse_val', aggfunc='mean')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, cmap='RdYlGn_r')
    
    # We want to show the values
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            text = ax.text(j, i, f"{pivot.values[i, j]:.3f}",
                           ha="center", va="center", color="black")
                           
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)
    plt.title('MSE Validation Heatmap (Lower is Better)')
    plt.tight_layout()
    plt.savefig('results/figures/3_heatmap.png')
    plt.close()

# 4. Bar Plot MSE
def plot_bar_mse():
    try:
        pivot = df_metrics.pivot_table(index='function', columns='algorithm', values='mse_val', aggfunc='mean')
        pivot.plot(kind='bar', figsize=(12, 6), logy=True, color=[ALGO_COLORS[c] for c in pivot.columns])
        plt.title('Validation MSE by Function and Algorithm')
        plt.ylabel('MSE (log scale)')
        plt.tight_layout()
        plt.savefig('results/figures/4_barplot_mse.png')
        plt.close()
    except Exception as e:
        print(f"Error plotting bar mse: {e}")

# 5. Bar Plot Time
def plot_bar_time():
    try:
        pivot = df_metrics.pivot_table(index='algorithm', columns='function', values='time', aggfunc='mean')
        pivot.plot(kind='bar', figsize=(12, 6))
        plt.title('Execution Time Comparison')
        plt.ylabel('Time (s)')
        plt.tight_layout()
        plt.savefig('results/figures/5_barplot_time.png')
        plt.close()
    except Exception as e:
        print(f"Error plotting bar time: {e}")

# 6. Gradient Magnitude
def plot_gradients():
    bp_data = df_metrics[df_metrics['algorithm'] == 'backprop']
    if not bp_data.empty:
        plt.figure(figsize=(10, 6))
        # Simple bar plot
        plt.bar(bp_data['function'], bp_data['grad_mean'], color='red')
        plt.yscale('log')
        plt.title('Average Gradient Magnitude (Backprop)')
        plt.ylabel('Gradient Norm (log)')
        plt.axhline(y=1e-6, color='black', linestyle='--', label='Threshold 1e-6')
        plt.legend()
        plt.tight_layout()
        plt.savefig('results/figures/6_gradients.png')
        plt.close()

# 8. Generalization
def plot_generalization():
    plt.figure(figsize=(8, 8))
    
    # Custom scatter
    for algo in ALGOS:
        subset = df_metrics[df_metrics['algorithm'] == algo]
        plt.scatter(subset['mse_train'], subset['mse_val'], 
                   label=ALGO_LABELS[algo], color=ALGO_COLORS[algo], s=50)
    
    lims = [
        np.min([plt.xlim(), plt.ylim()]),
        np.max([plt.xlim(), plt.ylim()]),
    ]
    plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Generalization: Train vs Val MSE')
    plt.xlabel('Train MSE')
    plt.ylabel('Val MSE')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/figures/8_generalization.png')
    plt.close()

if __name__ == "__main__":
    print("Generating Plots...")
    plot_functions()
    plot_convergence()
    plot_heatmap()
    plot_bar_mse()
    plot_bar_time()
    plot_gradients()
    plot_generalization()
    print("Done.")

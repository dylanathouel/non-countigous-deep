"""Benchmark dim=50: python -m experiments.run_50d"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.runner import run_benchmark

if __name__ == "__main__":
    run_benchmark(
        dim=50,
        output_dir="results/50d",
        eval_budget=25000,
        n_samples=2000,
        seed=42,
    )

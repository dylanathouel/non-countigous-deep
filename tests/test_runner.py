"""Test de runner.run_benchmark sur un cas minimal (dim=3, budget réduit)."""
import sys, os, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.runner import run_benchmark


def test_run_benchmark_smoke():
    out = "/tmp/heaviside_test_run"
    if os.path.exists(out):
        shutil.rmtree(out)
    df = run_benchmark(dim=3, output_dir=out, eval_budget=500, n_samples=300,
                       func_names=['gl'], seed=42)
    assert len(df) == 4
    assert set(df['algo']) == {'backprop', 'ray', 'gwo', 'hybrid'}
    assert os.path.exists(os.path.join(out, 'summary.csv'))
    assert os.path.exists(os.path.join(out, 'figures', 'gl.png'))
    print("PASS: test_run_benchmark_smoke")


if __name__ == "__main__":
    test_run_benchmark_smoke()
    print("\n[runner] 1 test passed.")

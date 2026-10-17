import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
import itertools
import time
from tqdm import tqdm
import theory
import config

# Sweeping over omega_b while defining ranges for beta and eta
OMEGA_BS = np.linspace(0.01, 7.0, 500)  # 500 points are sufficient for extremely smooth lines
GAMMAS = [0.3, 0.5, 0.7, 0.9]
BETAS = [0.15, 0.20, 0.25]
ETAS = [0.10, 0.15, 0.20]
MODES = ['BDoS', 'PDoS']


def compute_single_point(task_params):
    """
    Single-core computation task: Calculates the critical hash power alpha* for a single coordinate point in the grid.
    """
    w, g, b, e, mode = task_params
    calc = theory.PDoSTheoryCalculator(alpha=0.1, beta=b, eta=e, gamma=g, omega_b=w)

    # Invokes the purely mathematical analytical solution (includes 2D optimization)
    a_crit = calc.get_epsilon_alpha_crit(w, epsilon=0.0, gamma=g, mode=mode)

    return {
        'omega_b': w,
        'gamma': g,
        'beta': b,
        'eta': e,
        'mode': mode,
        'alpha_crit': a_crit
    }


def build_cache():
    print("=" * 50)
    print("Starting CTMC High-Performance Concurrent Pre-computation Engine")
    print("=" * 50)

    # Generate all required task combinations using Cartesian product
    tasks = list(itertools.product(OMEGA_BS, GAMMAS, BETAS, ETAS, MODES))
    total_tasks = len(tasks)
    print(f"Total computation tasks: {total_tasks}")

    # Get CPU core count, reserving one core for the OS to prevent freezing
    cores = max(1, cpu_count() - 1)
    print(f"Allocated CPU cores: {cores}")

    start_time = time.time()

    # Initialize and run the multiprocessing pool
    with Pool(cores) as pool:
        results = list(tqdm(pool.imap(compute_single_point, tasks), total=total_tasks))

    elapsed = time.time() - start_time
    print(f"Computation complete! Elapsed time: {elapsed:.2f} seconds")

    # Convert results to a DataFrame and save to disk
    df = pd.DataFrame(results)
    cache_path = config.CACHE_DIR / 'alpha_crit.csv'

    df.to_csv(cache_path, index=False)
    print(f"Global cache table successfully saved to: {cache_path}")


if __name__ == '__main__':
    build_cache()
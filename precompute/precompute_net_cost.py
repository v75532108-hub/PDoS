import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import numpy as np
from tqdm import tqdm
import theory, config


def get_optimal_net_cost_theory(omega_b, alpha, gamma, beta, eta, mode='PDoS'):
    """
    Invokes the CTMC MRP engine to calculate the cost boundary.
    Only returns valid net cost when deterrence is feasible (Delta U_T < 0);
    otherwise returns NaN to avoid plotting non-deterrent points.
    """
    calc = theory.PDoSTheoryCalculator(alpha=alpha, beta=beta, eta=eta, gamma=gamma, omega_b=omega_b)
    if mode == 'BDoS':
        _, u_m = calc.get_utilities(omega_b, 0.0, 0.0, 'Mine')
        _, u_s = calc.get_utilities(omega_b, 0.0, 0.0, 'Stop')
        if u_m - u_s < 0:
            u_a, _ = calc.get_utilities(omega_b, r1=0.0, r2=0.0, target_strategy='Stop')
            return -u_a / calc.c
        return np.nan
    else:
        _, _, u_a, is_feasible = calc.get_joint_policy(omega_b, target_strategy='Stop', resolution=201)
        if is_feasible:
            return -u_a / calc.c
        return np.nan


def build_net_cost_cache():
    print("=" * 60)
    print("Preparing to generate Figure 3 attack cost pre-computed cache (CTMC Grid Search)")
    print("=" * 60)

    # Experimental settings
    omega_bs = np.linspace(1.0, 2.25, 500)
    alphas = [0.20, 0.15, 0.10]
    params = config.get_experiment_defaults()
    beta = params['BETA']
    eta = params['ETA']
    gamma = params['GAMMA']

    results = []

    for a in alphas:
        for mode in ['BDoS', 'PDoS']:
            desc = f"Calculating alpha={a} ({mode})"
            for w in tqdm(omega_bs, desc=desc, leave=False):
                cost = get_optimal_net_cost_theory(w, a, gamma, beta, eta, mode)
                results.append({
                    'omega_b': w,
                    'alpha': a,
                    'mode': mode,
                    'net_cost': cost
                })

    # Save to disk
    df_cache = pd.DataFrame(results)
    cache_path = config.CACHE_DIR / 'net_cost.csv'
    cache_path.parent.mkdir(exist_ok=True, parents=True)
    df_cache.to_csv(cache_path, index=False)
    print(f"\n[+] Cache successfully saved to: {cache_path}")


if __name__ == '__main__':
    build_net_cost_cache()
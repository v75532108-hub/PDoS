import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

import config
import theory
from simulator import Simulator

# Retrieve baseline parameters from config
params = config.get_experiment_defaults()
beta_val = params['BETA']
eta_val = params['ETA']


def find_simulated_alpha_crit_robust(task_params):
    """
    Wraps the parameters into a single argument for multiprocessing map execution.
    Implements noise-resistant physical simulation validation and polynomial fitting logic.
    Uses least squares fitting to filter Monte Carlo noise and improve the precision
    of locating the zero-crossing point.
    """
    omega_b, gamma, mode = task_params

    # Calculate a priori theoretical coordinates
    calc_dummy = theory.PDoSTheoryCalculator(alpha=0.1, beta=beta_val, eta=eta_val, gamma=gamma, omega_b=omega_b)
    alpha_theory = calc_dummy.get_epsilon_alpha_crit(omega_b, epsilon=0.0, gamma=gamma, mode=mode)

    if np.isnan(alpha_theory) or alpha_theory <= 0.01 or alpha_theory >= 0.49:
        return {'omega_b': omega_b, 'gamma': gamma, 'mode': mode, 'alpha_crit_sim': np.nan}

    # Uniformly sample 11 test points within the neighborhood
    alphas_to_test = np.linspace(max(0.005, alpha_theory - 0.02), min(0.495, alpha_theory + 0.02), 11)
    diffs = []

    # Number of state transition events per simulation
    sim_events = 1000000

    for a in alphas_to_test:
        if mode == 'BDoS':
            sim_mine = Simulator(alpha=a, beta=beta_val, eta=eta_val, gamma=gamma, r1=0.0, omega_b=omega_b,
                                 target_strategy='Mine')
            sim_mine.r2 = 0.0
            u_mine = sim_mine.run(num_events=sim_events)['Target_Utility']

            sim_stop = Simulator(alpha=a, beta=beta_val, eta=eta_val, gamma=gamma, r1=0.0, omega_b=omega_b,
                                 target_strategy='Stop')
            sim_stop.r2 = 0.0
            u_stop = sim_stop.run(num_events=sim_events)['Target_Utility']
        else:
            calc = theory.PDoSTheoryCalculator(alpha=a, beta=beta_val, eta=eta_val, gamma=gamma, omega_b=omega_b)
            best_r1, best_r2, _ = calc.get_deterrence_r(omega_b)

            sim_mine = Simulator(alpha=a, beta=beta_val, eta=eta_val, gamma=gamma, r1=best_r1, omega_b=omega_b,
                                 target_strategy='Mine')
            sim_mine.r2 = best_r2
            u_mine = sim_mine.run(num_events=sim_events)['Target_Utility']

            sim_stop = Simulator(alpha=a, beta=beta_val, eta=eta_val, gamma=gamma, r1=best_r1, omega_b=omega_b,
                                 target_strategy='Stop')
            sim_stop.r2 = best_r2
            u_stop = sim_stop.run(num_events=sim_events)['Target_Utility']

        diffs.append(u_mine - u_stop)

    diffs = np.array(diffs)
    alphas_to_test = np.array(alphas_to_test)

    # Filter out invalid values
    valid_mask = ~np.isnan(diffs)

    # Require at least three valid points to perform polynomial fitting
    if np.sum(valid_mask) < 3:
        print(
            f"[Warning] {mode}: Insufficient valid scatter points at omega_b={omega_b:.2f}, skipping polynomial fitting.")
        return {'omega_b': omega_b, 'gamma': gamma, 'mode': mode, 'alpha_crit_sim': np.nan}

    diffs_clean = diffs[valid_mask]
    alphas_clean = alphas_to_test[valid_mask]

    # Zero-crossing validation
    if np.max(diffs_clean) < 0 or np.min(diffs_clean) > 0:
        print(
            f"[Warning] {mode}: No zero-crossing detected at omega_b={omega_b:.2f}, unable to determine the physical boundary.")
        return {'omega_b': omega_b, 'gamma': gamma, 'mode': mode, 'alpha_crit_sim': np.nan}

    # Map the utility difference back to hash power alpha, finding the x corresponding to y=0.
    # Using polynomial fitting provides stronger noise resistance than simple np.interp.
    coef = np.polyfit(diffs_clean, alphas_clean, deg=1)
    alpha_crit_sim = np.polyval(coef, 0.0)

    return {'omega_b': omega_b, 'gamma': gamma, 'mode': mode, 'alpha_crit_sim': alpha_crit_sim}


def build_sim_cache():
    print("=" * 50)
    print("Preparing to generate simulation scatter point cache (Simulation Checkpoints)")
    print("=" * 50)

    # Load the previously generated theoretical cache to determine sampling intervals
    df_theory = config.load_csv(config.CACHE_DIR / 'alpha_crit.csv')

    gammas = [0.3, 0.5, 0.7, 0.9]
    modes = ['BDoS', 'PDoS']
    num_points = 10
    tasks = []

    # Intelligently extract 10 uniform sampling coordinates for each line
    for g in gammas:
        for m in modes:
            sub_df = df_theory[(df_theory['gamma'] == g) & (df_theory['mode'] == m) &
                               (df_theory['beta'] == beta_val) & (df_theory['eta'] == eta_val)]

            # Filter out the valid interval strictly away from the extreme boundaries (0.02 ~ 0.48)
            valid_df = sub_df[(sub_df['alpha_crit'] > 0.02) & (sub_df['alpha_crit'] < 0.48)]
            if not valid_df.empty:
                w_min, w_max = valid_df['omega_b'].min(), valid_df['omega_b'].max()
                sim_w_list = np.linspace(w_min, w_max, num_points)
                for w in sim_w_list:
                    tasks.append((w, g, m))

    print(f"Extracted {len(tasks)} coordinate points requiring heavy simulation...")

    # Execute simulations using multiprocessing
    cores = max(1, cpu_count() - 1)
    with Pool(cores) as pool:
        results = list(tqdm(pool.imap_unordered(find_simulated_alpha_crit_robust, tasks), total=len(tasks)))

    # Save results
    df_sim = pd.DataFrame(results).dropna()  # Remove NaN points from failed fittings
    output_path = config.CACHE_DIR / 'sim_checkpoints.csv'
    df_sim.to_csv(output_path, index=False)
    print(f"\nSimulation scatter point cache successfully saved to: {output_path}")


if __name__ == '__main__':
    build_sim_cache()
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import theory, config

# 2024-2025 Real-world mining pool hash power distribution data
BETA_BASE = 0.3

TARGET_POOLS = {
    'AntPool': 0.2, 'ViaBTC': 0.13, 'F2Pool': 0.11,
    'Luxor': 0.03, 'Binance Pool': 0.02, 'SBI Crypto': 0.02,
}
for i in range(5): TARGET_POOLS[f'Mid_{i}'] = 0.005
for i in range(10): TARGET_POOLS[f'Small_{i}'] = 0.001
for i in range(50): TARGET_POOLS[f'Retail_{i}'] = 0.0001
ETA_BASE = sum(TARGET_POOLS.values())


def _calc_single_block(params):
    """
    Multiprocessing Worker function: Independently calculates the Nash equilibrium
    and utility under a single-block environment.
    """
    # Added prev_omega_b and prev_m to retrieve the decision environment of the preceding block
    alpha, gamma, omega_b, m, prev_omega_b, prev_m, mode = params
    beta = BETA_BASE * (1 - alpha)
    eta_total = ETA_BASE * (1 - alpha)

    calc_prev = theory.PDoSTheoryCalculator(alpha=alpha, beta=beta, eta=eta_total, gamma=gamma, omega_b=prev_omega_b)
    calc_prev.M = prev_m

    if mode == 'PDoS':
        r2_prev = calc_prev.get_optimal_r2(prev_omega_b, target_strategy='Mine')
        r1_prev, _ = calc_prev.get_constrained_optimal_r1(prev_omega_b, r2_prev, resolution=100)
    else:
        r1_prev, r2_prev = 0.0, 0.0

    # Nash equilibrium network-exit evolution (based on the deterrence parameter r1_prev from the previous block)
    pools = [{'name': k, 'power': v * (1 - alpha), 'active': True} for k, v in TARGET_POOLS.items()]
    pools = sorted(pools, key=lambda x: x['power'])

    while True:
        current_active_power = sum([float(p['power']) for p in pools if p['active']])
        current_x = current_active_power / eta_total if eta_total > 0 else 0

        if current_x <= 0: break

        changed = False
        for p in pools:
            if p['active']:
                active_eta_others = current_active_power - p['power']
                delta_active = active_eta_others + calc_prev.delta
                delta_u = calc_prev.get_exact_pool_utility_diff(
                    eta_i=p['power'], delta_active=delta_active, r1=r1_prev, mode=mode
                )
                if delta_u < 0:
                    p['active'] = False
                    changed = True
                    break  # After one drops out, the deteriorating environment caused by the hash power vacuum must be recalculated
        if not changed: break

    final_x = sum([float(p['power']) for p in pools if p['active']]) / eta_total if eta_total > 0 else 0

    # Calculate actual utility based on the current block's parameters
    calc_curr = theory.PDoSTheoryCalculator(alpha=alpha, beta=beta, eta=eta_total, gamma=gamma, omega_b=omega_b)
    calc_curr.M = m

    if mode == 'PDoS':
        r2_curr = calc_curr.get_optimal_r2(omega_b, target_strategy='Mine')
        r1_curr, _ = calc_curr.get_constrained_optimal_r1(omega_b, r2_curr, resolution=100)
    else:
        r1_curr, r2_curr = 0.0, 0.0

    # Theoretical utility when the target mines vs. theoretical utility when the target shuts down
    u_a_mine, _ = calc_curr.get_utilities(omega_b, r1_curr, r2_curr, 'Mine')
    u_a_stop, _ = calc_curr.get_utilities(omega_b, r1_curr, r2_curr, 'Stop')

    # Actual expected revenue of the attacker executing the attack
    # (utilizing the final_x from the previous block to settle the current block's dividends)
    u_a_actual = calc_curr.get_actual_u_a(omega_b, r1_curr, r2_curr, final_x)

    # Baseline utility of the attacker performing honest mining
    u_a_honest = calc_curr.c * (omega_b - 1.0)

    return final_x, u_a_actual, u_a_honest


def simulate_dynamic_avalanche_with_utility(alpha, gamma, replay_df, mode='PDoS', multiplier=1.0):
    """
    Executes dynamic concurrent simulation, returning result arrays
    for external prefix sum (cumulative sum) calculation.
    """
    tasks = []
    # The first block has no history; treat itself as the preceding block
    prev_omega_b = replay_df.iloc[0]['omega_b'] * multiplier
    prev_m = replay_df.iloc[0]['M']

    # Wrap tasks, integrating sliding historical states
    for _, row in replay_df.iterrows():
        curr_omega_b = row['omega_b'] * multiplier
        curr_m = row['M']
        tasks.append((alpha, gamma, curr_omega_b, curr_m, prev_omega_b, prev_m, mode))
        # State update
        prev_omega_b = curr_omega_b
        prev_m = curr_m

    cores = max(1, cpu_count() - 1)
    with Pool(cores) as p:
        results = list(tqdm(p.imap(_calc_single_block, tasks), total=len(tasks),
                            desc=f"{mode} (alpha={alpha}, bull_x={multiplier})"))

    results_x = [r[0] for r in results]

    results_u_actual = [r[1] for r in results]
    results_u_honest = [r[2] for r in results]

    cum_u_actual = np.cumsum(results_u_actual).tolist()
    cum_u_honest = np.cumsum(results_u_honest).tolist()

    return results_x, cum_u_actual, cum_u_honest


def build_cache():
    print("=" * 60)
    print("Preparing to generate Figure 5 Trace-Driven dynamic replay pre-computed cache")
    print("=" * 60)

    print("Loading raw block-by-block replay data...")
    df_replay = config.load_csv(config.DATA_DIR / 'block_replay_data.csv')

    df_replay['timestamp'] = pd.to_datetime(df_replay['timestamp'], utc=True)
    start_date = pd.to_datetime('2024-01-01', utc=True)
    end_date = pd.to_datetime('2026-01-01', utc=True)

    df_plot = df_replay[(df_replay['timestamp'] >= start_date) & (df_replay['timestamp'] < end_date)].copy()
    print(f"A total of {len(df_plot)} blocks participating in the evolution.")

    heights = df_plot['height'].tolist()
    gamma_fixed = config.get_experiment_defaults().get('GAMMA')

    # Iterating computations across multiple hash power levels
    alphas = [0.1, 0.15, 0.2]

    # Initialize the main cache DataFrame
    df_cache = pd.DataFrame({'height': heights})

    for alpha in alphas:
        # Calculate the two modes separately
        x_p, u_p, u_h_p = simulate_dynamic_avalanche_with_utility(alpha, gamma_fixed, df_plot, mode='PDoS')
        x_b, u_b, u_h_b = simulate_dynamic_avalanche_with_utility(alpha, gamma_fixed, df_plot, mode='BDoS')

        # Name fields using alpha to prevent overwriting
        df_cache[f'x_p_{alpha}'] = x_p
        df_cache[f'u_p_{alpha}'] = u_p
        df_cache[f'x_b_{alpha}'] = x_b
        df_cache[f'u_b_{alpha}'] = u_b

        # The cumulative utility of honest mining depends only on omega_b;
        # results are identical across different alphas, so record it just once.
        if 'u_h' not in df_cache.columns:
            df_cache['u_h'] = u_h_p

    cache_path = config.CACHE_DIR / 'replay.csv'
    df_cache.to_csv(cache_path, index=False)
    print(f"\n✅ Physical replay cache successfully saved to: {cache_path}")


if __name__ == '__main__':
    build_cache()
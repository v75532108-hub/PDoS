import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import theory, config

# Constants and target pool distribution configuration
BETA_BASE = 0.3  # Foundry USA (Baseline victim pool)

TARGET_POOLS = {
    'AntPool': 0.2,
    'ViaBTC': 0.13,
    'F2Pool': 0.11,
    'Luxor': 0.03,
    'Binance Pool': 0.02,
    'SBI Crypto': 0.02,
}
# Granular distribution for mid, small, and retail miners
for i in range(5): TARGET_POOLS[f'Mid_{i}'] = 0.005
for i in range(10): TARGET_POOLS[f'Small_{i}'] = 0.001
for i in range(50): TARGET_POOLS[f'Retail_{i}'] = 0.0001

ETA_BASE = sum(TARGET_POOLS.values())
DELTA_BASE = max(0.0, 1.0 - BETA_BASE - ETA_BASE)


def run_trace_driven_nash_replay_task(task_params):
    """
    Single-task wrapper for the multiprocessing pool.
    Executes trace-driven game replay on historical blocks to find the Nash equilibrium.
    """
    alpha, gamma, mode, replay_df = task_params
    surviving_x_list = []

    beta = BETA_BASE * (1 - alpha)
    eta_total = ETA_BASE * (1 - alpha)

    if eta_total <= 0:
        return {'alpha': alpha, 'gamma': gamma, 'mode': mode, 'throughput': 0.0}

    # Map the distribution to absolute hash power and sort in ascending order
    # (Simulating that marginal miners with lower risk tolerance drop off first)
    pools = sorted([(name, share * (1 - alpha)) for name, share in TARGET_POOLS.items()], key=lambda x: x[1])

    for _, block in replay_df.iterrows():
        omega_b = block['omega_b']
        m = block['M']

        calc = theory.PDoSTheoryCalculator(alpha=alpha, beta=beta, eta=eta_total, gamma=gamma, omega_b=omega_b)
        calc.M = m

        if mode == 'PDoS':
            r1, _, _ = calc.get_deterrence_r(omega_b, resolution=10)
        else:
            r1 = 0.0

        # Initial state: All miners default to staying online and mining
        active_status = {name: True for name, _ in pools}

        # Nash equilibrium evolution loop
        while True:
            # Total hash power and proportion of target miners still surviving in the network
            current_active_power = sum(power for name, power in pools if active_status[name])
            current_x = current_active_power / eta_total if eta_total > 0 else 0

            if current_x <= 0:  # Target population is fully paralyzed
                break

            changed = False

            # Iterate through each active pool and accurately evaluate its utility difference
            for name, power in pools:
                if active_status[name]:
                    # Isolate the remaining rational hash power still mining
                    active_eta_others = current_active_power - power
                    delta_active = active_eta_others + calc.delta

                    # Invoke the discrete pool utility evaluation function
                    delta_u = calc.get_exact_pool_utility_diff(
                        eta_i=power,
                        delta_active=delta_active,
                        r1=r1,
                        mode=mode
                    )

                    # If shutting down is more profitable than mining, the marginal miner shuts down
                    if delta_u < 0:
                        active_status[name] = False
                        changed = True
                        break  # After one drops out, the deteriorating environment (hash power vacuum) must be recalculated

            # If no one shuts down in this iteration, the Nash equilibrium is reached
            if not changed:
                break

        # Record the final surviving throughput for this block
        final_x = sum(power for name, power in pools if active_status[name]) / eta_total
        surviving_x_list.append(final_x)

    # Return the average expected throughput for this historical period
    return {'alpha': alpha, 'gamma': gamma, 'mode': mode, 'throughput': float(np.mean(surviving_x_list))}


def build_cache():
    print("=" * 60)
    print("Preparing to generate Trace-Driven Nash Equilibrium pre-computed cache")
    print("=" * 60)

    df_replay = config.load_csv(config.DATA_DIR / 'block_replay_data.csv')

    # Formal experiment sample size set to 10,000 to ensure smoothing accuracy
    sample_size = 10000
    if len(df_replay) > sample_size:
        df_sampled = df_replay.iloc[::len(df_replay) // sample_size].copy()
    else:
        df_sampled = df_replay.copy()

    print(f"Sampled {len(df_sampled)} representative blocks, starting multiprocessing replay...")

    alphas = np.linspace(0, 0.5, 50)
    gamma_fixed = config.get_experiment_defaults().get('GAMMA')
    modes = ['BDoS', 'PDoS']

    # Wrap tasks to dispatch into the multiprocessing pool
    tasks = []
    for a in alphas:
        for m in modes:
            tasks.append((a, gamma_fixed, m, df_sampled))

    cores = max(1, cpu_count() - 1)
    print(f"Waking up {cores} CPU cores...")

    with Pool(cores) as pool:
        results = list(tqdm(pool.imap_unordered(run_trace_driven_nash_replay_task, tasks), total=len(tasks),
                            desc="🚀 Nash Equilibrium Replay Progress"))

    # Save to disk
    df_sim = pd.DataFrame(results).dropna()
    output_path = config.CACHE_DIR / 'trace_nash_equilibrium.csv'
    output_path.parent.mkdir(exist_ok=True, parents=True)
    df_sim.to_csv(output_path, index=False)
    print(f"\n[+] Physical replay cache successfully saved to: {output_path}")


if __name__ == '__main__':
    build_cache()
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import config, theory


def _distribute_pool_reward(reward, shares_a, shares_v):
    """
    Returns the attacker's and honest miners' share payouts.
    Core mechanism: The pool distributes and settles the shares for the current round.
    """
    if reward <= 0:
        return 0.0, 0.0
    total = shares_a + shares_v
    if total > 0:
        return reward * (shares_a / total), reward * (shares_v / total)
    return 0.0, reward


class Simulator:
    def __init__(self, alpha, beta, eta, gamma, r1, omega_b, target_strategy='Mine'):
        # Hash power distribution
        self.alpha = alpha  # Attacker hash power
        self.beta = beta  # Victim pool hash power
        self.eta = eta  # Target hash power
        self.delta = max(0.0, 1.0 - alpha - beta - eta)  # Other miners' hash power
        self.r1 = r1  # Initial infiltration ratio

        # Network parameters
        self.gamma = gamma  # Propagation advantage (rushing ability)
        self.target_strategy = target_strategy  # Target strategy: 'Mine', 'Stop', 'SPV'
        self.omega_b = omega_b  # Baseline profitability factor

        # Economic parameters
        defaults = config.get_experiment_defaults()
        self.K = defaults['K']
        self.LAMBDA = defaults['LAMBDA']
        self.lambda_t = defaults['lambda_t']
        self.lambda_w = defaults['lambda_w']
        self.e_rw = defaults['e_rw']

        r_b = self.K + (self.lambda_t + self.lambda_w * self.e_rw) / self.LAMBDA
        self.c_block = r_b / self.omega_b

        # Local optimization of r2
        calc = theory.PDoSTheoryCalculator(
            alpha=self.alpha, beta=self.beta, eta=self.eta, gamma=self.gamma,
            omega_b=self.omega_b, lambda_t=self.lambda_t, lambda_w=self.lambda_w,
            e_rw=self.e_rw, k=self.K, lam=self.LAMBDA
        )
        self.r2 = calc.get_optimal_r2(self.omega_b, target_strategy=self.target_strategy)

    def _sample_block_reward(self, dt):
        """
        Core physical engine: Dynamically generates the absolute BTC reward contained
        in the block based on the actual physical elapsed time dt.
        """
        # 1. Baseline fixed Coinbase reward
        reward = self.K

        # 2. Linearly accumulating regular transaction fees over time
        reward += self.lambda_t * dt

        # 3. Bursty high-value whale rewards arriving via a Poisson process over time
        # np.random.poisson samples the number of whale transactions occurring in this
        # period based on the arrival rate lambda_w * dt
        num_whales = np.random.poisson(self.lambda_w * dt)
        if num_whales > 0:
            reward += num_whales * self.e_rw

        return reward

    def run(self, num_events=100000):
        # Current system state and cumulative time
        state = 0
        time_elapsed = 0.0

        # Absolute revenue statistics
        revenue_atk = 0.0
        revenue_vic = 0.0
        revenue_tar = 0.0
        revenue_oth = 0.0

        # Reward pool for suspended triggering blocks
        suspended_pri = 0.0
        suspended_inf = 0.0
        suspended_vic_atk_cut = 0.0
        suspended_vic_vic_cut = 0.0
        suspended_inf_atk_cut = 0.0
        suspended_inf_vic_cut = 0.0
        suspended_tar = 0.0
        suspended_oth = 0.0

        # Share statistics
        shares_atk = 0.0
        shares_vic = 0.0

        # Hash power time integral
        time_atk = 0.0
        time_tar = 0.0

        # Main control loop, each iteration corresponds to a block generation event
        for _ in range(num_events):
            # The network's active hash power is always 1 except in states 1 and 2
            active_hashrate = self.alpha + self.beta + self.eta + self.delta

            if state == 0:  # Initial state
                # Sample physical time (incorporates time inflation effect implicitly via active_hashrate)
                dt = random.expovariate(self.LAMBDA * active_hashrate)
                time_elapsed += dt

                # Accumulate shares and time
                shares_atk += self.r1 * self.alpha * dt
                shares_vic += self.beta * dt
                time_atk += self.alpha * dt
                time_tar += self.eta * dt

                # Simulate block discovery via random event
                rand = random.random() * active_hashrate
                if rand < self.delta:  # Other miners find a race-resolving block
                    revenue_oth += self._sample_block_reward(dt)
                elif rand < self.delta + self.eta:  # Target miners find a race-resolving block
                    revenue_tar += self._sample_block_reward(dt)
                elif rand < self.delta + self.eta + self.beta:  # Victim pool finds a race-resolving block
                    reward = self._sample_block_reward(dt)
                    atk_cut, vic_cut = _distribute_pool_reward(reward, shares_atk, shares_vic)
                    revenue_atk += atk_cut
                    revenue_vic += vic_cut
                    shares_atk, shares_vic = 0.0, 0.0
                elif rand < self.delta + self.eta + self.beta + self.alpha * (
                        1 - self.r1):  # Attacker's private power finds a race-triggering block
                    suspended_pri = self._sample_block_reward(dt)
                    state = 1
                else:  # Attacker's infiltration power finds a race-triggering block
                    suspended_inf = self._sample_block_reward(dt)
                    state = 2

            elif state == 1:  # Private Lead state
                eta_act = 0.0 if self.target_strategy == 'Stop' else self.eta  # Active status of target miners
                active_hashrate = self.beta + eta_act + self.delta

                # Sample physical time
                dt = random.expovariate(self.LAMBDA * active_hashrate)
                time_elapsed += dt

                # Accumulate shares and time
                shares_atk += self.r2 * self.alpha * dt
                shares_vic += self.beta * dt
                time_atk += self.r2 * self.alpha * dt
                time_tar += eta_act * dt

                # Random event
                rand = random.random() * active_hashrate
                if rand < self.beta:  # Victim pool finds a race-triggering block
                    reward = self._sample_block_reward(dt)
                    suspended_vic_atk_cut, suspended_vic_vic_cut = _distribute_pool_reward(reward, shares_atk,
                                                                                           shares_vic)
                    shares_atk, shares_vic = 0.0, 0.0
                    state = 3
                elif rand < self.beta + eta_act:
                    if self.target_strategy == 'SPV':  # Target falls into the SPV trap
                        suspended_pri = 0.0  # Attacker discards the private block
                        state = 0
                    else:  # Target miners find a race-triggering block
                        suspended_tar = self._sample_block_reward(dt)
                        state = 4
                else:  # Other miners find a race-triggering block
                    suspended_oth = self._sample_block_reward(dt)
                    state = 4

            elif state == 2:  # Infiltration Lead state
                eta_act = 0.0 if self.target_strategy == 'Stop' else self.eta
                active_hashrate = self.beta + eta_act + self.delta

                # Sample physical time
                dt = random.expovariate(self.LAMBDA * active_hashrate)
                time_elapsed += dt

                # Accumulate shares and time
                shares_atk += self.r2 * self.alpha * dt
                shares_vic += self.beta * dt
                time_atk += self.r2 * self.alpha * dt
                time_tar += eta_act * dt

                # Random event
                rand = random.random() * active_hashrate
                if rand < self.beta:  # Victim pool finds a race-resolving block
                    reward = self._sample_block_reward(dt)
                    atk_cut, vic_cut = _distribute_pool_reward(reward, shares_atk, shares_vic)
                    revenue_atk += atk_cut
                    revenue_vic += vic_cut
                    shares_atk, shares_vic = 0.0, 0.0
                    suspended_inf = 0.0  # Stale infiltration block is passively invalidated
                    state = 0
                elif rand < self.beta + eta_act:
                    if self.target_strategy == 'SPV':  # Target falls into the SPV trap
                        suspended_inf = 0.0  # Attacker discards the infiltration block
                        state = 0
                    else:  # Target miners find a race-triggering block
                        suspended_tar = self._sample_block_reward(dt)
                        suspended_inf_atk_cut, suspended_inf_vic_cut = _distribute_pool_reward(suspended_inf,
                                                                                               shares_atk, shares_vic)
                        shares_atk, shares_vic = 0.0, 0.0
                        state = 5
                else:  # Other miners find a race-triggering block
                    suspended_oth = self._sample_block_reward(dt)
                    suspended_inf_atk_cut, suspended_inf_vic_cut = _distribute_pool_reward(suspended_inf, shares_atk,
                                                                                           shares_vic)
                    shares_atk, shares_vic = 0.0, 0.0
                    state = 5

            elif state == 3:  # Race PvV
                # Sample physical time
                dt = random.expovariate(self.LAMBDA * active_hashrate)
                time_elapsed += dt

                # Accumulate shares and time
                shares_atk += 0.0
                shares_vic += self.beta * dt
                time_atk += self.alpha * dt
                time_tar += self.eta * dt

                # Random event
                rand = random.random() * active_hashrate
                reward = self._sample_block_reward(dt)
                if rand < self.alpha:  # Attacker's private hash power finds a race-resolving block
                    revenue_atk += suspended_pri + reward
                elif rand < self.alpha + self.beta:  # Victim pool finds a race-resolving block
                    atk_cut, vic_cut = _distribute_pool_reward(reward, shares_atk, shares_vic)
                    revenue_atk += suspended_vic_atk_cut + atk_cut
                    revenue_vic += suspended_vic_vic_cut + vic_cut
                    shares_atk, shares_vic = 0.0, 0.0
                elif rand < self.alpha + self.beta + self.eta:  # Target miners find a race-resolving block
                    revenue_tar += reward
                    if random.random() < self.gamma:  # Supports the private branch
                        revenue_atk += suspended_pri
                    else:
                        revenue_atk += suspended_vic_atk_cut
                        revenue_vic += suspended_vic_vic_cut
                else:  # Other miners find a race-resolving block
                    revenue_oth += reward
                    if random.random() < self.gamma:  # Supports the private branch
                        revenue_atk += suspended_pri
                    else:
                        revenue_atk += suspended_vic_atk_cut
                        revenue_vic += suspended_vic_vic_cut

                suspended_pri = suspended_vic_atk_cut = suspended_vic_vic_cut = 0.0  # Clear suspended rewards
                state = 0

            elif state == 4:  # Race PvE
                # Sample physical time
                dt = random.expovariate(self.LAMBDA * active_hashrate)
                time_elapsed += dt

                # Accumulate shares and time
                shares_atk += 0.0
                shares_vic += self.beta * dt
                time_atk += self.alpha * dt
                time_tar += self.eta * dt

                # Random event
                rand = random.random() * active_hashrate
                reward = self._sample_block_reward(dt)
                if rand < self.alpha:  # Attacker's private hash power finds a race-resolving block
                    revenue_atk += suspended_pri + reward
                elif rand < self.alpha + self.beta:  # Victim pool finds a race-resolving block
                    atk_cut, vic_cut = _distribute_pool_reward(reward, shares_atk, shares_vic)
                    revenue_atk += atk_cut
                    revenue_vic += vic_cut
                    shares_atk, shares_vic = 0.0, 0.0
                    if random.random() < self.gamma:  # Supports the private branch
                        revenue_atk += suspended_pri
                    else:
                        revenue_tar += suspended_tar
                        revenue_oth += suspended_oth
                elif rand < self.alpha + self.beta + self.eta:  # Target miners find a race-resolving block
                    revenue_tar += reward
                    if random.random() < self.gamma:  # Supports the private branch
                        revenue_atk += suspended_pri
                    else:
                        revenue_tar += suspended_tar
                        revenue_oth += suspended_oth
                else:  # Other miners find a race-resolving block
                    revenue_oth += reward
                    if random.random() < self.gamma:  # Supports the private branch
                        revenue_atk += suspended_pri
                    else:
                        revenue_tar += suspended_tar
                        revenue_oth += suspended_oth

                suspended_pri = suspended_tar = suspended_oth = 0.0
                state = 0

            elif state == 5:  # Race IvE
                # Sample physical time
                dt = random.expovariate(self.LAMBDA * active_hashrate)
                time_elapsed += dt

                # Accumulate shares and time
                shares_atk += self.alpha * dt
                shares_vic += self.beta * dt
                time_atk += self.alpha * dt
                time_tar += self.eta * dt

                # Random event
                rand = random.random() * active_hashrate
                reward = self._sample_block_reward(dt)
                if rand < self.alpha + self.beta:  # Attacker's infiltration hash power or victim pool finds a race-resolving block
                    atk_cut, vic_cut = _distribute_pool_reward(reward, shares_atk, shares_vic)
                    revenue_atk += suspended_inf_atk_cut + atk_cut
                    revenue_vic += suspended_inf_vic_cut + vic_cut
                    shares_atk, shares_vic = 0.0, 0.0
                elif rand < self.alpha + self.beta + self.eta:  # Target miners find a race-resolving block
                    revenue_tar += reward
                    if random.random() < self.gamma:  # Supports the infiltration branch
                        revenue_atk += suspended_inf_atk_cut
                        revenue_vic += suspended_inf_vic_cut
                    else:
                        revenue_tar += suspended_tar
                        revenue_oth += suspended_oth
                else:  # Other miners find a race-resolving block
                    revenue_oth += reward
                    if random.random() < self.gamma:  # Supports the infiltration branch
                        revenue_atk += suspended_inf_atk_cut
                        revenue_vic += suspended_inf_vic_cut
                    else:
                        revenue_tar += suspended_tar
                        revenue_oth += suspended_oth

                suspended_inf = suspended_inf_atk_cut = suspended_inf_vic_cut = 0.0
                suspended_tar = suspended_oth = 0.0
                state = 0

        # Final utility settlement
        cost_rate = self.c_block * self.LAMBDA / 1.0  # Normalized cost per unit time per unit hash power
        profit_rate_atk = (revenue_atk / time_elapsed) - (time_atk / time_elapsed) * cost_rate
        profit_rate_tar = (revenue_tar / time_elapsed) - (time_tar / time_elapsed) * cost_rate

        u_atk = (profit_rate_atk / self.alpha) / cost_rate
        u_tar = (profit_rate_tar / self.eta) / cost_rate

        return {
            'alpha': self.alpha, 'eta': self.eta, 'gamma': self.gamma,
            'Target_Strategy': self.target_strategy,
            'Optimal_r1': self.r1,
            'Optimal_r2': self.r2,
            'Attacker_Utility': u_atk, 'Target_Utility': u_tar
        }


def find_optimal_r(alpha, beta, eta, gamma, strategy, omega_b):
    """
    Uses discrete grid search to find the optimal r* that maximizes
    the attacker's utility under current environmental parameters.
    """
    best_r = 0.0
    max_utility = -np.inf

    # Build search grid for r (from 0.0 to 1.0, step 0.05 balances accuracy and speed)
    r_candidates = np.linspace(0.0, 1.0, 21)

    for r_test in r_candidates:
        # Instantiate simulator for exploratory runs (use a smaller num_events to accelerate search)
        sim = Simulator(
            alpha=alpha,
            beta=beta,
            eta=eta,
            gamma=gamma,
            r1=r_test,
            target_strategy=strategy,
            omega_b=omega_b
        )
        result = sim.run(num_events=10000)
        u_atk_test = result['Attacker_Utility']

        # Update optimal solution
        if u_atk_test > max_utility:
            max_utility = u_atk_test
            best_r = r_test

    return best_r


def run_experiment(params):
    alpha, omega_b, gamma, strategy = params
    best_r1 = find_optimal_r(alpha, 0.20, 0.10, gamma, strategy, omega_b)
    sim = Simulator(
        alpha=alpha, beta=0.20, eta=0.10, gamma=gamma,
        r1=best_r1, target_strategy=strategy, omega_b=omega_b
    )
    return sim.run(num_events=50000)


if __name__ == '__main__':
    print("Starting large-scale simulation data generation (CSV)...")

    # Build multi-dimensional parameter grid
    alphas = np.linspace(0.05, 0.45, 15)
    omega_bs = np.linspace(1.0, 2.5, 15)
    gammas = np.linspace(0.0, 1.0, 11)
    strategies = ['Mine', 'SPV', 'Stop']

    tasks = []
    for a in alphas:
        for w in omega_bs:
            for g in gammas:
                for s in strategies:
                    tasks.append((a, w, g, s))

    num_cores = max(1, cpu_count() - 1)
    with Pool(num_cores) as p:
        results = list(tqdm(p.imap_unordered(run_experiment, tasks), total=len(tasks)))

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv('./dataset/simulation_results.csv', index=False)
    print("Data generation complete! Saved to simulation_results.csv")
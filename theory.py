import numpy as np
import config


class PDoSTheoryCalculator:
    """
    Core theoretical equation calculator for the PDoS attack.
    Strictly based on the continuous-time Markov chain (CTMC) and normalized
    utility derivation formulas from the paper.
    """

    def __init__(self, alpha=None, beta=None, eta=None, gamma=None,
                 omega_b=None, lambda_t=None, lambda_w=None, e_rw=None,
                 k=None, lam=None, c=None):
        defaults = config.get_experiment_defaults()

        # Hash power allocation and validation
        self.alpha = float(alpha if alpha is not None else defaults['ALPHA'])
        self.beta = float(beta if beta is not None else defaults['BETA'])
        self.eta = float(eta if eta is not None else defaults['ETA'])
        self.gamma = float(gamma if gamma is not None else defaults['GAMMA'])
        self.delta = max(0.0, 1.0 - self.alpha - self.beta - self.eta)

        # Economic model parameters
        self.K = float(k if k is not None else defaults['K'])
        self.c = float(c if c is not None else defaults['c'])
        self.lam = float(lam if lam is not None else defaults['LAMBDA'])
        self.lambda_t = float(lambda_t if lambda_t is not None else defaults['lambda_t'])
        self.lambda_w = float(lambda_w if lambda_w is not None else defaults['lambda_w'])
        self.E_Rw = float(e_rw if e_rw is not None else defaults['e_rw'])
        # Baseline parameters
        self.R_b = self.K + (self.lambda_t + self.lambda_w * self.E_Rw) / self.lam
        self.omega_b = float(omega_b if omega_b is not None else defaults['omega_b'])
        self.M = (self.lambda_t + self.lambda_w * self.E_Rw) / (self.lam * self.R_b) if self.R_b > 0 else 0

        # Race winning probabilities
        self.p_A_3 = self.alpha + self.gamma * (self.eta + self.delta)
        self.p_A_4 = self.alpha + self.gamma * (self.beta + self.eta + self.delta)
        self.p_A_5 = self.alpha + self.beta + self.gamma * (self.eta + self.delta)

        self.p_V_3 = 1.0 - self.p_A_3
        self.p_E_4 = 1.0 - self.p_A_4
        self.p_E_5 = 1.0 - self.p_A_5

    def get_optimal_r2(self, omega_b, target_strategy='Mine'):
        """
        Calculates the optimal local infiltration ratio r2*.
        Based on the derivation of local profit maximization.
        """
        if self.alpha == 0: return 0.0

        # Hash power vacuum and inflation coefficient
        if target_strategy == 'Mine':
            delta_vac = self.alpha
        else:  # Stop or SPV
            delta_vac = self.alpha + self.eta

        e_s = 1 + self.M * (delta_vac / (1 - delta_vac)) if delta_vac < 1 else 1.0

        r2_opt = (self.beta * (np.sqrt(omega_b * e_s) - 1)) / self.alpha
        return max(0.0, min(1.0, r2_opt))

    def get_steady_state(self, r1, target_strategy):
        """
        Calculates the steady-state probability distribution vector pi of the CTMC
        under different target strategies.
        Strictly corresponds to Appendix A: Derivation of Steady-State Distributions.
        """
        pi = np.zeros(6)

        if target_strategy == 'Mine':
            part_func = 1.0 + self.alpha * (1.0 - self.alpha - r1 * self.beta)
            pi[0] = (1.0 - self.alpha) / part_func
            pi[1] = (1.0 - r1) * self.alpha / part_func
            pi[2] = r1 * self.alpha / part_func
            pi[3] = self.beta * (1.0 - r1) * self.alpha / part_func
            pi[4] = (self.eta + self.delta) * (1.0 - r1) * self.alpha / part_func
            pi[5] = (self.eta + self.delta) * r1 * self.alpha / part_func
        elif target_strategy == 'SPV':
            part_func = 1.0 + self.alpha * (1.0 - self.alpha - self.eta - r1 * self.beta)
            pi[0] = (1.0 - self.alpha) / part_func
            pi[1] = (1.0 - r1) * self.alpha / part_func
            pi[2] = r1 * self.alpha / part_func
            pi[3] = self.beta * (1.0 - r1) * self.alpha / part_func
            pi[4] = self.delta * (1.0 - r1) * self.alpha / part_func
            pi[5] = self.delta * r1 * self.alpha / part_func
        else:  # target_strategy == 'Stop'
            part_func = 1.0 - self.eta + self.alpha * (1.0 - self.alpha - self.eta - r1 * self.beta)
            pi[0] = (1.0 - self.alpha - self.eta) / part_func
            pi[1] = (1.0 - r1) * self.alpha / part_func
            pi[2] = r1 * self.alpha / part_func
            pi[3] = self.beta * (1.0 - r1) * self.alpha / part_func
            pi[4] = self.delta * (1.0 - r1) * self.alpha / part_func
            pi[5] = self.delta * r1 * self.alpha / part_func

        return pi

    def get_utilities(self, omega_b, r1, r2, target_strategy):
        """
        Calculates the normalized expected utility of the attacker and the target miner.
        Strictly corresponds to Section 5.4: Expected Utility Analysis.
        """
        pi = self.get_steady_state(r1, target_strategy)

        # Hash power vacuum
        if target_strategy == 'Mine':
            delta_vac = self.alpha
        else:  # Stop or SPV
            delta_vac = self.alpha + self.eta

        # Inflation coefficient
        e_s = 1 + self.M * (delta_vac / (1 - delta_vac)) if delta_vac < 1 else 1.0

        # Attacker's utility
        theta_a = 1.0 - (pi[1] + pi[2]) * (1.0 - r2)
        v_a = pi[3] + pi[4] + (1.0 / self.alpha) * (pi[3] * self.p_A_3 + pi[4] * self.p_A_4) if self.alpha > 0 else 0

        # Local time-weighted infiltration ratio
        r1_bar = (r1 * pi[0] + r2 * pi[1]) / (pi[0] + pi[1]) if (pi[0] + pi[1]) > 0 else 0
        r2_bar = (r1 * pi[0] + r2 * pi[2]) / (pi[0] + pi[2]) if (pi[0] + pi[2]) > 0 else 0

        # Normalized share extraction function
        def f(r):
            return r / (self.beta + r * self.alpha)

        s_a = (pi[0] * self.beta * f(r1) +
               e_s * (pi[3] * self.p_V_3 * f(r1_bar) + pi[2] * self.beta * f(r2_bar)) +
               pi[5] * (self.p_A_5 * f(r2_bar) + 1))

        # Final attacker utility
        u_a = self.c * (omega_b * (v_a + s_a) - theta_a)

        # Target miner's utility
        if target_strategy == 'Mine':
            v_t = (pi[0] + pi[3] + pi[4] + pi[5]) + e_s * (pi[1] * self.p_E_4 + pi[2] * self.p_E_5)
            theta_t = 1.0
        elif target_strategy == 'SPV':
            v_t = pi[0] + pi[3] + pi[4] + pi[5]
            theta_t = 1.0
        else:  # Stop
            v_t = pi[0] + pi[3] + pi[4] + pi[5]
            theta_t = v_t

        u_t = self.c * (omega_b * v_t - theta_t)

        return u_a, u_t

    def get_optimal_r1(self, omega_b, r2, target_strategy, resolution=200):
        """
        Finds the optimal initial infiltration ratio r1* that maximizes U_A
        using grid search.
        """
        best_r1 = 0.0
        max_u_a = -np.inf

        r1_candidates = np.linspace(0.0, 1.0, resolution)
        for r1_test in r1_candidates:
            u_a, _ = self.get_utilities(omega_b, r1_test, r2, target_strategy)
            if u_a > max_u_a:
                max_u_a = u_a
                best_r1 = r1_test

        return best_r1, max_u_a

    def get_constrained_optimal_r1(self, omega_b, r2, resolution=20):
        """
        Constrained optimization for r1:
        Prioritizes satisfying the target shutdown condition; if met, maximizes
        the attacker's utility.
        """
        best_r1 = 0.0
        max_u_a = -np.inf
        valid_r1s = []

        r1_candidates = np.linspace(0.0, 1.0, resolution)

        # 1. Filter out the set of r1 that can establish deterrence
        for r1_test in r1_candidates:
            _, u_t_mine = self.get_utilities(omega_b, r1_test, r2, 'Mine')
            _, u_t_stop = self.get_utilities(omega_b, r1_test, r2, 'Stop')
            if u_t_mine - u_t_stop < 0:
                valid_r1s.append(r1_test)

        # 2. If effective deterrence exists, select the one that maximizes self-interest
        if valid_r1s:
            for r1_test in valid_r1s:
                u_a, _ = self.get_utilities(omega_b, r1_test, r2, 'Stop')
                if u_a > max_u_a:
                    max_u_a = u_a
                    best_r1 = r1_test
        else:
            # 3. Deterrence failure (fallback to unconstrained optimization under 'Mine')
            for r1_test in r1_candidates:
                u_a_mine, _ = self.get_utilities(omega_b, r1_test, r2, 'Mine')
                if u_a_mine > max_u_a:
                    max_u_a = u_a_mine
                    best_r1 = r1_test

        return best_r1, max_u_a

    def get_deterrence_r(self, omega_b, resolution=100):
        """
        Globally searches for the (r1, r2) combination that minimizes the
        target's utility difference (Mine - Stop) to calculate the critical
        attacker hash power.
        """
        best_r1 = 0.0
        best_r2 = 0.0
        min_diff = float('inf')

        r_grid = np.linspace(0.0, 1.0, resolution)
        for r1 in r_grid:
            for r2 in r_grid:
                _, u_m = self.get_utilities(omega_b, r1, r2, 'Mine')
                _, u_s = self.get_utilities(omega_b, r1, r2, 'Stop')
                diff = u_m - u_s

                if diff < min_diff:
                    min_diff = diff
                    best_r1 = r1
                    best_r2 = r2

        return best_r1, best_r2, min_diff

    def get_epsilon_alpha_crit(self, omega_b, epsilon, gamma, mode='PDoS'):
        """
        Finds the critical attacker hash power threshold penetrating friction.
        """
        from scipy.optimize import brentq

        def epsilon_diff(alpha_test):
            calc = PDoSTheoryCalculator(alpha_test, self.beta, self.eta, gamma, omega_b,
                                        lambda_t=self.lambda_t, lambda_w=self.lambda_w,
                                        e_rw=self.E_Rw, k=self.K, lam=self.lam)

            if mode == 'BDoS':
                r1, r2 = 0.0, 0.0
                _, u_m = calc.get_utilities(omega_b, r1, r2, 'Mine')
                _, u_s = calc.get_utilities(omega_b, r1, r2, 'Stop')
                min_diff = u_m - u_s
            else:
                _, _, min_diff = calc.get_deterrence_r(omega_b)

            # Target only shuts down when the loss exceeds epsilon
            # min_diff < -epsilon is the lethal condition
            return min_diff + epsilon

        try:
            # Require the solver's error on the x-axis to be less than 1e-6
            return brentq(epsilon_diff, 0.001, 0.499, xtol=1e-6)
        except (ValueError, RuntimeError):
            return np.nan

    def get_micro_stop_utility_diff(self, omega_b, x, r1):
        """
        Calculates the shutdown utility difference of a microscopic individual
        under a given active target proportion x.
        """
        # 1. Macroscopic system parameter C(x) and hash power vacuum
        active_hash = self.beta + self.delta + x * self.eta
        vacuum = 1.0 - active_hash  # Equivalent to alpha + (1-x)*eta

        # 2. Normalized partition function D(x) and deterrence residence probability pi_det(x)
        d_x = 1.0 - (1.0 - x) * self.eta + self.alpha * (1.0 - self.alpha - (1.0 - x) * self.eta - r1 * self.beta)
        pi_det = self.alpha / d_x

        # 3. Dynamic MEV inflation coefficient e(x)
        e_x = 1.0 + self.M * (vacuum / active_hash)

        # 4. Microscopic weighted average winning rate p_bar_Ti(x) (assuming eta_i -> 0 under non-atomic game)
        p_bar = (1.0 - self.gamma) * ((1.0 - r1) * self.beta + self.delta + x * self.eta)

        # 5. Core: Individual microscopic shutdown utility difference Delta U_Ti^Stop(x)
        delta_u_stop = self.c * pi_det * (omega_b * e_x * p_bar - 1.0)

        return delta_u_stop

    def get_exact_pool_utility_diff(self, eta_i, delta_active, r1, mode='PDoS'):
        """
        Accurately calculates the utility difference (Mine vs Stop) for a discrete pool eta_i.
        Accounts for the modified winning rate when the target pool mines a block and supports its own branch.
        """
        if mode == 'BDoS':
            r1 = 0.0

        active_hash = self.beta + eta_i + delta_active
        eta_stopped = 1.0 - self.alpha - active_hash

        # ==========================================
        # 1. Evaluate expected utility of MINE strategy (U_Mine)
        # ==========================================
        vac_m = self.alpha + eta_stopped
        part_m = 1.0 - eta_stopped + self.alpha * (1.0 - self.alpha - eta_stopped - r1 * self.beta)

        # Steady-state probabilities, scenarios where target mines a block split from original 4 & 5
        pi_m = np.zeros(8)
        pi_m[0] = (1.0 - self.alpha - eta_stopped) / part_m
        pi_m[1] = (1.0 - r1) * self.alpha / part_m
        pi_m[2] = r1 * self.alpha / part_m
        pi_m[3] = self.beta * (1.0 - r1) * self.alpha / part_m
        pi_m[4] = eta_i * (1.0 - r1) * self.alpha / part_m
        pi_m[5] = eta_i * r1 * self.alpha / part_m
        pi_m[6] = delta_active * (1.0 - r1) * self.alpha / part_m
        pi_m[7] = delta_active * r1 * self.alpha / part_m

        # Survival probability when eta_i discovers a block itself
        p_e_4_eta = 1.0 - (self.alpha + self.gamma * (self.beta + delta_active))
        p_e_5_eta = 1.0 - (self.alpha + self.beta + self.gamma * delta_active)

        # Inflation coefficient
        e_s_m = 1.0 + self.M * (vac_m / (1.0 - vac_m)) if vac_m < 1.0 else 1.0

        v_t_mine = ((pi_m[0] + pi_m[3] + pi_m[4] + pi_m[5] + pi_m[6] + pi_m[7])
                    + e_s_m * (pi_m[1] * p_e_4_eta + pi_m[2] * p_e_5_eta))

        theta_t_mine = 1.0
        u_t_mine = self.c * (self.omega_b * v_t_mine - theta_t_mine)

        # ==========================================
        # 2. Evaluate expected utility of STOP strategy (U_Stop)
        # ==========================================
        eta_stopped_s = eta_stopped + eta_i
        part_s = 1.0 - eta_stopped_s + self.alpha * (1.0 - self.alpha - eta_stopped_s - r1 * self.beta)

        # Target shuts down, the two scenarios where target produces triggering blocks disappear
        pi_s = np.zeros(6)
        pi_s[0] = (1.0 - self.alpha - eta_stopped_s) / part_s
        pi_s[3] = self.beta * (1.0 - r1) * self.alpha / part_s
        pi_s[4] = delta_active * (1.0 - r1) * self.alpha / part_s
        pi_s[5] = delta_active * r1 * self.alpha / part_s

        v_t_stop = pi_s[0] + pi_s[3] + pi_s[4] + pi_s[5]
        theta_t_stop = v_t_stop
        u_t_stop = self.c * (self.omega_b * v_t_stop - theta_t_stop)

        return u_t_mine - u_t_stop

    def get_actual_u_a(self, omega_b, r1, r2, x):
        """
        Recalculates the non-linear Markov chain steady state based on the final
        survival ratio x. Accurately captures the difficulty adjustment and MEV
        inflation bonus brought by the hash power vacuum.
        """
        if self.alpha == 0: return 0.0

        # Reconstruct microscopic hash power distribution
        eta_act = x * self.eta
        eta_stop = (1.0 - x) * self.eta
        vacuum = self.alpha + eta_stop

        # Inflation coefficient
        e_s = 1.0 + self.M * (vacuum / (1.0 - vacuum)) if vacuum < 1.0 else 1.0

        # Partition function
        part = 1.0 - eta_stop + self.alpha * (1.0 - self.alpha - eta_stop - r1 * self.beta)

        # Reconstruct CTMC steady-state probabilities
        pi = np.zeros(6)
        pi[0] = (1.0 - self.alpha - eta_act) / part
        pi[1] = (1.0 - r1) * self.alpha / part
        pi[2] = r1 * self.alpha / part
        pi[3] = self.beta * (1.0 - r1) * self.alpha / part
        pi[4] = (eta_act + self.delta) * (1.0 - r1) * self.alpha / part
        pi[5] = (eta_act + self.delta) * r1 * self.alpha / part

        # Race winning probabilities
        p_a_3 = self.alpha + self.gamma * (eta_act + self.delta)
        p_a_4 = self.alpha + self.gamma * (self.beta + eta_act + self.delta)
        p_a_5 = self.alpha + self.beta + self.gamma * (eta_act + self.delta)
        p_v_3 = 1.0 - p_a_3

        # Calculate attacker's cost and revenue
        theta_a = 1.0 - (pi[1] + pi[2]) * (1.0 - r2)
        v_a = pi[3] + pi[4] + (1.0 / self.alpha) * (pi[3] * p_a_3 + pi[4] * p_a_4)

        # Reconstruct expected infiltration dividend
        r1_bar = (r1 * pi[0] + r2 * pi[1]) / (pi[0] + pi[1]) if (pi[0] + pi[1]) > 0 else 0
        r2_bar = (r1 * pi[0] + r2 * pi[2]) / (pi[0] + pi[2]) if (pi[0] + pi[2]) > 0 else 0

        def f(r):
            return r / (self.beta + r * self.alpha)

        s_a = (pi[0] * self.beta * f(r1) +
               e_s * (pi[3] * p_v_3 * f(r1_bar) + pi[2] * self.beta * f(r2_bar)) +
               pi[5] * (p_a_5 * f(r2_bar) + 1))

        return self.c * (omega_b * (v_a + s_a) - theta_a)
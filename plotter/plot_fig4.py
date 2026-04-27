import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as m_patches
from scipy.integrate import odeint
import theory, config

# Retrieve experimental baseline parameters
params = config.get_experiment_defaults()
beta_val = params.get('BETA')
eta_val = params.get('ETA')
gamma_fixed = params.get('GAMMA')
omega_b_val = params.get('OMEGA_B')
alpha_atk = params.get('ALPHA')

def replicator_dynamics(x, _t, calc, r1, omega_b):
    """
    Invokes the closed-form analytical solution from the theory model.
    """
    if isinstance(x, np.ndarray):
        x = x[0] if x.ndim > 0 else x.item()

    x_safe = np.clip(float(x), 1e-5, 0.9999)
    delta_u_stop = calc.get_micro_stop_utility_diff(omega_b, x_safe, r1)

    kappa = 2.25  # Evolution rate scaling constant
    dx_dt = kappa * x_safe * (1.0 - x_safe) * delta_u_stop
    return dx_dt

def plot_figure_4():
    # Apply global plot style
    config.apply_plot_style(22, 20, 26)
    print("Generating Figure 4: Evolutionary Dynamics and Hash Power Threshold Analysis...")

    # Create a 1x2 subplot layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.5, 5.5), dpi=300, sharey=True, constrained_layout=True)

    # Left subplot: Evolutionary logic based on replicator dynamics
    calc_base = theory.PDoSTheoryCalculator(alpha=alpha_atk, beta=beta_val, eta=eta_val, gamma=gamma_fixed,
                                            omega_b=omega_b_val)
    r1_p_dos, _, _ = calc_base.get_deterrence_r(omega_b_val, resolution=15)
    r1_b_dos = 0.0

    t = np.linspace(0, 100, 1000)
    x0 = 0.99

    x_b_dos = odeint(replicator_dynamics, x0, t, args=(calc_base, r1_b_dos, omega_b_val)).flatten()
    x_p_dos = odeint(replicator_dynamics, x0, t, args=(calc_base, r1_p_dos, omega_b_val)).flatten()

    zero_threshold = 0.001
    idx_p = np.where(x_p_dos <= zero_threshold)[0]
    idx_b = np.where(x_b_dos <= zero_threshold)[0]

    if len(idx_p) > 0 and len(idx_b) > 0:
        t_p_zero = t[idx_p[0]]
        t_b_zero = t[idx_b[0]]
        delta_t = t_b_zero - t_p_zero
        speed_up = t_b_zero / t_p_zero

        ax1.annotate('', xy=(t_p_zero, 0.10), xytext=(t_b_zero, 0.10),
                    arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
        ax1.text((t_p_zero + t_b_zero) / 2, 0.12, rf'$\Delta t \approx {delta_t:.1f}$',
                ha='center', fontweight='bold')
        ax1.text((t_p_zero + t_b_zero) / 2, 0.2, rf'{speed_up:.1f}$\times$ Faster',
                ha='center', color=config.COLOR_P, style='italic')
        ax1.vlines([t_p_zero, t_b_zero], 0, 0.10, colors=['gray'], linestyles='--', alpha=0.4)

    line_b, = ax1.plot(t, x_b_dos, label=rf'BDoS', color=config.COLOR_B, linestyle='--', linewidth=2.5,
                       marker='s', markevery=100, markersize=7, zorder=4)
    line_p, = ax1.plot(t, x_p_dos, label=rf'PDoS', color=config.COLOR_P, linestyle='-', linewidth=2.5,
                       marker='o', markevery=100, markersize=7, zorder=5)

    ax1.fill_between(t, x_p_dos, x_b_dos, facecolor='#FF7F0E', alpha=0.25, hatch='//', edgecolor='white', zorder=3)
    fill_proxy = m_patches.Patch(facecolor='#FF7F0E', alpha=0.25, hatch='//', edgecolor='white',
                                 label='Accelerated Loss')
    ax1.set_xlabel(r'Evolutionary Time ($t$)')
    ax1.set_ylabel(r'Target Surviving Proportion ($x$)')
    ax1.set_ylim(0, 1.05)
    ax1.set_xlim(0, 100)
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend(handles=[line_b, line_p, fill_proxy], loc='upper right', framealpha=0.95, edgecolor='gray', borderaxespad=0.2)
    ax1.set_title("(a) Evolutionary Dynamics", pad=15)

    # Right subplot: Hash power threshold and trace-driven Nash equilibrium
    print("Loading Trace-Driven replay pre-computed cache...")
    cache_path = config.CACHE_DIR / 'trace_nash_equilibrium.csv'
    try:
        df_cache = config.load_csv(cache_path)
    except FileNotFoundError:
        print(f"Error: Cache file not found at {cache_path}")
        return

    df_b = df_cache[(df_cache['mode'] == 'BDoS') & (df_cache['gamma'] == gamma_fixed)].sort_values('alpha')
    df_p = df_cache[(df_cache['mode'] == 'PDoS') & (df_cache['gamma'] == gamma_fixed)].sort_values('alpha')

    alphas = df_b['alpha'].values
    throughput_b = df_b['throughput'].values
    throughput_p = df_p['throughput'].values
    custom_marks = [0, 1] + list(range(4, len(alphas), 3))
    ax2.plot(alphas, throughput_b, label='BDoS', color=config.COLOR_B,
             linestyle='--', linewidth=2.5, marker='s', markevery=custom_marks, markersize=7)
    ax2.plot(alphas, throughput_p, label='PDoS', color=config.COLOR_P,
             linestyle='-', linewidth=2.5, marker='o', markevery=custom_marks, markersize=7)

    ax2.fill_between(alphas, throughput_p, throughput_b, facecolor='#FF7F0E', alpha=0.25,
                     hatch='//', edgecolor='white', label='Accelerated Loss')

    ax2.set_xlabel(r"Attacker's Mining Power ($\alpha$)")
    ax2.set_xlim((0.0, 0.45))
    ax2.grid(True, linestyle=':', alpha=0.6)

    ax2.set_title("(b) Trace-driven Nash Equilibrium", pad=15)

    # Save the plot
    save_path = config.FIGURES_DIR / 'Figure_4.pdf'
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"Figure successfully saved to: {save_path}")

if __name__ == '__main__':
    plot_figure_4()
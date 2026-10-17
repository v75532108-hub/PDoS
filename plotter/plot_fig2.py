import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as m_patches
import config

# Retrieve experimental parameters
params = config.get_experiment_defaults()
beta_val = params['BETA']
eta_val = params['ETA']


def get_curve_robust(cache_df, omega_bs, gamma, mode, beta=beta_val, eta=eta_val):
    """
    Processes theoretical curve boundaries based on pre-computed cache.
    Strictly filters required parameters and applies physical truncation logic.
    """
    # 1. Filter the corresponding parameter slice and strictly sort by omega_b to align the 500 grid points.
    df_sub = cache_df[
        (cache_df['gamma'] == gamma) &
        (cache_df['mode'] == mode) &
        (cache_df['beta'] == beta) &
        (cache_df['eta'] == eta)
        ].sort_values('omega_b')

    if df_sub.empty:
        raise ValueError(f"Missing parameter combination in cache: gamma={gamma}, mode={mode}, beta={beta}, eta={eta}")

    raw_alphas = df_sub['alpha_crit'].values
    alphas = []

    # 2. Iterate through the 500 pre-computed results to execute visual and physical truncation.
    for w, a in zip(omega_bs, raw_alphas):
        if w <= 1.0:
            alphas.append(0.0)
            continue

        if np.isnan(a):
            if not any(x > 0.001 for x in alphas):
                alphas.append(0.0)
            else:
                alphas.append(0.5)
        else:
            alphas.append(a)

    fill_a = np.array(alphas)

    # 3. Line clipping logic (fully preserving the original lossless logic)
    cross_0_idx = np.where(fill_a > 0.001)[0]
    cross_05_idx = np.where(fill_a >= 0.499)[0]

    line_a = fill_a.copy()
    if len(cross_05_idx) > 0:
        line_a[cross_05_idx[0] + 1:] = np.nan
    if len(cross_0_idx) > 0:
        first_non_zero = cross_0_idx[0]
        if first_non_zero > 0:
            line_a[:first_non_zero - 1] = np.nan

    return fill_a, line_a


def plot_figure_2():
    print("Plotting Figure 2...")
    config.apply_plot_style(24, 22, 18)
    theory_cache_path = config.CACHE_DIR / 'alpha_crit.csv'
    sim_cache_path = config.CACHE_DIR / 'sim_checkpoints.csv'
    theory_df = config.load_csv(theory_cache_path)
    sim_df = config.load_csv(sim_cache_path)

    omega_bs = np.linspace(0.01, 7.0, 500)
    gamma_list = [0.3, 0.5, 0.7, 0.9]
    subplot_labels = ['(a)', '(b)', '(c)', '(d)']

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=300, sharex=True, sharey=True)
    axes = axes.flatten()

    for i, gamma in enumerate(gamma_list):
        ax = axes[i]

        fill_b, line_b_a = get_curve_robust(theory_df, omega_bs, gamma, 'BDoS', beta_val, eta_val)
        fill_p, line_p_a = get_curve_robust(theory_df, omega_bs, gamma, 'PDoS', beta_val, eta_val)

        # Shaded regions
        ax.fill_between(omega_bs, 0, fill_p, color='#2ca02c', alpha=0.25, zorder=1)
        ax.fill_between(omega_bs, fill_p, fill_b, facecolor='#ff7f0e', alpha=0.35, hatch='//', edgecolor='white',
                        zorder=2)
        ax.fill_between(omega_bs, fill_b, 0.5, color='#d62728', alpha=0.35, zorder=1)

        # Theoretical solid lines
        ax.plot(omega_bs, line_b_a, color=config.COLOR_B, linestyle='-', linewidth=1.8, zorder=4)
        ax.plot(omega_bs, line_p_a, color=config.COLOR_P, linestyle='-', linewidth=1.8, zorder=4)

        print(f"Rendering simulation validation scatter points for gamma={gamma}...")

        sim_b = sim_df[(sim_df['gamma'] == gamma) & (sim_df['mode'] == 'BDoS')]
        if not sim_b.empty:  # BDoS scatter points: hollow blue squares
            ax.plot(sim_b['omega_b'], sim_b['alpha_crit_sim'], marker='s', color='none',
                    markeredgecolor=config.COLOR_B, markerfacecolor='white', markersize=6.5,
                    markeredgewidth=1.5, linestyle='None', zorder=5)

        sim_p = sim_df[(sim_df['gamma'] == gamma) & (sim_df['mode'] == 'PDoS')]
        if not sim_p.empty:  # PDoS scatter points: hollow red circles
            ax.plot(sim_p['omega_b'], sim_p['alpha_crit_sim'], marker='o', color='none',
                    markeredgecolor=config.COLOR_P, markerfacecolor='white', markersize=7,
                    markeredgewidth=1.5, linestyle='None', zorder=5)

        ax.set_title(rf'{subplot_labels[i]} Propagation Advantage $\gamma={gamma}$', pad=10)
        ax.set_ylim(0, 0.5)
        ax.set_xlim(0, 7.0)
        ax.margins(x=0, y=0)

        ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
        ax.grid(True, linestyle=':', alpha=0.5, zorder=0)

        if i % 2 == 0:
            ax.set_ylabel(r'Critical Threshold ($\alpha_{\mathcal{A}}^*$)')
        if i >= 2:
            ax.set_xlabel(r'Profitability Factor ($\omega_b$)')

    # Legend patches
    green_patch = m_patches.Patch(color='#2ca02c', alpha=0.25, label='Secure Region for both')
    red_patch = m_patches.Patch(color='#d62728', alpha=0.35, label='Vulnerable to both')
    orange_patch = m_patches.Patch(facecolor='#ff7f0e', alpha=0.35, hatch='//', edgecolor='white',
                                   label='Vulnerable to PDoS (Ours)')

    line_b_dos = plt.Line2D([0], [0], color=config.COLOR_B, linestyle='-', linewidth=1.8, label='Theory: BDoS Boundary')
    line_p_dos = plt.Line2D([0], [0], color=config.COLOR_P, linestyle='-', linewidth=1.8, label='Theory: PDoS Boundary')

    # Legend rendering
    handles: list = [line_b_dos, line_p_dos]
    scatter_b_dos = plt.Line2D([0], [0], color='none', marker='s', markeredgecolor=config.COLOR_B,
                               markerfacecolor='white', markersize=6.5, markeredgewidth=1.5,
                               label='Sim: BDoS Checkpoint')
    scatter_p_dos = plt.Line2D([0], [0], color='none', marker='o', markeredgecolor=config.COLOR_P,
                               markerfacecolor='white', markersize=7, markeredgewidth=1.5,
                               label='Sim: PDoS Checkpoint')
    handles.extend([scatter_b_dos, scatter_p_dos])
    handles.extend([green_patch, red_patch, orange_patch])
    axes[3].legend(handles=handles, loc='lower left', framealpha=0.95, borderaxespad=0.2)

    plt.tight_layout()
    save_path = config.FIGURES_DIR / 'Figure_2.pdf'
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"Figure successfully saved to: {save_path}")


if __name__ == '__main__':
    plot_figure_2()
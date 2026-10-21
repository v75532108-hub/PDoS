import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import config


def plot_figure_3():
    # Apply global plot style
    config.apply_plot_style(24, 22, 18)
    print("Loading cost cache to generate Figure 3...")

    df_cache = config.load_csv(config.CACHE_DIR / 'net_cost.csv')
    alphas = [0.20, 0.15, 0.10]
    line_alphas = [1.0, 0.65, 0.35]
    markers = ['s', 'o', '^']  # Square, circle, triangle

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.5, 5.5), dpi=300, constrained_layout=True)

    # Left subplot: Attack net cost rate
    ax1.axhline(0, color='black', linestyle='-', linewidth=1.5, zorder=3)
    ax1.axhspan(0, 0.45, facecolor='#d62728', alpha=0.1, zorder=0)
    ax1.axhspan(-0.25, 0, facecolor='#2ca02c', alpha=0.15, zorder=0)

    for i, a in enumerate(alphas):
        data_b = df_cache[(df_cache['alpha'] == a) & (df_cache['mode'] == 'BDoS')]
        data_p = df_cache[(df_cache['alpha'] == a) & (df_cache['mode'] == 'PDoS')]

        w_vals = data_b['omega_b'].values
        cost_b = data_b['net_cost'].values
        cost_p = data_p['net_cost'].values

        ax1.plot(w_vals, cost_b, label=rf'BDoS $\alpha={a}$',
                 color=config.COLOR_B, alpha=line_alphas[i], linestyle='--',
                 marker=markers[i], markevery=50, markersize=8, linewidth=2.5, zorder=4)

        ax1.plot(w_vals, cost_p, label=rf'PDoS $\alpha={a}$',
                 color=config.COLOR_P, alpha=line_alphas[i], linestyle='-',
                 marker=markers[i], markevery=50, markersize=8, linewidth=2.5, zorder=4)

        valid_idx = np.where(~np.isnan(cost_b))[0]
        if len(valid_idx) > 0 and valid_idx[-1] < len(w_vals) - 1:
            last_i = valid_idx[-1]
            if -0.25 <= cost_b[last_i] <= 0.45:
                ax1.plot(w_vals[last_i], cost_b[last_i], marker='X', color=config.COLOR_B,
                         markeredgecolor='black', markeredgewidth=1.2, markersize=10, zorder=6)

    text_x, text_y = 2.23, -0.14
    ax1.text(text_x, text_y, 'Deterrence fails', fontsize=12, fontweight='bold', color=config.COLOR_B,
             ha='right', va='center', zorder=7,
             bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor=config.COLOR_B, lw=1.2, alpha=0.95))

    # Arrow 1: to alpha=0.10 endpoint
    ax1.annotate('', xy=(1.985, -0.108), xytext=(2.03, text_y - 0.01),
                 arrowprops=dict(facecolor=config.COLOR_B, edgecolor=config.COLOR_B, arrowstyle='->', lw=1.8),
                 zorder=6)

    # Arrow 2: to alpha=0.15 endpoint
    ax1.annotate('', xy=(2.075, -0.208), xytext=(2.08, text_y - 0.02),
                 arrowprops=dict(facecolor=config.COLOR_B, edgecolor=config.COLOR_B, arrowstyle='->', lw=1.8),
                 zorder=6)

    ax1.text(2.23, 0.015, 'Cost-Incurring', color='darkred', fontweight='bold', ha='right', va='bottom', zorder=6)
    ax1.text(2.23, -0.015, 'Self-Sustaining', color='darkgreen', fontweight='bold', ha='right', va='top', zorder=6)

    ax1.set_xlabel(r'Profitability Factor ($\omega_b$)')
    ax1.set_ylabel(r'Net Cost Rate ($\mathcal{C}_{\mathcal{A}}^* / c$)')
    ax1.set_xlim(1.0, 2.25)
    ax1.set_ylim(-0.25, 0.45)
    ax1.margins(x=0)

    ax1.set_xticks(np.arange(1.0, 2.3, 0.25))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.grid(True, linestyle=':', alpha=0.5, zorder=1)

    handles_list_ax1 = []
    for i, a in enumerate(alphas):
        line_b = plt.Line2D([0], [0], color=config.COLOR_B, alpha=line_alphas[i], linestyle='--', marker=markers[i],
                            markersize=8, linewidth=2.5, label=rf'BDoS ($\alpha={a}$)')
        line_p = plt.Line2D([0], [0], color=config.COLOR_P, alpha=line_alphas[i], linestyle='-', marker=markers[i],
                            markersize=8, linewidth=2.5, label=rf'PDoS ($\alpha={a}$)')
        handles_list_ax1.extend([line_b, line_p])

    ax1.legend(handles=handles_list_ax1, loc='upper right', framealpha=0.95, edgecolor='gray',
               fancybox=False,
               borderaxespad=0.1)
    ax1.set_title("(a) Minimum Net Cost Rate", pad=15)

    # Right subplot: Attack sustainable time
    for i, a in enumerate(alphas):
        df_sub = df_cache[
            (df_cache['alpha'] == a) & (df_cache['mode'] == 'BDoS') & (df_cache['omega_b'] <= 1.8)].sort_values(
            'omega_b')
        omega_bs = df_sub['omega_b'].values
        costs = df_sub['net_cost'].tolist()

        costs = np.array(costs)
        for j in range(1, len(costs)): costs[j] = min(costs[j], costs[j - 1])

        t_max = np.full_like(costs, 1e8)
        asym_w = None

        for j in range(len(costs)):
            if costs[j] > 1e-9:
                t_max[j] = a / costs[j]
            else:
                t_max[j] = 1e8  # Break out of bounds (infinity)
                asym_w = omega_bs[j]
                break

        ax2.plot(omega_bs, t_max, label=rf'BDoS $\alpha={a}$', color=config.COLOR_B,
                 alpha=line_alphas[i], linestyle='--', linewidth=2.0,
                 marker=markers[i], markevery=30, markersize=8)

        if asym_w is not None:
            ax2.axvline(asym_w, color=config.COLOR_B, alpha=line_alphas[i] * 0.8, linestyle=':', linewidth=2.0)

    for i, a in enumerate(alphas):
        df_sub = df_cache[
            (df_cache['alpha'] == a) & (df_cache['mode'] == 'PDoS') & (df_cache['omega_b'] <= 1.8)].sort_values(
            'omega_b')
        omega_bs = df_sub['omega_b'].values
        costs = df_sub['net_cost'].tolist()

        costs = np.array(costs)
        for j in range(1, len(costs)): costs[j] = min(costs[j], costs[j - 1])

        t_max = np.full_like(costs, 1e8)
        asym_w = None

        for j in range(len(costs)):
            if costs[j] > 1e-9:
                t_max[j] = a / costs[j]
            else:
                t_max[j] = 1e8  # Break out of bounds (infinity)
                asym_w = omega_bs[j]
                break

        ax2.plot(omega_bs, t_max, label=rf'PDoS ($\alpha={a}$)', color=config.COLOR_P,
                 alpha=line_alphas[i], linestyle='-',
                 linewidth=2.5, zorder=5,
                 marker=markers[i], markevery=15, markersize=8)

        if asym_w is not None:
            ax2.axvline(asym_w, color=config.COLOR_P, alpha=line_alphas[i] * 0.8, linestyle=':', linewidth=2.0)

    # Axis truncation and infinity annotations
    ax2.set_yscale('log')
    ax2.set_ylim(1e-1, 1e6)
    ax2.set_xlim(1.0, 1.8)

    y_ticks = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
    y_labels = [r'$10^{-1}$', r'$10^0$', r'$10^1$', r'$10^2$', r'$10^3$', r'$10^4$', r'$\vdots$', r'$\infty$']
    ax2.set_yticks(y_ticks)
    ax2.set_yticklabels(y_labels, fontsize=18)

    gridlines = ax2.yaxis.get_gridlines()
    if len(gridlines) > 2:
        gridlines[-1].set_visible(False)
        gridlines[-2].set_visible(False)

    ax2.set_xlabel(r'Profitability Factor ($\omega_b$)')
    ax2.set_ylabel(r'Sustainable Time ($T_{\mathcal{A}}^*$)')
    ax2.grid(True, which="major", linestyle='-', alpha=0.3, zorder=1)
    ax2.grid(True, which="minor", linestyle=':', alpha=0.15, zorder=1)

    ax2.set_title("(b) Maximum Sustainable Time", pad=15)

    save_path = config.FIGURES_DIR / 'Figure_3.pdf'
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"Figure successfully saved to: {save_path}")


if __name__ == '__main__':
    plot_figure_3()
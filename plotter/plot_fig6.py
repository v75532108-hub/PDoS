import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from theory import PDoSTheoryCalculator


R_VALUES = np.round(np.linspace(0.0, 1.0, 6), 1)
FONT_BUMP = 5


def _joint_policy(calc, omega_b, share_retention=1.0, resolution=101):
    """Deterrence first, then attacker utility, using one deployed policy."""
    return calc.get_joint_policy(
        omega_b, target_strategy='Stop', share_retention=share_retention,
        resolution=resolution
    )


def _annotate_heatmap(ax, image, matrix, fmt):
    count = len(R_VALUES)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            red, green, blue, _ = image.cmap(image.norm(value))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            color = 'white' if luminance < 0.53 else 'black'
            ax.text(col, row, format(value, fmt), ha='center', va='center',
                    color=color,
                    fontsize=config.PAPER_HEATMAP_SIZE + FONT_BUMP,
                    fontweight='bold')
    tick_labels = [f'{value:.1f}'.rstrip('0').rstrip('.')
                   for value in R_VALUES]
    ax.set_xticks(range(count), labels=tick_labels)
    ax.set_yticks(range(count), labels=tick_labels)
    ax.set_xticks(np.arange(count + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(count + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=0.8, alpha=0.85)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.set_xlabel(r'$r_1$')
    ax.set_ylabel(r'$r_2$')


def plot_figure_6():
    config.apply_plot_style(
        axes_size=config.PAPER_AXES_SIZE + FONT_BUMP,
        tick_size=config.PAPER_TICK_SIZE + FONT_BUMP,
        legend_size=config.PAPER_LEGEND_SIZE + FONT_BUMP,
        title_size=config.PAPER_TITLE_SIZE + FONT_BUMP,
    )
    plt.rcParams['font.size'] = config.PAPER_FONT_SIZE + FONT_BUMP
    full = PDoSTheoryCalculator()

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 4.75), dpi=300,
                             constrained_layout=True)
    fig.set_constrained_layout_pads(
        w_pad=0.025, h_pad=0.02, wspace=0.035, hspace=0.02
    )

    # (a) Reward-source sensitivity: sweep non-coinbase share M while holding physical cost fixed.
    empirical_omega = 1.6
    empirical_m = full.M
    coinbase_omega = empirical_omega * (1.0 - empirical_m)
    reward_shares = np.linspace(0.0, 0.05, 101)
    reward_costs = []
    for reward_share in reward_shares:
        full.M = reward_share
        omega = coinbase_omega / (1.0 - reward_share)
        _, _, u_a, _ = _joint_policy(full, omega, resolution=201)
        reward_costs.append(-u_a / full.c)
    full.M = empirical_m

    ax = axes[0]
    ax.axhline(0, color='black', linewidth=1.2)
    ax.plot(100.0 * reward_shares, reward_costs, color='#6A3D9A',
            linewidth=2.8)
    marker_shares = np.array([0.0, empirical_m])
    marker_costs = np.interp(marker_shares, reward_shares, reward_costs)
    ax.scatter(100.0 * marker_shares, marker_costs, s=95,
               color=['#333333', '#E66101'], edgecolor='white', zorder=4)
    ax.annotate('Coinbase only\n$M=0$',
                (0.0, marker_costs[0]), xytext=(0.04, 0.82),
                textcoords='axes fraction',
                arrowprops={'arrowstyle': '-', 'color': '#555555'},
                fontsize=config.PAPER_ANNOTATION_SIZE + FONT_BUMP)
    ax.annotate(
                rf'Empirical mix' + '\n'
                + rf'$M={100 * empirical_m:.3f}\%$',
                (100.0 * empirical_m, marker_costs[1]),
                xytext=(0.30, 0.68), textcoords='axes fraction',
                arrowprops={'arrowstyle': '-', 'color': '#555555'},
                fontsize=config.PAPER_ANNOTATION_SIZE + FONT_BUMP)
    ax.set_xlabel(r'Non-coinbase share $M$ (%)')
    ax.set_ylabel(r'Net cost $\mathcal{C}_{\mathcal{A}}/c$')
    ax.set_title('(a) Reward Sensitivity')
    ax.set_xlim(0.0, 5.0)
    ax.set_ylim(min(reward_costs) - 0.012, max(reward_costs) + 0.038)
    ax.grid(True, linestyle=':', alpha=0.45)

    # (b,c) A finer 6x6 grid of fixed schedules.
    omega = 1.6
    count = len(R_VALUES)
    net_cost = np.zeros((count, count))
    scheduling_saving = np.zeros((count, count))
    for row, r2 in enumerate(R_VALUES):
        for col, r1 in enumerate(R_VALUES):
            u_a, _ = full.get_utilities(omega, r1, r2, 'Stop')
            net_cost[row, col] = -u_a / full.c
            u_static, _ = full.get_utilities(omega, r1, r1, 'Stop')
            static_cost = -u_static / full.c
            scheduling_saving[row, col] = static_cost - net_cost[row, col]

    r1_star, r2_star, _, _ = _joint_policy(full, omega)

    cost_norm = TwoSlopeNorm(
        vmin=float(net_cost.min()), vcenter=0.0, vmax=float(net_cost.max())
    )
    im1 = axes[1].imshow(net_cost, cmap='RdYlGn_r', norm=cost_norm,
                         aspect='auto')
    _annotate_heatmap(axes[1], im1, net_cost, '.2f')
    selected_col = int(np.argmin(np.abs(R_VALUES - r1_star)))
    selected_row = int(np.argmin(np.abs(R_VALUES - r2_star)))
    axes[1].add_patch(Rectangle(
        (selected_col - 0.48, selected_row - 0.48), 0.96, 0.96,
        fill=False, edgecolor='gold', linewidth=4.0,
    ))
    axes[1].set_title('(b) Fixed-Schedule Cost')
    axes[1].text(
        selected_col, selected_row + 0.29, 'OPT',
        ha='center', va='center', color='#7A5A00',
        fontsize=config.PAPER_HEATMAP_SIZE + FONT_BUMP - 2,
        fontweight='bold'
    )
    cbar1 = fig.colorbar(im1, ax=axes[1], shrink=0.86, pad=0.025)
    cbar1.ax.tick_params(labelsize=config.PAPER_TICK_SIZE + FONT_BUMP - 2)

    saving_limit = float(np.max(np.abs(scheduling_saving)))
    im2 = axes[2].imshow(scheduling_saving, cmap='coolwarm',
                         vmin=-saving_limit, vmax=saving_limit,
                         aspect='auto')
    _annotate_heatmap(axes[2], im2, scheduling_saving, '+.2f')
    axes[2].add_patch(Rectangle(
        (selected_col - 0.48, selected_row - 0.48), 0.96, 0.96,
        fill=False, edgecolor='gold', linewidth=4.0
    ))
    axes[2].text(
        selected_col, selected_row + 0.29, 'OPT',
        ha='center', va='center', color='#7A5A00',
        fontsize=config.PAPER_HEATMAP_SIZE + FONT_BUMP - 2,
        fontweight='bold'
    )
    axes[2].set_title('(c) Saving vs. Static')
    cbar2 = fig.colorbar(im2, ax=axes[2], shrink=0.86, pad=0.025)
    cbar2.ax.tick_params(labelsize=config.PAPER_TICK_SIZE + FONT_BUMP - 2)

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_path = config.FIGURES_DIR / 'Figure_6.pdf'
    fig.savefig(save_path, format='pdf')
    print(f'Figure successfully saved to: {save_path}')


if __name__ == '__main__':
    plot_figure_6()

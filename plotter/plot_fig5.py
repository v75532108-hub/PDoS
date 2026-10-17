import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from matplotlib.ticker import FuncFormatter
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


FONT_BUMP = 5


def plot_figure_5():
    print("Loading trace-driven replay pre-computed cache...")

    config.apply_plot_style(
        axes_size=config.PAPER_AXES_SIZE + FONT_BUMP,
        tick_size=config.PAPER_TICK_SIZE + FONT_BUMP,
        legend_size=config.PAPER_LEGEND_SIZE + FONT_BUMP,
        title_size=config.PAPER_TITLE_SIZE + FONT_BUMP,
    )
    plt.rcParams['font.size'] = config.PAPER_FONT_SIZE + FONT_BUMP

    df_plot = config.load_csv(config.CACHE_DIR / 'replay.csv')
    heights = df_plot['height']

    # Sort hash power in descending order
    alphas = [0.2, 0.15, 0.1]

    line_alphas = {0.2: 1.0, 0.15: 0.75, 0.1: 0.5}
    markers = {0.2: 's', 0.15: 'o', 0.1: '^'}

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(17.5, 4.75), dpi=300, sharex=True,
        constrained_layout=True
    )
    fig.set_constrained_layout_pads(
        w_pad=0.025, h_pad=0.02, wspace=0.04, hspace=0.02
    )

    # Subplot 1: Network liveness degradation
    base_a = 0.15
    line_survival_p, = ax1.plot(
        heights, df_plot[f'x_p_{base_a}'], color=config.COLOR_P, lw=2,
        alpha=line_alphas[base_a], marker=markers[base_a], markersize=5,
        markevery=25, rasterized=True, label='PDoS'
    )

    line_survival_b, = ax1.plot(
        heights, df_plot[f'x_b_{base_a}'], color=config.COLOR_B, lw=2,
        linestyle='--', alpha=line_alphas[base_a], marker=markers[base_a],
        markersize=5, markevery=25, rasterized=True, label='BDoS'
    )

    ax1.set_ylabel(r"Target survival $x$")
    ax1.set_xlabel("Block Height")
    ax1.set_ylim((0.0, 1.0))
    ax1.grid(True, linestyle=':', alpha=0.4)
    ax1.legend(
        handles=[line_survival_p, line_survival_b],
        loc='upper right', ncol=2, frameon=False
    )
    ax1.set_title(r'(a) Target Survival ($\alpha=0.15$)')

    # Subplot 2: Cumulative utility. Highlight the cumulative gap between PDoS and BDoS at baseline alpha.
    base_u_p = np.asarray(df_plot[f'u_p_{base_a}'])
    base_u_b = np.asarray(df_plot[f'u_b_{base_a}'])
    ax2.fill_between(heights, base_u_b, base_u_p,
                     color='#E66101', alpha=0.16, zorder=0,
                     label=rf'PDoS advantage ($\alpha={base_a}$)')

    for a in alphas:
        ax2.plot(heights, df_plot[f'u_p_{a}'], color=config.COLOR_P, lw=2.5,
                 alpha=line_alphas[a], marker=markers[a], markersize=5, markevery=0.03,
                 label=rf"PDoS ($\alpha={a}$)")
    for a in alphas:
        ax2.plot(heights, df_plot[f'u_b_{a}'], color=config.COLOR_B, lw=2.5,
                 linestyle='--', alpha=line_alphas[a], marker=markers[a], markersize=5, markevery=0.03,
                 label=rf"BDoS ($\alpha={a}$)")

    # Add zero-axis reference line
    ax2.axhline(0, color='black', linestyle='-', lw=1.5, zorder=1)

    ax2.set_ylabel(r"Cumulative utility $\Sigma U_{\mathcal{A}}$")
    ax2.set_xlabel("Block Height")
    ax2.set_xlim((heights.iloc[0], heights.iloc[-1]))

    def format_k(x, _pos):
        if abs(x) >= 1000:
            return f"{x / 1000:g}k"
        return f"{x:g}"

    ax2.xaxis.set_major_formatter(FuncFormatter(format_k))
    ax1.xaxis.set_major_formatter(FuncFormatter(format_k))
    ax2.yaxis.set_major_formatter(FuncFormatter(format_k))
    ax2.grid(True, linestyle=':', alpha=0.4)

    # Halving vertical line
    halving_block = 840000
    for ax in [ax1, ax2]:
        ax.axvline(x=halving_block, color='black', linestyle='--', linewidth=2.5, alpha=0.8, zorder=5)

    ax1.text(halving_block - 1500, ax1.get_ylim()[1] * 0.5, 'Halving', color='black',
             fontweight='bold', fontstyle='italic', ha='right', va='top',
             fontsize=config.PAPER_ANNOTATION_SIZE + FONT_BUMP,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3),
             zorder=6)

    advantage_proxy = plt.Rectangle(
        (0, 0), 1, 1, facecolor='#E66101', alpha=0.16,
        label=rf'PDoS advantage ($\alpha={base_a}$)'
    )
    alpha_handles = [
        plt.Line2D(
            [0], [0], color='#333333', linestyle='None', marker=markers[a],
            markersize=7, label=rf'$\alpha={a:.2f}$'
        )
        for a in alphas
    ]
    ax2.legend(
        handles=[advantage_proxy] + alpha_handles,
        loc='upper left', ncol=2, columnspacing=0.8, handlelength=1.6
    )
    ax2.set_title('(b) Cumulative Attacker Utility')

    # Save the plot
    save_path = config.FIGURES_DIR / 'Figure_5.pdf'
    fig.savefig(save_path, format='pdf')
    print(f"\nTrace-driven replay rendering complete! Figure saved to: {save_path}")


if __name__ == '__main__':
    plot_figure_5()

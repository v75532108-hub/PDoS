import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import config


def plot_figure_5():
    print("Loading trace-driven replay pre-computed cache...")

    # Increase font sizes to fit academic paper standards
    config.apply_plot_style(24, 22, 22)

    df_plot = config.load_csv(config.CACHE_DIR / 'replay.csv')
    heights = df_plot['height']

    # Sort hash power in descending order
    alphas = [0.2, 0.15, 0.1]

    line_alphas = {0.2: 1.0, 0.15: 0.75, 0.1: 0.5}
    markers = {0.2: 's', 0.15: 'o', 0.1: '^'}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), dpi=300, sharex=True,
                                   gridspec_kw={'height_ratios': [1, 1.2], 'hspace': 0.0})

    # Subplot 1: Network liveness degradation
    base_a = 0.15
    ax1.plot(heights, df_plot[f'x_p_{base_a}'], color=config.COLOR_P, lw=2,
             alpha=line_alphas[base_a], marker=markers[base_a], markersize=5, rasterized=True)

    ax1.plot(heights, df_plot[f'x_b_{base_a}'], color=config.COLOR_B, lw=2,
             linestyle='--', alpha=line_alphas[base_a], marker=markers[base_a], markersize=5, rasterized=True)

    ax1.set_ylabel("Target Surviving Proportion ($x$)")
    ax1.set_ylim((0.0, 1.0))
    ax1.grid(True, linestyle=':', alpha=0.4)

    # Subplot 2: Cumulative utility
    for a in sorted(alphas):
        u_p = np.array(df_plot[f'u_p_{a}'])
        u_b = np.array(df_plot[f'u_b_{a}'])

        # PDoS green fill (profitable regime)
        ax2.fill_between(heights, 0, u_p, where=(u_p >= 0), color='green', alpha=0.1, zorder=0, rasterized=True)
        # BDoS gray fill (positive but declining utility regime)
        ax2.fill_between(heights, 0, u_b, where=(u_b >= 0), color='dimgray', alpha=0.5, zorder=0, rasterized=True)
        # BDoS dark red fill (net-negative cumulative-utility regime)
        ax2.fill_between(heights, 0, u_b, where=(u_b < 0), color='darkred', alpha=0.4, zorder=0, rasterized=True)

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

    ax2.set_ylabel(r"Cumulative Utility ($\Sigma U_{\mathcal{A}}$)")
    ax2.set_xlabel("Block Height")
    ax2.set_xlim((heights.iloc[0], heights.iloc[-1]))

    def format_k(x, _pos):
        if abs(x) >= 1000:
            return f"{x / 1000:g}k"
        return f"{x:g}"

    ax2.xaxis.set_major_formatter(FuncFormatter(format_k))
    ax2.yaxis.set_major_formatter(FuncFormatter(format_k))
    ax2.grid(True, linestyle=':', alpha=0.4)

    # Halving vertical line
    halving_block = 840000
    for ax in [ax1, ax2]:
        ax.axvline(x=halving_block, color='black', linestyle='--', linewidth=2.5, alpha=0.8, zorder=5)

    ax1.text(halving_block - 1500, ax1.get_ylim()[1] * 0.5, 'Halving', color='black',
             fontweight='bold', fontstyle='italic', ha='right', va='top',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3), zorder=6)

    # Legend configuration: placed in the lower subplot
    ax2.legend(loc='upper left', framealpha=0.9, ncol=2)

    plt.tight_layout()

    # Save the plot
    save_path = config.FIGURES_DIR / 'Figure_5.pdf'
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"\nTrace-driven replay rendering complete! Figure saved to: {save_path}")


if __name__ == '__main__':
    plot_figure_5()
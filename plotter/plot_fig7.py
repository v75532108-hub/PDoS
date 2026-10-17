import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from theory import PDoSTheoryCalculator


FONT_BUMP = 5


def _net_cost(calc, omega_b, r1, r2, retention=1.0):
    u_a, _ = calc.get_utilities(
        omega_b, r1, r2, 'Stop', share_retention=retention
    )
    return -u_a / calc.c


def _target_gap(calc, omega_b, r1):
    _, u_m = calc.get_utilities(omega_b, r1, 0.0, 'Mine')
    _, u_s = calc.get_utilities(omega_b, r1, 0.0, 'Stop')
    return (u_m - u_s) / calc.c


def _zero_crossing(value_at):
    """Exact crossing for an affine fixed-policy utility in omega_b."""
    y0 = float(value_at(0.0))
    slope = float(value_at(1.0)) - y0
    return -y0 / slope if slope > 0 else np.inf


def _attacker_threshold(calc, r1, retention):
    # For evaluated parameters, optimal deterrence policy satisfies r2*=0 at the break-even boundary.
    return _zero_crossing(
        lambda omega: calc.get_utilities(
            omega, r1, 0.0, 'Stop', share_retention=retention
        )[0] / calc.c
    )


def _target_threshold(calc, r1):
    return _zero_crossing(lambda omega: _target_gap(calc, omega, r1))


def plot_figure_7():
    config.apply_plot_style(
        axes_size=config.PAPER_AXES_SIZE + FONT_BUMP,
        tick_size=config.PAPER_TICK_SIZE + FONT_BUMP,
        legend_size=config.PAPER_LEGEND_SIZE + FONT_BUMP,
        title_size=config.PAPER_TITLE_SIZE + FONT_BUMP,
    )
    plt.rcParams['font.size'] = config.PAPER_FONT_SIZE + FONT_BUMP
    calc = PDoSTheoryCalculator()
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 4.75), dpi=300,
                             constrained_layout=True)
    fig.set_constrained_layout_pads(
        w_pad=0.025, h_pad=0.02, wspace=0.035, hspace=0.02
    )

    # (a) Immediate payout: deterrence and attacker cost trade-off along infiltration policy frontier.
    omega = 1.6
    r1_grid = np.linspace(0.0, 1.0, 201)
    r2_grid = np.array([
        calc.get_optimal_r2(omega, r1, 'Stop', share_retention=1.0)
        for r1 in r1_grid
    ])
    costs = np.array([
        _net_cost(calc, omega, r1, r2)
        for r1, r2 in zip(r1_grid, r2_grid)
    ])
    gaps = np.array([_target_gap(calc, omega, r1) for r1 in r1_grid])
    amplification = 100.0 * ((-gaps) / (-gaps[0]) - 1.0)

    ax = axes[0]
    ax.axhline(0, color='#333333', linewidth=1.2)
    line_cost, = ax.plot(r1_grid, costs, color=config.COLOR_P,
                         linewidth=2.8, linestyle='-', marker='o',
                         markevery=40, markersize=6.5,
                         label='Net cost')
    ax.fill_between(r1_grid, costs, 0, where=costs < 0,
                    color=config.COLOR_P, alpha=0.09)
    ax.scatter([0.0, 0.5, 1.0], np.interp([0.0, 0.5, 1.0], r1_grid, costs),
               color=config.COLOR_P, edgecolor='white', s=55, zorder=4)
    ax.set_xlabel(r'$r_1$')
    ax.set_ylabel(r'Net cost $\mathcal{C}_{\mathcal{A}}/c$',
                  color=config.COLOR_P)
    ax.tick_params(axis='y', labelcolor=config.COLOR_P)
    ax.grid(True, linestyle=':', alpha=0.45)
    ax.set_title('(a) Immediate Payout')

    ax_right = ax.twinx()
    line_gap, = ax_right.plot(r1_grid, amplification,
                              color=config.COLOR_B, linestyle='--',
                              linewidth=2.5, marker='s',
                              markevery=40, markersize=6.0,
                              markerfacecolor='white',
                              label='Loss amp.')
    ax_right.set_ylabel('Loss amplification (%)', color=config.COLOR_B)
    ax_right.tick_params(axis='y', labelcolor=config.COLOR_B)
    ax.legend([line_cost, line_gap],
              [line_cost.get_label(), line_gap.get_label()],
              loc='lower left', frameon=False)

    # (b) PPLNS defense: finite-window payout risk where for K ~ Binomial(N, F), CV(K/N) = sqrt((1-F)/(N*F)).
    ax = axes[1]
    window = np.logspace(1, 5, 240)
    pplns_styles = [
        (0.25, '#542788', '--', 's'),
        (0.50, '#E66101', '-', 'o'),
        (1.00, '#1B9E77', '-.', '^'),
    ]
    for r1, color, linestyle, marker in pplns_styles:
        share_fraction = r1 * calc.alpha / (calc.beta + r1 * calc.alpha)
        cv = np.sqrt((1.0 - share_fraction) /
                     (window * share_fraction))
        ax.plot(window, cv, linewidth=2.5, color=color,
                linestyle=linestyle, marker=marker,
                markevery=40, markersize=6.0,
                markerfacecolor='white',
                label=rf'$r_1={r1:.2g}$, $F={share_fraction:.2f}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(bottom=3e-4)
    ax.set_xlabel(r'PPLNS window $N$')
    ax.set_ylabel('Payout CV')
    ax.set_title('(b) PPLNS Risk')
    ax.grid(True, which='both', linestyle=':', alpha=0.38)
    ax.legend(loc='lower left', frameon=False)

    # (c) Conditional payout: feasible deterrence window surplus over baseline BDoS interval.
    ax = axes[2]
    q_grid = np.linspace(0.0, 1.0, 101)
    policy_grid = np.linspace(0.0, 1.0, 1001)
    target_upper = np.array([
        _target_threshold(calc, r1) for r1 in policy_grid
    ])
    bdos_width = (_target_threshold(calc, 0.0)
                  - _attacker_threshold(calc, 0.0, 1.0))
    surplus = []
    selected_r1 = []
    for q in q_grid:
        attacker_lower = np.array([
            _attacker_threshold(calc, r1, q) for r1 in policy_grid
        ])
        widths = target_upper - attacker_lower
        best = int(np.nanargmax(widths))
        extra = max(0.0, widths[best] - bdos_width)
        surplus.append(extra)
        selected_r1.append(policy_grid[best] if extra > 1e-7 else 0.0)
    surplus = np.asarray(surplus)
    selected_r1 = np.asarray(selected_r1)
    # Locate the exact active-set transition: the derivative of the feasible
    # window at the BDoS boundary r1=0 changes sign here.
    h = 1e-5
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        local_slope = (
            (_target_threshold(calc, h) - _attacker_threshold(calc, h, mid))
            - bdos_width
        ) / h
        if local_slope > 0:
            hi = mid
        else:
            lo = mid
    q_critical = (lo + hi) / 2.0

    line_surplus, = ax.plot(q_grid, surplus, color=config.COLOR_P,
                            linewidth=2.8, linestyle='-', marker='o',
                            markevery=20, markersize=6.5,
                            label='PDoS')
    ax.fill_between(q_grid, 0, surplus, color=config.COLOR_P, alpha=0.10)
    ax.axvline(q_critical, color='#555555', linestyle=':', linewidth=1.5)
    ax.annotate(
        rf'$q^*={q_critical:.2f}$',
        xy=(q_critical, 0.0), xytext=(q_critical - 0.08, 0.18),
        fontsize=config.PAPER_ANNOTATION_SIZE + FONT_BUMP,
        ha='right', va='bottom',
        color='#444444',
        arrowprops={'arrowstyle': '->', 'color': '#555555', 'lw': 1.1}
    )
    ax.set_xlabel(r'Released fraction $q$')
    ax.set_ylabel('Feasibility surplus', color=config.COLOR_P)
    ax.tick_params(axis='y', labelcolor=config.COLOR_P)
    ax.set_title('(c) Delayed Payout')
    ax.grid(True, linestyle=':', alpha=0.45)

    ax_right = ax.twinx()
    line_policy, = ax_right.plot(q_grid, selected_r1, color=config.COLOR_B,
                                 linestyle='--', linewidth=2.3, marker='s',
                                 markevery=20, markersize=6.0,
                                 markerfacecolor='white',
                                 label=r'Best $r_1$')
    ax_right.set_ylabel(r'Selected $r_1$',
                        color=config.COLOR_B)
    ax_right.tick_params(axis='y', labelcolor=config.COLOR_B)
    ax.legend([line_surplus, line_policy],
              [line_surplus.get_label(), line_policy.get_label()],
              loc='upper left', frameon=False)

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_path = config.FIGURES_DIR / 'Figure_7.pdf'
    fig.savefig(save_path, format='pdf')
    print(f'Figure successfully saved to: {save_path}')


if __name__ == '__main__':
    plot_figure_7()

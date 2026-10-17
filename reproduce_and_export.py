"""Run the original plotters and export the exact arrays displayed in each panel."""
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from theory import PDoSTheoryCalculator
from plotter import plot_fig6, plot_fig7


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'plot_data'


def save_csv(name, columns):
    pd.DataFrame(columns).to_csv(OUTPUT / name, index=False, float_format='%.17g')


def main():
    OUTPUT.mkdir(exist_ok=True)
    calc = PDoSTheoryCalculator()
    metadata = {
        'python': platform.python_version(),
        'versions': {name: module.__version__ for name, module in
                     [('numpy', np), ('pandas', pd), ('matplotlib', matplotlib)]},
        'experiment_defaults': config.get_experiment_defaults(),
        'effective_model_parameters': {
            key: getattr(calc, key) for key in
            ['alpha', 'beta', 'eta', 'delta', 'gamma', 'K', 'c', 'lam',
             'lambda_t', 'lambda_w', 'E_Rw', 'R_b', 'M']
        },
        'note': 'Panel CSV values are extracted from the original plotting objects. '
                'Historical inputs calibrate the model; the plotted curves are theoretical.',
    }

    plot_fig6.plot_figure_6()
    fig = plt.gcf()
    ax_a, ax_b, ax_c = fig.axes[:3]
    line = ax_a.lines[1]
    shares_percent, costs = map(np.asarray, line.get_data())
    m = shares_percent / 100.0
    save_csv('Figure_6a_reward_sensitivity.csv', {
        'M': m, 'noncoinbase_share_percent': shares_percent,
        'omega_b': 1.6 * (1.0 - calc.M) / (1.0 - m),
        'normalized_net_cost': costs,
    })
    markers = np.asarray(ax_a.collections[0].get_offsets(), dtype=float)
    save_csv('Figure_6a_marked_points.csv', {
        'scenario': ['coinbase_only', 'empirical_mix'],
        'M': markers[:, 0] / 100.0,
        'noncoinbase_share_percent': markers[:, 0],
        'normalized_net_cost': markers[:, 1],
    })
    cost_grid = np.asarray(ax_b.images[0].get_array(), dtype=float)
    saving_grid = np.asarray(ax_c.images[0].get_array(), dtype=float)
    r1, r2 = np.meshgrid(plot_fig6.R_VALUES, plot_fig6.R_VALUES)
    save_csv('Figure_6b_fixed_policy_net_cost.csv', {
        'r1': r1.ravel(), 'r2': r2.ravel(),
        'normalized_net_cost': cost_grid.ravel(),
    })
    save_csv('Figure_6c_state_dependent_savings.csv', {
        'r1': r1.ravel(), 'r2': r2.ravel(),
        'normalized_dynamic_cost': cost_grid.ravel(),
        'normalized_static_cost': (cost_grid + saving_grid).ravel(),
        'normalized_cost_saving': saving_grid.ravel(),
    })
    policy = calc.get_joint_policy(1.6, target_strategy='Stop', resolution=101)
    metadata['figure_6'] = {
        'omega_b_for_schedules': 1.6,
        'joint_policy_r1': float(policy[0]), 'joint_policy_r2': float(policy[1]),
        'marker_values_use_interpolation': True,
        'saving_definition': 'static cost at (r1,r1) minus dynamic cost at (r1,r2)',
    }
    plt.close(fig)

    plot_fig7.plot_figure_7()
    fig = plt.gcf()
    ax_a, ax_b, ax_c = fig.axes[:3]
    right_a, right_c = fig.axes[3:5]
    r1_grid, costs = map(np.asarray, ax_a.lines[1].get_data())
    _, amplification = map(np.asarray, right_a.lines[0].get_data())
    r2_grid = np.array([calc.get_optimal_r2(1.6, r, 'Stop', share_retention=1.0)
                        for r in r1_grid])
    gaps = np.array([plot_fig7._target_gap(calc, 1.6, r) for r in r1_grid])
    save_csv('Figure_7a_immediate_payout.csv', {
        'r1': r1_grid, 'optimal_r2': r2_grid, 'omega_b': 1.6,
        'normalized_net_cost': costs, 'normalized_target_mine_minus_stop_gap': gaps,
        'target_loss_amplification_percent': amplification,
    })
    pplns_rows = []
    for r, line in zip([0.25, 0.50, 1.00], ax_b.lines):
        n, cv = map(np.asarray, line.get_data())
        f = r * calc.alpha / (calc.beta + r * calc.alpha)
        pplns_rows.append(pd.DataFrame({
            'r1': r, 'share_fraction_F': f, 'window_N': n,
            'payout_CV': cv, 'payout_fraction_variance': f * (1.0 - f) / n,
        }))
    pd.concat(pplns_rows, ignore_index=True).to_csv(
        OUTPUT / 'Figure_7b_pplns_risk.csv', index=False, float_format='%.17g')
    q, surplus = map(np.asarray, ax_c.lines[0].get_data())
    _, selected_r1 = map(np.asarray, right_c.lines[0].get_data())
    omega_a = np.array([plot_fig7._attacker_threshold(calc, r, retained)
                        for r, retained in zip(selected_r1, q)])
    omega_t = np.array([plot_fig7._target_threshold(calc, r) for r in selected_r1])
    bdos_width = (plot_fig7._target_threshold(calc, 0.0)
                  - plot_fig7._attacker_threshold(calc, 0.0, 1.0))
    save_csv('Figure_7c_conditional_payout.csv', {
        'released_fraction_q': q, 'window_surplus_WP_minus_WB': surplus,
        'selected_r1': selected_r1, 'selected_r2': 0.0,
        'attacker_break_even_omega': omega_a,
        'target_deterrence_upper_omega': omega_t,
        'bdos_window_width_WB': bdos_width,
        'selected_policy_window_width': np.maximum(omega_t - omega_a, 0.0),
    })
    metadata['figure_7'] = {
        'omega_b_for_immediate_payout': 1.6,
        'q_critical': float(ax_c.lines[1].get_xdata()[0]),
        'bdos_window_width': float(bdos_width),
        'r2_at_boundary': 0.0,
        'pplns_panel_plots': 'coefficient of variation; variance is also exported',
        'boundary_policy_grid_points': 1001,
    }
    plt.close(fig)
    (OUTPUT / 'parameters_and_summary.json').write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

    for path in OUTPUT.glob('*.csv'):
        df = pd.read_csv(path)
        assert not df.isna().any().any(), f'Missing value in {path.name}'
    assert cost_grid.shape == saving_grid.shape == (6, 6)
    assert len(q) == 101 and len(r1_grid) == 201
    assert np.allclose(amplification, 100.0 * ((-gaps) / (-gaps[0]) - 1.0))
    assert np.allclose(surplus, np.maximum(omega_t - omega_a - bdos_width, 0.0))

    print('Exported all six panels to plot_data/ and regenerated both PDFs in figures/.')


if __name__ == '__main__':
    main()

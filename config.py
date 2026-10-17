import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Absolute paths, resolving all other paths based on the physical location of this config.py file.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'dataset'
CACHE_DIR = BASE_DIR / 'cache'
FIGURES_DIR = BASE_DIR / 'figures'

# ==========================================
# 1. Static Network and Physical Economic Baseline Constants
# ==========================================
LAMBDA = 1.0 / 600  # Baseline Bitcoin block generation rate
c = 1.0  # Normalized operational cost

# Default game-theoretic parameters
ALPHA = 0.15
BETA = 0.20
ETA = 0.10
GAMMA = 0.5
OMEGA_B = 1.6

# Baseline hardware and electricity price (Single Source of Truth)
EFFICIENCY = 22.0  # T19 Pro efficiency (22 J/TH)
ELECTRICITY_PRICE = 0.05  # $0.05 / kWh
HASHES_PER_DIFF = 2 ** 32  # Number of hashes per difficulty unit
J_PER_KWH = 3.6e6  # Joules per kWh
OPEX = 1.15  # 15% OPEX accounting for cooling, pool fees, labor, and maintenance


# Physical cost calculation function
def calculate_cost_btc(difficulty, btc_price_usd, efficiency_j_th=EFFICIENCY, electricity_price=ELECTRICITY_PRICE):
    """
    Calculates the expected physical cost of mining a block, denominated in BTC.
    Uses the baseline hardware and electricity price from the config by default,
    which can be overridden during plotting.
    """
    energy_j = difficulty * HASHES_PER_DIFF * (efficiency_j_th * 1e-12)
    cost_usd = (energy_j / J_PER_KWH) * electricity_price * OPEX
    return cost_usd / btc_price_usd


# ==========================================
# 2. Data Loaders
# ==========================================
def get_global_economic_params():
    """
    Reads the preprocessed daily data and returns the global average economic parameters.
    Serves as the single source of truth for theoretical calculations and simulations.
    """
    df_daily = load_csv(DATA_DIR / 'daily_params.csv')

    baseline = df_daily[df_daily['date'] >= '2025-07-01']

    # Extract global averages
    lambda_t = baseline['lambda_t'].mean()
    lambda_w = baseline['lambda_w'].mean()

    e_rw = baseline['e_rw'].mean()
    omega_b = baseline['omega_b'].mean()
    mean_m = baseline['M'].mean()
    c_block = baseline['c_block'].mean()
    mean_k = baseline['K'].mean()

    return {
        'lambda_t': lambda_t,
        'lambda_w': lambda_w,
        'e_rw': e_rw,
        'omega_b': omega_b,
        'K': mean_k,
        'M': mean_m,
        'c_block': c_block
    }


def get_experiment_defaults():
    """
    Returns all fixed parameters used for the paper's tables and default experiments.
    Includes pure mathematical settings, physical baseline hardware, and economic
    parameters derived from on-chain statistics.
    """
    # Retrieve the mean values of the dynamically calculated on-chain economic parameters
    economic_params = get_global_economic_params()

    return {
        # Game-theoretic distribution parameters
        'ALPHA': ALPHA,
        'BETA': BETA,
        'ETA': ETA,
        'GAMMA': GAMMA,
        'OMEGA_B': OMEGA_B,

        # Physical and protocol baselines
        'K': economic_params['K'],
        'LAMBDA': LAMBDA,
        'c': c,

        # Hardware and electricity price baselines
        'EFFICIENCY': EFFICIENCY,
        'ELECTRICITY_PRICE': ELECTRICITY_PRICE,

        # Macroeconomic on-chain averages
        'lambda_t': economic_params['lambda_t'],
        'lambda_w': economic_params['lambda_w'],
        'e_rw': economic_params['e_rw'],
        'omega_b': economic_params['omega_b'],
        'M': economic_params['M'],
        'c_block': economic_params['c_block']
    }


def load_csv(path: Path, dtype=None) -> pd.DataFrame:
    """
    Unified data loader:
    1. Ignores type warnings.
    2. Eliminates attribute warnings via type assertion.
    """
    df = pd.read_csv(path, dtype=dtype)  # type: ignore
    assert isinstance(df, pd.DataFrame)
    return df


# ==========================================
# 3. Unified Paper Color Palette
# ==========================================
COLOR_B = '#1F77B4'  # Classic academic blue (Represents BDoS Baseline)
COLOR_P = '#D62728'  # Classic academic red (Represents PDoS)

# Unified typography parameters for paper figures
PAPER_FONT_SIZE = 18
PAPER_AXES_SIZE = 21
PAPER_TITLE_SIZE = 20
PAPER_TICK_SIZE = 17
PAPER_LEGEND_SIZE = 15
PAPER_ANNOTATION_SIZE = 15
PAPER_HEATMAP_SIZE = 13


def apply_plot_style(axes_size=PAPER_AXES_SIZE,
                     tick_size=PAPER_TICK_SIZE,
                     legend_size=PAPER_LEGEND_SIZE,
                     title_size=PAPER_TITLE_SIZE):
    """
    Configures one common, publication-scale Matplotlib style for the paper.
    """
    # Font settings
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    # Font sizes
    plt.rcParams['font.size'] = PAPER_FONT_SIZE  # General text/annotations
    plt.rcParams['axes.labelsize'] = axes_size  # X and Y axis labels
    plt.rcParams['axes.titlesize'] = title_size  # Panel titles
    plt.rcParams['xtick.labelsize'] = tick_size  # X-axis tick labels
    plt.rcParams['ytick.labelsize'] = tick_size  # Y-axis tick labels
    plt.rcParams['legend.fontsize'] = legend_size  # Legend size
    plt.rcParams['axes.titlepad'] = 7
    plt.rcParams['axes.labelpad'] = 6
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.88
    plt.rcParams['legend.facecolor'] = 'white'
    plt.rcParams['legend.edgecolor'] = 'none'
    plt.rcParams['legend.borderaxespad'] = 0.45
    plt.rcParams['legend.handlelength'] = 2.0
    plt.rcParams['savefig.pad_inches'] = 0.02

    # Figure dimensions
    plt.rcParams['figure.figsize'] = (10, 6)

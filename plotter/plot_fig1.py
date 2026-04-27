import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as m_dates
from matplotlib.ticker import MaxNLocator
import config


def plot_figure_1(input_block, input_daily, input_price):
    print("Plotting Figure 1...")
    config.apply_plot_style(26, 22, 20)

    # Load block-level data
    df_block = config.load_csv(input_block)
    df_block['timestamp'] = pd.to_datetime(df_block['timestamp'], utc=True)
    df_block['date'] = df_block['timestamp'].dt.normalize()

    # Load price data
    df_price = config.load_csv(input_price)
    df_price['Date'] = pd.to_datetime(df_price['Date'], utc=True)
    df_price['btc_price_usd'] = df_price['Price'].astype(str).str.replace(',', '').astype(float)

    # Load daily macro parameters
    df_daily = config.load_csv(input_daily)
    df_daily['date'] = pd.to_datetime(df_daily['date'], utc=True)

    # Merge the three dataframes
    df_block = df_block.merge(df_price[['Date', 'btc_price_usd']], left_on='date', right_on='Date', how='left')
    df_block['btc_price_usd'] = df_block['btc_price_usd'].ffill().bfill()
    df_block = df_block.merge(df_daily[['date', 'whale_threshold']], on='date', how='inner')

    # Define mainstream ASIC models and electricity prices
    miners = {
        'Bitmain Antminer S21 XP+ Hyd (Advanced, 11 J/TH)': {'e': 11, 'color': '#2ca02c'},
        'Bitmain Antminer S21+ Hyd (Sub-Adv, 15 J/TH)': {'e': 15, 'color': '#98df8a'},
        'Bitmain Antminer T19 Pro Hyd (Mainstream, 22 J/TH)': {'e': 22, 'color': '#1f77b4'},
        'Bitmain Antminer S19 Pro++ (Marginal, 26 J/TH)': {'e': 26, 'color': '#ff7f0e'},
        'Bitmain Antminer S19 Pro (Obsolete, 29.5 J/TH)': {'e': 29.5, 'color': '#d62728'}
    }

    prices = {
        'Cheap ($0.04/kWh)': {'p': 0.04, 'color': '#2ca02c', 'style': '-'},
        'Average ($0.05/kWh)': {'p': 0.05, 'color': '#1f77b4', 'style': '-'},
        'Expensive ($0.07/kWh)': {'p': 0.07, 'color': '#d62728', 'style': '-'}
    }

    # Calculate daily smoothed parameters
    daily_df = df_block.groupby('date').agg({
        'difficulty': 'mean',
        'revenue': 'mean',
        'btc_price_usd': 'mean'
    }).reset_index()

    # Capture true "local whales" based on the 30-day sliding window threshold
    print("Capturing dynamic whale blocks...")
    whales = df_block[df_block['fee'] > df_block['whale_threshold']].copy()

    # ================= Plotting =================
    print("Generating subplots...")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), dpi=300, sharex=True, gridspec_kw={'hspace': 0})

    # Define the Bitcoin Halving date
    halving_date = pd.Timestamp('2024-04-20', tz='UTC')

    # --- Subplot 1: Transaction Fee Trend and Whale Annotations ---
    ax1.plot(df_block['timestamp'], df_block['fee'], color='gray', linewidth=0.8, alpha=0.75, label='Transaction Fee',
             zorder=1)
    ax1.plot(df_daily['date'], df_daily['whale_threshold'], label='Dynamic Whale Threshold (P99.9)',
             color='purple', linestyle='--', linewidth=1.0, zorder=2)
    ax1.scatter(whales['timestamp'], whales['fee'], color='darkred', marker='*', s=60, alpha=0.8, edgecolors='black',
                linewidth=0.5, label='Whale Events', zorder=3)

    ax1.set_ylabel('Transaction Fee (BTC)')
    ax1.set_ylim(bottom=0)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))

    # --- Subplot 2: Fixed Electricity Price, Evolution of 5 ASIC Models ---
    fixed_price = 0.05
    for name, params in miners.items():
        c_block = config.calculate_cost_btc(daily_df['difficulty'], daily_df['btc_price_usd'], params['e'], fixed_price)
        omega_b = daily_df['revenue'] / c_block
        ax2.plot(daily_df['date'], omega_b, label=name, color=params['color'], linewidth=1.5, alpha=0.8, zorder=3)
    ax2.axhline(1.0, color='red', linestyle='--', linewidth=1.5, zorder=2, label=r'Shutdown Threshold ($\omega_b = 1$)')
    ax2.set_ylabel(r'Profitability Factor ($\omega_b$)')
    ax2.set_ylim(bottom=0)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))

    # --- Subplot 3: Fixed ASIC Model, Evolution of 3 Electricity Prices ---
    fixed_efficiency = 22
    for name, params in prices.items():
        c_block = config.calculate_cost_btc(daily_df['difficulty'], daily_df['btc_price_usd'], fixed_efficiency,
                                            params['p'])
        omega_b = daily_df['revenue'] / c_block
        ax3.plot(daily_df['date'], omega_b, label=name, color=params['color'], linestyle=params['style'],
                 linewidth=1.5, alpha=0.8, zorder=3)

    ax3.legend(loc='upper right', framealpha=0.9)

    # Halving vertical line logic
    for i, ax in enumerate([ax1, ax2, ax3]):
        # Black dashed line spanning all subplots
        ax.axvline(x=halving_date, color='black', linestyle='--', linewidth=2.0, alpha=1.0, zorder=1.5)

        # Display the "Halving" text annotation only in the middle subplot (ax2)
        if i == 1:
            ax.text(halving_date + pd.Timedelta(days=8), ax.get_ylim()[1] * 0.8, 'Halving',
                    color='black', fontweight='bold', fontstyle='italic', ha='left', alpha=1.0)

    ax3.axhline(1.0, color='red', linestyle='--', linewidth=1.5, zorder=2, label=r'Shutdown Threshold ($\omega_b = 1$)')
    ax3.set_ylabel(r'Profitability Factor ($\omega_b$)')
    ax3.set_ylim(bottom=0, top=8.6)
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.yaxis.set_major_locator(MaxNLocator(nbins=5))

    # Fit tightly to the boundaries to eliminate whitespace
    ax3.set_xlim([df_block['timestamp'].min(), df_block['timestamp'].max()])

    # X-axis time formatting
    ax3.xaxis.set_major_locator(m_dates.MonthLocator(interval=2))
    ax3.xaxis.set_major_formatter(m_dates.DateFormatter('%Y-%m'))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha='center')

    plt.tight_layout()
    save_path = config.FIGURES_DIR / 'Figure_1.pdf'
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"Figure successfully saved to: {save_path}")


if __name__ == "__main__":
    input_block_filename = config.DATA_DIR / 'block_metrics.csv'
    input_daily_filename = config.DATA_DIR / 'daily_params.csv'
    input_price_filename = config.DATA_DIR / 'btc-usd.csv'
    plot_figure_1(input_block_filename, input_daily_filename, input_price_filename)
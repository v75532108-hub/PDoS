import pandas as pd
import numpy as np

import config
from config import calculate_cost_btc


def parse_bits_to_difficulty(bits):
    """
    Converts the Bitcoin 'bits' field to a difficulty value.
    :param bits: The 'bits' field from the Bitcoin blockchain.
    :return: The calculated difficulty value.
    """
    try:
        bits_int = int(bits, 16)
        exponent = bits_int >> 24
        coefficient = bits_int & 0xffffff
        if coefficient == 0: return np.nan
        target = coefficient * (2 ** (8 * (exponent - 3)))
        difficulty_1_target = 0xffff * (2 ** 208)
        if target == 0: return np.nan
        return difficulty_1_target / target
    except Exception as e:
        print(f"Error parsing bits {bits}: {e}")
        return np.nan


def generate_block_metrics(input_chain, output_file):
    """
    Cleans block-level data and exports the results.
    :param input_chain: Raw Bitcoin blockchain data.
    :param output_file: Path to the output file.
    :return: A DataFrame for the next processing stage.
    """
    print(f"Reading {input_chain} ...")

    # Load data, forcing 'bits' to be read as strings to prevent pandas from misinterpreting it as scientific notation.
    df = config.load_csv(input_chain, dtype={'bits': str})
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    # Sort by height to ensure correct time difference calculations.
    df = df.sort_values('height').reset_index(drop=True)

    # Calculate block time intervals, fill NaN, and handle negative time drifts.
    df['delta_t'] = df['timestamp'].diff().dt.total_seconds().fillna(600)
    df['delta_t'] = df['delta_t'].apply(lambda x: max(x, 1.0))

    # Process specific fields.
    df['difficulty'] = df['bits'].apply(parse_bits_to_difficulty)
    df['fee'] = df['total_fee_satoshi'] / 1e8
    df['revenue'] = df['miner_revenue_satoshi'] / 1e8
    df['block_reward'] = ((df['miner_revenue_satoshi'] - df['total_fee_satoshi']) / 1e8).round(8)

    # Save the results.
    cols = ['height', 'timestamp', 'delta_t', 'difficulty', 'block_reward', 'fee', 'revenue']
    df[cols].to_csv(output_file, index=False)
    print(f"Block-level data saved to {output_file}")
    return df


def generate_daily_metrics(df, input_file, output_file, window_days=30):
    """
    Calculates daily macroeconomic parameters using a centered sliding window
    based on cleaned block-level data.
    :param df: Cleaned block-level DataFrame.
    :param input_file: Input price file.
    :param output_file: Output file path.
    :param window_days: Size of the sliding window in days.
    """
    print(f"Estimating parameters using a {window_days}-day centered sliding window...")

    # Load and align price data.
    df_price = config.load_csv(input_file)
    df_price['Date'] = pd.to_datetime(df_price['Date'], utc=True)
    df_price['btc_price_usd'] = df_price['Price'].astype(str).str.replace(',', '').astype(float)
    df['date'] = df['timestamp'].dt.normalize()
    df = df.merge(df_price[['Date', 'btc_price_usd']], left_on='date', right_on='Date', how='left')

    # Build a timestamp index to accelerate sliding window slicing.
    df_indexed = df.set_index('timestamp').sort_index()

    # Experimental evaluation period.
    start_date = pd.Timestamp('2024-01-01', tz='UTC')
    end_date = pd.Timestamp('2025-12-31', tz='UTC')
    daily_dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Centered window: Take half of window_days before and after (e.g., +/- 15 days).
    half_window = pd.Timedelta(days=window_days / 2)
    results = []

    for current_date in daily_dates:
        # Get the data slice within the sliding window.
        window_start = current_date - half_window
        window_end = current_date + half_window
        df_window = df_indexed[(df_indexed.index >= window_start) & (df_indexed.index < window_end)]

        # Calculate total time.
        total_time = df_window['delta_t'].sum()

        # Determine the whale threshold to separate linear transaction fees from whale transactions.
        whale_threshold = df_window['fee'].quantile(0.999)

        # Baseline linear fee accumulation rate (clipping excess above the threshold).
        baseline_fees = df_window['fee'].clip(upper=whale_threshold)
        lambda_t = baseline_fees.sum() / total_time if total_time > 0 else 0

        # Extract whale transactions.
        whales = df_window[df_window['fee'] > whale_threshold]

        # Whale arrival rate.
        lambda_w = len(whales) / total_time if total_time > 0 else 0

        # Expected excess fee from whales.
        e_rw = (whales['fee'] - whale_threshold).mean() if len(whales) > 0 else 0

        # Get the exact data slice for the current day (24 hours).
        daily_start = current_date
        daily_end = current_date + pd.Timedelta(days=1)
        df_daily = df_indexed[(df_indexed.index >= daily_start) & (df_indexed.index < daily_end)]

        daily_diff = df_daily['difficulty'].mean()
        daily_price = df_daily['btc_price_usd'].mean()
        daily_revenue = df_daily['revenue'].mean()

        cost_btc = calculate_cost_btc(daily_diff, daily_price)
        omega_b = daily_revenue / cost_btc

        daily_k = df_daily['block_reward'].mode()[0]
        expected_fee = (lambda_t + lambda_w * e_rw) / config.LAMBDA
        r_b = daily_k + expected_fee
        daily_m = expected_fee / r_b if r_b > 0 else 0

        results.append({
            'date': current_date.date(),
            'lambda_t': lambda_t,
            'lambda_w': lambda_w,
            'e_rw': e_rw,
            'K': daily_k,
            'whale_threshold': whale_threshold,
            'c_block': cost_btc,
            'omega_b': omega_b,
            'M': daily_m,
        })

    df_daily = pd.DataFrame(results)
    df_daily.to_csv(output_file, index=False)
    print(f"Daily macroeconomic parameters saved to: {output_file}")
    return df_daily


def generate_replay_metrics(df, input_price_file, output_replay_file):
    """
    Generates block-by-block replay data based on pure historical block traces.
    """
    print("Preparing pure trace-driven replay data...")

    # 1. Load and align price data.
    df_price = config.load_csv(input_price_file)
    df_price['Date'] = pd.to_datetime(df_price['Date'], utc=True)
    df_price['btc_price_usd'] = df_price['Price'].astype(str).str.replace(',', '').astype(float)

    df['date'] = df['timestamp'].dt.normalize()
    df = df.merge(df_price[['Date', 'btc_price_usd']], left_on='date', right_on='Date', how='left')
    df['btc_price_usd'] = df['btc_price_usd'].ffill().bfill()

    # 2. Strictly filter the 2024-2025 experimental evaluation period.
    start_date = pd.Timestamp('2024-01-01', tz='UTC')
    end_date = pd.Timestamp('2025-12-31', tz='UTC')
    df_replay = df[(df['timestamp'] >= start_date) & (df['timestamp'] < end_date)].copy()

    # 3. Accurately calculate the micro-physical cost and profitability factor (omega_b) block-by-block.
    # Note: calculate_cost_btc implicitly uses the baseline hardware efficiency from config.
    df_replay['c_block'] = calculate_cost_btc(df_replay['difficulty'], df_replay['btc_price_usd'])
    df_replay['omega_b'] = df_replay['revenue'] / df_replay['c_block']

    # 4. Calculate the instantaneous MEV ratio (M) for the subsequent PDoS time inflation formula.
    # If revenue is 0 (extreme edge case), set M to 0.
    df_replay['M'] = np.where(df_replay['revenue'] > 0, df_replay['fee'] / df_replay['revenue'], 0)

    # 5. Organize export columns: retain block height.
    cols_to_save = [
        'height', 'timestamp', 'delta_t', 'difficulty', 'btc_price_usd',
        'fee', 'revenue', 'block_reward', 'c_block', 'omega_b', 'M'
    ]

    df_replay = df_replay[cols_to_save].sort_values('height').reset_index(drop=True)
    df_replay.to_csv(output_replay_file, index=False)

    print(f"Block replay data saved to: {output_replay_file} (Contains {len(df_replay)} micro-state blocks)")
    return df_replay


if __name__ == "__main__":
    input_chain_filename = config.DATA_DIR / 'btc_chain.csv'
    input_price_filename = config.DATA_DIR / 'btc-usd.csv'
    output_block_filename = config.DATA_DIR / 'block_metrics.csv'
    output_daily_filename = config.DATA_DIR / 'daily_params.csv'
    output_replay_filename = config.DATA_DIR / 'block_replay_data.csv'

    df_block_metrics = generate_block_metrics(input_chain_filename, output_block_filename)
    generate_daily_metrics(df_block_metrics, input_price_filename, output_daily_filename)
    generate_replay_metrics(df_block_metrics, input_price_filename, output_replay_filename)

    print("Data preprocessing completed successfully.")

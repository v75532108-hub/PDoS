import pandas as pd
import config


def print_tables(input_blocks):
    """
    Prints the core dataset statistics and parameter estimation tables for the paper.
    """
    # Read block data
    df_blocks = config.load_csv(input_blocks)
    df_blocks['timestamp'] = pd.to_datetime(df_blocks['timestamp'])

    # Select the experimental evaluation window
    start_date = pd.Timestamp('2024-01-01', tz='UTC')
    end_date = pd.Timestamp('2025-12-31', tz='UTC')
    df_blocks = df_blocks[(df_blocks['timestamp'] >= start_date) & (df_blocks['timestamp'] <= end_date)].copy()
    min_time_str = df_blocks['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')
    max_time_str = df_blocks['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')

    empty_blocks = df_blocks[df_blocks['fee'] == 0]
    empty_block_count = len(empty_blocks)
    last_empty_height = empty_blocks['height'].max()
    empty_block_ratio = empty_block_count / len(df_blocks)

    # Print Table 1
    print("\n" + "=" * 50)
    print("Table 1: Dataset Statistics")
    print("=" * 50)
    print(f"Time Range: {min_time_str} to {max_time_str}")
    print(f"Total Blocks: {len(df_blocks):,}")
    print(f"Block Interval Avg.: {df_blocks['delta_t'].mean():.2f} s")
    print(f"Block Interval Med.: {df_blocks['delta_t'].median():.2f} s")
    print(f"Block Interval P95: {df_blocks['delta_t'].quantile(0.95):.2f} s")
    print(f"Tx Fee Avg.: {df_blocks['fee'].mean():.4f} BTC")
    print(f"Tx Fee P99.9: {df_blocks['fee'].quantile(0.999):.4f} BTC")
    print(f"Empty Blocks: {empty_block_count}")
    print(f"Empty Block Ratio: {empty_block_ratio:.2%}")
    print(f"Last Empty Block Height: {last_empty_height}")
    print("=" * 50 + "\n")

    # Print Table 2
    params = config.get_experiment_defaults()
    print("\n" + "=" * 50)
    print("Table 2: Parameter Estimation")
    print("=" * 50)
    print(f"Tx Fee Accum. Rate (lambda_t): {params['lambda_t']:.8e} BTC/s")
    print(f"Whale Arrival Rate (lambda_w): {params['lambda_w']:.8e} /s")
    print(f"Exp. Whale Reward (e_rw): {params['e_rw']:.4f} BTC")
    print(f"MEV Ratio (M): {params['M']:.4f}")
    print(f"Block Mining Cost (c_block): {params['c_block']:.4f} BTC")
    print(f"Profitability Factor (omega_b): {params['omega_b']:.4f}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    input_blocks_filename = 'dataset/block_metrics.csv'
    print_tables(input_blocks_filename)
-- Define a temporary table BlockFees to calculate the total transaction fees for each block
WITH BlockFees AS (
  SELECT
    block_number,
    SUM(fee) as total_fee_satoshi
  FROM
    `bigquery-public-data.crypto_bitcoin.transactions`
  WHERE
    block_timestamp >= '2023-01-01 00:00:00'
  GROUP BY
    block_number
),

-- Retrieve the total output value of the Coinbase transaction for each block
CoinbaseTx AS (
  SELECT
    block_number,
    SUM(output_value) as coinbase_output_satoshi
  FROM
    `bigquery-public-data.crypto_bitcoin.transactions`
  WHERE
    block_timestamp >= '2023-01-01 00:00:00'
    AND is_coinbase = TRUE
  GROUP BY
    block_number
)

SELECT
  b.number as height,
  b.timestamp,
  b.bits,
  -- Total miner revenue
  COALESCE(c.coinbase_output_satoshi, 0) AS miner_revenue_satoshi,
  -- Total transaction fees
  COALESCE(f.total_fee_satoshi, 0) AS total_fee_satoshi
FROM
  `bigquery-public-data.crypto_bitcoin.blocks` b
LEFT JOIN
  BlockFees f ON b.number = f.block_number
LEFT JOIN
  CoinbaseTx c ON b.number = c.block_number
WHERE
  b.timestamp >= '2023-01-01 00:00:00'
ORDER BY
  b.number ASC;
# Model notes

- Separate stacked LSTM per resolved stock ticker.
- Historical adjusted close data fetched via yfinance; no hardcoded/demo stock-price data.
- Up to five years of history, 60-day sequences, chronological holdout evaluation.
- Reports MAE, RMSE and holdout directional accuracy. These are historical metrics, not a probability of profit.
- Forecast includes next trading-day and recursive seven-step estimates; recursive error can accumulate.
- A symbol must have enough valid historical prices; unsupported or insufficient-history symbols return an error.

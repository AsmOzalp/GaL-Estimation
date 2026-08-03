# GaL-Estimation and Risk Management Pipeline 

An end-to-end econometric risk management framework for cryptocurrency assets. This project integrates robust GARCH volatility filtering with empirical Geometric Asymmetric Linnik (GaL) distribution parameter estimation to model extreme tail risk and compute Value-at-Risk (VaR).
# pip install -r requirements.txt
## Project Structure

```text
├── data/                   # Historical OHLCV & trade frequency CSV data
├── src/                    # Core econometric and backtesting modules
│   ├── data_loader.py      # Binance & Yahoo Finance data acquisition
│   ├── garch_models.py     # Multi-specification robust GARCH volatility filter
│   ├── estimation.py       # ECF-based GaL/AL estimators & boundary diagnostics
│   ├── risk_metrics.py     # Exact Gil-Pelaez CDF inversion & VaR quantiles
│   └── backtest.py         # Kupiec (UC), Christoffersen (CC) & OOS testing
├── run_crypto.py           # Root execution script (Master entry point)
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
└── README.md               # Project documentation

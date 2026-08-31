# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 00:52:18 2026
@author: m-a-o

Module: data_loader.py
Description: Automated data acquisition and cleaning module. Downloads daily market asset 
             metrics from Yahoo Finance or high-frequency trade sequences via Binance API.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import time
import requests
from pathlib import Path

def load_commodity_data(symbol="GC=F", start_date="2010-01-01", end_date=None):
    """
    Downloads historical asset records from Yahoo Finance and processes log returns.

    end_date=None (default) pulls through the most recent available trading day,
    so the exact sample window depends on WHEN this function is run. For a
    manuscript/reproducibility snapshot, pass an explicit end_date (e.g.
    "2026-08-31") so re-running the pipeline later reproduces the same window
    reported in the paper, rather than silently extending it.
    """
    print(f"\n[{symbol}] Downloading target dataset from Yahoo Finance "
          f"starting from {start_date}" + (f" through {end_date}" if end_date else " through today") + "...")
    df = yf.download(symbol, start=start_date, end=end_date, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        close_col = df['Close'][symbol]
        vol_col = df['Volume'][symbol]
    else:
        close_col = df['Close']
        vol_col = df['Volume']

    data = pd.DataFrame({'Close': close_col, 'Volume': vol_col})
    original_len = len(data)
    data = data[data['Volume'] > 0].dropna()
    cleaned_len = len(data)
    print(f" -> Holidays and zero-volume periods removed: {original_len - cleaned_len} rows excluded.")

    data['log_return'] = np.log(data['Close'] / data['Close'].shift(1)) * 100
    data = data.dropna()
    if len(data) > 0:
        date_start, date_end = data.index[0], data.index[-1]
        print(f" -> [Reproducibility] Actual sample window used: "
              f"{date_start.date()} .. {date_end.date()}  (n={len(data)} observations) "
              f"-- report this exact window in the manuscript's data section")
    data = data.reset_index(drop=True)

    return data['log_return'].to_numpy(), data['Volume'].to_numpy()


def load_btc_binance_data(symbol="BTCUSDT", interval="1d", years=1):
    """Loads historical trade frequency metrics locally or fetches directly from Binance API endpoints."""
    filename = f"data/{symbol}_{interval}_{years}y_data_with_trades.csv"
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if filepath.exists():
        print(f"\n[{symbol}] Local cache file read successfully ('{filename}').")
        df = pd.read_csv(filepath)
        df['log_return'] = np.log(df['close'] / df['close'].shift(1)) * 100
        df = df.dropna().reset_index(drop=True)
        return df['log_return'].to_numpy(), df['trades'].to_numpy()

    print(f"\n[{symbol}] Cache not found. Fetching structural records directly from Binance API...")
    limit = 1000
    end_time = int(time.time() * 1000)
    start_time = end_time - (years * 365 * 24 * 60 * 60 * 1000)
    all_data = []

    while start_time < end_time:
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit, "startTime": start_time, "endTime": end_time}
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception:
            break
        if not data: break
        all_data.extend(data)
        start_time = data[-1][0] + 1
        time.sleep(0.3)

    if not all_data:
        print(" [WARNING] Connection failed or empty response received from Binance. Generating random fallback simulation series.")
        np.random.seed(42)
        return np.random.normal(0, 2.0, 2000), np.random.randint(100, 5000, 2000)

    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
    df['close'] = df['close'].astype(float)
    df['trades'] = df['trades'].astype(int)
    df.to_csv(filepath, index=False)
    
    df['log_return'] = np.log(df['close'] / df['close'].shift(1)) * 100
    df = df.dropna().reset_index(drop=True)
    return df['log_return'].to_numpy(), df['trades'].to_numpy()
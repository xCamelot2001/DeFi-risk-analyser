from __future__ import annotations
import os
import json
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# -------------------------------
# Config
# -------------------------------
BASE_URL = "https://api.coingecko.com/api/v3"
CACHE_DIR = "data/cache"          # keep repo clean; add this path to .gitignore
CACHE_TTL = 300                   # seconds

os.makedirs(CACHE_DIR, exist_ok=True)

# -------------------------------
# Retry session
# -------------------------------
def _retry_session(
    retries: int = 3,
    backoff_factor: float = 0.5,
    status_forcelist: tuple = (500, 502, 503, 504),
) -> requests.Session:
    s = requests.Session()
    r = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        raise_on_status=False,
    )
    a = HTTPAdapter(max_retries=r)
    s.mount("http://", a)
    s.mount("https://", a)
    return s

# -------------------------------
# Cache helpers
# -------------------------------
def _cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, name)

def _load_cache(name: str):
    p = _cache_path(name)
    if not os.path.exists(p):
        return None
    age = datetime.now().timestamp() - os.path.getmtime(p)
    if age > CACHE_TTL:
        return None
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception:
        return None

def _save_cache(name: str, data) -> None:
    with open(_cache_path(name), "w") as f:
        json.dump(data, f)

# -------------------------------
# Public API
# -------------------------------
def get_market_data(vs_currency: str = "usd", per_page: int = 10, page: int = 1):
    """
    Top coins by market cap.
    """
    key = f"cg_market_{vs_currency}_{per_page}_{page}.json"
    cached = _load_cache(key)
    if cached is not None:
        return cached

    url = f"{BASE_URL}/coins/markets"
    params = {
        "vs_currency": vs_currency,
        "order": "market_cap_desc",
        "per_page": per_page,
        "page": page,
        "sparkline": False,
    }
    session = _retry_session()
    r = session.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    _save_cache(key, data)
    return data

def get_historical_data(coin_id: str, vs_currency: str = "usd", days: int = 30):
    """
    Historical daily prices from CoinGecko.
    """
    key = f"cg_hist_{coin_id}_{vs_currency}_{days}.json"
    cached = _load_cache(key)
    if cached is not None:
        return cached

    url = f"{BASE_URL}/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days, "interval": "daily"}
    session = _retry_session()
    r = session.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    _save_cache(key, data)
    return data

def parse_price_history(data: dict) -> pd.DataFrame:
    """
    Convert CoinGecko chart data -> DataFrame with ['date','price'].
    """
    prices = data.get("prices", [])
    if not prices:
        return pd.DataFrame(columns=["date", "price"])
    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.drop(columns=["timestamp"], inplace=True)
    return df[["date", "price"]].sort_values("date").reset_index(drop=True)

# -------------------------------
# Analytics
# -------------------------------
def compute_volatility(df: pd.DataFrame) -> float:
    """
    Annualized volatility from daily log returns.
    """
    if df.empty:
        return float("nan")
    r = np.log(df["price"] / df["price"].shift(1)).dropna()
    if r.empty:
        return float("nan")
    return float(r.std() * np.sqrt(365))

def compute_drawdown(df: pd.DataFrame) -> float:
    """
    Maximum drawdown (negative number).
    """
    if df.empty:
        return float("nan")
    s = df["price"].astype(float)
    cummax = s.cummax()
    dd = (s - cummax) / cummax
    return float(dd.min())

def compute_sharpe(df: pd.DataFrame, rf_annual: float = 0.0) -> float:
    """
    Annualized Sharpe ratio using daily log returns.
    """
    if df.empty:
        return float("nan")
    r = np.log(df["price"] / df["price"].shift(1)).dropna()
    if r.empty:
        return float("nan")
    rf_daily = (1 + rf_annual) ** (1 / 365) - 1
    mean_d = r.mean()
    std_d = r.std()
    if std_d == 0 or np.isnan(std_d):
        return float("nan")
    return float((mean_d - rf_daily) / std_d * np.sqrt(365))

def compute_30d_return(df: pd.DataFrame) -> float:
    if len(df) < 31:
        return float("nan")
    p0 = df["price"].iloc[-31]
    p1 = df["price"].iloc[-1]
    return float(p1 / p0 - 1)

def rolling_volatility(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    out = df[["date", "price"]].copy()
    r = np.log(out["price"] / out["price"].shift(1))
    out["rolling_vol"] = r.rolling(window).std() * np.sqrt(365)
    return out.dropna(subset=["rolling_vol"])

def rolling_max_drawdown(df: pd.DataFrame, window: int = 90) -> pd.DataFrame:
    out = df[["date", "price"]].copy()
    s = out["price"]
    roll_cummax = s.rolling(window).apply(lambda a: np.maximum.accumulate(a).max(), raw=False)
    out["rolling_mdd"] = (s - roll_cummax) / roll_cummax
    return out.dropna(subset=["rolling_mdd"])

def risk_score(df: pd.DataFrame) -> float:
    """
    Heuristic 0–100 risk score combining volatility, drawdown, and 30d momentum.
    """
    vol = compute_volatility(df)       # e.g., 0.80 = 80% ann.
    mdd = compute_drawdown(df)         # negative
    mdd_pos = -mdd if not np.isnan(mdd) else np.nan
    r30 = compute_30d_return(df)

    VOL_CAP = 1.5
    MDD_CAP = 0.90

    vol_norm = np.clip((vol or 0) / VOL_CAP, 0, 1) if not np.isnan(vol) else 0.0
    mdd_norm = np.clip((mdd_pos or 0) / MDD_CAP, 0, 1) if not np.isnan(mdd_pos) else 0.0
    mom_norm = ((np.tanh(-(r30 or 0) * 3) + 1) / 2) if not np.isnan(r30) else 0.5

    score = 100 * (0.45 * vol_norm + 0.45 * mdd_norm + 0.10 * mom_norm)
    return float(score)

def risk_bucket(score: float) -> str:
    if np.isnan(score):
        return "N/A"
    if score < 33:
        return "Low"
    if score < 66:
        return "Medium"
    return "High"


# for end-to-end usage
def compute_coin_risk_summary(
    coin_id: str,
    vs_currency: str = "usd",
    days: int = 90,
) -> dict:
    """
    End-to-end: download history, compute metrics, return a summary dict.
    """
    data = get_historical_data(coin_id=coin_id, vs_currency=vs_currency, days=days)
    df = parse_price_history(data)

    vol = compute_volatility(df)
    mdd = compute_drawdown(df)
    sharpe = compute_sharpe(df)
    ret_30d = compute_30d_return(df)
    score = risk_score(df)
    bucket = risk_bucket(score)

    return {
        "coin_id": coin_id,
        "vs_currency": vs_currency,
        "days": days,
        "volatility": vol,
        "max_drawdown": mdd,
        "sharpe": sharpe,
        "return_30d": ret_30d,
        "score": score,
        "bucket": bucket,
    }

from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .coins_api import get_historical_data, parse_price_history

# ---------- Returns helpers ----------
def log_returns(price_df: pd.DataFrame) -> pd.Series:
    """
    Input: DataFrame with ['date','price'].
    Output: daily log returns indexed by date.
    """
    if price_df.empty:
        return pd.Series(dtype=float)
    r = np.log(price_df["price"] / price_df["price"].shift(1)).dropna()
    r.index = price_df.loc[r.index, "date"]
    return r

# ---------- Single-asset risk ----------
def var_historic(returns: pd.Series, alpha: float = 0.95) -> float:
    """
    Historical VaR at confidence alpha (returns are log returns).
    Returns positive loss magnitude (e.g., 0.042 means -4.2%).
    """
    if returns.empty:
        return float("nan")
    q = np.quantile(returns, 1 - alpha)
    return float(-q)

def cvar_historic(returns: pd.Series, alpha: float = 0.95) -> float:
    """
    Conditional VaR (Expected Shortfall). Positive loss magnitude.
    """
    if returns.empty:
        return float("nan")
    thresh = np.quantile(returns, 1 - alpha)
    tail = returns[returns <= thresh]
    if tail.empty:
        return float("nan")
    return float(-tail.mean())

def beta_to(asset_returns: pd.Series, bench_returns: pd.Series) -> float:
    """
    OLS beta of asset to benchmark (cov/var).
    """
    df = pd.concat([asset_returns, bench_returns], axis=1).dropna()
    if df.shape[0] < 5:
        return float("nan")
    cov = np.cov(df.iloc[:, 0], df.iloc[:, 1], ddof=1)[0, 1]
    var_b = np.var(df.iloc[:, 1], ddof=1)
    if var_b == 0 or np.isnan(var_b):
        return float("nan")
    return float(cov / var_b)

# ---------- Multi-asset helpers ----------
def fetch_price_series(coin_ids: List[str], vs_currency: str, days: int) -> Dict[str, pd.DataFrame]:
    """
    Returns {coin_id: DataFrame['date','price']}.
    """
    out: Dict[str, pd.DataFrame] = {}
    for cid in coin_ids:
        try:
            data = get_historical_data(cid, vs_currency=vs_currency, days=days)
            out[cid] = parse_price_history(data)
        except Exception:
            out[cid] = pd.DataFrame(columns=["date", "price"])
    return out

def returns_matrix(prices: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Wide DF of log returns (index=date, columns=coin ids).
    """
    cols = []
    for cid, df in prices.items():
        r = log_returns(df).rename(cid)
        cols.append(r)
    if not cols:
        return pd.DataFrame()
    R = pd.concat(cols, axis=1)
    return R.dropna(how="all")

def corr_matrix(prices: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    R = returns_matrix(prices)
    if R.empty:
        return pd.DataFrame()
    return R.corr()

# ---------- Portfolio ----------
def equal_weighted_portfolio_returns(prices: Dict[str, pd.DataFrame]) -> pd.Series:
    R = returns_matrix(prices)
    if R.empty:
        return pd.Series(dtype=float)
    w = np.ones(R.shape[1]) / R.shape[1]
    port = R.fillna(0).values @ w
    return pd.Series(port, index=R.index)

def portfolio_var_cvar(prices: Dict[str, pd.DataFrame], alpha: float = 0.95) -> Tuple[float, float]:
    pr = equal_weighted_portfolio_returns(prices)
    if pr.empty:
        return float("nan"), float("nan")
    return var_historic(pr, alpha), cvar_historic(pr, alpha)

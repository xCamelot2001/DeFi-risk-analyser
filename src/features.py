from __future__ import annotations
import numpy as np
import pandas as pd

def _max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    cummax = series.cummax()
    dd = (series - cummax) / cummax
    return float(dd.min())

def _ann_vol(series: pd.Series) -> float:
    r = series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return np.nan
    return float(r.std() * np.sqrt(365))

def _momentum(series: pd.Series, window: int = 30) -> float:
    if len(series) < window + 1:
        return np.nan
    return float(series.iloc[-1] / series.iloc[-window - 1] - 1)

def _liquidity_proxy(series: pd.Series) -> float:
    # crude: higher median TVL -> deeper liquidity
    if series.empty:
        return np.nan
    return float(series.median())

def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Input long DF: [date, protocol, tvl]
    Output per-protocol features:
      vol_ann, max_dd, mom_30d, tvl_median, tvl_latest, tvl_change_7d
    """
    if raw.empty:
        return pd.DataFrame(columns=[
            "protocol","vol_ann","max_dd","mom_30d","tvl_median","tvl_latest","tvl_change_7d"
        ])

    feats = []
    for proto, g in raw.groupby("protocol"):
        g = g.sort_values("date")
        s = g["tvl"].astype(float)

        feats.append({
            "protocol": proto,
            "vol_ann": _ann_vol(s),
            "max_dd": _max_drawdown(s),
            "mom_30d": _momentum(s, 30),
            "tvl_median": _liquidity_proxy(s),
            "tvl_latest": float(s.iloc[-1]),
            "tvl_change_7d": float(s.iloc[-1] / s.iloc[max(0, len(s)-8)] - 1) if len(s) > 7 else np.nan,
        })

    df = pd.DataFrame(feats)

    # Add a simple cross-sectional correlation feature (optional)
    pivot = raw.pivot_table(index="date", columns="protocol", values="tvl")
    corr = pivot.pct_change().corr(min_periods=20) if pivot.shape[1] >= 2 else None
    if corr is not None:
        mean_corr = corr.mean(skipna=True).rename("mean_corr").reset_index().rename(columns={"index": "protocol"})
        df = df.merge(mean_corr, on="protocol", how="left")
    else:
        df["mean_corr"] = np.nan

    # Clean
    return df.replace([np.inf, -np.inf], np.nan)

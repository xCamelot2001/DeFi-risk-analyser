# app_coins.py
# AI-Powered Coin Risk Analyzer (Streamlit MVP)
# ---------------------------------------------
# Install deps (inside your venv):
#   pip install streamlit pandas numpy requests
#
# Run:
#   streamlit run app_coins.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
from typing import Optional
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="Coin Risk Analyzer — MVP", layout="wide")
st.title("Coin Risk Analyzer — MVP (Price-based)")

st.sidebar.header("Settings")
read_timeout = st.sidebar.slider(
    "Read timeout (sec)", 30, 180, 90, step=15, key="read_timeout_sec"
)

vs_currency = st.sidebar.selectbox(
    "Fiat", ["usd", "eur", "gbp"], index=0, key="fiat_select"
)

days_back = st.sidebar.selectbox(
    "History window (days)", ["14","30","90","180","365"], index=1, key="history_window"
)

# ------------- ROBUST HTTP SESSION ----------
CG_SESSION: Optional[requests.Session] = None
def _cg_session() -> requests.Session:
    """Create a single session with retries/backoff."""
    global CG_SESSION
    if CG_SESSION is None:
        s = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1.5,  # 0s, 1.5s, 3s...
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        s.headers.update({"User-Agent": "defi-coin-risk-mvp/0.1"})
        s.mount("https://", HTTPAdapter(max_retries=retries))
        CG_SESSION = s
    return CG_SESSION

# ------------- DATA FETCH -------------------
@st.cache_data(ttl=3600)
def fetch_coingecko_market_chart(coin_id: str, vs="usd", days="max", read_to=90, interval="daily") -> pd.DataFrame:
    """
    Returns DataFrame with columns: date, price, mcap, volume for a coin.
    Uses the free CoinGecko market_chart endpoint.
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    s = _cg_session()
    try:
        r = s.get(
            url,
            params={"vs_currency": vs, "days": days, "interval": interval},
            timeout=(10, read_to),  # (connect, read)
        )
        if r.status_code == 429:
            # rate limited — return empty but don't crash the UI
            st.session_state.setdefault("errors", []).append(f"{coin_id}: 429 rate limited")
            return pd.DataFrame(columns=["date", "price", "mcap", "volume"])
        r.raise_for_status()
        js = r.json()

        def to_df(key, col):
            arr = js.get(key, [])
            df = pd.DataFrame(arr, columns=["ts_ms", col])
            if df.empty:
                return df
            df["date"] = pd.to_datetime(df["ts_ms"], unit="ms", errors="coerce")
            return df[["date", col]]

        p = to_df("prices", "price")
        m = to_df("market_caps", "mcap")
        v = to_df("total_volumes", "volume")
        df = p.merge(m, on="date", how="outer").merge(v, on="date", how="outer")
        df = df.dropna().sort_values("date")
        return df
    except requests.exceptions.RequestException as e:
        st.session_state.setdefault("errors", []).append(f"{coin_id}: {e.__class__.__name__}")
        return pd.DataFrame(columns=["date", "price", "mcap", "volume"])

# ------------- FEATURE ENGINEERING ----------
def compute_coin_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna().sort_values("date")
    # basic daily return (row-to-row), OK to keep for vol calc
    df["ret"] = df["price"].pct_change().fillna(0.0)

    # precise 7-day return via timestamp, not row count
    ret7 = _ret_over_days(df, 7)
    df["ret_7d_exact"] = ret7  # same value on all rows; we’ll read the last one

    # realized vol proxy (still row-based; that’s fine for signal)
    df["vol_7d"] = df["ret"].rolling(7).std().fillna(0.0).clip(0, 1)

    # 90-day max drawdown (use daily “as-of” to be robust)
    # resample to 1D "close" to reduce intraday noise
    d = df.set_index(pd.to_datetime(df["date"], utc=True)).sort_index()
    d_daily = d["price"].resample("1D").last().dropna()
    roll_max = d_daily.rolling(90, min_periods=1).max()
    mdd_daily = (d_daily / roll_max) - 1.0
    # map back last daily mdd to the dataframe’s last row
    df["mdd_90d"] = mdd_daily.iloc[-1] if not mdd_daily.empty else 0.0

    # volume spike vs 30d average (use daily sum for stability)
    if "volume" in d.columns:
        v_daily = d["volume"].resample("1D").sum().replace(0, np.nan)
        v30 = v_daily.rolling(30).mean()
        spike = ((v_daily.iloc[-1] / v30.iloc[-1]) - 1.0) if (len(v_daily) >= 30 and v30.iloc[-1] not in [0, np.nan]) else 0.0
        df["vol_spike"] = float(np.clip((spike if np.isfinite(spike) else 0.0), 0, 3) / 3.0)
    else:
        df["vol_spike"] = 0.0

    return df

def _ret_over_days(df: pd.DataFrame, days: int) -> float:
    """
    Compute calendar-day return using timestamps:
    ret_days = (P_t / P_{t-days} - 1)
    We match the price at or BEFORE the target timestamp.
    """
    if df.empty or "date" not in df or "price" not in df:
        return np.nan

    # ensure sorted and datetime index
    d = df[["date", "price"]].dropna().sort_values("date").copy()
    d["date"] = pd.to_datetime(d["date"], utc=True)
    d = d.set_index("date")

    t_last = d.index[-1]
    t_target = t_last - pd.Timedelta(days=days)

    # get the last price at or before t_target (asof)
    # reindex with the union so asof works cleanly
    d_target = d.reindex(d.index.union([t_target]))
    # .asof returns the last valid value before/at the label
    p_target = d_target["price"].asof(t_target)
    p_last = float(d["price"].iloc[-1])

    if p_target is None or np.isnan(p_target) or p_target == 0:
        return np.nan
    return (p_last / float(p_target)) - 1.0

def coin_risk_score(ret_7d_exact, vol_7d, mdd_90d, vol_spike) -> int:
    """
    0..100 risk score combining:
      - recent drop (ret_7d negative),
      - drawdown (mdd_90d),
      - volatility,
      - volume spike.
    """
    drawdown_term = np.clip(-float(ret_7d_exact), 0, 1)  # only penalize negative
    mdd_term      = np.clip(-float(mdd_90d), 0, 1)
    vol_term      = np.clip(float(vol_7d), 0, 1)
    spike_term    = np.clip(float(vol_spike), 0, 1)
    raw = 0.45*drawdown_term + 0.25*mdd_term + 0.20*vol_term + 0.10*spike_term
    return float(100 * np.clip(raw, 0, 1))

def risk_bucket(score: float) -> str:
    if score >= 61: return "High"
    if score >= 31: return "Medium"
    return "Low"

# ------------- UI: COIN PICKER -------------
coin_ids = st.multiselect(
    "Pick coins (CoinGecko ids):",
    [
        "bitcoin", "ethereum", "solana", "chainlink", "uniswap",
        "aave", "dogecoin", "pepe", "arbitrum", "optimism"
    ],
    default=["bitcoin", "ethereum", "chainlink"]
)

if st.button("Update data"):
    st.cache_data.clear()

# ------------- MAIN LOOP -------------------
rows = []
charts = st.container()

with st.spinner("Fetching coin data & computing risk..."):
    for cid in coin_ids:
        df = fetch_coingecko_market_chart(cid, vs=vs_currency, days="7", read_to=read_timeout, interval="daily")
        if df.empty:
            rows.append({"coin": cid, "risk": None, "bucket": "-", "note": "No data"})
            continue

        feats = compute_coin_features(df)
        latest = feats.iloc[-1]
        score = coin_risk_score(latest["ret_7d_exact"], latest["vol_7d"], latest["mdd_90d"], latest["vol_spike"])

        rows.append({
            "coin": cid,
            "risk": score,
            "ret_7d_exact%": round(100*latest["ret_7d_exact"], 2),
            "vol_7d": round(latest["vol_7d"], 4),
            "mdd_90d%": round(100*latest["mdd_90d"], 2),
            "vol_spike": round(latest["vol_spike"], 3),
            "price": round(df["price"].iloc[-1], 4),
        })

        with charts.expander(f"{cid} — charts"):
            st.line_chart(df.set_index("date")["price"], height=180)
            st.line_chart(
                feats.set_index("date")[["ret_7d_exact", "vol_7d", "mdd_90d", "vol_spike"]],
                height=220
            )

# ------------- TABLE + ERRORS --------------
table = pd.DataFrame(rows)
st.subheader("Coin Risk Table (price-based MVP)")
st.dataframe(table, width="stretch")

errs = st.session_state.get("errors", [])
if errs:
    st.warning("Some requests failed:\n- " + "\n- ".join(errs))
    st.session_state["errors"] = []

st.caption(
    "Notes: risk uses price & volume features only (no on-chain yet). "
    "Next steps: add DEX liquidity depth, holder concentration, funding/OI, and sentiment."
)

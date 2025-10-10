import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

st.set_page_config(page_title="DeFi Risk Analyzer", layout="wide")
st.title("DeFi Risk Analyzer — MVP")

# --- 1) helpers -------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_defillama_tvl(protocol: str) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: date (datetime), tvl_usd (float)
    NOTE: Check DeFiLlama API docs for the exact endpoint. Common pattern:
    https://api.llama.fi/protocol/{slug}  or a charts/tvl endpoint.
    """
    url = f"https://api.llama.fi/protocol/{protocol}"   # adjust if needed
    r = requests.get(url, timeout=3000)
    r.raise_for_status()
    js = r.json()
    # Expect a 'tvl' list of { date, totalLiquidityUSD } or similar
    tvl_list = js.get("tvl", [])
    df = pd.DataFrame(tvl_list)
    # common keys vary; adapt if keys differ
    if "date" in df and "totalLiquidityUSD" in df:
        df["date"] = pd.to_datetime(df["date"], unit="s")
        df = df.rename(columns={"totalLiquidityUSD": "tvl_usd"})
        return df[["date","tvl_usd"]].sort_values("date")
    return pd.DataFrame(columns=["date","tvl_usd"])

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tvl_7d_change_pct"] = df["tvl_usd"].pct_change(7).fillna(0.0).clip(-1, 1)
    df["tvl_vol_7d"] = df["tvl_usd"].pct_change().rolling(7).std().fillna(0.0).clip(0, 1)
    return df

def simple_risk_score(tvl_change_pct, tvl_vol):
    """
    Simple rule-based score in [0,100].
    Higher drawdown and higher vol => higher risk.
    """
    # negative change => risk; flip sign and clip
    drawdown = float(np.clip(-tvl_change_pct, 0, 1))
    vol_term = float(np.clip(tvl_vol, 0, 1))
    raw = 0.7 * drawdown + 0.3 * vol_term
    return int(100 * np.clip(raw, 0, 1))

# --- 2) UI ------------------------------------------------------
protocol_slugs = st.multiselect(
    "Pick protocols (start with a few):", 
    ["uniswap", "aave", "curve"], 
    default=["uniswap", "aave"]
)

if st.button("Update data"):
    st.cache_data.clear()  # force refresh

rows = []
charts = st.container()
for slug in protocol_slugs:
    tvl = fetch_defillama_tvl(protocol=slug)
    if tvl.empty:
        rows.append({"protocol": slug, "risk": None, "note": "No TVL data"})
        continue
    feats = compute_features(tvl)
    latest = feats.iloc[-1]
    risk = simple_risk_score(latest["tvl_7d_change_pct"], latest["tvl_vol_7d"])
    rows.append({"protocol": slug, "risk": risk, 
                 "tvl_7d_change%": round(100*latest["tvl_7d_change_pct"],2),
                 "tvl_vol_7d": round(latest["tvl_vol_7d"],4)})
    with charts.expander(f"{slug} — charts"):
        st.line_chart(tvl.set_index("date")["tvl_usd"], height=180)
        st.line_chart(feats.set_index("date")[["tvl_7d_change_pct","tvl_vol_7d"]], height=180)

table = pd.DataFrame(rows)
st.subheader("Risk Table (simple MVP)")
st.dataframe(table, use_container_width=True)

st.caption("MVP note: risk uses only TVL signals. Add price/whale/oracle/sentiment later.")

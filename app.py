import os
import streamlit as st
import pandas as pd

from src.ingest import fetch_protocol_stats
from src.features import build_features
from src.modeling import load_model, score_risk
from src.utils import plot_timeseries, plot_shap

st.set_page_config(page_title="DeFi Risk Analyzer", layout="wide")
st.title("DeFi Risk Analyzer — Risk Scoring & Explainability")

with st.sidebar:
    st.header("Settings")
    protos = st.multiselect(
        "Select protocols",
        ["uniswap", "aave", "curve", "makerdao", "compound"],
        default=["uniswap", "aave"],
    )
    lookback = st.slider("Lookback (days)", 30, 365, 180, step=15)
    st.caption("Data from public APIs (DeFiLlama). Cached to reduce rate limits.")

@st.cache_data(ttl=60*30, show_spinner=True)
def load_data(protocols: tuple, days: int):
    raw_df = fetch_protocol_stats(protocols, days=days)  # handles retries + caching
    feats = build_features(raw_df)                       # clean, align, engineer
    return raw_df, feats

if len(protos) == 0:
    st.info("Select at least one protocol to begin.")
    st.stop()

raw, feats = load_data(tuple(sorted(set(p.lower() for p in protos))), lookback)

# Guard rails

from src.ingest import fetch_protocol_stats
_raw, dbg = fetch_protocol_stats(tuple(sorted(set(p.lower() for p in protos))), days=lookback, debug=True)
feats = build_features(_raw)
raw = _raw  # keep variable name used later

if raw.empty or feats.empty:
    st.warning("No data returned. Debug details below.")
    with st.expander("Show debug details"):
        st.write(dbg)  # shows per-protocol url, rows, min/max dates, errors if any
    st.stop()

model = load_model()  # local file or fallback heuristic
risk_df, shap_values = score_risk(model, feats)

tab1, tab2, tab3 = st.tabs(["Overview", "Explainability", "Data"])
with tab1:
    st.subheader("Risk Scores (higher = riskier)")
    st.dataframe(
        risk_df.sort_values("risk_score", ascending=False).reset_index(drop=True),
        use_container_width=True,
    )
    st.plotly_chart(plot_timeseries(raw), use_container_width=True)

with tab2:
    st.subheader("Why is a protocol risky?")
    fig = plot_shap(shap_values, feats.columns)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Export")
    st.download_button("Download features CSV", feats.to_csv(index=False), "features.csv", "text/csv")
    st.caption("Features are per-protocol aggregates built from TVL time series (volatility, drawdown, momentum, etc.).")

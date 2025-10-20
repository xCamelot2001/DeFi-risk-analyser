import streamlit as st
import pandas as pd

from src.coins_api import (
    get_market_data,
    get_historical_data,
    parse_price_history,
    compute_volatility,
    compute_drawdown,
    compute_sharpe,
    compute_30d_return,
    risk_score,
    risk_bucket,
    rolling_volatility,
    rolling_max_drawdown,
)

from src.coins_risk import (
    fetch_price_series,
    corr_matrix,
    var_historic,
    cvar_historic,
    beta_to,
    log_returns,
    portfolio_var_cvar,
)


st.set_page_config(page_title="Coin Risk Analyzer", layout="wide")
st.title("🪙 Coin Risk Analyzer")

# ---------- Sidebar ----------
with st.sidebar:
    vs_currency = st.selectbox("Fiat Currency", ["usd", "eur", "gbp"], index=0, key="vs_currency_select")
    days = st.slider("Lookback (days)", min_value=30, max_value=365, value=120, step=10, key="lookback_slider")
    rf = st.number_input("Risk-free (annual, %)", min_value=0.0, max_value=20.0, value=4.5, step=0.1, key="rf_input") / 100.0
    roll_vol_win = st.slider("Rolling Vol Window (days)", 14, 120, 30, step=7, key="roll_vol_win")
    roll_mdd_win = st.slider("Rolling MDD Window (days)", 30, 180, 90, step=15, key="roll_mdd_win")
    topn = st.slider("Compare Top N by Market Cap", 5, 30, 10, step=5, key="topn_slider")

# ---------- Select coin ----------
market = get_market_data(vs_currency=vs_currency, per_page=topn)
coins = [c["id"] for c in market]
coin_choice = st.selectbox("Select a Coin", coins, key="coin_select")

# ---------- Fetch & compute ----------
raw = get_historical_data(coin_choice, vs_currency=vs_currency, days=days)
df = parse_price_history(raw)

# Main chart
st.subheader(f"Price — {coin_choice} ({vs_currency.upper()})")
st.line_chart(df.set_index("date")["price"])

# Metrics
vol = compute_volatility(df)
mdd = compute_drawdown(df)
sr = compute_sharpe(df, rf_annual=rf)
r30 = compute_30d_return(df)
score = risk_score(df)
bucket = risk_bucket(score)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Volatility (Ann.)", f"{vol:.2%}" if pd.notna(vol) else "—")
c2.metric("Max Drawdown", f"{mdd:.2%}" if pd.notna(mdd) else "—")
c3.metric("Sharpe (Ann.)", f"{sr:.2f}" if pd.notna(sr) else "—")
c4.metric("30-day Return", f"{r30:.2%}" if pd.notna(r30) else "—")

st.info(f"Risk Score: **{score:.1f} / 100** → **{bucket}**")

# Rolling risk charts
rv = rolling_volatility(df, window=roll_vol_win)
rm = rolling_max_drawdown(df, window=roll_mdd_win)

st.subheader("Rolling Risk")
tab1, tab2 = st.tabs(["Rolling Volatility (Ann.)", "Rolling Max Drawdown"])
with tab1:
    if not rv.empty:
        st.line_chart(rv.set_index("date")["rolling_vol"])
with tab2:
    if not rm.empty:
        st.line_chart(rm.set_index("date")["rolling_mdd"])

st.header("🧪 Risk Lab")

# Controls
alpha = st.slider("Confidence (VaR/CVaR)", 0.80, 0.99, 0.95, 0.01, key="alpha_slider")
benchmark = st.selectbox("Benchmark for Beta", ["bitcoin", "ethereum"], index=0, key="bench_select")

# Fetch prices for selected set (Top N for matrix + current coin for single metrics)
price_map = fetch_price_series(coins, vs_currency=vs_currency, days=days)

# Correlation matrix
st.subheader("Correlation (Top N)")
corr = corr_matrix(price_map)
if not corr.empty:
    st.dataframe(corr.style.format("{:.2f}"), use_container_width=True)
else:
    st.write("Not enough data to compute correlations.")

# Single-asset VaR/CVaR & Beta
st.subheader(f"{coin_choice} — VaR / CVaR / Beta")
coin_returns = log_returns(parse_price_history(get_historical_data(coin_choice, vs_currency=vs_currency, days=days)))
bench_returns = log_returns(parse_price_history(get_historical_data(benchmark, vs_currency=vs_currency, days=days)))

VaR = var_historic(coin_returns, alpha=alpha)
CVaR = cvar_historic(coin_returns, alpha=alpha)
beta = beta_to(coin_returns, bench_returns)

c1, c2, c3 = st.columns(3)
c1.metric(f"VaR {int(alpha*100)}% (daily)", f"{VaR:.2%}" if pd.notna(VaR) else "—")
c2.metric(f"CVaR {int(alpha*100)}% (daily)", f"{CVaR:.2%}" if pd.notna(CVaR) else "—")
c3.metric(f"Beta vs {benchmark.upper()}", f"{beta:.2f}" if pd.notna(beta) else "—")

# Portfolio VaR/CVaR (equal weight across Top N)
st.subheader(f"Equal-Weight Portfolio (Top {len(coins)}) — VaR / CVaR")
p_VaR, p_CVaR = portfolio_var_cvar(price_map, alpha=alpha)
d1, d2 = st.columns(2)
d1.metric(f"Portfolio VaR {int(alpha*100)}% (daily)", f"{p_VaR:.2%}" if pd.notna(p_VaR) else "—")
d2.metric(f"Portfolio CVaR {int(alpha*100)}% (daily)", f"{p_CVaR:.2%}" if pd.notna(p_CVaR) else "—")

# ---------- Top-N comparison ----------
st.subheader(f"Top {topn} Coins — Risk Snapshot")
rows = []
for c in coins:
    h = get_historical_data(c, vs_currency=vs_currency, days=days)
    d = parse_price_history(h)
    if len(d) < 5:
        continue
    v = compute_volatility(d)
    dd = compute_drawdown(d)
    s = compute_sharpe(d, rf_annual=rf)
    r = compute_30d_return(d)
    sc = risk_score(d)
    rows.append({
        "coin": c,
        "vol_ann": v,
        "mdd": dd,
        "sharpe": s,
        "ret_30d": r,
        "risk_score": sc,
        "bucket": risk_bucket(sc),
    })

if rows:
    table = pd.DataFrame(rows)
    # Sort by risk_score ascending (safer at top)
    table = table.sort_values("risk_score", ascending=True, na_position="last")
    # Pretty formatting for display
    show = table.copy()
    show["vol_ann"] = show["vol_ann"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "—")
    show["mdd"] = show["mdd"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "—")
    show["ret_30d"] = show["ret_30d"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "—")
    show["sharpe"] = show["sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    show["risk_score"] = show["risk_score"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "—")

    st.dataframe(show.reset_index(drop=True), use_container_width=True)
else:
    st.write("No data available for comparison.")

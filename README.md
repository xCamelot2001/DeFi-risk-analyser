---
title: DeFi Risk Analyzer
emoji: 🪙
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: "1.38.0"
app_file: app_coins.py
pinned: false
---

# DeFi Risk Analyzer — Streamlit + DeFiLlama

Live demo: **https://huggingface.co/spaces/hosseinmasjedi/defi-risk-analyser**

## What it does

- Pulls TVL time series from **DeFiLlama** (public API, cached)
- Builds per-protocol risk features: **volatility**, **max drawdown**, **momentum**, **liquidity proxy**, **crowding (mean corr)**
- Scores risk via a baseline **heuristic** or your **own model** (`models/risk_model.pkl`)
- Optional **SHAP** explainability for model-based scores
- Clean UX: protocol multi-select, lookback slider, plots, CSV export

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app_coins.py
```

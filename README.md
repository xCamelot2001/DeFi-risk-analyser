# 🪙 DeFi Risk Analyser

> A real-time DeFi protocol risk dashboard powered by DeFiLlama, built with Streamlit.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38.0-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace%20Spaces-yellow?logo=huggingface)](https://huggingface.co/spaces/hosseinmasjedi/defi-risk-analyser)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Live demo →** [huggingface.co/spaces/hosseinmasjedi/defi-risk-analyser](https://huggingface.co/spaces/hosseinmasjedi/defi-risk-analyser)

---

## Overview

DeFi Risk Analyser is an interactive dashboard that quantifies the risk profile of DeFi protocols using on-chain TVL (Total Value Locked) data from the [DeFiLlama](https://defillama.com/) public API. It computes a suite of financial risk metrics per protocol, scores them via a heuristic baseline or a pluggable ML model, and surfaces optional SHAP-based explainability — all through a clean, no-auth Streamlit UI.

---

## Features

- **Live TVL data** — Pulls and caches time-series TVL from the DeFiLlama public API (no API key required)
- **Per-protocol risk metrics:**
  - Volatility (rolling standard deviation of TVL returns)
  - Maximum drawdown
  - Momentum (short-term vs. long-term TVL trend)
  - Liquidity proxy
  - Crowding score (mean pairwise correlation across selected protocols)
- **Dual scoring modes** — Baseline heuristic scoring out of the box, or drop in your own trained model (`models/risk_model.pkl`) for ML-based scores
- **SHAP explainability** — Optional feature attribution for model-based risk scores
- **Interactive controls** — Multi-protocol selector, adjustable lookback window, interactive Plotly charts
- **CSV export** — Download the computed risk feature table for offline analysis
- **FastAPI backend** — Lightweight API layer (`main.py`) for programmatic access to risk scores

---

## Project Structure

```
DeFi-risk-analyser/
├── app.py               # Alternative Streamlit entry point
├── app_coins.py         # Primary Streamlit dashboard
├── main.py              # FastAPI backend
├── src/                 # Core logic (data fetching, feature engineering, scoring)
├── tests/               # Pytest test suite
├── .streamlit/          # Streamlit theme / config
├── .github/workflows/   # CI pipeline
├── requirements.txt
└── runtime.txt
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/xCamelot2001/DeFi-risk-analyser.git
cd DeFi-risk-analyser

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run app_coins.py
```

The app will open at `http://localhost:8501`.

### Run the API

```bash
uvicorn main:app --reload
```

API docs available at `http://localhost:8000/docs`.

---

## Bring Your Own Model

To use a custom ML risk-scoring model instead of the heuristic baseline:

1. Train a model on the feature set (volatility, drawdown, momentum, liquidity proxy, crowding).
2. Serialise it with `joblib`:
   ```python
   import joblib
   joblib.dump(model, "models/risk_model.pkl")
   ```
3. Place `risk_model.pkl` in a `models/` directory at the project root.
4. Restart the app — it will auto-detect and use the model, with optional SHAP explanations.

---

## Running Tests

```bash
pytest tests/
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Dashboard | Streamlit 1.38 |
| Data source | DeFiLlama API |
| Data processing | Pandas, NumPy |
| ML / scoring | scikit-learn, joblib |
| Explainability | SHAP |
| Visualisation | Plotly |
| API | FastAPI, Uvicorn |
| Resilience | Tenacity (retry logic) |
| Testing | Pytest |
| Deployment | Hugging Face Spaces |

---

## Contributing

Contributions are welcome! Please open an issue to discuss what you'd like to change, then submit a pull request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## Disclaimer

This project is for **educational and research purposes only**. Nothing in this dashboard constitutes financial advice. DeFi carries significant risk — always do your own research.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# adjust import path if needed (e.g. from src.coins_api import ...)
from src.coins_api import compute_coin_risk_summary

app = FastAPI(
    title="DeFi Risk Analyzer API",
    description="FastAPI wrapper around my CoinGecko-based risk engine",
    version="0.1.0",
)

class CoinRiskResponse(BaseModel):
    coin_id: str
    vs_currency: str
    days: int
    volatility: float
    max_drawdown: float
    sharpe: float
    return_30d: float
    score: float
    bucket: str


@app.get("/")
def root():
    return {"message": "DeFi Risk Analyzer API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/risk", response_model=CoinRiskResponse)
def get_risk(
    coin_id: str,
    vs_currency: str = "usd",
    days: int = 90,
):
    try:
        result = compute_coin_risk_summary(
            coin_id=coin_id,
            vs_currency=vs_currency,
            days=days,
        )
    except Exception as e:
        # good enough error handling
        raise HTTPException(status_code=500, detail=str(e))

    return CoinRiskResponse(**result)

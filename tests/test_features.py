import pandas as pd
from src.features import build_features

def test_build_features_basic():
    # synthetic 40 days for 2 protocols
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    df = pd.DataFrame({
        "date": list(dates)*2,
        "protocol": ["uniswap"]*40 + ["aave"]*40,
        "tvl": list(range(100, 140)) + list(range(200, 240)),
    })
    feats = build_features(df)
    assert set(["protocol","vol_ann","max_dd","mom_30d","tvl_median","tvl_latest","tvl_change_7d","mean_corr"]).issubset(feats.columns)
    assert len(feats) == 2

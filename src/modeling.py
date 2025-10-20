from __future__ import annotations
import os
import numpy as np
import pandas as pd
import joblib

try:
    import shap
except Exception:
    shap = None  # SHAP is optional at runtime

MODEL_PATH = os.getenv("RISK_MODEL_PATH", "models/risk_model.pkl")

def load_model():
    """
    Load a pickled model if present. Otherwise return None (we'll use a heuristic).
    """
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None

def _heuristic_score(feats: pd.DataFrame) -> pd.Series:
    """
    Simple, explainable baseline: higher risk with higher vol + deeper drawdown, lower median TVL.
    Scaled to ~[0,100].
    """
    f = feats.copy()
    # Normalize with caps
    vol = np.clip(f["vol_ann"].fillna(0) / 1.5, 0, 1)
    mdd = np.clip((-f["max_dd"]).fillna(0) / 0.9, 0, 1)
    liq = 1 - np.clip((np.log1p(f["tvl_median"].fillna(0)) / 20.0), 0, 1)  # more TVL -> less risk
    mom = np.clip(-f["mom_30d"].fillna(0), 0, 1)  # negative momentum increases risk
    corr = np.clip(f["mean_corr"].fillna(0.5), 0, 1)  # higher crowding -> slightly higher risk

    score = 100 * (0.40*vol + 0.35*mdd + 0.15*liq + 0.05*mom + 0.05*corr)
    return score

def score_risk(model, feats: pd.DataFrame):
    """
    Returns:
      risk_df: protocol, risk_score
      shap_values: array-like (n_samples, n_features) or None
    """
    X = feats.drop(columns=["protocol"])
    if model is None:
        scores = _heuristic_score(feats)
        shap_vals = np.zeros((len(feats), X.shape[1])) if len(feats) else None
    else:
        # Model inference
        scores = pd.Series(model.predict_proba(X)[:, 1] * 100 if hasattr(model, "predict_proba")
                           else model.predict(X), index=feats.index)
        # SHAP
        if shap is not None:
            try:
                explainer = shap.Explainer(model, X, feature_names=X.columns)
                sv = explainer(X)
                shap_vals = sv.values
            except Exception:
                shap_vals = None
        else:
            shap_vals = None

    risk_df = pd.DataFrame({"protocol": feats["protocol"], "risk_score": scores})
    return risk_df, shap_vals

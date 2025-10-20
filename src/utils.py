from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def plot_timeseries(raw: pd.DataFrame):
    """
    Expects columns [date, protocol, tvl] — plots TVL per protocol.
    """
    if raw.empty:
        fig = go.Figure()
        fig.update_layout(title="No data")
        return fig
    fig = px.line(raw, x="date", y="tvl", color="protocol", title="TVL (USD)")
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def plot_shap(shap_values, feature_names):
    """
    Simple SHAP bar for mean |impact|. If shap_values is None, return a blank fig.
    """
    if shap_values is None or len(shap_values) == 0:
        fig = go.Figure()
        fig.update_layout(title="No SHAP values to display (using heuristic model).")
        return fig
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    names = np.array(list(feature_names))[order]
    vals = mean_abs[order]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h"))
    fig.update_layout(title="Feature importance (mean |SHAP|)", yaxis=dict(autorange="reversed"))
    return fig

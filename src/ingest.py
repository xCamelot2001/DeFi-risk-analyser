from __future__ import annotations
import os, json, logging
import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import Iterable, List
from .cache import Cache

BASE = "https://api.llama.fi"

logger = logging.getLogger(__name__)
_cache = Cache(root="data/cache", ttl_seconds=60 * 30)

class ApiError(Exception):
    pass

def _slugify(name: str) -> str:
    return name.strip().lower().replace(" ", "-")

@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.7, min=1, max=6),
    retry=retry_if_exception_type((requests.RequestException, ApiError)),
)
def _get(url: str, timeout: int = 60) -> dict:
    r = requests.get(url, timeout=timeout)
    if r.status_code != 200:
        raise ApiError(f"HTTP {r.status_code}: {url}")
    return r.json()

def _fetch_protocol_tvl_series(protocol_slug: str) -> pd.DataFrame:
    """
    Build a daily TVL timeseries (sum across chains) for a protocol.
    Output: DataFrame with columns ['date','protocol','tvl']
    """
    cache_key = f"tvl_series_{protocol_slug}.json"
    cached = _cache.get(cache_key)
    if cached is None:
        # Correct FREE endpoint per docs
        data = _get(f"{BASE}/protocol/{protocol_slug}")
        _cache.set(cache_key, data)
    else:
        data = cached

    # Per docs: data["chainTvls"][<chain>]["tvl"] = [{date, totalLiquidityUSD}, ...]
    chain_tvls = data.get("chainTvls", {}) if isinstance(data, dict) else {}
    rows = []

    for _chain, obj in chain_tvls.items():
        tvl_list = (obj or {}).get("tvl", [])
        if not isinstance(tvl_list, list):
            continue
        for e in tvl_list:
            # Keep only clean entries with both date and value
            if not isinstance(e, dict):
                continue
            ts = e.get("date")
            val = e.get("totalLiquidityUSD", e.get("tvl"))
            if ts is None or val is None:
                continue
            rows.append((ts, val))

    if not rows:
        return pd.DataFrame(columns=["date", "protocol", "tvl"])

    df = pd.DataFrame(rows, columns=["date", "tvl"])
    df["date"] = pd.to_datetime(df["date"], unit="s", utc=True).dt.tz_localize(None)
    df["tvl"] = pd.to_numeric(df["tvl"], errors="coerce")
    df = df.dropna(subset=["tvl"])

    # sum across chains by date
    df = df.groupby("date", as_index=False)["tvl"].sum().sort_values("date")
    df["protocol"] = protocol_slug
    return df[["date", "protocol", "tvl"]]


def fetch_protocol_stats(protocols: Iterable[str], days: int = 180) -> pd.DataFrame:
    """
    Fetch TVL series for all protocols and truncate to lookback window.
    Returns a long DataFrame: [date, protocol, tvl]
    """
    frames: List[pd.DataFrame] = []
    for p in protocols:
        slug = _slugify(p)
        try:
            df = _fetch_protocol_tvl_series(slug)
        except Exception:
            # Graceful fail for unknown slug
            df = pd.DataFrame(columns=["date", "protocol", "tvl"])
        if not df.empty:
            # keep last N days
            cutoff = df["date"].max() - pd.Timedelta(days=days)
            frames.append(df[df["date"] >= cutoff])

    if not frames:
        return pd.DataFrame(columns=["date", "protocol", "tvl"])
    all_df = pd.concat(frames, ignore_index=True)
    # Basic sanity
    all_df["tvl"] = pd.to_numeric(all_df["tvl"], errors="coerce")
    all_df = all_df.dropna(subset=["tvl"])
    return all_df

def fetch_protocol_stats(protocols, days: int = 180, debug: bool = False):
    """
    Fetch TVL series for all protocols and truncate to lookback.
    Returns:
      - if debug is False: DataFrame [date, protocol, tvl]
      - if debug is True:  (DataFrame, debug_dict)
    """
    frames = []
    dbg = {"items": []}

    for p in protocols:
        slug = _slugify(p)
        item_dbg = {"input": p, "slug": slug, "status": "ok", "rows": 0, "url": f"{BASE}/protocol/{slug}"}
        try:
            df = _fetch_protocol_tvl_series(slug)
            if df.empty:
                item_dbg["status"] = "empty"
            else:
                cutoff = df["date"].max() - pd.Timedelta(days=days)
                df = df[df["date"] >= cutoff]
                item_dbg["rows"] = len(df)
                if not df.empty:
                    item_dbg["min_date"] = str(df["date"].min())
                    item_dbg["max_date"] = str(df["date"].max())
            frames.append(df)
        except Exception as e:
            item_dbg["status"] = f"error: {type(e).__name__}"
            item_dbg["error"] = str(e)
            frames.append(pd.DataFrame(columns=["date","protocol","tvl"]))
        dbg["items"].append(item_dbg)

    if not frames:
        out = pd.DataFrame(columns=["date", "protocol", "tvl"])
    else:
        out = pd.concat(frames, ignore_index=True)
        out["tvl"] = pd.to_numeric(out["tvl"], errors="coerce")
        out = out.dropna(subset=["tvl"])

    if debug:
        dbg["total_rows"] = len(out)
        return out, dbg
    return out
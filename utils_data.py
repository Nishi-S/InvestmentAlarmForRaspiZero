#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import datetime as dt
from typing import Dict, List, Optional

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def _cache_path(cache_dir: str, ticker: str) -> str:
    safe = ticker.replace("/", "_")
    return os.path.join(cache_dir, f"{safe}.csv")


def load_cache(cache_dir: str, ticker: str, max_age_days: Optional[int] = None) -> Optional[pd.DataFrame]:
    path = _cache_path(cache_dir, ticker)
    if not os.path.exists(path):
        return None
    try:
        if max_age_days is not None:
            mtime = os.path.getmtime(path)
            age_days = (time.time() - mtime) / 86400.0
            if age_days > max_age_days:
                return None
        df = pd.read_csv(path, parse_dates=True, index_col=0)
        # Ensure columns are in expected dtype
        return df
    except Exception:
        return None


def save_cache(cache_dir: str, ticker: str, df: pd.DataFrame) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(cache_dir, ticker)
    try:
        df.to_csv(path)
    except Exception:
        pass


def _download_one(ticker: str, start: dt.date, end: Optional[dt.date]):
    try:
        import yfinance as yf
    except Exception as e:
        raise RuntimeError("yfinance not installed: pip install yfinance") from e
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return None
    return df.dropna(how="all")


def download_with_cache(
    tickers: List[str],
    start: dt.date,
    end: Optional[dt.date] = None,
    cache_dir: str = "cache",
    max_age_days: Optional[int] = 2,
    retries: int = 2,
    backoff_sec: float = 1.5,
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    missing: List[str] = []
    for t in tickers:
        # Try fresh download with retries
        df = None
        for i in range(max(1, retries + 1)):
            try:
                df = _download_one(t, start, end)
                if df is not None and not df.empty:
                    break
            except Exception:
                df = None
            time.sleep(backoff_sec * (i + 1))

        if df is None or df.empty:
            # Fallback to cache
            cached = load_cache(cache_dir, t, max_age_days=max_age_days)
            if cached is not None:
                out[t] = cached
            else:
                # last resort: try cache without staleness check
                cached = load_cache(cache_dir, t, max_age_days=None)
                if cached is not None:
                    out[t] = cached
                else:
                    missing.append(t)
        else:
            out[t] = df
            save_cache(cache_dir, t, df)
    if missing:
        logger.warning("Data unavailable for tickers (no download and no cache): %s", ", ".join(sorted(missing)))
    return out

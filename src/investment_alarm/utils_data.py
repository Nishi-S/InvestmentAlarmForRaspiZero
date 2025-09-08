#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import datetime as dt
from typing import Dict, List, Optional

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def _cache_path(cache_dir: str, ticker: str, ext: str = "pkl.gz") -> str:
    """キャッシュファイルパス（拡張子可変）。
    既定は pickle（gzip）: .pkl.gz
    CSV 互換読取用には ext="csv" を指定。
    """
    safe = ticker.replace("/", "_")
    return os.path.join(cache_dir, f"{safe}.{ext}")


def load_cache(cache_dir: str, ticker: str, max_age_days: Optional[int] = None) -> Optional[pd.DataFrame]:
    path_pkl = _cache_path(cache_dir, ticker, "pkl.gz")
    path_csv = _cache_path(cache_dir, ticker, "csv")

    def _is_stale(p: str) -> bool:
        if max_age_days is None:
            return False
        mtime = os.path.getmtime(p)
        age_days = (time.time() - mtime) / 86400.0
        return age_days > max_age_days
    try:
        # 1) pickle（優先）
        if os.path.exists(path_pkl) and not _is_stale(path_pkl):
            try:
                df = pd.read_pickle(path_pkl, compression="gzip")
                return df
            except Exception:
                pass
        # 2) CSV（後方互換）
        if os.path.exists(path_csv) and not _is_stale(path_csv):
            try:
                # pandas 2.x: 明示的に日付フォーマットを指定
                # 既存キャッシュには MultiIndex 風の2行ヘッダ（2行目が "Ticker,..."）が含まれる場合がある。
                # その場合は header=[0,1] + skiprows=[2] で読み取り、列は1階層に落とす。
                with open(path_csv, "r", encoding="utf-8") as f:
                    _first = f.readline()
                    _second = f.readline()
                if _second.startswith("Ticker,"):
                    df = pd.read_csv(
                        path_csv,
                        header=[0, 1],
                        index_col=0,
                        parse_dates=[0],
                        date_format="%Y-%m-%d",
                        skiprows=[2],
                    )
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                else:
                    df = pd.read_csv(
                        path_csv,
                        index_col=0,
                        parse_dates=[0],
                        date_format="%Y-%m-%d",
                    )
                return df
            except Exception:
                pass
        return None
    except Exception:
        return None


def save_cache(cache_dir: str, ticker: str, df: pd.DataFrame) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    # 既定は pickle(gzip)。
    path = _cache_path(cache_dir, ticker, "pkl.gz")
    try:
        # pickle は型・index をそのまま保持し、高速にロード可
        df.to_pickle(path, compression="gzip")
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
        # まず新鮮なキャッシュがあれば即使用（ネットワーク回避）
        cached_fresh = load_cache(cache_dir, t, max_age_days=max_age_days)
        if cached_fresh is not None:
            out[t] = cached_fresh
            # 移行のため pickle に保存（CSVの場合）
            try:
                save_cache(cache_dir, t, cached_fresh)
            except Exception:
                pass
            continue

        # 新鮮キャッシュが無い場合のみダウンロードを試行
        df = None
        for i in range(max(1, retries + 1)):
            try:
                df = _download_one(t, start, end)
                if df is not None and not df.empty:
                    break
            except Exception:
                df = None
            time.sleep(backoff_sec * (i + 1))

        if df is not None and not df.empty:
            out[t] = df
            save_cache(cache_dir, t, df)
            continue

        # ダウンロード失敗時は古いキャッシュでも採用
        cached_any = load_cache(cache_dir, t, max_age_days=None)
        if cached_any is not None:
            out[t] = cached_any
            try:
                save_cache(cache_dir, t, cached_any)
            except Exception:
                pass
        else:
            missing.append(t)
    if missing:
        logger.warning("Data unavailable for tickers (no download and no cache): %s", ", ".join(sorted(missing)))
    return out

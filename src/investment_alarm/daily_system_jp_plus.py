#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily Trading Helper JP PLUS (1-day cycle) — drop-in replacement 2025-09-06
- 日本株・個別株向けの簡易スクリーニング＆翌営業日の計画出力ツール
- 指標: EMA20/50, SMA100, ATR(14), Donchian20, Bollinger %B, VOLR, OBV傾き, RSI(2)
- ロット対応（1株/100株）、売買代金フィルタ、イベント除外（CSV）
- スコア合成: Trend/Momentum/Volume/Breakout/Structure（重みは設定可）
- 出力: ranks_*.csv（全銘柄スコア） / plan_*.csv（採用＆株数） / plan_latest.json（要約）
- メール本文を「人間可読の要約 + 機械可読YAMLブロック」にリッチ化
注意: 教育・情報提供のみ。実売買前に必ずご自身で検証してください。
"""

import os
import time
import logging
import sys
import math
import json
import argparse
import datetime as dt
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

from .utils_data import download_with_cache
from .notify import send_email as _send_email
from .status_page import generate_status_html

logger = logging.getLogger(__name__)

# --- optional: load /etc/jpplus.env when run outside systemd ---
def _load_env_file(path="/etc/jpplus.env"):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
    except Exception as e:
        logger.warning("failed to read env file %s: %s", path, e)
_load_env_file()

# ===== 指標関数 =====
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).std()

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]; low = df["Low"]; close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(),
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]; low = df["Low"]; close = df["Close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_ = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / (atr_ + 1e-9))
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / (atr_ + 1e-9))
    dx = (100 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-9)).fillna(0.0)
    adx_ = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx_

def rsi_wilder(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100 - (100 / (1 + rs))

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    sign = np.sign(close.diff().fillna(0))
    return (sign * volume.fillna(0)).cumsum()

def lin_slope(series: pd.Series, lookback: int = 20) -> float:
    s = series.dropna()
    if len(s) < lookback:
        return np.nan
    y = s.iloc[-lookback:].values
    x = np.arange(lookback)
    x = (x - x.mean()) / (x.std() + 1e-9)
    b = np.polyfit(x, y, 1)[0]
    return float(np.asarray(b).item())  # 0次元ndarray→確実にスカラー化

def donchian_high(series: pd.Series, window: int = 20) -> pd.Series:
    return series.rolling(window).max()

def bollinger_percent_b(close: pd.Series, window: int = 20, k: float = 2.0) -> pd.Series:
    mid = sma(close, window)
    sd = std(close, window)
    upper = mid + k * sd
    lower = mid - k * sd
    width = (upper - lower).replace(0, np.nan)
    return (close - lower) / width

def macd_hist(close: pd.Series) -> pd.Series:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal

# ===== 設定 =====
@dataclass
class Config:
    universe: List[str]
    benchmark: str = "1306.T"
    risk_on_assets: List[str] = field(default_factory=list)
    defensive_assets: List[str] = field(default_factory=list)  # JPでは休む想定
    data_days: int = 420
    regime_ma: int = 100
    atr_period: int = 14
    atr_mult_stop: float = 2.0
    top_k: int = 1
    capital: float = 200000.0
    risk_per_trade: float = 0.005
    min_turnover_jpy: float = 1.0e8  # 20日平均売買代金 下限
    lot_size_default: int = 1
    lot_size_map: Dict[str, int] = field(default_factory=dict)
    events_csv: Optional[str] = None  # "ticker,until,reason" (until=YYYY-MM-DD)
    outdir: str = "reports"
    email_enabled: bool = False
    email_to: Optional[str] = None
    email_subject: str = "DAILY-JP PLAN"
    # 追加オプション
    holiday_skip_jp: bool = True
    cache_dir: str = "cache"
    cache_max_age_days: int = 2
    # ホールド＆エグジット
    hold_days: int = 1
    trailing_stop_mult: float = 0.0  # 0=無効
    # サイジング
    min_notional_jpy: float = 0.0
    notional_cap_per_ticker_jpy: float = 1e18
    commission_per_trade_jpy: float = 0.0
    slippage_bps: float = 0.0
    # 再エントリークールオフ
    reentry_cooloff_days: int = 0
    picks_history_file: str = "reports/picks_history.json"
    # イベント自動ブロック拡張
    event_block_days_before: int = 0
    event_block_days_after: int = 0
    # スコア重み
    w_trend: float = 0.30
    w_momo: float = 0.25
    w_volume: float = 0.20
    w_breakout: float = 0.15
    w_structure: float = 0.10
    w_strength: float = 0.05
    # 分散制約・相関
    sector_map_csv: Optional[str] = "config/sector_map.csv"
    sector_max_fraction: float = 1.0
    corr_max: float = 1.0
    corr_lookback_days: int = 20
    # ポートフォリオ配分
    risk_total_pct: float = 0.0
    risk_allocation_mode: str = "per_trade"  # or "portfolio"
    # リスクオフ採用
    allow_pick_in_risk_off: bool = False
    risk_off_risk_scale: float = 0.5

def load_config(path: str) -> Config:
    try:
        import yaml
    except Exception:
        print("PyYAML が必要です: pip install pyyaml", file=sys.stderr)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(**raw)

# ===== データ取得 =====
def download_history(tickers: List[str], start: dt.date, cache_dir: str, max_age_days: int) -> Dict[str, pd.DataFrame]:
    return download_with_cache(tickers, start=start, end=None, cache_dir=cache_dir, max_age_days=max_age_days)

def _is_jp_trading_day() -> bool:
    # jpholiday があれば祝日も判定。なければ月〜金
    try:
        import jpholiday  # type: ignore
        today = dt.date.today()
        if today.weekday() >= 5:
            return False
        return not jpholiday.is_holiday(today)
    except Exception:
        return dt.date.today().weekday() < 5

# ===== レジーム判定 =====
def risk_on_regime(bench_df: pd.DataFrame, ma: int) -> bool:
    close = bench_df["Close"].dropna()
    if len(close) < ma + 25:
        return False
    ma_series = sma(close, ma)
    close_last = float(np.asarray(close.iloc[-1]).item())
    ma_last    = float(np.asarray(ma_series.iloc[-1]).item())
    above = (close_last > ma_last)
    slope = lin_slope(ma_series, lookback=20)
    if np.isnan(slope):
        return False
    return bool(above and (float(slope) > 0.0))

# ===== 特徴量 =====
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["EMA20"] = ema(out["Close"], 20)
    out["EMA50"] = ema(out["Close"], 50)
    out["SMA100"] = sma(out["Close"], 100)
    out["ATR"] = atr(out, 14)
    out["DonHigh20"] = donchian_high(out["High"], 20)
    out["PB"] = bollinger_percent_b(out["Close"], 20, 2.0)
    out["VOLR"] = out["Volume"] / (out["Volume"].rolling(20).mean() + 1e-9)
    out["OBV"] = obv(out["Close"], out["Volume"])
    # OBV傾きは直近だけ計算して軽量化
    out["OBV_SLOPE20"] = np.nan
    if len(out["OBV"]) >= 21:
        out.loc[out.index[-1], "OBV_SLOPE20"] = lin_slope(out["OBV"], 20)
    out["RSI2"] = rsi_wilder(out["Close"], 2)
    out["RET5"] = out["Close"].pct_change(5)
    out["RET20"] = out["Close"].pct_change(20)
    # 追加指標
    out["DIST52H"] = np.nan
    if len(out) >= 60:
        rollmax = out["Close"].rolling(252, min_periods=60).max()
        out["DIST52H"] = out["Close"] / (rollmax + 1e-9)
    out["ADX14"] = _ensure_series(adx(out, 14), out.index)
    out["MACD_HIST"] = _ensure_series(macd_hist(out["Close"]), out.index)
    return out

# ===== スコアリング =====
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def score_row(row: Dict[str, float], med_atr: float) -> Dict[str, float]:
    trend_components = [
        1.0 if row["Close"] > row["EMA20"]  else 0.0,
        1.0 if row["Close"] > row["EMA50"]  else 0.0,
        1.0 if row["Close"] > row["SMA100"] else 0.0,
    ]
    trend = sum(trend_components) / 3.0

    mom5  = row.get("RET5_Z", 0.0)
    mom20 = row.get("RET20_Z", 0.0)
    momo = clamp01(0.5 + 0.25 * float(mom5) + 0.25 * float(mom20))

    volr_score = clamp01((min(row["VOLR"], 2.0) - 0.5) / 1.5)
    obv_s = row.get("OBV_SLOPE20", np.nan)
    obv_score = 0.6 if (not np.isnan(obv_s) and obv_s > 0) else 0.4
    volume = 0.8 * volr_score + 0.2 * obv_score

    breakout = 0.0
    don = row.get("DonHigh20", np.nan)
    pb  = row.get("PB", np.nan)
    if not np.isnan(don) and don > 0:
        proximity = clamp01((row["Close"] - 0.98 * don) / (0.02 * don + 1e-9))
        breakout = 0.6 * proximity + 0.4 * (clamp01(pb) if not np.isnan(pb) else 0.0)
    elif not np.isnan(pb):
        breakout = clamp01(pb)

    if med_atr <= 0 or np.isnan(row["ATR"]):
        structure = 0.5
    else:
        ratio = row["ATR"] / med_atr
        structure = clamp01(1.0 - min(abs(np.log(ratio)), 2.0) / 2.0)

    # 追加: 強さ（52週高近さ、ADX、MACDヒスト）
    d52 = row.get("DIST52H", np.nan)
    adx14 = row.get("ADX14", np.nan)
    mh = row.get("MACD_HIST", np.nan)
    s_d52 = 0.5 if np.isnan(d52) else clamp01(float(d52))
    s_adx = 0.5 if np.isnan(adx14) else clamp01(float(adx14) / 50.0)
    s_mh = 0.5 if np.isnan(mh) else (0.7 if mh > 0 else 0.3)
    strength = 0.5 * s_d52 + 0.3 * s_adx + 0.2 * s_mh

    return {
        "trend": float(trend),
        "momo": float(momo),
        "volume": float(volume),
        "breakout": float(breakout),
        "structure": float(structure),
        "strength": float(strength),
    }

def build_scores(cfg: Config, frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    # 事前統計
    ret5_vals, ret20_vals, atr_vals = [], [], []
    feats_cache: Dict[str, pd.DataFrame] = {}
    for t, df in frames.items():
        if len(df) < 120:  # 最低データ長
            continue
        feats = compute_features(df)
        feats_cache[t] = feats
        if not np.isnan(feats["RET5"].iloc[-1]):   ret5_vals.append(float(np.asarray(feats["RET5"].iloc[-1]).item()))
        if not np.isnan(feats["RET20"].iloc[-1]):  ret20_vals.append(float(np.asarray(feats["RET20"].iloc[-1]).item()))
        atr_vals.append(float(np.asarray(feats["ATR"].iloc[-1]).item()))

    ret5_mean = np.mean(ret5_vals) if ret5_vals else 0.0
    ret5_std  = np.std(ret5_vals) + 1e-9
    ret20_mean = np.mean(ret20_vals) if ret20_vals else 0.0
    ret20_std  = np.std(ret20_vals) + 1e-9
    med_atr = float(np.median(atr_vals)) if atr_vals else 0.0

    rows = []
    for t, feats in feats_cache.items():
        # スカラー抽出
        c     = float(np.asarray(feats["Close"].iloc[-1]).item())
        e20   = float(np.asarray(feats["EMA20"].iloc[-1]).item())
        e50   = float(np.asarray(feats["EMA50"].iloc[-1]).item())
        s100  = float(np.asarray(feats["SMA100"].iloc[-1]).item())
        avtr  = float(np.asarray(feats["ATR"].iloc[-1]).item())
        d20   = feats["DonHigh20"].iloc[-1]
        pb    = feats["PB"].iloc[-1]
        volr  = float(np.asarray(feats["VOLR"].iloc[-1]).item())
        obvs  = feats["OBV_SLOPE20"].iloc[-1]
        rsi2  = feats["RSI2"].iloc[-1]
        ret5  = float(np.asarray(feats["RET5"].iloc[-1]).item())
        ret20 = float(np.asarray(feats["RET20"].iloc[-1]).item())
        dist52 = feats["DIST52H"].iloc[-1] if "DIST52H" in feats.columns else np.nan
        adx14  = feats["ADX14"].iloc[-1] if "ADX14" in feats.columns else np.nan
        mh     = feats["MACD_HIST"].iloc[-1] if "MACD_HIST" in feats.columns else np.nan

        ret5_z  = (ret5  - ret5_mean)  / ret5_std
        ret20_z = (ret20 - ret20_mean) / ret20_std

        row_vals = {
            "Close": c, "EMA20": e20, "EMA50": e50, "SMA100": s100,
            "ATR": avtr,
            "DonHigh20": float(d20) if pd.notna(d20) else np.nan,
            "PB": float(pb) if pd.notna(pb) else np.nan,
            "VOLR": volr,
            "OBV_SLOPE20": float(obvs) if pd.notna(obvs) else np.nan,
            "RSI2": float(rsi2) if pd.notna(rsi2) else np.nan,
            "RET5_Z": float(ret5_z), "RET20_Z": float(ret20_z),
            "DIST52H": float(dist52) if pd.notna(dist52) else np.nan,
            "ADX14": float(adx14) if pd.notna(adx14) else np.nan,
            "MACD_HIST": float(mh) if pd.notna(mh) else np.nan,
        }
        s = score_row(row_vals, med_atr)

        avg_vol = float(np.asarray(feats["Volume"].rolling(20).mean().iloc[-1]).item())
        turnover = avg_vol * c

        rows.append({
            "ticker": t,
            "close": round(c, 4),
            "ema20": round(e20, 4),
            "ema50": round(e50, 4),
            "sma100": round(s100, 4),
            "atr": round(avtr, 4),
            "ret5": round(ret5, 4),
            "ret20": round(ret20, 4),
            "volr": round(volr, 3),
            "obv_slope20": round(row_vals["OBV_SLOPE20"] if not np.isnan(row_vals["OBV_SLOPE20"]) else 0.0, 6),
            "pb": round(row_vals["PB"], 3) if not np.isnan(row_vals["PB"]) else np.nan,
            "don20": round(row_vals["DonHigh20"], 4) if not np.isnan(row_vals["DonHigh20"]) else np.nan,
            "rsi2": round(row_vals["RSI2"], 2) if not np.isnan(row_vals["RSI2"]) else np.nan,
            "score_trend": round(s["trend"], 4),
            "score_momo": round(s["momo"], 4),
            "score_volume": round(s["volume"], 4),
            "score_breakout": round(s["breakout"], 4),
            "score_structure": round(s["structure"], 4),
            "strength": round(s.get("strength", 0.0), 4),
            "score_total": round(
                cfg.w_trend * s["trend"] + cfg.w_momo * s["momo"] +
                cfg.w_volume * s["volume"] + cfg.w_breakout * s["breakout"] +
                cfg.w_structure * s["structure"] + cfg.w_strength * s.get("strength", 0.0), 4
            ),
            "turnover20": round(turnover, 2),
        })

    return pd.DataFrame(rows).sort_values("score_total", ascending=False)

# ===== イベント除外 =====
def load_event_block(cfg: Config) -> Dict[str, Tuple[Optional[dt.date], dt.date]]:
    block_range: Dict[str, Tuple[Optional[dt.date], dt.date]] = {}
    if cfg.events_csv and os.path.exists(cfg.events_csv):
        try:
            df = pd.read_csv(cfg.events_csv)
            for _, r in df.iterrows():
                try:
                    tkr = str(r["ticker"]).strip()
                    if "until" in r and not pd.isna(r["until"]):
                        until = dt.datetime.strptime(str(r["until"]), "%Y-%m-%d").date()
                        start = None
                        if cfg.event_block_days_before and cfg.event_block_days_before > 0:
                            start = until - dt.timedelta(days=cfg.event_block_days_before)
                    elif "date" in r and not pd.isna(r["date"]):
                        ev = dt.datetime.strptime(str(r["date"]), "%Y-%m-%d").date()
                        start = ev - dt.timedelta(days=cfg.event_block_days_before)
                        until = ev + dt.timedelta(days=cfg.event_block_days_after)
                    else:
                        continue
                    block_range[tkr] = (start, until)
                except Exception:
                    continue
        except Exception as e:
            logger.warning("events_csv 読み込み失敗: %s", e)
    return block_range

def load_reentry_block(cfg: Config) -> Dict[str, dt.date]:
    if cfg.reentry_cooloff_days <= 0:
        return {}
    path = cfg.picks_history_file
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            arr = json.load(f)
        block: Dict[str, dt.date] = {}
        for rec in arr:
            t = str(rec.get("ticker"))
            d = rec.get("date")
            if not t or not d:
                continue
            try:
                dt0 = dt.datetime.strptime(d, "%Y-%m-%d").date()
                until = dt0 + dt.timedelta(days=cfg.reentry_cooloff_days)
                prev = block.get(t)
                if (prev is None) or (until > prev):
                    block[t] = until
            except Exception:
                continue
        return block
    except Exception:
        return {}

# ===== 候補選定（却下理由つき） =====
def decide_candidates(
    cfg: Config,
    score_df: pd.DataFrame,
    is_risk_on: bool,
    block_ranges: Optional[Dict[str, Tuple[Optional[dt.date], dt.date]]] = None,
    corr: Optional[Dict[str, Dict[str, float]]] = None,
    block_until: Optional[Dict[str, dt.date]] = None,
):
    today = dt.date.today()
    if score_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # 互換: 古い引数 block_until の対応
    if block_ranges is None:
        if block_until:
            block_ranges = {k: (None, v) for k, v in block_until.items()}
        else:
            block_ranges = {}

    if is_risk_on:
        pool = cfg.risk_on_assets or cfg.universe
    else:
        pool = cfg.defensive_assets if cfg.allow_pick_in_risk_off else []

    # セクターマップを一度だけ読み込む（高速化）
    sector_map = None
    try:
        if cfg.sector_map_csv and os.path.exists(cfg.sector_map_csv):
            sm = pd.read_csv(cfg.sector_map_csv)
            sector_map = {str(x.get('ticker')).strip(): str(x.get('sector') or 'UNKNOWN').strip() for _, x in sm.iterrows()}
    except Exception:
        sector_map = None

    filt = []
    for _, r in score_df.iterrows():
        t = r["ticker"]
        if t not in pool:
            continue
        reasons = []
        if r["turnover20"] < cfg.min_turnover_jpy:
            reasons.append(f"low_turnover<{cfg.min_turnover_jpy:.0f}")
        if t in block_ranges:
            start, until = block_ranges[t]
            if (until and today <= until) and (start is None or today >= start):
                reasons.append(f"event_block_until {until}")
        if (r["close"] < r["ema20"]) or (r["close"] < r["ema50"]) or (r["close"] < r["sma100"]):
            reasons.append("below_MAs")
        # sector map（事前に読み込んだ dict を参照）
        sec = sector_map.get(t, 'UNKNOWN') if sector_map else 'UNKNOWN'
        filt.append((t, r, reasons, sec))

    kept = [ (t, r, reasons, sec) for (t, r, reasons, sec) in filt if len(reasons) == 0 ]
    kept_sorted = sorted(kept, key=lambda x: x[1]["score_total"], reverse=True)
    # diversification: sector cap and corr
    picks = []
    sector_counts: Dict[str, int] = {}
    sector_cap = max(1, int(np.ceil(cfg.sector_max_fraction * max(cfg.top_k, 1)))) if cfg.sector_max_fraction < 1.0 else max(cfg.top_k, 10**9)
    for t, r, reasons, sec in kept_sorted:
        if len(picks) >= cfg.top_k:
            break
        if sector_counts.get(sec, 0) >= sector_cap:
            continue
        ok_corr = True
        if corr and cfg.corr_max < 1.0 and picks:
            for pt, pr, _, psec in picks:
                c = None
                try:
                    c = corr.get(t, {}).get(pt, None)
                    if c is None:
                        c = corr.get(pt, {}).get(t, None)
                except Exception:
                    c = None
                if c is not None and c > cfg.corr_max:
                    ok_corr = False
                    break
        if not ok_corr:
            continue
        picks.append((t, r, reasons, sec))
        sector_counts[sec] = sector_counts.get(sec, 0) + 1

    plan_rows = []
    risk_scale = 1.0 if is_risk_on else (cfg.risk_off_risk_scale if cfg.allow_pick_in_risk_off else 0.0)
    if cfg.risk_allocation_mode == "portfolio" and cfg.risk_total_pct > 0 and len(picks) > 0:
        total_budget = max(cfg.capital * cfg.risk_total_pct * risk_scale - 2.0 * cfg.commission_per_trade_jpy * len(picks), 0.0)
        # weights = 1/ATR（低ボラ優遇）
        weights = []
        rps_list = []
        for t, r, _, _ in picks:
            w = 1.0 / max(float(r["atr"]), 1e-6)
            weights.append(max(w, 1e-9))
            rps = cfg.atr_mult_stop * max(float(r["atr"]), 1e-6) + (cfg.slippage_bps / 10000.0) * float(r["close"]) * 2.0
            rps_list.append(rps)
        wsum = sum(weights)
        for (t, r, _, sec), w, rps in zip(picks, weights, rps_list):
            rb = total_budget * (w / wsum)
            lot = cfg.lot_size_map.get(t, cfg.lot_size_default) if cfg.lot_size_default else 1
            raw_shares = int(max(math.floor(rb / max(rps, 1e-9)), 0))
            shares = max((raw_shares // max(lot, 1)) * max(lot, 1), 0)
            # 最低約定金額
            if cfg.min_notional_jpy > 0 and float(r["close"]) * shares < cfg.min_notional_jpy:
                need = int(math.ceil(cfg.min_notional_jpy / max(float(r["close"]), 1e-6)))
                shares = max((need // max(lot, 1)) * max(lot, 1), shares)
            # ティッカー上限
            if cfg.notional_cap_per_ticker_jpy < 1e17:
                cap_shares = int(math.floor(cfg.notional_cap_per_ticker_jpy / max(float(r["close"]), 1e-6)))
                cap_shares = max((cap_shares // max(lot, 1)) * max(lot, 1), 0)
                shares = min(shares, cap_shares)
            stop = float(r["close"]) - cfg.atr_mult_stop * float(r["atr"])
            target = float(r["close"]) + 3.0 * float(r["atr"])
            plan_rows.append({
                "date": today.isoformat(),
                "ticker": t,
                "regime": "RISK_ON" if is_risk_on else "RISK_OFF",
                "close_ref": float(r["close"]),
                "atr": float(r["atr"]),
                "stop_ref": round(float(stop), 4),
                "target_ref": round(float(target), 4),
                "shares": int(shares),
                "notional": round(shares * float(r["close"]), 2),
                "risk_per_trade": cfg.risk_per_trade,
                "atr_mult_stop": cfg.atr_mult_stop,
                "entry_plan": "next_open",
                "exit_plan": ("trail" if cfg.trailing_stop_mult and cfg.trailing_stop_mult > 0 else f"{cfg.hold_days}d_close"),
                "score_total": float(r["score_total"]),
                "score_breakdown": f"T{r['score_trend']:.2f}|M{r['score_momo']:.2f}|V{r['score_volume']:.2f}|B{r['score_breakout']:.2f}|S{r['score_structure']:.2f}",
                "volr": float(r["volr"]),
                "pb": float(r["pb"]) if not np.isnan(r["pb"]) else np.nan,
                "don20": float(r["don20"]) if not np.isnan(r["don20"]) else np.nan,
                "obv_slope20": float(r["obv_slope20"]),
                "turnover20": float(r["turnover20"]),
                "sector": sec,
            })
    else:
        for t, r, _, sec in picks:
            risk_budget = max(cfg.capital * cfg.risk_per_trade * risk_scale - 2.0 * cfg.commission_per_trade_jpy, 0.0)
            rps = cfg.atr_mult_stop * max(float(r["atr"]), 1e-6) + (cfg.slippage_bps / 10000.0) * float(r["close"]) * 2.0
            raw_shares = math.floor(risk_budget / max(rps, 1e-9))
            lot = cfg.lot_size_map.get(t, cfg.lot_size_default) if cfg.lot_size_default else 1
            shares = max((raw_shares // max(lot, 1)) * max(lot, 1), 0)
            # 最低約定金額を満たすよう調整
            if cfg.min_notional_jpy > 0 and r["close"] * shares < cfg.min_notional_jpy:
                need = int(math.ceil(cfg.min_notional_jpy / max(r["close"], 1e-6)))
                shares = max((need // max(lot, 1)) * max(lot, 1), shares)
            # ティッカー上限
            if cfg.notional_cap_per_ticker_jpy < 1e17:
                cap_shares = int(math.floor(cfg.notional_cap_per_ticker_jpy / max(r["close"], 1e-6)))
                cap_shares = max((cap_shares // max(lot, 1)) * max(lot, 1), 0)
                shares = min(shares, cap_shares)
            stop = r["close"] - cfg.atr_mult_stop * r["atr"]
            target = r["close"] + 3.0 * r["atr"]
            plan_rows.append({
                "date": today.isoformat(),
                "ticker": t,
                "regime": "RISK_ON" if is_risk_on else "RISK_OFF",
                "close_ref": float(r["close"]),
                "atr": float(r["atr"]),
                "stop_ref": round(float(stop), 4),
                "target_ref": round(float(target), 4),
                "shares": int(shares),
                "notional": round(shares * float(r["close"]), 2),
                "risk_per_trade": cfg.risk_per_trade,
                "atr_mult_stop": cfg.atr_mult_stop,
                "entry_plan": "next_open",
                "exit_plan": ("trail" if cfg.trailing_stop_mult and cfg.trailing_stop_mult > 0 else f"{cfg.hold_days}d_close"),
                "score_total": float(r["score_total"]),
                "score_breakdown": f"T{r['score_trend']:.2f}|M{r['score_momo']:.2f}|V{r['score_volume']:.2f}|B{r['score_breakout']:.2f}|S{r['score_structure']:.2f}",
                "volr": float(r["volr"]),
                "pb": float(r["pb"]) if not np.isnan(r["pb"]) else np.nan,
                "don20": float(r["don20"]) if not np.isnan(r["don20"]) else np.nan,
                "obv_slope20": float(r["obv_slope20"]),
                "turnover20": float(r["turnover20"]),
                "sector": sec,
            })

    # 却下上位（理由つき）
    rej_rows = []
    rejected_sorted = sorted([x for x in filt if len(x[2]) > 0],
                             key=lambda x: x[1]["score_total"], reverse=True)[:5]
    for t, r, reasons, _ in rejected_sorted:
        rej_rows.append({
            "ticker": t,
            "score_total": float(r["score_total"]),
            "reasons": "|".join(reasons)
        })

    return pd.DataFrame(plan_rows), pd.DataFrame(rej_rows)

# ===== メール本文生成（テスト容易化のため分離） =====
def format_daily_email(
    cfg: Config,
    is_risk_on: bool,
    plan_df: pd.DataFrame,
    rej_df: pd.DataFrame,
    plan_csv_path: str,
    ranks_csv_path: str,
) -> str:
    date_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M JST")
    body_lines = [
        f"Regime: {'RISK_ON' if is_risk_on else 'RISK_OFF'} (bench={cfg.benchmark})   Date: {date_str}",
        f"Capital: {cfg.capital:.0f}  Risk/Trade: {cfg.risk_per_trade*100:.1f}%  ATRx: {cfg.atr_mult_stop}  Lot: {cfg.lot_size_default}",
        f"Universe: {len(cfg.universe)} tickers",
        "",
        "[Plan]",
    ]
    if plan_df.empty:
        body_lines.append("No picks based on current rules.")
    else:
        for _, r in plan_df.iterrows():
            body_lines.append(
                f"- {r['ticker']}  shares={r['shares']}  close={r['close_ref']}  ATR={r['atr']}  "
                f"stop={r['stop_ref']}  target={r['target_ref']}  score={r['score_total']}  "
                f"[{r['score_breakdown']}]"
            )
            obv_txt = "pos" if r["obv_slope20"] > 0 else "neg"
            body_lines.append(
                f"  volr={r['volr']}  pb={r['don20'] if np.isnan(r['pb']) else r['pb']}  "
                f"don20={r['don20']}  obv_slope20={obv_txt}  turnover20={r['turnover20']:.2e}"
            )
            body_lines.append("  entry=next_open  exit=1d_close")

    body_lines.append("\n[Rejected top]")
    if rej_df.empty:
        body_lines.append("(none)")
    else:
        for _, r in rej_df.iterrows():
            body_lines.append(f"- {r['ticker']}  reason={r['reasons']}  score={r['score_total']}")

    body_lines.append("\n[Files]")
    body_lines.append(os.path.abspath(plan_csv_path))
    body_lines.append(os.path.abspath(ranks_csv_path))

    # YAML ブロック
    yaml_lines = []
    yaml_lines.append("```yaml")
    yaml_lines.append("jpplan:")
    yaml_lines.append(f"  date: {dt.date.today().isoformat()}")
    yaml_lines.append(f"  regime: {'RISK_ON' if is_risk_on else 'RISK_OFF'}")
    yaml_lines.append(f"  capital: {cfg.capital}")
    yaml_lines.append(f"  risk_per_trade: {cfg.risk_per_trade}")
    yaml_lines.append(f"  atr_mult_stop: {cfg.atr_mult_stop}")
    yaml_lines.append(f"  lot_size_default: {cfg.lot_size_default}")
    yaml_lines.append("  picks: []" if plan_df.empty else "  picks:")
    for _, r in plan_df.iterrows():
        obv_txt = "pos" if r["obv_slope20"] > 0 else "neg"
        yaml_lines.append(f"    - ticker: {r['ticker']}")
        yaml_lines.append(f"      shares: {int(r['shares'])}")
        yaml_lines.append(f"      close: {float(r['close_ref'])}")
        yaml_lines.append(f"      atr: {float(r['atr'])}")
        yaml_lines.append(f"      stop: {float(r['stop_ref'])}")
        yaml_lines.append(f"      target: {float(r['target_ref'])}")
        yaml_lines.append(f"      score_total: {float(r['score_total'])}")
        yaml_lines.append(f"      score_breakdown: \"{r['score_breakdown']}\"")
        yaml_lines.append(f"      signals: {{volr: {float(r['volr'])}, pb: {('null' if np.isnan(r['pb']) else float(r['pb']))}, don20: {('null' if np.isnan(r['don20']) else float(r['don20']))}, obv_slope20: {obv_txt}}}")
        yaml_lines.append(f"      turnover20: {float(r['turnover20'])}")
        yaml_lines.append(f"      entry: next_open")
        yaml_lines.append(f"      exit: 1d_close")
    yaml_lines.append("  rejected_top: []" if rej_df.empty else "  rejected_top:")
    for _, r in rej_df.iterrows():
        yaml_lines.append(f"    - ticker: {r['ticker']}")
        yaml_lines.append(f"      reason: \"{r['reasons']}\"")
        yaml_lines.append(f"      score_total: {float(r['score_total'])}")
    yaml_lines.append("```")

    body_lines.append("\n# 機械可読ブロック（このままChatGPTに貼ってください）")
    body_lines.extend(yaml_lines)
    return "\n".join(body_lines)

# ===== メイン =====
def main():
    # Logging 基本設定（環境変数 LOG_LEVEL で上書き可）
    if not logging.getLogger().handlers:
        logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="[%(levelname)s] %(message)s")

    p = argparse.ArgumentParser(description="Daily Trading Helper JP PLUS")
    p.add_argument("-c", "--config", default="config/config.jp.yml")
    p.add_argument("--outdir", default=None)
    p.add_argument("--no-email", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.outdir:
        cfg.outdir = args.outdir

    # 休日スキップ（日本）
    if cfg.holiday_skip_jp and not _is_jp_trading_day():
        logger.info("Skipping (not a JP trading day)")
        return

    start_all = dt.datetime.now()
    start = dt.date.today() - dt.timedelta(days=cfg.data_days)
    tickers = sorted(set([cfg.benchmark] + cfg.universe))
    logger.info("Downloading: %s since %s", ", ".join(tickers), start)

    t0 = time.time()
    frames = download_history(tickers, start, cache_dir=cfg.cache_dir, max_age_days=cfg.cache_max_age_days)
    logger.info("Timing: data fetch %.3fs", time.time() - t0)
    if cfg.benchmark not in frames or frames[cfg.benchmark].empty:
        logger.error("ベンチマークの価格系列を取得できませんでした。")
        sys.exit(2)

    t1 = time.time()
    is_risk_on = risk_on_regime(frames[cfg.benchmark], cfg.regime_ma)
    score_df = build_scores(cfg, {t: frames[t] for t in cfg.universe if t in frames})
    logger.info("Timing: scoring %.3fs", time.time() - t1)

    write_outputs = not args.dry_run
    if write_outputs:
        os.makedirs(cfg.outdir, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    ranks_csv = os.path.join(cfg.outdir, f"ranks_{ts}.csv")
    if write_outputs:
        score_df.to_csv(ranks_csv, index=False, encoding="utf-8")

    block_ranges = load_event_block(cfg)
    # 再エントリーブロックを併合（endのみ延長）
    re_block = load_reentry_block(cfg)
    for k, v in re_block.items():
        prev = block_ranges.get(k)
        if prev is None:
            block_ranges[k] = (None, v)
        else:
            start, end = prev
            if v > end:
                block_ranges[k] = (start, v)

    # 相関行列（必要時のみ）
    corr = None
    try:
        if cfg.corr_max < 1.0 and not score_df.empty:
            tickers_in_scores = [t for t in score_df["ticker"].tolist() if t in frames]
            cols = []
            for t in tickers_in_scores:
                df = frames[t]
                s = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
                cols.append(s.rename(t))
            if cols:
                t2 = time.time()
                mat = pd.concat(cols, axis=1, join="inner").dropna()
                if len(mat) > cfg.corr_lookback_days + 2:
                    mat = mat.iloc[-cfg.corr_lookback_days - 1:]
                rets = mat.pct_change().dropna(how="all").fillna(0.0)
                cm = rets.corr().fillna(0.0)
                corr = {i: cm.loc[i].to_dict() for i in cm.index}
                logger.info("Timing: correlation %.3fs", time.time() - t2)
    except Exception:
        corr = None

    t3 = time.time()
    plan_df, rej_df = decide_candidates(cfg, score_df, is_risk_on, block_ranges, corr=corr)
    logger.info("Timing: decide %.3fs", time.time() - t3)
    plan_csv = os.path.join(cfg.outdir, f"plan_{ts}.csv")
    if write_outputs:
        plan_df.to_csv(plan_csv, index=False, encoding="utf-8")

    # 最新要約JSON
    latest_json = os.path.join(cfg.outdir, "plan_latest.json")
    summary = {
        "date": dt.date.today().isoformat(),
        "regime": "RISK_ON" if is_risk_on else "RISK_OFF",
        "top": plan_df.to_dict(orient="records") if not plan_df.empty else [],
    }
    if write_outputs:
        with open(latest_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    # メール本文（要約＋YAML）
    body = format_daily_email(cfg, is_risk_on, plan_df, rej_df, plan_csv, ranks_csv)

    if not args.no_email and write_outputs:
        _send_email(cfg.email_enabled, cfg.email_to, cfg.email_subject, body)

    # コンソール要約
    logger.info("\n=== SUMMARY ===")
    logger.info("Regime: %s (benchmark: %s)", "RISK_ON" if is_risk_on else "RISK_OFF", cfg.benchmark)
    if plan_df.empty:
        logger.info("No picks for tomorrow based on current regime/rules.")
    else:
        logger.info("\n%s", plan_df.to_string(index=False))

    # 履歴に追記（再エントリー制御用）
    if write_outputs:
        try:
            hist_path = cfg.picks_history_file
            os.makedirs(os.path.dirname(hist_path), exist_ok=True)
            prev = []
            if os.path.exists(hist_path):
                with open(hist_path, "r", encoding="utf-8") as f:
                    prev = json.load(f)
            for _, r in plan_df.iterrows():
                prev.append({"date": dt.date.today().isoformat(), "ticker": r["ticker"]})
            with open(hist_path, "w", encoding="utf-8") as f:
                json.dump(prev[-1000:], f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ステータスHTMLとメトリクス
    if write_outputs:
        try:
            generate_status_html(os.path.join(cfg.outdir, "status.html"), daily_json_path=latest_json, alarms_json_path="reports_alarms/alarms_latest.json")
        except Exception:
            pass
        try:
            runtime_sec = (dt.datetime.now() - start_all).total_seconds()
            picks_count = len(plan_df) if not plan_df.empty else 0
            rejects_count = len(rej_df) if not rej_df.empty else 0
            metrics = [
                f"daily_picks_count {picks_count}",
                f"daily_rejects_count {rejects_count}",
                f"script_run_seconds{{script=\"daily\"}} {runtime_sec:.3f}",
            ]
            # セクター分布
            try:
                if not plan_df.empty and "sector" in plan_df.columns:
                    vc = plan_df["sector"].value_counts()
                    for sec, cnt in vc.items():
                        metrics.append(f"daily_picks_sector_count{{sector=\"{sec}\"}} {int(cnt)}")
            except Exception:
                pass
            with open(os.path.join(cfg.outdir, "metrics.prom"), "w", encoding="utf-8") as f:
                f.write("\n".join(metrics) + "\n")
        except Exception:
            pass

    logger.info("\nSaved:\n - %s\n - %s\n - %s", plan_csv, ranks_csv, latest_json)
    logger.info("\n※ 実売買前に、取引コスト・税制・流動性・決算/材料・約定方法等を必ずご確認ください。")

if __name__ == "__main__":
    main()
def _ensure_series(x: Any, index: pd.Index) -> pd.Series:
    """返り値を安全に Series 化して、指定 index に合わせる。
    DataFrame が来た場合は先頭列を使用。
    """
    try:
        if isinstance(x, pd.Series):
            return x.reindex(index)
        if isinstance(x, pd.DataFrame):
            if x.shape[1] >= 1:
                return x.iloc[:, 0].reindex(index)
        # スカラや配列の場合
        s = pd.Series(x, index=index)
        return s
    except Exception:
        return pd.Series(index=index, dtype=float)

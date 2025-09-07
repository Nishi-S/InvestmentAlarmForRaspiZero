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
import sys
import math
import json
import argparse
import datetime as dt
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

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
        print(f"[WARN] failed to read env file {path}: {e}")
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
    # スコア重み
    w_trend: float = 0.30
    w_momo: float = 0.25
    w_volume: float = 0.20
    w_breakout: float = 0.15
    w_structure: float = 0.10

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
def download_history(tickers: List[str], start: dt.date) -> Dict[str, pd.DataFrame]:
    try:
        import yfinance as yf
    except Exception:
        print("yfinance が必要です: pip install yfinance", file=sys.stderr)
        sys.exit(1)
    data = {}
    for t in tickers:
        df = yf.download(t, start=start, auto_adjust=False, progress=False)
        if df is None or df.empty:
            print(f"[WARN] データなし: {t}")
            continue
        data[t] = df.dropna(how="all")
    return data

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

    return {
        "trend": float(trend),
        "momo": float(momo),
        "volume": float(volume),
        "breakout": float(breakout),
        "structure": float(structure),
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
            "score_total": round(
                cfg.w_trend * s["trend"] + cfg.w_momo * s["momo"] +
                cfg.w_volume * s["volume"] + cfg.w_breakout * s["breakout"] +
                cfg.w_structure * s["structure"], 4
            ),
            "turnover20": round(turnover, 2),
        })

    return pd.DataFrame(rows).sort_values("score_total", ascending=False)

# ===== イベント除外 =====
def load_event_block(cfg: Config) -> Dict[str, dt.date]:
    block_until = {}
    if cfg.events_csv and os.path.exists(cfg.events_csv):
        try:
            df = pd.read_csv(cfg.events_csv)
            for _, r in df.iterrows():
                try:
                    until = dt.datetime.strptime(str(r["until"]), "%Y-%m-%d").date()
                    block_until[str(r["ticker"]).strip()] = until
                except Exception:
                    continue
        except Exception as e:
            print(f"[WARN] events_csv 読み込み失敗: {e}")
    return block_until

# ===== 候補選定（却下理由つき） =====
def decide_candidates(cfg: Config, score_df: pd.DataFrame, is_risk_on: bool, block_until: Dict[str, dt.date]):
    today = dt.date.today()
    if score_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    pool = cfg.risk_on_assets if is_risk_on else cfg.defensive_assets
    if not pool:
        pool = cfg.universe if is_risk_on else []

    filt = []
    for _, r in score_df.iterrows():
        t = r["ticker"]
        if t not in pool:
            continue
        reasons = []
        if r["turnover20"] < cfg.min_turnover_jpy:
            reasons.append(f"low_turnover<{cfg.min_turnover_jpy:.0f}")
        if t in block_until and today <= block_until[t]:
            reasons.append(f"event_block_until {block_until[t]}")
        if (r["close"] < r["ema20"]) or (r["close"] < r["ema50"]) or (r["close"] < r["sma100"]):
            reasons.append("below_MAs")
        filt.append((t, r, reasons))

    kept = [ (t, r, reasons) for (t, r, reasons) in filt if len(reasons) == 0 ]
    kept_sorted = sorted(kept, key=lambda x: x[1]["score_total"], reverse=True)
    picks = kept_sorted[: cfg.top_k]

    plan_rows = []
    for t, r, _ in picks:
        risk_budget = cfg.capital * cfg.risk_per_trade
        risk_per_share = cfg.atr_mult_stop * max(r["atr"], 1e-6)
        raw_shares = math.floor(risk_budget / risk_per_share)
        lot = cfg.lot_size_map.get(t, cfg.lot_size_default) if cfg.lot_size_default else 1
        shares = max((raw_shares // max(lot, 1)) * max(lot, 1), 0)
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
            "exit_plan": "1d_close",
            "score_total": float(r["score_total"]),
            "score_breakdown": f"T{r['score_trend']:.2f}|M{r['score_momo']:.2f}|V{r['score_volume']:.2f}|B{r['score_breakout']:.2f}|S{r['score_structure']:.2f}",
            "volr": float(r["volr"]),
            "pb": float(r["pb"]) if not np.isnan(r["pb"]) else np.nan,
            "don20": float(r["don20"]) if not np.isnan(r["don20"]) else np.nan,
            "obv_slope20": float(r["obv_slope20"]),
            "turnover20": float(r["turnover20"]),
        })

    # 却下上位（理由つき）
    rej_rows = []
    rejected_sorted = sorted([x for x in filt if len(x[2]) > 0],
                             key=lambda x: x[1]["score_total"], reverse=True)[:5]
    for t, r, reasons in rejected_sorted:
        rej_rows.append({
            "ticker": t,
            "score_total": float(r["score_total"]),
            "reasons": "|".join(reasons)
        })

    return pd.DataFrame(plan_rows), pd.DataFrame(rej_rows)

# ===== メール送信 =====
def send_email(cfg: Config, subject: str, body: str):
    if not cfg.email_enabled or not cfg.email_to:
        return
    user = os.environ.get("EMAIL_USER")
    pwd  = os.environ.get("EMAIL_PASS")
    if not user or not pwd:
        print("[WARN] EMAIL_USER / EMAIL_PASS が未設定のためメール送信をスキップ")
        return
    try:
        import smtplib
        from email.mime.text import MIMEText
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = user
        msg["To"] = cfg.email_to
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as smtp:
            smtp.login(user, pwd)
            smtp.sendmail(user, [cfg.email_to], msg.as_string())
        print(f"[INFO] メール送信完了: {cfg.email_to}")
    except Exception as e:
        print(f"[WARN] メール送信に失敗: {e}")

# ===== メイン =====
def main():
    p = argparse.ArgumentParser(description="Daily Trading Helper JP PLUS")
    p.add_argument("-c", "--config", default="config.jp.yml")
    args = p.parse_args()

    cfg = load_config(args.config)
    start = dt.date.today() - dt.timedelta(days=cfg.data_days)
    tickers = sorted(set([cfg.benchmark] + cfg.universe))
    print(f"[INFO] Downloading: {', '.join(tickers)}  since {start}")

    frames = download_history(tickers, start)
    if cfg.benchmark not in frames or frames[cfg.benchmark].empty:
        print("[ERROR] ベンチマークの価格系列を取得できませんでした。", file=sys.stderr)
        sys.exit(2)

    is_risk_on = risk_on_regime(frames[cfg.benchmark], cfg.regime_ma)
    score_df = build_scores(cfg, {t: frames[t] for t in cfg.universe if t in frames})

    os.makedirs(cfg.outdir, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    ranks_csv = os.path.join(cfg.outdir, f"ranks_{ts}.csv")
    score_df.to_csv(ranks_csv, index=False, encoding="utf-8")

    block_until = load_event_block(cfg)
    plan_df, rej_df = decide_candidates(cfg, score_df, is_risk_on, block_until)
    plan_csv = os.path.join(cfg.outdir, f"plan_{ts}.csv")
    plan_df.to_csv(plan_csv, index=False, encoding="utf-8")

    # 最新要約JSON
    latest_json = os.path.join(cfg.outdir, "plan_latest.json")
    summary = {
        "date": dt.date.today().isoformat(),
        "regime": "RISK_ON" if is_risk_on else "RISK_OFF",
        "top": plan_df.to_dict(orient="records") if not plan_df.empty else [],
    }
    with open(latest_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # メール本文（要約＋YAML）
    date_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M JST")
    body_lines = [
        f"Regime: {'RISK_ON' if is_risk_on else 'RISK_OFF'} (bench={cfg.benchmark})   Date: {date_str}",
        f"Capital: {cfg.capital:.0f}  Risk/Trade: {cfg.risk_per_trade*100:.1f}%  ATRx: {cfg.atr_mult_stop}  Lot: {cfg.lot_size_default}",
        f"Universe: {len(cfg.universe)} tickers",
        "",
        "[Plan]"
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
    body_lines.append(os.path.abspath(plan_csv))
    body_lines.append(os.path.abspath(ranks_csv))

    # YAMLブロック
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
    body = "\n".join(body_lines)

    send_email(cfg, cfg.email_subject, body)

    # コンソール要約
    print("\n=== SUMMARY ===")
    print(f"Regime: {'RISK_ON' if is_risk_on else 'RISK_OFF'} (benchmark: {cfg.benchmark})")
    if plan_df.empty:
        print("No picks for tomorrow based on current regime/rules.")
    else:
        print(plan_df.to_string(index=False))

    print(f"\nSaved:\n - {plan_csv}\n - {ranks_csv}\n - {latest_json}")
    print("\n※ 実売買前に、取引コスト・税制・流動性・決算/材料・約定方法等を必ずご確認ください。")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Investment Alarms (Long-term) — 10Y, VIX, USDJPY, SMH/XLK

想定するアラーム条件（デフォルト値）
 - 米10年金利（US10Y=^TNX/10）: 4.60% 上抜け(Hard) / 3.90% 割れ(Hard)
 - VIX: 22 超(Soft) / 28 超(Hard)
 - USD/JPY: 145.00 円割れ（円高加速）
 - SMH/XLK: 50日線割れ（終値ベースのデッドクロス）

出力:
 - reports_alarms/alarms_*.csv
 - reports_alarms/alarms_latest.json

メール送信（任意）: Gmail アプリパスワード（環境変数 EMAIL_USER / EMAIL_PASS）
"""

import os
import sys
import json
import argparse
import datetime as dt
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

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


# ==== Config ====
@dataclass
class AlarmConfig:
    data_days: int = 260
    outdir: str = "reports_alarms"

    # thresholds
    us10y_cross_up_hard: float = 4.60
    us10y_cross_down_hard: float = 3.90

    vix_cross_up_soft: float = 22.0
    vix_cross_up_hard: float = 28.0

    usdjpy_cross_down: float = 145.0

    ratio_short: str = "SMH"
    ratio_long: str = "XLK"
    ratio_sma_days: int = 50

    email_enabled: bool = False
    email_to: Optional[str] = None
    email_subject: str = "INVESTMENT ALARM"


def load_config(path: str) -> AlarmConfig:
    try:
        import yaml
    except Exception:
        print("PyYAML が必要です: pip install pyyaml", file=sys.stderr)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return AlarmConfig(**raw)


# ==== Data ====
def download_history(tickers: List[str], start: dt.date) -> Dict[str, pd.DataFrame]:
    try:
        import yfinance as yf
    except Exception:
        print("yfinance が必要です: pip install yfinance", file=sys.stderr)
        sys.exit(1)
    data: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            df = yf.download(t, start=start, auto_adjust=False, progress=False)
            if df is None or df.empty:
                print(f"[WARN] データなし: {t}")
                continue
            df = df.dropna(how="all")
            data[t] = df
        except Exception as e:
            print(f"[WARN] 取得失敗: {t}: {e}")
    return data


def _last_two(series: pd.Series):
    s = series.dropna()
    if len(s) < 2:
        return None, None
    return float(np.asarray(s.iloc[-2]).item()), float(np.asarray(s.iloc[-1]).item())


def _adj_close(df: pd.DataFrame) -> pd.Series:
    # ETF などは Adj Close が望ましいが、無ければ Close
    if "Adj Close" in df.columns:
        return df["Adj Close"]
    return df["Close"]


# ==== Alarms ====
def evaluate_alarms(cfg: AlarmConfig, frames: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    now_ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
    results: List[Dict[str, Any]] = []

    # 1) US10Y (^TNX) → percent = close/10
    if "^TNX" in frames:
        tnx = frames["^TNX"]["Close"]/10.0
        prev, curr = _last_two(tnx)
        if prev is not None:
            # cross up hard 4.60
            th = cfg.us10y_cross_up_hard
            cross_up_hard = (prev <= th) and (curr > th)
            results.append({
                "name": "US10Y_UP_HARD",
                "severity": "HARD",
                "desc": f"US10Y crosses above {th:.2f}%",
                "triggered": bool(cross_up_hard),
                "value_prev": round(prev, 4),
                "value": round(curr, 4),
                "threshold": th,
                "ts": now_ts,
            })
            # cross down hard 3.90
            th2 = cfg.us10y_cross_down_hard
            cross_down_hard = (prev >= th2) and (curr < th2)
            results.append({
                "name": "US10Y_DOWN_HARD",
                "severity": "HARD",
                "desc": f"US10Y crosses below {th2:.2f}%",
                "triggered": bool(cross_down_hard),
                "value_prev": round(prev, 4),
                "value": round(curr, 4),
                "threshold": th2,
                "ts": now_ts,
            })
    else:
        results.append({"name": "US10Y", "error": "^TNX data missing"})

    # 2) VIX (^VIX)
    if "^VIX" in frames:
        vix = frames["^VIX"]["Close"]
        prev, curr = _last_two(vix)
        if prev is not None:
            th_soft = cfg.vix_cross_up_soft
            th_hard = cfg.vix_cross_up_hard
            cross_up_soft = (prev <= th_soft) and (curr > th_soft)
            cross_up_hard = (prev <= th_hard) and (curr > th_hard)
            results.append({
                "name": "VIX_UP_SOFT",
                "severity": "SOFT",
                "desc": f"VIX crosses above {th_soft}",
                "triggered": bool(cross_up_soft),
                "value_prev": round(prev, 4),
                "value": round(curr, 4),
                "threshold": th_soft,
                "ts": now_ts,
            })
            results.append({
                "name": "VIX_UP_HARD",
                "severity": "HARD",
                "desc": f"VIX crosses above {th_hard}",
                "triggered": bool(cross_up_hard),
                "value_prev": round(prev, 4),
                "value": round(curr, 4),
                "threshold": th_hard,
                "ts": now_ts,
            })
    else:
        results.append({"name": "VIX", "error": "^VIX data missing"})

    # 3) USDJPY (USDJPY=X)
    if "USDJPY=X" in frames:
        jpy = frames["USDJPY=X"]["Close"]
        prev, curr = _last_two(jpy)
        if prev is not None:
            th = cfg.usdjpy_cross_down
            cross_down = (prev >= th) and (curr < th)
            results.append({
                "name": "USDJPY_DOWN",
                "severity": "INFO",
                "desc": f"USDJPY crosses below {th}",
                "triggered": bool(cross_down),
                "value_prev": round(prev, 4),
                "value": round(curr, 4),
                "threshold": th,
                "ts": now_ts,
            })
    else:
        results.append({"name": "USDJPY", "error": "USDJPY=X data missing"})

    # 4) SMH/XLK ratio vs SMA50
    sh = cfg.ratio_short; lg = cfg.ratio_long
    if (sh in frames) and (lg in frames):
        df_sh = _adj_close(frames[sh]).rename("short")
        df_lg = _adj_close(frames[lg]).rename("long")
        df = pd.concat([df_sh, df_lg], axis=1, join="inner").dropna()
        if not df.empty:
            ratio = (df["short"] / df["long"]).rename("ratio")
            sma = ratio.rolling(cfg.ratio_sma_days).mean()
            r_prev, r_curr = _last_two(ratio)
            s_prev, s_curr = _last_two(sma)
            if r_prev is not None and s_prev is not None and not (np.isnan(s_prev) or np.isnan(s_curr)):
                cross_down = (r_prev >= s_prev) and (r_curr < s_curr)
                results.append({
                    "name": f"{sh}/{lg}_SMA{cfg.ratio_sma_days}_DOWN",
                    "severity": "INFO",
                    "desc": f"{sh}/{lg} closes below SMA{cfg.ratio_sma_days}",
                    "triggered": bool(cross_down),
                    "value_prev": round(r_prev, 6),
                    "value": round(r_curr, 6),
                    "sma_prev": round(s_prev, 6),
                    "sma": round(s_curr, 6),
                    "ts": now_ts,
                })
    else:
        results.append({"name": "SMH/XLK", "error": f"{sh} or {lg} data missing"})

    # summary
    triggered = [r for r in results if r.get("triggered")]
    return {
        "date": dt.date.today().isoformat(),
        "triggers": triggered,
        "all": results,
    }


# ==== Output / Email ====
def send_email(cfg: AlarmConfig, subject: str, body: str):
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


def format_summary(res: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("[Investment Alarms]")
    if not res["triggers"]:
        lines.append("No new triggers.")
    else:
        for r in res["triggers"]:
            name = r.get("name", "")
            sev  = r.get("severity", "")
            desc = r.get("desc", "")
            val  = r.get("value")
            th   = r.get("threshold")
            lines.append(f"- {name} [{sev}] {desc} value={val}{'' if th is None else f' thr={th}'}")
    return "\n".join(lines)


# ==== Main ====
def main():
    ap = argparse.ArgumentParser(description="Long-term Investment Alarms")
    ap.add_argument("-c", "--config", default="config.alarms.yml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    start = dt.date.today() - dt.timedelta(days=cfg.data_days)

    tickers = ["^TNX", "^VIX", "USDJPY=X", cfg.ratio_short, cfg.ratio_long]
    print(f"[INFO] Downloading: {', '.join(tickers)} since {start}")
    frames = download_history(tickers, start)

    res = evaluate_alarms(cfg, frames)

    os.makedirs(cfg.outdir, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    latest_json = os.path.join(cfg.outdir, "alarms_latest.json")
    with open(latest_json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    # flat CSV of all results
    rows = []
    for r in res["all"]:
        base = {
            "name": r.get("name"),
            "severity": r.get("severity"),
            "desc": r.get("desc"),
            "triggered": r.get("triggered"),
            "value": r.get("value"),
            "value_prev": r.get("value_prev"),
            "threshold": r.get("threshold"),
            "sma": r.get("sma"),
            "sma_prev": r.get("sma_prev"),
            "ts": r.get("ts"),
            "error": r.get("error"),
        }
        rows.append(base)
    out_csv = os.path.join(cfg.outdir, f"alarms_{ts}.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")

    body = format_summary(res)
    send_email(cfg, cfg.email_subject, body)

    print("\n=== ALARMS SUMMARY ===")
    print(body)
    print(f"\nSaved:\n - {out_csv}\n - {latest_json}")


if __name__ == "__main__":
    main()

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
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

from .utils_data import download_with_cache
from .notify import send_email as _send_email
from .status_page import generate_status_html
import logging

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


# ==== Config ====
@dataclass
class AlarmConfig:
    data_days: int = 260
    outdir: str = "reports_alarms"
    cache_dir: str = "cache"
    cache_max_age_days: int = 2

    # thresholds
    us10y_cross_up_hard: float = 4.60
    us10y_cross_down_hard: float = 3.90

    vix_cross_up_soft: float = 22.0
    vix_cross_up_hard: float = 28.0

    usdjpy_cross_down: float = 145.0

    ratio_short: str = "SMH"
    ratio_long: str = "XLK"
    ratio_sma_days: int = 50

    # Additional ratio signals
    ratio_signals: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"short": "HYG", "long": "IEF", "sma_days": 50, "name": "HYG/IEF_SMA50_DOWN", "severity": "INFO"},
        {"short": "RSP", "long": "SPY", "sma_days": 50, "name": "RSP/SPY_SMA50_DOWN", "severity": "INFO"},
        {"short": "IWM", "long": "QQQ", "sma_days": 50, "name": "IWM/QQQ_SMA50_DOWN", "severity": "INFO"},
    ])

    # Baseline regime checks
    spy_sma200_down: bool = True
    topix_t1306_sma200_down: bool = False

    # Holidays / schedule
    holiday_skip_us: bool = True

    # Notifications (email only)
    email_enabled: bool = False
    email_to: Optional[str] = None
    email_subject: str = "INVESTMENT ALARM"

    # Stateful alarms
    notify_on_change_only: bool = True
    cooldown_hours: int = 12
    state_file: str = "reports_alarms/alarms_state.json"

    # Digest with daily plan
    digest_include_daily: bool = True
    daily_latest_json: str = "reports/plan_latest.json"


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
def download_history(tickers: List[str], start: dt.date, cache_dir: str, max_age_days: int) -> Dict[str, pd.DataFrame]:
    return download_with_cache(tickers, start=start, end=None, cache_dir=cache_dir, max_age_days=max_age_days)


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


def _as_col(x: pd.Series | pd.DataFrame, name: str) -> pd.DataFrame:
    """Robustly return a single-column DataFrame named `name` from Series/DataFrame.
    Handles environments where upstream returns DataFrame unexpectedly.
    """
    if isinstance(x, pd.Series):
        return x.to_frame(name)
    if isinstance(x, pd.DataFrame):
        # Prefer standard close columns if present
        for c in ("Adj Close", "Close", "close", "adjclose"):
            if c in x.columns:
                return x[[c]].rename(columns={c: name})
        # Single-column case
        if x.shape[1] == 1:
            return x.rename(columns={x.columns[0]: name})
        # Fallback to first column
        return x.iloc[:, [0]].rename(columns={x.columns[0]: name})
    # Last resort: coerce to Series
    s = pd.Series(x)
    return s.to_frame(name)


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
        df_sh = _as_col(_adj_close(frames[sh]), "short")
        df_lg = _as_col(_adj_close(frames[lg]), "long")
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

    # 5) Additional ratio signals
    for sig in cfg.ratio_signals:
        sh = sig.get("short"); lg = sig.get("long"); sma_days = int(sig.get("sma_days", 50))
        name = sig.get("name") or f"{sh}/{lg}_SMA{sma_days}_DOWN"
        sev = sig.get("severity", "INFO")
        if (sh in frames) and (lg in frames):
            df_sh = _as_col(_adj_close(frames[sh]), "short")
            df_lg = _as_col(_adj_close(frames[lg]), "long")
            df = pd.concat([df_sh, df_lg], axis=1, join="inner").dropna()
            if not df.empty:
                ratio = (df["short"] / df["long"]).rename("ratio")
                sma = ratio.rolling(sma_days).mean()
                r_prev, r_curr = _last_two(ratio)
                s_prev, s_curr = _last_two(sma)
                if r_prev is not None and s_prev is not None and not (np.isnan(s_prev) or np.isnan(s_curr)):
                    cross_down = (r_prev >= s_prev) and (r_curr < s_curr)
                    results.append({
                        "name": name,
                        "severity": sev,
                        "desc": f"{sh}/{lg} closes below SMA{sma_days}",
                        "triggered": bool(cross_down),
                        "value_prev": round(r_prev, 6),
                        "value": round(r_curr, 6),
                        "sma_prev": round(s_prev, 6),
                        "sma": round(s_curr, 6),
                        "ts": now_ts,
                    })
        else:
            results.append({"name": name, "error": f"{sh} or {lg} data missing"})

    # 6) Baseline regimes
    def _sma_down(name: str, sym: str, days: int = 200):
        if sym not in frames:
            results.append({"name": name, "error": f"{sym} data missing"})
            return
        close = _adj_close(frames[sym])
        sma = close.rolling(days).mean()
        c_prev, c_curr = _last_two(close)
        s_prev, s_curr = _last_two(sma)
        if c_prev is None or s_prev is None or np.isnan(s_prev) or np.isnan(s_curr):
            return
        cross_down = (c_prev >= s_prev) and (c_curr < s_curr)
        results.append({
            "name": name,
            "severity": "INFO",
            "desc": f"{sym} closes below SMA{days}",
            "triggered": bool(cross_down),
            "value_prev": round(c_prev, 6),
            "value": round(c_curr, 6),
            "sma_prev": round(s_prev, 6),
            "sma": round(s_curr, 6),
            "ts": now_ts,
        })

    if cfg.spy_sma200_down:
        _sma_down("SPY_SMA200_DOWN", "SPY", 200)
    if cfg.topix_t1306_sma200_down:
        _sma_down("1306.T_SMA200_DOWN", "1306.T", 200)

    # summary
    triggered = [r for r in results if r.get("triggered")]
    return {
        "date": dt.date.today().isoformat(),
        "triggers": triggered,
        "all": results,
    }


# ==== Output / Email ====
def send_all_notifications(cfg: AlarmConfig, subject: str, body: str, payload: Dict[str, Any]):
    _send_email(cfg.email_enabled, cfg.email_to, subject, body)


def should_send_email(cfg: AlarmConfig, changes: List[Dict[str, Any]]) -> bool:
    """Return True if we should send an email for alarms.
    Policy: 1) email feature enabled, 2) at least one change (threshold crossed).
    """
    return bool(cfg.email_enabled and changes)


def format_summary(res: Dict[str, Any], changes_only: Optional[List[Dict[str, Any]]] = None, daily_summary: Optional[str] = None) -> str:
    lines: List[str] = []
    lines.append("[Investment Alarms]")
    changes = changes_only if changes_only is not None else res.get("triggers", [])
    if not changes:
        lines.append("No state changes.")
    else:
        for r in changes:
            name = r.get("name", "")
            sev  = r.get("severity", "")
            desc = r.get("desc", "")
            val  = r.get("value")
            th   = r.get("threshold")
            lines.append(f"- {name} [{sev}] {desc} value={val}{'' if th is None else f' thr={th}'}")
    if daily_summary:
        lines.append("\n[Daily Plan]")
        lines.append(daily_summary)
    return "\n".join(lines)


def _is_us_trading_day() -> bool:
    # Use pandas_market_calendars if available, else Mon-Fri
    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar("XNYS")
        # NOTE: Use naive UTC date to avoid tz_localize TypeError on aware timestamps
        #   (cf. pd.Timestamp.today(tz="UTC").tz_localize(None) raises on aware ts)
        #   A stable approach is utcnow().normalize() which returns naive midnight UTC.
        today = pd.Timestamp.utcnow().normalize()
        sched = nyse.schedule(start_date=today, end_date=today)
        return not sched.empty
    except Exception:
        wd = dt.date.today().weekday()
        return wd < 5


def _load_state(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_state(path: str, state: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _compute_changes(state: Dict[str, Any], results: List[Dict[str, Any]], cooldown_hours: int) -> List[Dict[str, Any]]:
    now = dt.datetime.now()
    changes: List[Dict[str, Any]] = []
    for r in results:
        name = r.get("name")
        if not name or "triggered" not in r:
            continue
        prev = state.get(name, {}).get("triggered", False)
        last_ts = state.get(name, {}).get("last_notified")
        changed = (bool(r["triggered"]) != bool(prev))
        within_cooldown = False
        if last_ts:
            try:
                last_dt = dt.datetime.fromisoformat(last_ts)
                within_cooldown = (now - last_dt).total_seconds() < cooldown_hours * 3600
            except Exception:
                within_cooldown = False
        if changed or not within_cooldown:
            changes.append(r)
    return changes


# ==== Main ====
def main():
    # Logging 基本設定
    if not logging.getLogger().handlers:
        logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="[%(levelname)s] %(message)s")

    ap = argparse.ArgumentParser(description="Long-term Investment Alarms")
    ap.add_argument("-c", "--config", default="config.alarms.yml")
    ap.add_argument("--outdir", default=None, help="override output dir")
    ap.add_argument("--no-email", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.outdir:
        cfg.outdir = args.outdir

    # Holiday-aware skip
    if cfg.holiday_skip_us and not _is_us_trading_day():
        logger.info("Skipping (not a US trading day)")
        return

    start_all = dt.datetime.now()
    start = dt.date.today() - dt.timedelta(days=cfg.data_days)

    tickers = ["^TNX", "^VIX", "USDJPY=X", cfg.ratio_short, cfg.ratio_long]
    for sig in cfg.ratio_signals:
        tickers.extend([sig.get("short"), sig.get("long")])
    if cfg.spy_sma200_down:
        tickers.append("SPY")
    if cfg.topix_t1306_sma200_down:
        tickers.append("1306.T")
    tickers = [t for t in sorted(set(tickers)) if t]

    logger.info("Downloading: %s since %s", ", ".join(tickers), start)
    frames = download_history(tickers, start, cache_dir=cfg.cache_dir, max_age_days=cfg.cache_max_age_days)

    res = evaluate_alarms(cfg, frames)

    # Stateful changes
    state = _load_state(cfg.state_file)
    changes = _compute_changes(state, res["all"], cfg.cooldown_hours) if cfg.notify_on_change_only else res["triggers"]
    now_iso = dt.datetime.now().isoformat(timespec="seconds")
    for r in res["all"]:
        n = r.get("name"); trig = bool(r.get("triggered", False))
        if not n:
            continue
        st = state.get(n, {})
        st["triggered"] = trig
        if r in changes:
            st["last_notified"] = now_iso
        state[n] = st
    _save_state(cfg.state_file, state)

    write_outputs = not args.dry_run
    if write_outputs:
        os.makedirs(cfg.outdir, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    latest_json = os.path.join(cfg.outdir, "alarms_latest.json")
    if write_outputs:
        with open(latest_json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

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
    if write_outputs:
        pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")

    # Digest with daily plan
    daily_summary = None
    if cfg.digest_include_daily and os.path.exists(cfg.daily_latest_json):
        try:
            with open(cfg.daily_latest_json, "r", encoding="utf-8") as f:
                daily = json.load(f)
            picks = daily.get("top", [])
            if not picks:
                daily_summary = "No picks."
            else:
                lines = []
                for r in picks:
                    lines.append(f"- {r['ticker']} shares={r['shares']} close={r['close_ref']} score={r['score_total']}")
                daily_summary = "\n".join(lines)
        except Exception:
            pass

    body = format_summary(res, changes_only=changes, daily_summary=daily_summary)
    payload = {"changes": changes, "all": res.get("all", []), "date": res.get("date")}

    # Send only when there is at least one threshold-crossing change
    if not args.no_email and write_outputs and should_send_email(cfg, changes):
        send_all_notifications(cfg, cfg.email_subject, body, payload)
    else:
        logger.info("No new triggers. Email suppressed.")

    if write_outputs:
        try:
            generate_status_html(os.path.join(cfg.outdir, "status.html"), daily_json_path=cfg.daily_latest_json, alarms_json_path=latest_json)
        except Exception:
            pass
        try:
            runtime_sec = (dt.datetime.now() - start_all).total_seconds()
            metrics = [
                f"alarms_triggers_count {len([x for x in res.get('all', []) if x.get('triggered')])}",
                f"alarms_changes_count {len(changes)}",
                f"script_run_seconds{{script=\"alarms\"}} {runtime_sec:.3f}",
            ]
            with open(os.path.join(cfg.outdir, "metrics.prom"), "w", encoding="utf-8") as f:
                f.write("\n".join(metrics) + "\n")
        except Exception:
            pass

    logger.info("\n=== ALARMS SUMMARY ===\n%s", body)
    if write_outputs:
        logger.info("\nSaved:\n - %s\n - %s", out_csv, latest_json)


if __name__ == "__main__":
    main()

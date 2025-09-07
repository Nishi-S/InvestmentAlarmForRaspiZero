import os
import sys
import datetime as dt
import numpy as np
import pandas as pd

# Ensure repo root on path for direct module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from investment_alarms import evaluate_alarms, AlarmConfig, should_send_email


def df_from_series(values, start=dt.date(2024, 1, 1), col="Close"):
    idx = pd.bdate_range(start=start, periods=len(values))
    return pd.DataFrame({col: pd.Series(values, index=idx).astype(float)})


def test_evaluate_alarms_triggers_expected():
    # ^TNX in tenths of percent: 45.9 -> 46.1 means 4.59% -> 4.61% (cross up 4.60)
    tnx = df_from_series([45.9, 46.1])

    # VIX: cross up soft 22, not hard 28
    vix = df_from_series([21.0, 22.1])

    # USDJPY: cross down 145
    jpy = df_from_series([145.1, 144.9])

    # SMH/XLK ratio crosses below SMA50 at the end
    # Build 60 days ratio: 59 days at 1.0, last day 0.9
    ratio_days = 60
    ratio_prev_vals = [1.0] * (ratio_days - 1) + [0.9]
    long_price = 100.0
    xlk_prices = [long_price] * ratio_days
    smh_prices = [v * long_price for v in ratio_prev_vals]
    idx = pd.bdate_range(start=dt.date(2024, 1, 1), periods=ratio_days)
    smh_df = pd.DataFrame({"Adj Close": pd.Series(smh_prices, index=idx).astype(float)})
    xlk_df = pd.DataFrame({"Adj Close": pd.Series(xlk_prices, index=idx).astype(float)})

    frames = {
        "^TNX": tnx,
        "^VIX": vix,
        "USDJPY=X": jpy,
        "SMH": smh_df,
        "XLK": xlk_df,
    }

    cfg = AlarmConfig(
        data_days=120,
        us10y_cross_up_hard=4.60,
        us10y_cross_down_hard=3.90,
        vix_cross_up_soft=22.0,
        vix_cross_up_hard=28.0,
        usdjpy_cross_down=145.0,
        ratio_short="SMH",
        ratio_long="XLK",
        ratio_sma_days=50,
        email_enabled=False,
    )

    res = evaluate_alarms(cfg, frames)
    names = {r.get("name") for r in res["triggers"]}

    assert "US10Y_UP_HARD" in names
    assert "VIX_UP_SOFT" in names
    assert "USDJPY_DOWN" in names
    assert f"{cfg.ratio_short}/{cfg.ratio_long}_SMA{cfg.ratio_sma_days}_DOWN" in names

    # Ensure VIX hard is not triggered here
    assert "VIX_UP_HARD" not in names

    # Email policy: with triggers and email_enabled=True -> should send
    cfg2 = AlarmConfig(email_enabled=True)
    assert should_send_email(cfg2, res["triggers"]) is True


def test_no_triggers_suppresses_email():
    # All series flat -> no crosses
    flat = df_from_series([100.0, 100.0])
    frames = {
        "^TNX": df_from_series([46.0, 46.0]),  # 4.60% stays flat
        "^VIX": df_from_series([20.0, 20.0]),
        "USDJPY=X": df_from_series([150.0, 150.0]),
        "SMH": flat.rename(columns={"Close": "Adj Close"}),
        "XLK": flat.rename(columns={"Close": "Adj Close"}),
    }
    cfg = AlarmConfig(ratio_short="SMH", ratio_long="XLK", ratio_sma_days=2, email_enabled=True)
    res = evaluate_alarms(cfg, frames)
    assert res["triggers"] == []
    assert should_send_email(cfg, res["triggers"]) is False

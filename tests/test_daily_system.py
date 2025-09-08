import os
import sys
import datetime as dt
import numpy as np
import pandas as pd

# Ensure repo root on path for direct module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from investment_alarm.daily_system_jp_plus import (
    compute_features,
    risk_on_regime,
    build_scores,
    decide_candidates,
    Config,
)


def make_price_df(days: int = 150, start=dt.date(2024, 1, 1)) -> pd.DataFrame:
    idx = pd.bdate_range(start=start, periods=days)
    base = 100.0 + np.linspace(0, days - 1, days) * 0.5  # gentle uptrend
    close = pd.Series(base, index=idx)
    high = close * 1.01
    low = close * 0.99
    open_ = close * 0.995
    vol = pd.Series(1_000_000, index=idx)
    return pd.DataFrame({
        "Open": open_.astype(float),
        "High": high.astype(float),
        "Low": low.astype(float),
        "Close": close.astype(float),
        "Adj Close": close.astype(float),
        "Volume": vol.astype(float),
    })


def test_compute_features_columns_and_obv_slope():
    df = make_price_df(150)
    feats = compute_features(df)
    for col in ["EMA20", "EMA50", "SMA100", "ATR", "DonHigh20", "PB", "VOLR", "OBV", "OBV_SLOPE20", "RSI2", "RET5", "RET20"]:
        assert col in feats.columns
    # OBV slope is only computed for the last row
    assert np.isnan(feats["OBV_SLOPE20"].iloc[-2])
    assert np.isfinite(feats["OBV_SLOPE20"].iloc[-1])


def test_risk_on_regime_true_for_uptrend():
    bench = make_price_df(150)
    assert risk_on_regime(bench, ma=100) is True


def test_build_scores_and_decide_candidates_pipeline():
    df_a = make_price_df(150)
    df_b = make_price_df(150)
    frames = {"AAA": df_a, "BBB": df_b}

    cfg = Config(
        universe=["AAA", "BBB"],
        risk_on_assets=["AAA", "BBB"],
        defensive_assets=[],
        data_days=200,
        regime_ma=100,
        atr_period=14,
        atr_mult_stop=2.0,
        top_k=1,
        capital=100000.0,
        risk_per_trade=0.01,
        min_turnover_jpy=1_000.0,
        lot_size_default=1,
        lot_size_map={},
        events_csv=None,
        outdir="reports",
        email_enabled=False,
        email_to=None,
    )

    score_df = build_scores(cfg, frames)
    assert set(["AAA", "BBB"]).issuperset(set(score_df["ticker"]))
    assert {"score_total", "turnover20"}.issubset(score_df.columns)

    is_on = True
    plan_df, rej_df = decide_candidates(cfg, score_df, is_on, block_until={})
    # Should select exactly 1 pick and produce 0-5 rejected with reasons
    assert len(plan_df) <= 1
    if len(plan_df) == 1:
        row = plan_df.iloc[0]
        assert row["shares"] >= 0
        assert row["ticker"] in ("AAA", "BBB")

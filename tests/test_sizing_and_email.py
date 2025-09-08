import os
import sys
import math
import datetime as dt
import numpy as np
import pandas as pd

# Ensure repo root on path for direct module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from investment_alarm.daily_system_jp_plus import (
    build_scores,
    decide_candidates,
    Config,
    format_daily_email,
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


def _single_ticker_cfg(**overrides) -> Config:
    base = dict(
        universe=["AAA"],
        risk_on_assets=["AAA"],
        defensive_assets=[],
        data_days=200,
        regime_ma=100,
        atr_period=14,
        atr_mult_stop=2.0,
        top_k=1,
        capital=100000.0,
        risk_per_trade=0.01,
        min_turnover_jpy=0.0,  # フィルタ無効化
        lot_size_default=1,
        lot_size_map={},
        events_csv=None,
        outdir="reports",
        email_enabled=False,
        email_to=None,
    )
    base.update(overrides)
    return Config(**base)


def _prepare_scores():
    df = make_price_df(150)
    frames = {"AAA": df}
    return frames


def test_min_notional_adjustment():
    frames = _prepare_scores()
    # 小さい資金で raw_shares を少なめにし、最低約定額で引き上げられることを確認
    cfg = _single_ticker_cfg(capital=1000.0, risk_per_trade=0.01, min_notional_jpy=1000.0)
    score_df = build_scores(cfg, frames)
    plan_df, _ = decide_candidates(cfg, score_df, is_risk_on=True, block_until={})
    if plan_df.empty:
        # 上記条件で銘柄がフィルタされることは想定しない
        assert False, "plan_df should not be empty"
    row = plan_df.iloc[0]
    close = float(row["close_ref"])
    shares = int(row["shares"])
    assert shares * close >= cfg.min_notional_jpy - 1e-6


def test_lot_size_enforcement():
    frames = _prepare_scores()
    # 100株単位に丸め込まれることを確認
    cfg = _single_ticker_cfg(capital=100000.0, risk_per_trade=0.01, lot_size_default=100, min_notional_jpy=0.0)
    score_df = build_scores(cfg, frames)
    plan_df, _ = decide_candidates(cfg, score_df, is_risk_on=True, block_until={})
    assert not plan_df.empty
    row = plan_df.iloc[0]
    # 期待値: floor(raw_shares/100)*100
    atr = float(row["atr"])  # decide_candidatesが出力した計算値を利用
    risk_budget = cfg.capital * cfg.risk_per_trade
    risk_per_share = cfg.atr_mult_stop * max(atr, 1e-6)
    raw = math.floor(risk_budget / risk_per_share)
    expected = (raw // 100) * 100
    assert int(row["shares"]) == expected
    assert int(row["shares"]) % 100 == 0


def test_notional_cap_enforcement():
    frames = _prepare_scores()
    # 銘柄ごとの金額上限で shares が抑制されることを確認
    cfg = _single_ticker_cfg(capital=200000.0, risk_per_trade=0.05, notional_cap_per_ticker_jpy=1000.0, lot_size_default=1)
    score_df = build_scores(cfg, frames)
    plan_df, _ = decide_candidates(cfg, score_df, is_risk_on=True, block_until={})
    assert not plan_df.empty
    row = plan_df.iloc[0]
    close = float(row["close_ref"])
    cap_shares = int(math.floor(cfg.notional_cap_per_ticker_jpy / max(close, 1e-6)))
    assert int(row["shares"]) <= cap_shares


def test_format_daily_email_contains_sections():
    frames = _prepare_scores()
    cfg = _single_ticker_cfg()
    score_df = build_scores(cfg, frames)
    plan_df, rej_df = decide_candidates(cfg, score_df, is_risk_on=True, block_until={})
    # 仮のパス
    plan_csv = "/tmp/plan.csv"
    ranks_csv = "/tmp/ranks.csv"
    body = format_daily_email(cfg, True, plan_df, rej_df, plan_csv, ranks_csv)
    # セクションとYAMLブロック、ティッカーが含まれること
    assert "Regime: RISK_ON" in body
    assert "```yaml" in body and "jpplan:" in body
    if not plan_df.empty:
        assert plan_df.iloc[0]["ticker"] in body
    assert os.path.abspath(plan_csv) in body and os.path.abspath(ranks_csv) in body

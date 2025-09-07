import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from investment_alarms import _is_us_trading_day


def test_is_us_trading_day_returns_bool_without_exception():
    # 返り値がブールであることだけを確認（実日の市場開場状況に依存しない）
    assert isinstance(_is_us_trading_day(), bool)


"""Trading days and the point in time membership of the stocks."""
from __future__ import annotations

import pandas as pd


def trading_days(close: pd.DataFrame, stale_share: float = 0.5) -> pd.Series:
    """Flags the days that are trading days rather than holiday fills.

    A day is a holiday fill when more than stale_share of the stocks with
    a close show the same close as on the previous row.

    Args:
        close: Closing prices, dates by stocks.
        stale_share: Largest share of unchanged closes on a trading day.

    Returns:
        Boolean Series on the index of close, True on trading days.
    """
    same = (close == close.shift(1)) & close.notna()
    stale = (same.sum(axis=1)
             / close.notna().sum(axis=1).where(lambda s: s > 0)) > stale_share
    return ~stale


def membership(close: pd.DataFrame, volume: pd.DataFrame,
               market_cap: pd.DataFrame, window: int = 1260,
               cover: float = 0.95) -> pd.DataFrame:
    """Point in time membership of each stock on each day.

    A stock is a member from the first day on which the trailing window
    has a close, a volume and a market value on at least the cover share
    of its days, the day itself included, while the stock has a close.

    Args:
        close: Closing prices, dates by stocks.
        volume: Shares traded, same shape.
        market_cap: Market value, same shape.
        window: Length of the trailing window in trading days.
        cover: Smallest share of days with all three fields present.

    Returns:
        Boolean frame of the members, dates by stocks.
    """
    avail = (close.notna() & volume.notna() & market_cap.notna()).astype(float)
    ok = avail.rolling(window, min_periods=window).mean() >= cover
    return ok & close.notna()

"""Ranks of the signals across stocks and along each stock's own history."""
from __future__ import annotations

from typing import Dict, Optional

import pandas as pd


def cross_stock_ranks(signals: Dict[str, pd.DataFrame],
                      members: Optional[pd.DataFrame] = None
                      ) -> Dict[str, pd.DataFrame]:
    """Percentile rank of each signal across the stocks of the day.

    Args:
        signals: Dict of signal name to frame, dates by stocks.
        members: Boolean frame of the members of each day, or None when
          every stock with a value is ranked. A non member gets NaN.

    Returns:
        Dict of signal name plus "_cs" to frame of ranks in (0, 1], ties
        at their average rank.
    """
    if members is None:
        return {k + "_cs": v.rank(axis=1, pct=True)
                for k, v in signals.items()}
    return {k + "_cs": v.where(members).rank(axis=1, pct=True)
            for k, v in signals.items()}


def own_history_percentiles(signals: Dict[str, pd.DataFrame],
                            window: int = 252, min_periods: int = 200
                            ) -> Dict[str, pd.DataFrame]:
    """Percentile of each value within the stock's own trailing window.

    Args:
        signals: Dict of signal name to frame, dates by stocks.
        window: Length of the trailing window in trading days.
        min_periods: Smallest number of values for a percentile.

    Returns:
        Dict of signal name plus "_ts" to frame of percentiles in (0, 1].
    """
    return {k + "_ts": v.rolling(window, min_periods=min_periods).rank(pct=True)
            for k, v in signals.items()}


def availability_flags(signals: Dict[str, pd.DataFrame]
                       ) -> Dict[str, pd.DataFrame]:
    """Flags of the two signals that are often missing.

    Args:
        signals: Dict of signal name to frame, with the keys "rating" and
          "ltrev_1260_252".

    Returns:
        Dict with "rating_av" and "ltrev_av", frames of 1.0 where the
        signal has a value and 0.0 where it has none.
    """
    return {"rating_av": signals["rating"].notna().astype(float),
            "ltrev_av": signals["ltrev_1260_252"].notna().astype(float)}

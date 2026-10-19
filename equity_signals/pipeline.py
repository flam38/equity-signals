"""From daily fields to signals, ranks and forward returns in one call."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from .fields import Fields
from .labels import forward_returns
from .ranks import availability_flags, cross_stock_ranks
from .ranks import own_history_percentiles
from .signals import raw_signals
from .universe import membership, trading_days


@dataclass
class SignalSet:
    """Signals and labels of a set of stocks over a window of days.

    Attributes:
        raw: Dict of signal name to frame of raw values.
        ranks: Dict of signal name plus "_cs" to frame of cross stock
          percentile ranks among the members of the day.
        history: Dict of signal name plus "_ts" to frame of percentiles
          within the stock's own trailing window.
        availability: Dict of flag name to frame of 0.0 or 1.0.
        forward: Frame of forward simple returns, NaN for non members.
        close: Closing prices on the trading days of the window.
        market_cap: Market values on the trading days of the window.
        members: Boolean frame of the members, or None without the entry
          rule.
    """

    raw: Dict[str, pd.DataFrame]
    ranks: Dict[str, pd.DataFrame]
    history: Dict[str, pd.DataFrame]
    availability: Dict[str, pd.DataFrame]
    forward: pd.DataFrame
    close: pd.DataFrame
    market_cap: pd.DataFrame
    members: Optional[pd.DataFrame]


def build_signals(fields: Fields, start: str, last_day: str,
                  horizon: int = 21, lag: int = 1, stale_share: float = 0.5,
                  entry_rule: bool = True, window: int = 1260,
                  cover: float = 0.95) -> SignalSet:
    """Builds the signals, their ranks and the forward returns.

    The fields are aligned, the holiday fills are dropped, the membership
    is computed on the trading days, the signals use the whole history
    before the window, and every output is cut to the window.

    Args:
        fields: Daily fields over the whole history.
        start: First day of the window, as "YYYY-MM-DD".
        last_day: Last day of the window, as "YYYY-MM-DD".
        horizon: Holding period of the forward return in trading days.
        lag: Days between the signal and the first close held.
        stale_share: Largest share of unchanged closes on a trading day.
        entry_rule: Whether to compute the membership and rank among the
          members only.
        window: Trailing window of the entry rule in trading days.
        cover: Smallest share of days with all fields for the entry rule.

    Returns:
        SignalSet on the trading days from start to last_day.
    """
    aligned = fields.aligned()
    frames = aligned.rows(trading_days(aligned.close, stale_share))
    members = None
    if entry_rule:
        members = membership(frames.close, frames.volume, frames.market_cap,
                             window, cover)
    raw = raw_signals(frames, members)
    history = {k: v.loc[start:last_day]
               for k, v in own_history_percentiles(raw).items()}
    availability = {k: v.loc[start:last_day]
                    for k, v in availability_flags(raw).items()}
    raw = {k: v.loc[start:last_day] for k, v in raw.items()}
    members_window = None if members is None else members.loc[start:last_day]
    ranks = cross_stock_ranks(raw, members_window)
    forward = forward_returns(frames.close, horizon, lag, members)
    return SignalSet(raw=raw, ranks=ranks, history=history,
                     availability=availability,
                     forward=forward.loc[start:last_day],
                     close=frames.close.loc[start:last_day],
                     market_cap=frames.market_cap.loc[start:last_day],
                     members=members_window)

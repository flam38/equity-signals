"""Forward returns and the cross sectional residual on beta."""
from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd


def forward_returns(close: pd.DataFrame, horizon: int = 21, lag: int = 1,
                    members: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Simple return from the close lag days ahead to lag plus horizon.

    Args:
        close: Closing prices, dates by stocks.
        horizon: Holding period in trading days.
        lag: Days between the signal and the first close held.
        members: Boolean frame of the members, or None. A non member
          gets NaN.

    Returns:
        Frame of forward returns, NaN where a later close is missing.
    """
    fwd = close.shift(-(lag + horizon)) / close.shift(-lag) - 1
    return fwd if members is None else fwd.where(members)


def index_beta(stock_log_returns: pd.DataFrame,
               index_log_returns: pd.Series, window: int = 252,
               min_periods: int = 200) -> pd.DataFrame:
    """Rolling beta of each stock to an index.

    Args:
        stock_log_returns: Daily log returns of the stocks, dates by
          stocks.
        index_log_returns: Daily log returns of the index on the same
          dates.
        window: Length of the trailing window in trading days.
        min_periods: Smallest number of days for an estimate.

    Returns:
        Frame of betas known at the close of each day.
    """
    return stock_log_returns.rolling(window, min_periods=min_periods).cov(
        index_log_returns).div(
        index_log_returns.rolling(window, min_periods=min_periods).var(),
        axis=0)


def beta_residual(y: np.ndarray, beta: np.ndarray, min_stocks: int = 20,
                  return_slope: bool = False
                  ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Removes the cross sectional fit on beta from each day's values.

    For each day, the fitted value is the mean of y plus the slope times
    the deviation of beta from its mean, over the stocks with a finite
    y. A missing beta takes the day's mean.

    Args:
        y: Array [days, stocks] of values, NaN where missing.
        beta: Array [days, stocks] of betas.
        min_stocks: Smallest number of stocks for a day to count.
        return_slope: Whether to return the slope of each day as well.

    Returns:
        Array [days, stocks] of residuals, NaN for a missing value and on
        days with fewer than min_stocks stocks; with return_slope also
        the array [days] of slopes.
    """
    y = np.asarray(y, dtype=float)
    beta = np.asarray(beta, dtype=float)
    with np.errstate(all="ignore"):
        b_mean = np.nanmean(np.where(np.isfinite(beta) & np.isfinite(y),
                                     beta, np.nan), axis=1, keepdims=True)
    bf = np.where(np.isfinite(beta), beta, b_mean)
    m = np.isfinite(y) & np.isfinite(bf)
    n = m.sum(axis=1, keepdims=True)
    nn_ = np.maximum(n, 1)
    y_bar = np.where(m, y, 0.0).sum(axis=1, keepdims=True) / nn_
    b_bar = np.where(m, bf, 0.0).sum(axis=1, keepdims=True) / nn_
    dy = np.where(m, y - y_bar, 0.0)
    db = np.where(m, bf - b_bar, 0.0)
    slope = (dy * db).sum(axis=1, keepdims=True) / np.maximum(
        (db * db).sum(axis=1, keepdims=True), 1e-12)
    res = np.where(m, dy - slope * db, np.nan)
    res[(n < min_stocks).ravel()] = np.nan
    return (res, slope.ravel()) if return_slope else res

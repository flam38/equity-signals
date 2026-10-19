"""The thirty signals from daily prices, volume, market value and ratings."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .fields import Fields

SIGNALS = (
    ("ret_5", "reversal", "Log return over the last 5 trading days."),
    ("ret_21", "reversal", "Log return over the last 21 trading days."),
    ("resid_ret_21", "reversal",
     "21 day log return minus the 252 day beta times the market's 21 day "
     "log return."),
    ("ltrev_1260_252", "reversal",
     "Log return from 1,260 to 252 trading days ago."),
    ("mom_63_21", "momentum", "Log return from 63 to 21 trading days ago."),
    ("mom_126_21", "momentum", "Log return from 126 to 21 trading days ago."),
    ("mom_252_21", "momentum", "Log return from 252 to 21 trading days ago."),
    ("mom_252_21_vs", "momentum",
     "mom_252_21 over the 252 day volatility times the square root of 231."),
    ("high52", "momentum",
     "Log close minus the highest log close of the last 252 days."),
    ("resid_mom_252_21", "momentum",
     "mom_252_21 minus the 252 day beta times the market's return over the "
     "same window."),
    ("tstat_mom_252_21", "momentum",
     "resid_mom_252_21 over the idiosyncratic volatility times the square "
     "root of 231."),
    ("ups_252_21", "momentum",
     "Share of up days over the 231 days ending 21 days ago."),
    ("smooth_mom", "momentum",
     "mom_252_21 times one plus its sign times (2 ups_252_21 minus 1)."),
    ("vol_21", "volatility",
     "Standard deviation of daily log returns over 21 days."),
    ("vol_63", "volatility",
     "Standard deviation of daily log returns over 63 days."),
    ("vol_252", "volatility",
     "Standard deviation of daily log returns over 252 days."),
    ("downvol_63", "volatility",
     "Standard deviation over 63 days of the daily log returns with the "
     "positive ones set to zero."),
    ("ivol_252", "volatility",
     "Square root of the 252 day variance net of beta squared times the "
     "market variance."),
    ("beta_63", "beta",
     "63 day covariance with the market over the 63 day market variance."),
    ("beta_252", "beta",
     "252 day covariance with the market over the 252 day market variance."),
    ("maxret_21", "volatility",
     "Largest daily log return of the last 21 days."),
    ("rsi_14", "technical",
     "Relative strength index with 14 day exponential averages of gains "
     "and losses."),
    ("park_21", "volatility", "Parkinson range volatility over 21 days."),
    ("logcap", "size and liquidity", "Log of the market value."),
    ("dvol_21", "size and liquidity",
     "Log of the 21 day average traded value, volume times close."),
    ("volratio_21_63", "size and liquidity",
     "21 day average volume over the 63 day average volume."),
    ("amihud_63", "size and liquidity",
     "63 day mean of the absolute log return over the traded value, times "
     "one million."),
    ("rating", "analyst", "Consensus analyst rating."),
    ("rating_chg_21", "analyst", "Change of the rating over 21 days."),
    ("rating_chg_63", "analyst", "Change of the rating over 63 days."),
)
SIGNAL_NAMES = tuple(name for name, _, _ in SIGNALS)


def raw_signals(fields: Fields,
                members: Optional[pd.DataFrame] = None
                ) -> Dict[str, pd.DataFrame]:
    """Computes the thirty raw signals on trading days.

    Args:
        fields: Aligned fields restricted to trading days.
        members: Boolean frame of the members of each day, or None when
          every stock with data counts. The market return of a day is
          the mean log return over the members.

    Returns:
        Dict of signal name to frame, dates by stocks. Infinite values
        are NaN.
    """
    close, high, low = fields.close, fields.high, fields.low
    volume, market_cap, rating = fields.volume, fields.market_cap, fields.rating
    log_close = np.log(close)
    returns = log_close.diff()
    if members is None:
        market = returns.mean(axis=1)
    else:
        market = returns.where(members).mean(axis=1)
    market_cum = market.cumsum()

    def past_return(a: int, b: int) -> pd.DataFrame:
        """Log return of each stock from a days ago to b days ago.

        Args:
            a: Start of the window, in trading days before the day.
            b: End of the window, in trading days before the day.

        Returns:
            Frame of log returns, dates by stocks.
        """
        return log_close.shift(b) - log_close.shift(a)

    def market_past_return(a: int, b: int) -> pd.Series:
        """Log return of the market from a days ago to b days ago.

        Args:
            a: Start of the window, in trading days before the day.
            b: End of the window, in trading days before the day.

        Returns:
            Series of market log returns by date.
        """
        return market_cum.shift(b) - market_cum.shift(a)

    def rolling_vol(w: int) -> pd.DataFrame:
        """Standard deviation of the daily log returns over w days.

        Args:
            w: Length of the window in trading days; 80 percent of the
              days are needed for a value.

        Returns:
            Frame of volatilities, dates by stocks.
        """
        return returns.rolling(w, min_periods=int(0.8 * w)).std()

    def rolling_beta(w: int) -> pd.DataFrame:
        """Covariance with the market over the market variance, w days.

        Args:
            w: Length of the window in trading days; 80 percent of the
              days are needed for a value.

        Returns:
            Frame of betas, dates by stocks.
        """
        return returns.rolling(w, min_periods=int(0.8 * w)).cov(market).div(
            market.rolling(w, min_periods=int(0.8 * w)).var(), axis=0)

    beta_252, beta_63, vol_252 = (rolling_beta(252), rolling_beta(63),
                                  rolling_vol(252))
    market_var_252 = market.rolling(252, min_periods=200).var()
    idio_vol = np.sqrt((vol_252 ** 2
                        - (beta_252 ** 2).mul(market_var_252, axis=0))
                       .clip(lower=0))
    residual_momentum = (past_return(252, 21)
                         - beta_252.mul(market_past_return(252, 21), axis=0))
    up_share = ((returns > 0).astype(float)).where(returns.notna()).rolling(
        231, min_periods=180).mean().shift(21)
    price_change = close.diff()
    avg_gain = price_change.clip(lower=0).ewm(alpha=1 / 14, adjust=False,
                                              min_periods=14).mean()
    avg_loss = (-price_change).clip(lower=0).ewm(alpha=1 / 14, adjust=False,
                                                 min_periods=14).mean()
    dollar_volume = (volume * close).where(volume > 0)
    signals = {
        "ret_5": past_return(5, 0),
        "ret_21": past_return(21, 0),
        "resid_ret_21": (past_return(21, 0)
                         - beta_252.mul(market_past_return(21, 0), axis=0)),
        "ltrev_1260_252": past_return(1260, 252),
        "mom_63_21": past_return(63, 21),
        "mom_126_21": past_return(126, 21),
        "mom_252_21": past_return(252, 21),
        "mom_252_21_vs": past_return(252, 21) / (vol_252 * np.sqrt(231)),
        "high52": log_close - log_close.rolling(252, min_periods=200).max(),
        "resid_mom_252_21": residual_momentum,
        "tstat_mom_252_21": residual_momentum / (idio_vol * np.sqrt(231)),
        "ups_252_21": up_share,
        "smooth_mom": (past_return(252, 21)
                       * (1 + np.sign(past_return(252, 21))
                          * (2 * up_share - 1))),
        "vol_21": rolling_vol(21),
        "vol_63": rolling_vol(63),
        "vol_252": vol_252,
        "downvol_63": returns.clip(upper=0).rolling(63, min_periods=50).std(),
        "ivol_252": idio_vol,
        "beta_63": beta_63,
        "beta_252": beta_252,
        "maxret_21": returns.rolling(21, min_periods=17).max(),
        "rsi_14": 100 - 100 / (1 + avg_gain / avg_loss),
        "park_21": np.sqrt(((np.log(high / low) ** 2) / (4 * np.log(2)))
                           .rolling(21, min_periods=17).mean()),
        "logcap": np.log(market_cap.where(market_cap > 0)),
        "dvol_21": np.log(dollar_volume.rolling(21, min_periods=17).mean()),
        "volratio_21_63": (volume.rolling(21, min_periods=17).mean()
                           / volume.rolling(63, min_periods=50).mean()),
        "amihud_63": ((returns.abs() / dollar_volume)
                      .rolling(63, min_periods=50).mean() * 1e6),
        "rating": rating,
        "rating_chg_21": rating - rating.shift(21),
        "rating_chg_63": rating - rating.shift(63),
    }
    return {k: v.replace([np.inf, -np.inf], np.nan)
            for k, v in signals.items()}

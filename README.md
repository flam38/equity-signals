# equity-signals

Thirty cross sectional equity signals computed from daily prices, volume, market value and analyst ratings, with their percentile ranks among the stocks of each day, a point in time membership rule, and the forward returns and residual labels used to evaluate them. Code only: no data is included.

## What it computes

Given daily fields for a set of stocks (close, high, low, volume, market value and, optionally, a consensus analyst rating), `build_signals` returns, for a window of days:

- the 30 raw signals listed below;
- their cross stock percentile ranks among the members of the day (`_cs`), ties at their average rank, NaN for a non member;
- the percentile of each value within the stock's own trailing 252 days (`_ts`);
- two availability flags (`rating_av`, `ltrev_av`);
- the forward simple return from the close one day ahead to the close 22 days ahead (a 21 day holding period after a one day gap), NaN for non members;
- the closes, market values and membership on the same days.

Conventions:

- Holiday fills are dropped first: a day on which more than half of the stocks with a close repeat the previous close is not a trading day.
- A stock becomes a member on the first day on which the trailing 1,260 trading days have a close, a volume and a market value on at least 95 percent of their days, and stays one while it has a close.
- The market return of a day is the mean daily log return over the members of that day; betas are rolling covariances with it over its rolling variance.
- Rolling windows require 80 percent of their days for volatility and beta, 200 of 252 for the one year windows, 17 of 21 and 50 of 63 for the short windows.
- Infinite values become NaN. A missing rank is left as NaN; a consumer that needs a filled value decides the fill.

## The signals

| Name | Family | Definition |
|---|---|---|
| `ret_5` | reversal | Log return over the last 5 trading days. |
| `ret_21` | reversal | Log return over the last 21 trading days. |
| `resid_ret_21` | reversal | 21 day log return minus the 252 day beta times the market's 21 day log return. |
| `ltrev_1260_252` | reversal | Log return from 1,260 to 252 trading days ago. |
| `mom_63_21` | momentum | Log return from 63 to 21 trading days ago. |
| `mom_126_21` | momentum | Log return from 126 to 21 trading days ago. |
| `mom_252_21` | momentum | Log return from 252 to 21 trading days ago. |
| `mom_252_21_vs` | momentum | `mom_252_21` over the 252 day volatility times the square root of 231. |
| `high52` | momentum | Log close minus the highest log close of the last 252 days. |
| `resid_mom_252_21` | momentum | `mom_252_21` minus the 252 day beta times the market's return over the same window. |
| `tstat_mom_252_21` | momentum | `resid_mom_252_21` over the idiosyncratic volatility times the square root of 231. |
| `ups_252_21` | momentum | Share of up days over the 231 days ending 21 days ago. |
| `smooth_mom` | momentum | `mom_252_21` times one plus its sign times (2 `ups_252_21` minus 1). |
| `vol_21` | volatility | Standard deviation of daily log returns over 21 days. |
| `vol_63` | volatility | Standard deviation of daily log returns over 63 days. |
| `vol_252` | volatility | Standard deviation of daily log returns over 252 days. |
| `downvol_63` | volatility | Standard deviation over 63 days of the daily log returns with the positive ones set to zero. |
| `ivol_252` | volatility | Square root of the 252 day variance net of beta squared times the market variance. |
| `beta_63` | beta | 63 day covariance with the market over the 63 day market variance. |
| `beta_252` | beta | 252 day covariance with the market over the 252 day market variance. |
| `maxret_21` | volatility | Largest daily log return of the last 21 days. |
| `rsi_14` | technical | Relative strength index with 14 day exponential averages of gains and losses. |
| `park_21` | volatility | Parkinson range volatility over 21 days. |
| `logcap` | size and liquidity | Log of the market value. |
| `dvol_21` | size and liquidity | Log of the 21 day average traded value, volume times close. |
| `volratio_21_63` | size and liquidity | 21 day average volume over the 63 day average volume. |
| `amihud_63` | size and liquidity | 63 day mean of the absolute log return over the traded value, times one million. |
| `rating` | analyst | Consensus analyst rating. |
| `rating_chg_21` | analyst | Change of the rating over 21 days. |
| `rating_chg_63` | analyst | Change of the rating over 63 days. |

The same table is available in code as `equity_signals.SIGNALS`.

## Usage

```python
import pandas as pd
from equity_signals import Fields, build_signals

fields = Fields(close=close, high=high, low=low, volume=volume,
                market_cap=market_cap, rating=rating)   # DataFrames, dates by tickers
out = build_signals(fields, start="2007-01-01", last_day="2020-09-30")

out.ranks["mom_252_21_cs"]     # percentile ranks among the members of the day
out.forward                    # 21 day forward returns after a one day gap
out.members                    # point in time membership
```

Every frame is indexed by date with one column per stock. The whole history should be passed in, because the signals need up to 1,260 trading days before the first day of the window.

The residual label of a cross sectional study, the forward return net of its daily fit on beta, is `labels.beta_residual(forward.to_numpy(), beta.to_numpy())`, with `labels.index_beta` giving the rolling beta of each stock to an index from daily log returns.

## Readers

No vendor reader is included. A reader maps the vendor's daily fields to `Fields`: close, high, low, volume, market value (price times shares outstanding when the vendor gives shares), and the rating when a dated consensus is available. Prices spliced from several pulls should be checked for breaks before use.

## Tests

```bash
pip install -e ".[test]"
pytest -q
```

The tests run on made up data and check the names and shapes, the absence of look ahead (signals up to a day do not change when later rows change), known values, the holiday fill rule, the entry rule timing, the rank conventions, the forward return definition and the residual on beta.

## License

No license has been chosen yet; all rights reserved by the owner until one is added.

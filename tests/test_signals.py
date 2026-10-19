"""Tests of the signal functions on made up data."""
import numpy as np
import pandas as pd

from equity_signals import Fields, SIGNAL_NAMES, build_signals, raw_signals
from equity_signals.labels import beta_residual, forward_returns
from equity_signals.ranks import cross_stock_ranks
from equity_signals.universe import membership, trading_days


def make_fields(days=1500, stocks=6, seed=0):
    """Random walk prices with high, low, volume, market value and rating."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2010-01-04", periods=days)
    cols = ["S%d" % i for i in range(stocks)]
    log_close = np.cumsum(rng.normal(0.0003, 0.01, (days, stocks)), axis=0)
    close = pd.DataFrame(np.exp(log_close + 4.0), index=index, columns=cols)
    spread = np.abs(rng.normal(0.005, 0.002, (days, stocks)))
    volume = pd.DataFrame(rng.integers(10_000, 1_000_000,
                                       (days, stocks)).astype(float),
                          index=index, columns=cols)
    rating = pd.DataFrame(rng.uniform(2.0, 5.0, (days, stocks)),
                          index=index, columns=cols)
    return Fields(close=close, high=close * (1 + spread),
                  low=close * (1 - spread), volume=volume,
                  market_cap=close * 1e7, rating=rating)


def test_signal_names_and_shapes():
    fields = make_fields().aligned()
    signals = raw_signals(fields)
    assert set(signals) == set(SIGNAL_NAMES)
    assert len(signals) == 30
    for frame in signals.values():
        assert frame.shape == fields.close.shape
        assert frame.index.equals(fields.close.index)
        assert list(frame.columns) == list(fields.close.columns)
        assert not np.isinf(frame.to_numpy()).any()


def test_no_lookahead():
    fields = make_fields().aligned()
    later = make_fields(seed=1).aligned()
    cut = 900
    mixed = Fields(
        close=pd.concat([fields.close.iloc[:cut], later.close.iloc[cut:]]),
        high=pd.concat([fields.high.iloc[:cut], later.high.iloc[cut:]]),
        low=pd.concat([fields.low.iloc[:cut], later.low.iloc[cut:]]),
        volume=pd.concat([fields.volume.iloc[:cut],
                          later.volume.iloc[cut:]]),
        market_cap=pd.concat([fields.market_cap.iloc[:cut],
                              later.market_cap.iloc[cut:]]),
        rating=pd.concat([fields.rating.iloc[:cut], later.rating.iloc[cut:]]))
    a, b = raw_signals(fields), raw_signals(mixed)
    for name in SIGNAL_NAMES:
        x = a[name].iloc[:cut].to_numpy()
        y = b[name].iloc[:cut].to_numpy()
        assert np.allclose(x, y, equal_nan=True), name


def test_known_values():
    fields = make_fields().aligned()
    signals = raw_signals(fields)
    log_close = np.log(fields.close)
    expected = log_close - log_close.shift(5)
    assert np.allclose(signals["ret_5"].to_numpy(), expected.to_numpy(),
                       equal_nan=True)
    assert np.allclose(signals["logcap"].to_numpy(),
                       np.log(fields.market_cap).to_numpy())
    high52 = signals["high52"].to_numpy()
    assert np.nanmax(high52) <= 1e-12
    rsi = signals["rsi_14"].to_numpy()
    assert np.nanmin(rsi) >= 0.0 and np.nanmax(rsi) <= 100.0
    assert np.allclose(signals["rating_chg_21"].to_numpy(),
                       (fields.rating - fields.rating.shift(21)).to_numpy(),
                       equal_nan=True)


def test_trading_days_drops_holiday_fills():
    fields = make_fields(days=30)
    close = fields.close.copy()
    close.iloc[10] = close.iloc[9]
    keep = trading_days(close)
    assert not keep.iloc[10]
    assert keep.iloc[9] and keep.iloc[11]
    assert keep.iloc[0]


def test_membership_timing():
    fields = make_fields(days=12, stocks=2)
    volume = fields.volume.copy()
    volume.iloc[:2, 0] = np.nan
    members = membership(fields.close, volume, fields.market_cap, window=3,
                         cover=1.0)
    first = members["S0"].to_numpy()
    assert not first[:4].any()
    assert first[4:].all()
    assert members["S1"].to_numpy()[2:].all()
    assert not members["S1"].to_numpy()[:2].any()


def test_cross_stock_ranks():
    values = pd.DataFrame([[1.0, 2.0, 2.0, 4.0]], columns=list("abcd"))
    members = pd.DataFrame([[True, True, True, False]], columns=list("abcd"))
    ranks = cross_stock_ranks({"x": values}, members)["x_cs"]
    row = ranks.iloc[0]
    assert np.isnan(row["d"])
    assert abs(row["a"] - 1 / 3) < 1e-12
    assert abs(row["b"] - row["c"]) < 1e-12
    assert abs(row["b"] - 2.5 / 3) < 1e-12
    free = cross_stock_ranks({"x": values})["x_cs"].iloc[0]
    assert abs(free["d"] - 1.0) < 1e-12


def test_forward_returns_definition():
    fields = make_fields(days=40, stocks=2)
    close = fields.close
    fwd = forward_returns(close, horizon=2, lag=1)
    for t in (0, 5, 20):
        expected = close.iloc[t + 3] / close.iloc[t + 1] - 1
        assert np.allclose(fwd.iloc[t].to_numpy(), expected.to_numpy())
    assert fwd.iloc[-3:].isna().all().all()
    assert fwd.iloc[-4].notna().all()


def test_beta_residual_removes_the_fit():
    rng = np.random.default_rng(2)
    y = rng.normal(size=(50, 30))
    beta = rng.uniform(0.5, 1.5, (50, 30))
    y[:, 3] += 2.0 * beta[:, 3]
    y[0, :15] = np.nan
    res = beta_residual(y, beta, min_stocks=20)
    assert np.isnan(res[0]).all()
    for d in range(1, 50):
        r = res[d]
        assert np.abs(np.nanmean(r)) < 1e-10
        assert abs(np.nansum(r * (beta[d] - beta[d].mean()))) < 1e-8
    res2, slope = beta_residual(y, beta, min_stocks=20, return_slope=True)
    assert np.allclose(res2, res, equal_nan=True) and slope.shape == (50,)


def test_build_signals_window():
    fields = make_fields(days=1800, stocks=5)
    out = build_signals(fields, start="2014-01-01", last_day="2015-12-31",
                        window=252, cover=0.9)
    assert out.forward.index[0] >= pd.Timestamp("2014-01-01")
    assert out.forward.index[-1] <= pd.Timestamp("2015-12-31")
    assert set(out.ranks) == {n + "_cs" for n in SIGNAL_NAMES}
    assert set(out.history) == {n + "_ts" for n in SIGNAL_NAMES}
    assert set(out.availability) == {"rating_av", "ltrev_av"}
    assert out.members is not None and out.members.all().all()
    ranks = out.ranks["mom_252_21_cs"].to_numpy()
    assert np.nanmin(ranks) > 0.0 and np.nanmax(ranks) <= 1.0
    assert out.close.index.equals(out.forward.index)

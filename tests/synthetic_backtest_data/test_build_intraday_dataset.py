from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from Strategy_Auto_Trader.synthetic_backtest_data import build_intraday_dataset as build
from Strategy_Auto_Trader.synthetic_backtest_data.build_intraday_dataset import _finish, chain_link, splice_scale
from Strategy_Auto_Trader.core.trading_sessions import LSE


def _daily(values: list[float], start: str) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)))


class TestChainLink:
    def test_earlier_history_is_rescaled_to_the_primary_first_close(self):
        fallback = _daily([10, 11, 12, 13, 14], "2020-01-06")
        primary = _daily([260, 270], "2020-01-09")  # fallback on that date is 13
        out = chain_link(primary, fallback)
        assert out.loc["2020-01-08"] == pytest.approx(12 * 260 / 13)
        assert out.loc["2020-01-09"] == 260
        assert out.index.is_monotonic_increasing and not out.index.duplicated().any()

    def test_returns_before_the_link_are_preserved(self):
        fallback = _daily([10, 11, 12, 13, 14], "2020-01-06")
        out = chain_link(_daily([260, 270], "2020-01-09"), fallback)
        assert np.allclose(np.log(out.iloc[:4]).diff().dropna(), np.log(fallback.iloc[:4]).diff().dropna())


class TestDailyIbkrUnitShift:
    def test_fund_flag_back_adjusts_a_hundredfold_unit_change(self, tmp_path, monkeypatch):
        # IBKR daily ISF.L is quoted ~100x larger before 2004-04-16 than after it
        days = pd.bdate_range("2004-04-12", periods=8)
        close = [45800.0, 45600.0, 45700.0, 45600.0, 459.0, 458.0, 463.0, 461.0]
        pd.DataFrame({"Close": close}, index=days.tz_localize("UTC")).to_csv(tmp_path / "ISF.L.csv")
        monkeypatch.setattr(build, "_DAILY", tmp_path)

        raw = build._daily_ibkr("ISF.L")
        fixed = build._daily_ibkr("ISF.L", fund=True)

        assert raw.iloc[0] > 40000
        assert fixed.iloc[-1] == 461.0
        assert np.log(fixed).diff().abs().max() < 0.05

    def test_index_series_are_left_untouched_even_with_big_moves(self, tmp_path, monkeypatch):
        days = pd.bdate_range("2020-03-02", periods=4)
        pd.DataFrame({"Close": [15.0, 25.0, 40.0, 62.0]}, index=days.tz_localize("UTC")).to_csv(tmp_path / "INDEX_VIX.csv")
        monkeypatch.setattr(build, "_DAILY", tmp_path)
        assert build._daily_ibkr("INDEX_VIX").tolist() == [15.0, 25.0, 40.0, 62.0]


class TestSpliceScale:
    def test_uses_second_real_day_to_skip_a_partial_first_day(self):
        real = _daily([1.0, 200.0, 210.0], "2020-01-06")
        proxy = _daily([1.0, 100.0, 105.0], "2020-01-06")
        assert splice_scale(real, proxy) == pytest.approx(2.0)

    def test_falls_forward_to_next_common_date(self):
        real = _daily([1.0, 200.0, 210.0], "2020-01-06")
        proxy = pd.Series([105.0], index=[real.index[2]])
        assert splice_scale(real, proxy) == pytest.approx(2.0)

    def test_no_overlap_raises(self):
        with pytest.raises(ValueError):
            splice_scale(_daily([1.0, 2.0], "2020-01-06"), _daily([1.0], "2019-01-07"))


class TestFinish:
    def _bars(self, day: str, closes: list[float]) -> pd.DataFrame:
        idx = pd.date_range(f"{day} 08:00", periods=len(closes), freq="h", tz="UTC")
        return pd.DataFrame({"Close": closes, "bar_end": idx + pd.Timedelta(hours=1)}, index=idx)

    def test_schema_and_source_labels(self):
        bridged = self._bars("2020-01-06", [1.0, 1.1])
        real = pd.Series([2.0, 2.1], index=pd.date_range("2020-01-07 08:00", periods=2, freq="h", tz="UTC"))
        out = _finish(bridged, real, LSE)
        assert list(out.columns) == ["Close", "bar_end_utc", "source"]
        assert out.index.name == "bar_start_utc"
        assert out["source"].tolist() == ["bridged", "bridged", "real", "real"]

    def test_real_wins_on_overlapping_stamps(self):
        bridged = self._bars("2020-01-07", [1.0, 1.1])
        real = pd.Series([2.0], index=pd.DatetimeIndex(["2020-01-07 08:00"], tz="UTC"))
        out = _finish(bridged, real, LSE)
        assert out.loc[pd.Timestamp("2020-01-07 08:00", tz="UTC"), "Close"] == 2.0
        assert not out.index.duplicated().any()

from __future__ import annotations

import io
from unittest import mock

import numpy as np
import pandas as pd
import pytest


class TestLoadJournal:

    def test_load_journal_basic(self, tmp_path):
        from scripts import analyze_journal
        csv_file = tmp_path / "live.csv"
        csv_content = """date_opened,date_closed,strategy,ticker,entry_score,entry_price,regime_at_entry,rsi_at_entry,volume_ratio,exit_price,pnl_usd,return_pct,days_held,peak_gain,peak_loss,exit_reason
2025-01-01 10:00:00+00:00,2025-01-01 12:00:00+00:00,trend_follow,SPY,6.5,420.0,0.5,55.0,1.1,425.0,210.0,1.19,0.083,2.5,-0.5,target
2025-01-02 09:00:00+00:00,2025-01-02 11:00:00+00:00,conservative,QQQ,4.0,380.0,0.2,45.0,0.9,375.0,-190.0,-1.32,0.083,0.5,-2.0,stop
"""
        csv_file.write_text(csv_content)
        with mock.patch.object(analyze_journal, "JOURNAL", csv_file):
            df = analyze_journal.load_journal()
            assert len(df) == 2
            assert set(df["strategy"].unique()) == {"trend_follow", "conservative"}
            assert df["return_pct"].dtype == float
            assert "UTC" in str(df["date_closed"].dtype)

    def test_load_journal_filters_unclosed_trades(self, tmp_path):
        from scripts import analyze_journal
        csv_file = tmp_path / "live.csv"
        csv_content = """date_opened,date_closed,strategy,ticker,entry_score,entry_price,regime_at_entry,rsi_at_entry,volume_ratio,exit_price,pnl_usd,return_pct,days_held,peak_gain,peak_loss,exit_reason
2025-01-01 10:00:00+00:00,2025-01-01 12:00:00+00:00,trend_follow,SPY,6.5,420.0,0.5,55.0,1.1,425.0,210.0,1.19,0.083,2.5,-0.5,target
2025-01-02 09:00:00+00:00,,conservative,QQQ,4.0,380.0,0.2,45.0,0.9,375.0,0.0,,0.083,0.5,-2.0,
2025-01-03 14:00:00+00:00,,default,IWM,5.0,190.0,0.3,60.0,1.0,189.0,0.0,,0.042,0.1,-0.1,
"""
        csv_file.write_text(csv_content)
        with mock.patch.object(analyze_journal, "JOURNAL", csv_file):
            df = analyze_journal.load_journal()
            assert len(df) == 1
            assert df.iloc[0]["ticker"] == "SPY"

    def test_load_journal_numeric_coercion(self, tmp_path):
        from scripts import analyze_journal
        csv_file = tmp_path / "live.csv"
        csv_content = """date_opened,date_closed,strategy,ticker,entry_score,entry_price,regime_at_entry,rsi_at_entry,volume_ratio,exit_price,pnl_usd,return_pct,days_held,peak_gain,peak_loss,exit_reason
2025-01-01 10:00:00+00:00,2025-01-01 12:00:00+00:00,trend_follow,SPY,6.5,420.0,0.5,55.0,1.1,425.0,210.0,1.19,0.083,2.5,-0.5,target
2025-01-02 09:00:00+00:00,2025-01-02 11:00:00+00:00,conservative,QQQ,bad_score,380.0,0.2,45.0,0.9,375.0,-190.0,-1.32,0.083,0.5,-2.0,stop
"""
        csv_file.write_text(csv_content)
        with mock.patch.object(analyze_journal, "JOURNAL", csv_file):
            df = analyze_journal.load_journal()
            assert len(df) == 2
            assert pd.isna(df.iloc[1]["entry_score"])


class TestOutcomeStats:

    def test_outcome_stats_basic(self):
        from scripts import analyze_journal
        df = pd.DataFrame({
            "return_pct": [1.0, 2.0, -0.5, -1.5],
            "pnl_usd": [100.0, 200.0, -50.0, -150.0],
            "days_held": [1, 2, 0.5, 1.5],
        })
        stats = analyze_journal._outcome_stats(df)
        assert stats["n"] == 4
        assert stats["hit_rate"] == 0.5
        assert stats["avg_ret"] == 0.25
        assert stats["med_ret"] == 0.25
        assert stats["total_pnl"] == 100.0

    def test_outcome_stats_all_wins(self):
        from scripts import analyze_journal
        df = pd.DataFrame({
            "return_pct": [1.0, 2.0, 0.5],
            "pnl_usd": [100.0, 200.0, 50.0],
            "days_held": [1, 2, 0.5],
        })
        stats = analyze_journal._outcome_stats(df)
        assert stats["hit_rate"] == 1.0
        assert stats["total_pnl"] == 350.0
        assert np.isinf(stats["profit_factor"])

    def test_outcome_stats_all_losses(self):
        from scripts import analyze_journal
        df = pd.DataFrame({
            "return_pct": [-1.0, -2.0, -0.5],
            "pnl_usd": [-100.0, -200.0, -50.0],
            "days_held": [1, 2, 0.5],
        })
        stats = analyze_journal._outcome_stats(df)
        assert stats["hit_rate"] == 0.0
        assert stats["total_pnl"] == -350.0
        assert stats["profit_factor"] == 0.0

    def test_outcome_stats_single_win(self):
        from scripts import analyze_journal
        df = pd.DataFrame({
            "return_pct": [1.5],
            "pnl_usd": [150.0],
            "days_held": [1.0],
        })
        stats = analyze_journal._outcome_stats(df)
        assert stats["n"] == 1
        assert stats["hit_rate"] == 1.0
        assert stats["avg_ret"] == 1.5
        assert np.isinf(stats["profit_factor"])

    def test_outcome_stats_single_loss(self):
        from scripts import analyze_journal
        df = pd.DataFrame({
            "return_pct": [-2.0],
            "pnl_usd": [-200.0],
            "days_held": [0.5],
        })
        stats = analyze_journal._outcome_stats(df)
        assert stats["n"] == 1
        assert stats["hit_rate"] == 0.0
        assert stats["avg_ret"] == -2.0
        assert stats["profit_factor"] == 0.0

    def test_outcome_stats_empty(self):
        from scripts import analyze_journal
        df = pd.DataFrame({
            "return_pct": [],
            "pnl_usd": [],
            "days_held": [],
        })
        stats = analyze_journal._outcome_stats(df)
        assert stats["n"] == 0
        assert np.isnan(stats["hit_rate"])

    def test_outcome_stats_mixed_with_breakeven(self):
        from scripts import analyze_journal
        df = pd.DataFrame({
            "return_pct": [1.0, 0.0, -1.0],
            "pnl_usd": [100.0, 0.0, -100.0],
            "days_held": [1, 1, 1],
        })
        stats = analyze_journal._outcome_stats(df)
        assert stats["n"] == 3
        assert stats["hit_rate"] == 1.0 / 3
        assert stats["total_pnl"] == 0.0
        assert stats["profit_factor"] == 1.0

    def test_outcome_stats_profit_factor_calculation(self):
        from scripts import analyze_journal
        df = pd.DataFrame({
            "return_pct": [2.0, 1.0, -0.5, -0.2],
            "pnl_usd": [200.0, 100.0, -50.0, -20.0],
            "days_held": [1, 1, 1, 1],
        })
        stats = analyze_journal._outcome_stats(df)
        expected_pf = 300.0 / 70.0
        assert abs(stats["profit_factor"] - expected_pf) < 0.01


class TestA1PerStrategy:

    def test_a1_per_strategy(self, capsys, tmp_path):
        from scripts import analyze_journal
        df = pd.DataFrame({
            "strategy": ["trend_follow", "trend_follow", "conservative", "conservative"],
            "return_pct": [1.0, -0.5, 0.8, -1.0],
            "pnl_usd": [100.0, -50.0, 80.0, -100.0],
            "days_held": [1, 0.5, 1, 1],
        })
        result = analyze_journal.a1_per_strategy(df)
        assert len(result) == 2
        assert "trend_follow" in result.index
        assert "conservative" in result.index
        assert result.loc["trend_follow", "n"] == 2
        assert result.loc["conservative", "n"] == 2

    def test_a1_per_strategy_output(self, capsys):
        from scripts import analyze_journal
        df = pd.DataFrame({
            "strategy": ["s1", "s1"],
            "return_pct": [1.0, -0.5],
            "pnl_usd": [100.0, -50.0],
            "days_held": [1, 0.5],
        })
        analyze_journal.a1_per_strategy(df)
        captured = capsys.readouterr()
        assert "A1  Per-strategy outcomes" in captured.out


class TestBucketTable:

    def test_bucket_table_basic(self):
        from scripts import analyze_journal
        df = pd.DataFrame({
            "rsi_at_entry": [35, 45, 55, 65, 75, 85],
            "return_pct": [1.0, 0.5, -0.5, 1.5, -1.0, 2.0],
            "pnl_usd": [100, 50, -50, 150, -100, 200],
            "days_held": [1, 1, 1, 1, 1, 1],
        })
        result = analyze_journal._bucket_table(
            df, "rsi_at_entry",
            [0, 40, 50, 60, 70, 101],
            ["<40", "40-50", "50-60", "60-70", ">70"]
        )
        assert len(result) > 0
        assert "n" in result.columns

    def test_bucket_table_empty_bucket(self):
        from scripts import analyze_journal
        df = pd.DataFrame({
            "score": [0.1, 0.2, 0.9],
            "return_pct": [1.0, 0.5, -0.5],
            "pnl_usd": [100, 50, -50],
            "days_held": [1, 1, 1],
        })
        result = analyze_journal._bucket_table(
            df, "score",
            [0, 0.3, 0.6, 1.0],
            ["low", "mid", "high"]
        )
        assert "n" in result.columns

    def test_bucket_table_all_in_one_bucket(self):
        from scripts import analyze_journal
        df = pd.DataFrame({
            "vol": [1.0, 1.05, 1.1],
            "return_pct": [1.0, 0.5, -0.5],
            "pnl_usd": [100, 50, -50],
            "days_held": [1, 1, 1],
        })
        result = analyze_journal._bucket_table(
            df, "vol",
            [0.5, 1.2, 2.0],
            ["low", "high"]
        )
        assert result["n"].sum() == 3


class TestA3Rsi:

    def test_a3_rsi(self, capsys):
        from scripts import analyze_journal
        df = pd.DataFrame({
            "rsi_at_entry": [35, 45, 55, 65, 75, 85],
            "return_pct": [1.0, 0.5, -0.5, 1.5, -1.0, 2.0],
            "pnl_usd": [100, 50, -50, 150, -100, 200],
            "days_held": [1, 1, 1, 1, 1, 1],
        })
        analyze_journal.a3_rsi(df)
        captured = capsys.readouterr()
        assert "A3  rsi_at_entry vs outcome" in captured.out


class TestA4Regime:

    def test_a4_regime(self, capsys):
        from scripts import analyze_journal
        df = pd.DataFrame({
            "regime_at_entry": [-0.5, 0.1, 0.4, 0.6, 0.9],
            "return_pct": [1.0, 0.5, -0.5, 1.5, 2.0],
            "pnl_usd": [100, 50, -50, 150, 200],
            "days_held": [1, 1, 1, 1, 1],
        })
        analyze_journal.a4_regime(df)
        captured = capsys.readouterr()
        assert "A4  regime_at_entry" in captured.out


class TestA5Volume:

    def test_a5_volume(self, capsys):
        from scripts import analyze_journal
        df = pd.DataFrame({
            "volume_ratio": [0.5, 1.0, 1.5, 2.0],
            "return_pct": [1.0, 0.5, -0.5, 1.5],
            "pnl_usd": [100, 50, -50, 150],
            "days_held": [1, 1, 1, 1],
        })
        analyze_journal.a5_volume(df)
        captured = capsys.readouterr()
        assert "A5  volume_ratio" in captured.out


class TestA6ExitReasons:

    def test_a6_exit_reasons(self, capsys):
        from scripts import analyze_journal
        df = pd.DataFrame({
            "strategy": ["trend_follow", "trend_follow", "conservative"],
            "exit_reason": ["target", "stop", "target"],
            "return_pct": [1.0, -0.5, 0.8],
            "pnl_usd": [100, -50, 80],
            "peak_gain": [2.0, 0.5, 1.5],
            "peak_loss": [-0.5, -2.0, -0.2],
            "days_held": [1, 0.5, 1],
        })
        analyze_journal.a6_exit_reasons(df)
        captured = capsys.readouterr()
        assert "A6  exit_reason" in captured.out

    def test_a6_exit_reasons_calculation(self):
        from scripts import analyze_journal
        df = pd.DataFrame({
            "strategy": ["s1", "s1"],
            "exit_reason": ["target", "stop"],
            "return_pct": [1.0, -0.5],
            "pnl_usd": [100, -50],
            "peak_gain": [2.0, 0.5],
            "peak_loss": [-0.5, -2.0],
            "days_held": [1, 0.5],
        })
        g = df.groupby(["strategy", "exit_reason"]).agg(
            n=("return_pct", "size"),
            avg_ret=("return_pct", "mean"),
            avg_peak_gain=("peak_gain", "mean"),
            avg_peak_loss=("peak_loss", "mean"),
            avg_days=("days_held", "mean"),
            total_pnl=("pnl_usd", "sum"),
        )
        g["give_back"] = g["avg_peak_gain"] - g["avg_ret"]
        assert g.loc[("s1", "target"), "avg_peak_gain"] == 2.0
        assert g.loc[("s1", "target"), "avg_ret"] == 1.0
        assert g.loc[("s1", "target"), "give_back"] == 1.0


class TestA8DaysHeld:

    def test_a8_days_held(self, capsys):
        from scripts import analyze_journal
        df = pd.DataFrame({
            "strategy": ["trend_follow", "trend_follow", "conservative", "conservative"],
            "return_pct": [1.0, -0.5, 0.8, -1.0],
            "days_held": [1.5, 0.5, 2.0, 0.25],
        })
        analyze_journal.a8_days_held(df)
        captured = capsys.readouterr()
        assert "A8  days_held" in captured.out


class TestA9Concentration:

    def test_a9_concentration(self, capsys):
        from scripts import analyze_journal
        df = pd.DataFrame({
            "strategy": ["trend_follow"] * 6,
            "ticker": ["SPY", "SPY", "QQQ", "QQQ", "IWM", "VTI"],
            "pnl_usd": [500, 300, 200, 150, 50, 25],
        })
        analyze_journal.a9_concentration(df, "trend_follow")
        captured = capsys.readouterr()
        assert "A9  Per-ticker P&L" in captured.out

    def test_a9_concentration_no_warning_when_distributed(self, capsys):
        from scripts import analyze_journal
        df = pd.DataFrame({
            "strategy": ["s"] * 6,
            "ticker": ["A", "B", "C", "D", "E", "F"],
            "pnl_usd": [100, 100, 100, 100, 100, 100],
        })
        analyze_journal.a9_concentration(df, "s")
        captured = capsys.readouterr()
        assert "WARNING" not in captured.out

    def test_a9_concentration_warning_when_concentrated(self, capsys):
        from scripts import analyze_journal
        df = pd.DataFrame({
            "strategy": ["s"] * 4,
            "ticker": ["A", "B", "C", "D"],
            "pnl_usd": [400, 50, 30, 20],
        })
        analyze_journal.a9_concentration(df, "s")
        captured = capsys.readouterr()
        assert "WARNING" in captured.out

    def test_a9_concentration_zero_pnl(self, capsys):
        from scripts import analyze_journal
        df = pd.DataFrame({
            "strategy": ["s"],
            "ticker": ["A"],
            "pnl_usd": [0],
        })
        analyze_journal.a9_concentration(df, "s")
        captured = capsys.readouterr()
        assert "A9  Per-ticker P&L" in captured.out


class TestA10CapitalEfficiency:

    @staticmethod
    def _df(rows):
        df = pd.DataFrame(rows)
        for col in ("date_opened", "date_closed"):
            df[col] = pd.to_datetime(df[col], utc=True, format="mixed")
        return df

    def test_basic_single_trade(self, capsys):
        from scripts import analyze_journal
        # $1000 notional held 36.5 days for +$50 -> 5% per 0.1y = 50%/dollar-year
        df = self._df([{
            "date_opened": "2025-01-01", "date_closed": "2025-02-06 12:00:00",
            "ticker": "SPY", "pnl_usd": 50.0, "return_pct": 0.05,
        }])
        r = analyze_journal.a10_capital_efficiency(df)
        assert r is not None
        assert r["total_pnl"] == pytest.approx(50.0)
        assert r["dollar_days"] == pytest.approx(1000.0 * 36.5)
        assert r["ann_return_on_deployed"] == pytest.approx(0.50)
        assert r["max_concurrent"] == pytest.approx(1000.0)
        assert "A10" in capsys.readouterr().out

    def test_overlapping_trades_peak_exposure(self):
        from scripts import analyze_journal
        df = self._df([
            {"date_opened": "2025-01-01", "date_closed": "2025-01-20",
             "ticker": "A", "pnl_usd": 10.0, "return_pct": 0.01},   # 1000
            {"date_opened": "2025-01-10", "date_closed": "2025-01-30",
             "ticker": "B", "pnl_usd": 40.0, "return_pct": 0.02},   # 2000
            {"date_opened": "2025-02-05", "date_closed": "2025-02-15",
             "ticker": "C", "pnl_usd": -15.0, "return_pct": -0.03},  # 500
        ])
        r = analyze_journal.a10_capital_efficiency(df)
        assert r["max_concurrent"] == pytest.approx(3000.0)

    def test_disjoint_trades_peak_is_largest_single(self):
        from scripts import analyze_journal
        df = self._df([
            {"date_opened": "2025-01-01", "date_closed": "2025-01-10",
             "ticker": "A", "pnl_usd": 10.0, "return_pct": 0.01},   # 1000
            {"date_opened": "2025-02-01", "date_closed": "2025-02-10",
             "ticker": "B", "pnl_usd": 5.0, "return_pct": 0.02},    # 250
        ])
        r = analyze_journal.a10_capital_efficiency(df)
        assert r["max_concurrent"] == pytest.approx(1000.0)

    def test_zero_return_rows_excluded_from_notional(self):
        from scripts import analyze_journal
        df = self._df([
            {"date_opened": "2025-01-01", "date_closed": "2025-01-10",
             "ticker": "A", "pnl_usd": 10.0, "return_pct": 0.01},
            {"date_opened": "2025-01-02", "date_closed": "2025-01-05",
             "ticker": "B", "pnl_usd": 0.0, "return_pct": 0.0},
        ])
        r = analyze_journal.a10_capital_efficiency(df)
        assert r["dollar_days"] == pytest.approx(1000.0 * 9)
        assert r["max_concurrent"] == pytest.approx(1000.0)

    def test_all_zero_returns_skips(self, capsys):
        from scripts import analyze_journal
        df = self._df([{
            "date_opened": "2025-01-01", "date_closed": "2025-01-10",
            "ticker": "A", "pnl_usd": 0.0, "return_pct": 0.0,
        }])
        assert analyze_journal.a10_capital_efficiency(df) is None
        assert "skipping" in capsys.readouterr().out

    def test_bh_comparison_uses_last_row_per_ticker(self, capsys):
        from scripts import analyze_journal
        df = self._df([
            {"date_opened": "2025-01-01", "date_closed": "2025-07-02 12:00:00",
             "ticker": "A", "pnl_usd": 10.0, "return_pct": 0.01, "bh_return": 0.05},
            {"date_opened": "2025-07-03", "date_closed": "2026-01-01",
             "ticker": "A", "pnl_usd": 10.0, "return_pct": 0.01, "bh_return": 0.10},
        ])
        r = analyze_journal.a10_capital_efficiency(df)
        # span = 365d, last bh_return for A = 0.10 -> annualised 10%
        assert r["bh_annualised"] == pytest.approx(0.10, rel=1e-3)
        assert "Buy & hold" in capsys.readouterr().out

    def test_blended_return_with_capital(self, capsys):
        from scripts import analyze_journal
        # one trade, $1000 deployed the whole 365d window, +$100
        df = self._df([{
            "date_opened": "2025-01-01", "date_closed": "2026-01-01",
            "ticker": "A", "pnl_usd": 100.0, "return_pct": 0.10,
        }])
        r = analyze_journal.a10_capital_efficiency(df, capital=10_000.0, risk_free=0.04)
        # idle 9000 * 4% = 360; (100+360)/10000 = 4.6% over exactly 1 year
        assert r["blended_annualised"] == pytest.approx(0.046, rel=1e-3)
        assert "utilisation" in capsys.readouterr().out

    def test_no_capital_no_blended(self):
        from scripts import analyze_journal
        df = self._df([{
            "date_opened": "2025-01-01", "date_closed": "2025-02-01",
            "ticker": "A", "pnl_usd": 10.0, "return_pct": 0.01,
        }])
        r = analyze_journal.a10_capital_efficiency(df)
        assert r["blended_annualised"] is None
        assert r["bh_annualised"] is None


class TestRecommendedParameters:

    def test_recommended_parameters_basic(self, capsys):
        from scripts import analyze_journal
        df = pd.DataFrame({
            "rsi_at_entry": [35, 45, 55, 65, 75, 85],
            "regime_at_entry": [-0.5, 0.1, 0.4, 0.6, 0.9, 0.7],
            "volume_ratio": [0.5, 1.0, 1.5, 2.0, 1.2, 0.8],
            "pnl_usd": [100, 50, -50, 150, -100, 200],
        })
        analyze_journal.recommended_parameters(df, "trend_follow")
        captured = capsys.readouterr()
        assert "RECOMMENDED PARAMETERS" in captured.out
        assert "trend_follow" in captured.out


class TestSyntheticJournalIntegration:

    def test_full_pipeline_with_synthetic_data(self, tmp_path):
        from scripts import analyze_journal
        csv_file = tmp_path / "live.csv"
        csv_content = """date_opened,date_closed,strategy,ticker,entry_score,entry_price,regime_at_entry,rsi_at_entry,volume_ratio,exit_price,pnl_usd,return_pct,days_held,peak_gain,peak_loss,exit_reason
2025-01-01 10:00:00+00:00,2025-01-01 12:00:00+00:00,trend_follow,SPY,6.5,420.0,0.5,55.0,1.1,425.0,210.0,1.19,0.083,2.5,-0.5,target
2025-01-02 09:00:00+00:00,2025-01-02 11:00:00+00:00,trend_follow,SPY,7.0,415.0,0.6,50.0,1.2,410.0,-150.0,-1.2,0.083,1.0,-2.0,stop
2025-01-03 14:00:00+00:00,2025-01-03 16:00:00+00:00,conservative,QQQ,4.5,380.0,0.2,45.0,0.9,385.0,150.0,1.31,0.083,0.5,-0.3,target
2025-01-04 08:00:00+00:00,2025-01-04 10:00:00+00:00,conservative,QQQ,4.0,375.0,0.1,40.0,0.8,370.0,-200.0,-1.33,0.083,0.5,-1.5,stop
2025-01-05 11:00:00+00:00,2025-01-05 13:00:00+00:00,trend_follow,IWM,6.0,190.0,0.4,60.0,1.3,195.0,250.0,2.63,0.083,1.0,-0.5,target
"""
        csv_file.write_text(csv_content)
        with mock.patch.object(analyze_journal, "JOURNAL", csv_file):
            df = analyze_journal.load_journal()
            assert len(df) == 5
            a1 = analyze_journal.a1_per_strategy(df)
            assert len(a1) == 2
            assert a1.loc["trend_follow", "n"] == 3
            assert a1.loc["conservative", "n"] == 2

    def test_boundary_case_single_trade_win(self, tmp_path):
        from scripts import analyze_journal
        csv_file = tmp_path / "live.csv"
        csv_content = """date_opened,date_closed,strategy,ticker,entry_score,entry_price,regime_at_entry,rsi_at_entry,volume_ratio,exit_price,pnl_usd,return_pct,days_held,peak_gain,peak_loss,exit_reason
2025-01-01 10:00:00+00:00,2025-01-01 12:00:00+00:00,trend_follow,SPY,6.5,420.0,0.5,55.0,1.1,425.0,210.0,1.19,0.083,2.5,-0.5,target
"""
        csv_file.write_text(csv_content)
        with mock.patch.object(analyze_journal, "JOURNAL", csv_file):
            df = analyze_journal.load_journal()
            assert len(df) == 1
            stats = analyze_journal._outcome_stats(df)
            assert stats["n"] == 1
            assert stats["hit_rate"] == 1.0
            assert np.isinf(stats["profit_factor"])

    def test_boundary_case_all_losses(self, tmp_path):
        from scripts import analyze_journal
        csv_file = tmp_path / "live.csv"
        csv_content = """date_opened,date_closed,strategy,ticker,entry_score,entry_price,regime_at_entry,rsi_at_entry,volume_ratio,exit_price,pnl_usd,return_pct,days_held,peak_gain,peak_loss,exit_reason
2025-01-01 10:00:00+00:00,2025-01-01 12:00:00+00:00,trend_follow,SPY,6.5,420.0,0.5,55.0,1.1,415.0,-210.0,-1.19,0.083,2.5,-1.5,stop
2025-01-02 09:00:00+00:00,2025-01-02 11:00:00+00:00,trend_follow,SPY,7.0,415.0,0.6,50.0,1.2,410.0,-150.0,-1.2,0.083,1.0,-2.0,stop
2025-01-03 14:00:00+00:00,2025-01-03 16:00:00+00:00,trend_follow,QQQ,5.0,380.0,0.2,45.0,0.9,375.0,-100.0,-1.31,0.083,0.5,-0.8,stop
"""
        csv_file.write_text(csv_content)
        with mock.patch.object(analyze_journal, "JOURNAL", csv_file):
            df = analyze_journal.load_journal()
            assert len(df) == 3
            stats = analyze_journal._outcome_stats(df)
            assert stats["n"] == 3
            assert stats["hit_rate"] == 0.0
            assert stats["profit_factor"] == 0.0
            assert stats["total_pnl"] < 0

    def test_boundary_case_empty_journal(self, tmp_path):
        from scripts import analyze_journal
        csv_file = tmp_path / "live.csv"
        csv_content = """date_opened,date_closed,strategy,ticker,entry_score,entry_price,regime_at_entry,rsi_at_entry,volume_ratio,exit_price,pnl_usd,return_pct,days_held,peak_gain,peak_loss,exit_reason
"""
        csv_file.write_text(csv_content)
        with mock.patch.object(analyze_journal, "JOURNAL", csv_file):
            df = analyze_journal.load_journal()
            assert len(df) == 0
            stats = analyze_journal._outcome_stats(df)
            assert stats["n"] == 0

    def test_boundary_case_with_nan_numeric_values(self, tmp_path):
        from scripts import analyze_journal
        csv_file = tmp_path / "live.csv"
        csv_content = """date_opened,date_closed,strategy,ticker,entry_score,entry_price,regime_at_entry,rsi_at_entry,volume_ratio,exit_price,pnl_usd,return_pct,days_held,peak_gain,peak_loss,exit_reason
2025-01-01 10:00:00+00:00,2025-01-01 12:00:00+00:00,trend_follow,SPY,,420.0,0.5,55.0,1.1,425.0,210.0,1.19,0.083,2.5,-0.5,target
2025-01-02 09:00:00+00:00,2025-01-02 11:00:00+00:00,trend_follow,SPY,7.0,,0.6,50.0,1.2,410.0,-150.0,-1.2,0.083,1.0,-2.0,stop
"""
        csv_file.write_text(csv_content)
        with mock.patch.object(analyze_journal, "JOURNAL", csv_file):
            df = analyze_journal.load_journal()
            assert len(df) == 2
            assert pd.isna(df.iloc[0]["entry_score"])
            assert pd.isna(df.iloc[1]["entry_price"])

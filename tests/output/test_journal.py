from __future__ import annotations

import csv
from dataclasses import fields
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from Strategy_Auto_Trader.output.journal import (
    JOURNAL_FIELDNAMES,
    TradeRecord,
    _existing_keys,
    _trade_key,
    append_trades,
    extract_trades_from_csv,
    extract_trades_from_detail,
)


class TestTradeRecord:

    def test_defaults(self):
        record = TradeRecord(date_opened="2020-01-01", ticker="SPY")
        assert record.date_opened == "2020-01-01"
        assert record.ticker == "SPY"
        assert record.strategy == ""
        assert record.vol_filter == ""
        assert record.entry_signal == ""
        assert record.entry_score == 0.0
        assert record.entry_gate_flag == ""
        assert record.entry_price == 0.0
        assert record.regime_at_entry == 0.0
        assert record.rsi_at_entry == 0.0
        assert record.volume_ratio == 0.0
        assert record.kelly_fraction == 0.0
        assert record.date_closed == ""
        assert record.exit_reason == ""
        assert record.exit_price == 0.0
        assert record.stop_level == 0.0
        assert record.target_level == 0.0
        assert record.pnl_usd == 0.0
        assert record.return_pct == 0.0
        assert record.days_held == 0
        assert record.peak_gain == 0.0
        assert record.peak_loss == 0.0
        assert record.strategy_return == 0.0
        assert record.bh_return == 0.0
        assert record.notes == ""

    def test_journal_fieldnames_matches_dataclass(self):
        dataclass_fields = [f.name for f in fields(TradeRecord)]
        assert JOURNAL_FIELDNAMES == dataclass_fields


class TestTradeKey:

    def test_trade_key_basic(self):
        row = {
            "ticker": "SPY",
            "strategy": "trend_follow",
            "date_opened": "2020-01-01",
            "date_closed": "2020-01-05",
        }
        key = _trade_key(row)
        assert key == ("SPY", "trend_follow", "2020-01-01", "2020-01-05")

    def test_trade_key_missing_fields(self):
        row = {"ticker": "AAPL"}
        key = _trade_key(row)
        assert key == ("AAPL", "", "", "")

    def test_trade_key_empty_dict(self):
        key = _trade_key({})
        assert key == ("", "", "", "")


class TestExistingKeys:

    def test_nonexistent_path_returns_empty_set(self, tmp_path):
        journal_path = tmp_path / "nonexistent.csv"
        result = _existing_keys(journal_path)
        assert result == set()

    def test_empty_csv_returns_empty_set(self, tmp_path):
        journal_path = tmp_path / "empty.csv"
        journal_path.write_text("ticker,strategy,date_opened,date_closed\n")
        result = _existing_keys(journal_path)
        assert result == set()

    def test_reads_existing_keys(self, tmp_path):
        journal_path = tmp_path / "journal.csv"
        fieldnames = ["ticker", "strategy", "date_opened", "date_closed"]
        with open(journal_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({"ticker": "SPY", "strategy": "trend", "date_opened": "2020-01-01", "date_closed": "2020-01-05"})
            writer.writerow({"ticker": "QQQ", "strategy": "default", "date_opened": "2020-01-02", "date_closed": "2020-01-06"})

        result = _existing_keys(journal_path)
        assert ("SPY", "trend", "2020-01-01", "2020-01-05") in result
        assert ("QQQ", "default", "2020-01-02", "2020-01-06") in result
        assert len(result) == 2


class TestExtractTradesFromDetail:

    def test_empty_dataframe(self):
        detail = pd.DataFrame()
        result = extract_trades_from_detail("SPY", detail)
        assert result == []

    def test_missing_trade_event_column(self):
        detail = pd.DataFrame({
            "date": ["2020-01-01"],
            "close": [100.0],
        })
        result = extract_trades_from_detail("SPY", detail)
        assert result == []

    def test_buy_with_no_matching_sell_dropped(self):
        detail = pd.DataFrame({
            "date": ["2020-01-01", "2020-01-02"],
            "trade_event": ["BUY", None],
            "close": [100.0, 101.0],
            "entry_price": [100.0, None],
        })
        result = extract_trades_from_detail("SPY", detail)
        assert result == []

    def test_full_buy_sell_round_trip(self):
        detail = pd.DataFrame({
            "date": ["2020-01-01", "2020-01-05"],
            "trade_event": ["BUY", "SELL"],
            "close": [100.0, 110.0],
            "entry_price": [100.0, None],
            "signal_flag": ["momentum", None],
            "signal_score": [0.5, None],
            "gate_flag": ["ok", None],
            "regime_signal": [0.8, None],
            "rsi": [55.0, None],
            "volume_ratio": [1.2, None],
            "kelly_fraction": [0.1, None],
            "stop_level": [95.0, None],
            "target_level": [105.0, None],
            "portfolio_value": [10000.0, None],
            "sell_reason": [None, "take_profit"],
        })
        result = extract_trades_from_detail("SPY", detail, strategy="trend_follow", vol_filter="high")

        assert len(result) == 1
        trade = result[0]
        assert trade.date_opened == "2020-01-01"
        assert trade.ticker == "SPY"
        assert trade.strategy == "trend_follow"
        assert trade.vol_filter == "high"
        assert trade.entry_price == 100.0
        assert trade.exit_price == 110.0
        assert trade.date_closed == "2020-01-05"
        assert trade.exit_reason == "take_profit"
        assert trade.days_held == 4
        assert abs(trade.return_pct - 0.1) < 1e-9
        assert abs(trade.pnl_usd - 100.0) < 1e-6
        assert trade.entry_signal == "momentum"
        assert trade.entry_score == 0.5
        assert trade.entry_gate_flag == "ok"
        assert trade.regime_at_entry == 0.8
        assert trade.rsi_at_entry == 55.0
        assert trade.volume_ratio == 1.2
        assert trade.kelly_fraction == 0.1
        assert trade.stop_level == 95.0
        assert trade.target_level == 105.0

    def test_entry_price_zero_no_division_error(self):
        detail = pd.DataFrame({
            "date": ["2020-01-01", "2020-01-05"],
            "trade_event": ["BUY", "SELL"],
            "close": [100.0, 110.0],
            "signal_flag": ["", None],
            "signal_score": [0.0, None],
            "gate_flag": ["", None],
            "regime_signal": [0.0, None],
            "rsi": [0.0, None],
            "volume_ratio": [0.0, None],
            "kelly_fraction": [0.1, None],
            "stop_level": [0.0, None],
            "target_level": [0.0, None],
            "portfolio_value": [10000.0, None],
            "sell_reason": [None, "stop"],
        })
        result = extract_trades_from_detail("SPY", detail)

        assert len(result) == 1
        trade = result[0]
        assert trade.entry_price == 100.0
        assert trade.return_pct == 0.1
        assert trade.peak_gain == 0.1
        assert trade.peak_loss == 0.0
        assert abs(trade.pnl_usd - 100.0) < 1e-6

    def test_second_buy_while_open_is_ignored(self):
        detail = pd.DataFrame({
            "date": ["2020-01-01", "2020-01-02", "2020-01-05"],
            "trade_event": ["BUY", "BUY", "SELL"],
            "close": [100.0, 105.0, 110.0],
            "entry_price": [100.0, 105.0, None],
            "signal_flag": ["", "", None],
            "signal_score": [0.0, 0.0, None],
            "gate_flag": ["", "", None],
            "regime_signal": [0.0, 0.0, None],
            "rsi": [0.0, 0.0, None],
            "volume_ratio": [0.0, 0.0, None],
            "kelly_fraction": [0.1, 0.1, None],
            "stop_level": [0.0, 0.0, None],
            "target_level": [0.0, 0.0, None],
            "portfolio_value": [10000.0, 10000.0, None],
            "sell_reason": [None, None, "profit"],
        })
        result = extract_trades_from_detail("SPY", detail)

        assert len(result) == 1
        trade = result[0]
        assert trade.entry_price == 100.0
        assert trade.entry_signal == ""
        assert trade.date_opened == "2020-01-01"

    def test_sell_with_no_open_trade_is_ignored(self):
        detail = pd.DataFrame({
            "date": ["2020-01-01", "2020-01-02"],
            "trade_event": ["SELL", "SELL"],
            "close": [100.0, 105.0],
            "entry_price": [None, None],
            "signal_flag": [None, None],
            "signal_score": [0.0, 0.0],
            "gate_flag": [None, None],
            "regime_signal": [0.0, 0.0],
            "rsi": [0.0, 0.0],
            "volume_ratio": [0.0, 0.0],
            "kelly_fraction": [0.1, 0.1],
            "stop_level": [0.0, 0.0],
            "target_level": [0.0, 0.0],
            "portfolio_value": [10000.0, 10000.0],
            "sell_reason": [None, None],
        })
        result = extract_trades_from_detail("SPY", detail)

        assert result == []

    def test_same_day_open_close_days_held_zero(self):
        detail = pd.DataFrame({
            "date": ["2020-01-01", "2020-01-01"],
            "trade_event": ["BUY", "SELL"],
            "close": [100.0, 101.0],
            "entry_price": [100.0, None],
            "signal_flag": ["", None],
            "signal_score": [0.0, None],
            "gate_flag": ["", None],
            "regime_signal": [0.0, None],
            "rsi": [0.0, None],
            "volume_ratio": [0.0, None],
            "kelly_fraction": [0.1, None],
            "stop_level": [0.0, None],
            "target_level": [0.0, None],
            "portfolio_value": [10000.0, None],
            "sell_reason": [None, "day_trade"],
        })
        result = extract_trades_from_detail("SPY", detail)

        assert len(result) == 1
        trade = result[0]
        assert trade.days_held == 0

    def test_peak_gain_and_loss_tracking(self):
        detail = pd.DataFrame({
            "date": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"],
            "trade_event": ["BUY", None, None, None, "SELL"],
            "close": [100.0, 95.0, 115.0, 92.0, 110.0],
            "entry_price": [100.0, None, None, None, None],
            "signal_flag": ["", None, None, None, None],
            "signal_score": [0.0, None, None, None, None],
            "gate_flag": ["", None, None, None, None],
            "regime_signal": [0.0, None, None, None, None],
            "rsi": [0.0, None, None, None, None],
            "volume_ratio": [0.0, None, None, None, None],
            "kelly_fraction": [0.1, None, None, None, None],
            "stop_level": [0.0, None, None, None, None],
            "target_level": [0.0, None, None, None, None],
            "portfolio_value": [10000.0, None, None, None, None],
            "sell_reason": [None, None, None, None, "profit"],
        })
        result = extract_trades_from_detail("SPY", detail)

        assert len(result) == 1
        trade = result[0]
        assert abs(trade.peak_gain - 0.15) < 1e-9
        assert abs(trade.peak_loss - (-0.08)) < 1e-9

    def test_multiple_round_trips(self):
        detail = pd.DataFrame({
            "date": ["2020-01-01", "2020-01-05", "2020-01-10", "2020-01-15"],
            "trade_event": ["BUY", "SELL", "BUY", "SELL"],
            "close": [100.0, 110.0, 105.0, 115.0],
            "entry_price": [100.0, None, 105.0, None],
            "signal_flag": ["", None, "", None],
            "signal_score": [0.0, None, 0.0, None],
            "gate_flag": ["", None, "", None],
            "regime_signal": [0.0, None, 0.0, None],
            "rsi": [0.0, None, 0.0, None],
            "volume_ratio": [0.0, None, 0.0, None],
            "kelly_fraction": [0.1, None, 0.1, None],
            "stop_level": [0.0, None, 0.0, None],
            "target_level": [0.0, None, 0.0, None],
            "portfolio_value": [10000.0, None, 10000.0, None],
            "sell_reason": [None, "profit", None, "profit"],
        })
        result = extract_trades_from_detail("SPY", detail)

        assert len(result) == 2
        assert result[0].entry_price == 100.0
        assert result[0].exit_price == 110.0
        assert result[1].entry_price == 105.0
        assert result[1].exit_price == 115.0

    def test_missing_optional_fields_default_to_zero(self):
        detail = pd.DataFrame({
            "date": ["2020-01-01", "2020-01-05"],
            "trade_event": ["BUY", "SELL"],
            "close": [100.0, 110.0],
        })
        result = extract_trades_from_detail("SPY", detail)

        assert len(result) == 1
        trade = result[0]
        assert trade.entry_signal == ""
        assert trade.entry_score == 0.0
        assert trade.entry_gate_flag == ""
        assert trade.regime_at_entry == 0.0
        assert trade.rsi_at_entry == 0.0
        assert trade.volume_ratio == 0.0
        assert trade.kelly_fraction == 0.0
        assert trade.stop_level == 0.0
        assert trade.target_level == 0.0

    def test_strategy_and_bh_return_fields(self):
        detail = pd.DataFrame({
            "date": ["2020-01-01", "2020-01-05"],
            "trade_event": ["BUY", "SELL"],
            "close": [100.0, 110.0],
            "entry_price": [100.0, None],
            "signal_flag": ["", None],
            "signal_score": [0.0, None],
            "gate_flag": ["", None],
            "regime_signal": [0.0, None],
            "rsi": [0.0, None],
            "volume_ratio": [0.0, None],
            "kelly_fraction": [0.1, None],
            "stop_level": [0.0, None],
            "target_level": [0.0, None],
            "portfolio_value": [10000.0, None],
            "strategy_equity": [1.15, 1.2],
            "bh_equity": [1.08, 1.12],
            "sell_reason": [None, ""],
        })
        result = extract_trades_from_detail("SPY", detail)

        assert len(result) == 1
        trade = result[0]
        assert abs(trade.strategy_return - 0.2) < 1e-9
        assert abs(trade.bh_return - 0.12) < 1e-9


class TestExtractTradesFromCsv:

    def test_nonexistent_file(self):
        csv_path = Path("/nonexistent/file.csv")
        result = extract_trades_from_csv("SPY", csv_path)
        assert result == []

    def test_valid_csv_file(self, tmp_path):
        csv_path = tmp_path / "backtest.csv"
        csv_path.write_text(
            "date,trade_event,close,entry_price,signal_flag,signal_score,gate_flag,regime_signal,rsi,volume_ratio,kelly_fraction,stop_level,target_level,portfolio_value,sell_reason\n"
            "2020-01-01,BUY,100.0,100.0,mom,0.5,ok,0.8,55.0,1.2,0.1,95.0,105.0,10000.0,\n"
            "2020-01-05,SELL,110.0,,,,,,,,,,,,profit\n"
        )
        result = extract_trades_from_csv("SPY", csv_path, strategy="trend_follow")

        assert len(result) == 1
        trade = result[0]
        assert trade.ticker == "SPY"
        assert trade.strategy == "trend_follow"
        assert trade.entry_price == 100.0
        assert trade.exit_price == 110.0


class TestAppendTrades:

    def test_empty_trades_list_returns_zero(self, tmp_path):
        journal_path = tmp_path / "journal.csv"
        result = append_trades(journal_path, [])

        assert result == 0
        assert not journal_path.exists()

    def test_first_call_creates_header(self, tmp_path):
        journal_path = tmp_path / "journal.csv"
        trade = TradeRecord(date_opened="2020-01-01", ticker="SPY")
        result = append_trades(journal_path, [trade])

        assert result == 1
        assert journal_path.exists()
        content = journal_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 2
        assert lines[0].startswith("date_opened,ticker")

    def test_second_call_no_header(self, tmp_path):
        journal_path = tmp_path / "journal.csv"
        trade1 = TradeRecord(date_opened="2020-01-01", ticker="SPY")
        trade2 = TradeRecord(date_opened="2020-01-02", ticker="QQQ")

        append_trades(journal_path, [trade1])
        result = append_trades(journal_path, [trade2])

        assert result == 1
        content = journal_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 3
        assert lines[0].startswith("date_opened,ticker")

    def test_duplicate_trade_key_skipped(self, tmp_path):
        journal_path = tmp_path / "journal.csv"
        trade = TradeRecord(
            date_opened="2020-01-01",
            ticker="SPY",
            strategy="trend_follow",
            date_closed="2020-01-05",
        )

        append_trades(journal_path, [trade])
        result = append_trades(journal_path, [trade])

        assert result == 0
        content = journal_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 2

    def test_mixed_new_and_duplicate(self, tmp_path):
        journal_path = tmp_path / "journal.csv"
        trade1 = TradeRecord(
            date_opened="2020-01-01",
            ticker="SPY",
            strategy="trend",
            date_closed="2020-01-05",
        )
        trade2 = TradeRecord(
            date_opened="2020-01-02",
            ticker="QQQ",
            strategy="default",
            date_closed="2020-01-06",
        )

        append_trades(journal_path, [trade1])
        result = append_trades(journal_path, [trade1, trade2])

        assert result == 1
        content = journal_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 3

    def test_creates_parent_directories(self, tmp_path):
        journal_path = tmp_path / "subdir" / "nested" / "journal.csv"
        trade = TradeRecord(date_opened="2020-01-01", ticker="SPY")

        result = append_trades(journal_path, [trade])

        assert result == 1
        assert journal_path.exists()
        assert journal_path.parent.exists()

    def test_csv_content_has_all_fields(self, tmp_path):
        journal_path = tmp_path / "journal.csv"
        trade = TradeRecord(
            date_opened="2020-01-01",
            ticker="SPY",
            strategy="trend_follow",
            vol_filter="high",
            entry_signal="rsi_bounce",
            entry_score=0.75,
            entry_gate_flag="pass",
            entry_price=100.0,
            regime_at_entry=0.8,
            rsi_at_entry=30.0,
            volume_ratio=1.5,
            kelly_fraction=0.05,
            date_closed="2020-01-05",
            exit_reason="take_profit",
            exit_price=105.0,
            stop_level=98.0,
            target_level=110.0,
            pnl_usd=500.0,
            return_pct=0.05,
            days_held=4,
            peak_gain=0.08,
            peak_loss=-0.02,
            strategy_return=0.12,
            bh_return=0.03,
            notes="test trade",
        )

        append_trades(journal_path, [trade])

        with open(journal_path, newline="") as f:
            reader = csv.DictReader(f)
            row = next(reader)
            assert row["date_opened"] == "2020-01-01"
            assert row["ticker"] == "SPY"
            assert row["strategy"] == "trend_follow"
            assert row["vol_filter"] == "high"
            assert row["entry_signal"] == "rsi_bounce"
            assert row["entry_score"] == "0.75"
            assert row["entry_gate_flag"] == "pass"
            assert row["entry_price"] == "100.0"
            assert row["regime_at_entry"] == "0.8"
            assert row["rsi_at_entry"] == "30.0"
            assert row["volume_ratio"] == "1.5"
            assert row["kelly_fraction"] == "0.05"
            assert row["date_closed"] == "2020-01-05"
            assert row["exit_reason"] == "take_profit"
            assert row["exit_price"] == "105.0"
            assert row["stop_level"] == "98.0"
            assert row["target_level"] == "110.0"
            assert float(row["pnl_usd"]) == 500.0
            assert float(row["return_pct"]) == 0.05
            assert row["days_held"] == "4"
            assert float(row["peak_gain"]) == 0.08
            assert float(row["peak_loss"]) == -0.02
            assert float(row["strategy_return"]) == 0.12
            assert float(row["bh_return"]) == 0.03
            assert row["notes"] == "test trade"

    def test_multiple_calls_accumulate(self, tmp_path):
        journal_path = tmp_path / "journal.csv"
        trades_batch1 = [
            TradeRecord(
                date_opened="2020-01-01",
                ticker="SPY",
                strategy="trend",
                date_closed="2020-01-05",
            ),
            TradeRecord(
                date_opened="2020-01-06",
                ticker="QQQ",
                strategy="default",
                date_closed="2020-01-10",
            ),
        ]
        trades_batch2 = [
            TradeRecord(
                date_opened="2020-01-11",
                ticker="IWM",
                strategy="conservative",
                date_closed="2020-01-15",
            ),
        ]

        count1 = append_trades(journal_path, trades_batch1)
        count2 = append_trades(journal_path, trades_batch2)

        assert count1 == 2
        assert count2 == 1
        content = journal_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 4

    def test_append_with_none_values(self, tmp_path):
        journal_path = tmp_path / "journal.csv"
        trade = TradeRecord(
            date_opened="2020-01-01",
            ticker="SPY",
            entry_price=100.0,
            exit_price=0.0,
        )

        result = append_trades(journal_path, [trade])

        assert result == 1
        with open(journal_path, newline="") as f:
            reader = csv.DictReader(f)
            row = next(reader)
            assert row["exit_price"] == "0.0"

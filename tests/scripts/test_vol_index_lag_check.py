"""Tests for scripts/vol_index_lag_check.py — payload parsing only, no network."""

from __future__ import annotations

from scripts import vol_index_lag_check as check


def _yahoo_payload(closes, regular_market_time=1790972101):
    return {
        "chart": {
            "result": [
                {
                    "meta": {"regularMarketTime": regular_market_time},
                    "indicators": {"quote": [{"close": closes}]},
                }
            ]
        }
    }


class TestParseYahooChart:

    def test_returns_last_trade_epoch_and_last_close(self):
        parsed = check.parse_yahoo_chart(_yahoo_payload([21.0, 21.5, 21.2]))
        assert parsed == {"last_trade_epoch": 1790972101, "last_price": 21.2}

    def test_skips_trailing_null_closes(self):
        parsed = check.parse_yahoo_chart(_yahoo_payload([21.0, 21.5, None]))
        assert parsed["last_price"] == 21.5

    def test_all_null_closes_gives_none_price(self):
        parsed = check.parse_yahoo_chart(_yahoo_payload([None, None]))
        assert parsed["last_price"] is None


class TestParseCboeQuote:

    def test_extracts_timestamp_last_trade_and_price(self):
        payload = {
            "timestamp": "2026-10-04 10:30:06",
            "data": {"last_trade_time": "2026-10-02T00:00:00-05:00", "current_price": 21.2},
        }
        assert check.parse_cboe_quote(payload) == {
            "timestamp": "2026-10-04 10:30:06",
            "last_trade_time": "2026-10-02T00:00:00-05:00",
            "price": 21.2,
        }


class TestSnapshot:

    def test_formats_lag_in_minutes(self, monkeypatch):
        from datetime import datetime, timezone

        monkeypatch.setattr(check, "fetch_yahoo", lambda s: {"last_trade_epoch": 1000, "last_price": 21.2})
        monkeypatch.setattr(
            check,
            "fetch_cboe",
            lambda s: {"timestamp": "T", "last_trade_time": "L", "price": 21.0},
        )
        now = datetime.fromtimestamp(1000 + 900, timezone.utc)  # 15 minutes after Yahoo's last trade

        line = check.snapshot("VXN", now)

        assert "lag  15.0m" in line
        assert "yahoo" in line and "cboe" in line

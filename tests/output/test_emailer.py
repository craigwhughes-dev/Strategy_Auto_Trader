from __future__ import annotations

from unittest import mock

import pytest


class TestEmailer:

    def test_embed_chart_replaces_src(self, tmp_path):
        from Strategy_Auto_Trader.output.emailer import _embed_chart
        # Create a tiny PNG-like file
        chart_path = tmp_path / "backtest_chart.png"
        chart_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        html = '<img src="backtest_chart.png" alt="chart">'
        result = _embed_chart(html, tmp_path)
        assert 'src="data:image/png;base64,' in result
        assert 'src="backtest_chart.png"' not in result

    def test_embed_chart_no_file(self, tmp_path):
        from Strategy_Auto_Trader.output.emailer import _embed_chart
        html = '<img src="backtest_chart.png">'
        result = _embed_chart(html, tmp_path)
        assert result == html  # unchanged

    def test_embed_chart_no_match(self, tmp_path):
        from Strategy_Auto_Trader.output.emailer import _embed_chart
        html = '<img src="other.png">'
        chart_path = tmp_path / "backtest_chart.png"
        chart_path.write_bytes(b"png-data")
        result = _embed_chart(html, tmp_path)
        assert result == html  # no replacement because src doesn't match

    @mock.patch("Strategy_Auto_Trader.output.emailer._send")
    def test_send_trade_alert_with_summary_html(self, mock_send, tmp_path):
        from Strategy_Auto_Trader.output.emailer import send_trade_alert
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        summary = run_dir / "daily_summary.html"
        summary.write_text('<html><body><img src="backtest_chart.png"></body></html>')
        # No chart file -> no embedding, but should still work

        result_dict = {
            "ticker": "AAPL",
            "current_signal": "BUY",
            "close": 150.0,
            "score": 3,
            "run_dir": str(run_dir),
        }
        send_trade_alert(result_dict)
        mock_send.assert_called_once()
        call_args = mock_send.call_args
        assert "[BUY]" in call_args[0][0]  # subject
        assert "AAPL" in call_args[0][0]

    @mock.patch("Strategy_Auto_Trader.output.emailer._send")
    def test_send_trade_alert_no_summary(self, mock_send, tmp_path):
        from Strategy_Auto_Trader.output.emailer import send_trade_alert
        run_dir = tmp_path / "run_missing"
        run_dir.mkdir()

        result_dict = {
            "ticker": "GOOG",
            "current_signal": "SELL",
            "close": 2800.0,
            "score": -3,
            "run_dir": str(run_dir),
        }
        send_trade_alert(result_dict)
        mock_send.assert_called_once()
        call_args = mock_send.call_args
        assert "[SELL]" in call_args[0][0]
        assert "not found" in call_args[0][1]

    @mock.patch("Strategy_Auto_Trader.output.emailer._send")
    def test_send_daily_roundup(self, mock_send):
        from Strategy_Auto_Trader.output.emailer import send_daily_roundup
        results = [
            {"ticker": "AAPL", "current_signal": "BUY", "close": 150.0,
             "score": 3, "trade_event": "BUY", "portfolio_value": 21000,
             "strategy_return": 0.05, "bh_return": 0.03},
            {"ticker": "GOOG", "current_signal": "HOLD", "close": 2800.0,
             "score": 0, "trade_event": "", "portfolio_value": 19500,
             "strategy_return": -0.025, "bh_return": -0.01},
        ]
        failed = [{"ticker": "TSLA", "error": "timeout"}]
        send_daily_roundup(results, failed)
        mock_send.assert_called_once()
        call_args = mock_send.call_args
        subject = call_args[0][0]
        html = call_args[0][1]
        assert "Roundup" in subject
        assert "AAPL" in html
        assert "GOOG" in html
        assert "TSLA" in html  # failed ticker

    @mock.patch("Strategy_Auto_Trader.output.emailer._send")
    def test_send_daily_roundup_no_failures(self, mock_send):
        from Strategy_Auto_Trader.output.emailer import send_daily_roundup
        results = [
            {"ticker": "AAPL", "current_signal": "BUY", "close": 150.0,
             "score": 3, "trade_event": "", "portfolio_value": 20500,
             "strategy_return": 0.025, "bh_return": 0.02},
        ]
        send_daily_roundup(results, [])
        mock_send.assert_called_once()

    def test_get_smtp_creds_no_password(self):
        from Strategy_Auto_Trader.output.emailer import _get_smtp_creds
        with mock.patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="SMTP_PASSWORD"):
                _get_smtp_creds()

    def test_get_smtp_creds_with_password(self):
        from Strategy_Auto_Trader.output.emailer import _get_smtp_creds
        with mock.patch.dict("os.environ", {"SMTP_PASSWORD": "secret", "SMTP_USER": "user@example.com"}):
            user, password = _get_smtp_creds()
            assert user == "user@example.com"
            assert password == "secret"

    @mock.patch("Strategy_Auto_Trader.output.emailer._send")
    def test_send_execution_interrupted_alert_with_orders(self, mock_send):
        from Strategy_Auto_Trader.output.emailer import send_execution_interrupted_alert

        error = RuntimeError("Socket lost after place_order")
        send_execution_interrupted_alert(
            "ftse",
            error,
            buys=["AAPL x100 @ 150.0", "MSFT x50 @ 350.0"],
            sells=["GOOG x75 @ 2800.0"],
            unresolved=["TSLA", "NVDA"],
        )

        mock_send.assert_called_once()
        call_args = mock_send.call_args
        subject = call_args[0][0]
        html = call_args[0][1]

        assert "EXECUTION INTERRUPTED" in subject
        assert "ftse" in subject
        assert "3 order(s) placed" in subject  # 2 buys + 1 sell
        assert "AAPL" in html
        assert "MSFT" in html
        assert "GOOG" in html
        assert "TSLA" in html
        assert "NVDA" in html
        assert "Socket lost" in html

    @mock.patch("Strategy_Auto_Trader.output.emailer._send")
    def test_send_execution_interrupted_alert_no_sells(self, mock_send):
        from Strategy_Auto_Trader.output.emailer import send_execution_interrupted_alert

        error = TimeoutError("Connection timeout")
        send_execution_interrupted_alert(
            "sp500",
            error,
            buys=["AAPL x100 @ 150.0"],
            sells=[],
            unresolved=["MSFT", "GOOG", "TSLA"],
        )

        mock_send.assert_called_once()
        call_args = mock_send.call_args
        subject = call_args[0][0]
        html = call_args[0][1]

        assert "sp500" in subject
        assert "1 order(s) placed" in subject
        assert "AAPL" in html
        assert "MSFT" in html
        assert "Connection timeout" in html

    @mock.patch("Strategy_Auto_Trader.output.emailer._send")
    def test_send_execution_interrupted_alert_only_sells(self, mock_send):
        from Strategy_Auto_Trader.output.emailer import send_execution_interrupted_alert

        error = OSError("[Errno 10054] Connection reset by peer")
        send_execution_interrupted_alert(
            "ftse",
            error,
            buys=[],
            sells=["AAPL x100 @ 145.0", "MSFT x50 @ 340.0"],
            unresolved=["GOOG"],
        )

        mock_send.assert_called_once()
        call_args = mock_send.call_args
        subject = call_args[0][0]

        assert "2 order(s) placed" in subject
        assert "ftse" in subject

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import _rising_prices


class TestReport:

    def test_pct_formats_with_sign(self):
        from Strategy_Auto_Trader.output.report import _pct
        assert _pct(0.0532, sign=True) == "+5.3%"
        assert _pct(-0.0532, sign=True) == "-5.3%"
        assert _pct(0.0532) == "5.3%"

    def test_pct_non_finite_returns_dash(self):
        from Strategy_Auto_Trader.output.report import _pct
        assert _pct(float("nan")) == "—"

    def test_gbp_formats_with_sign(self):
        from Strategy_Auto_Trader.output.report import _gbp
        assert _gbp(1234.5, sign=True) == "+£1,234"
        assert _gbp(-1234.5, sign=True) == "-£1,234"
        assert _gbp(1234.5) == "£1,234"

    def test_f2_non_finite_returns_dash(self):
        from Strategy_Auto_Trader.output.report import _f2
        assert _f2(float("nan")) == "—"
        assert _f2(1.5) == "1.50"

    def test_build_vote_rows_html_contains_labels(self):
        from Strategy_Auto_Trader.output.report import _build_vote_rows_html
        votes = {"markov": 1, "rsi": -1, "trend": 0, "volume": 1, "hmm": -1}
        html = _build_vote_rows_html(votes, "Bull")
        assert "Markov (Bull)" in html
        assert "Trend" in html
        assert "Volume" in html
        assert "HiddenMarkovModel" in html
        assert "&#x25B2; Bull" in html  # markov=1
        assert "&#x25BC; Bear" in html  # rsi=-1
        assert "&#x25A0; Neutral" in html  # trend=0

    def test_build_trade_history_rows_html_empty(self):
        from Strategy_Auto_Trader.output.report import _build_trade_history_rows_html
        html = _build_trade_history_rows_html(pd.DataFrame())
        assert "No trades yet" in html

    def test_build_trade_history_rows_html_with_trades(self):
        from Strategy_Auto_Trader.output.report import _build_trade_history_rows_html
        df = pd.DataFrame({
            "close": [100.0, 110.0],
            "trade_event": ["BUY", "SELL"],
            "score": [2.0, -1.0],
            "sell_reason": ["", "signal"],
            "effective_stop": [0.1, None],
            "portfolio_value": [20000.0, 21000.0],
        }, index=pd.bdate_range("2024-01-01", periods=2))
        html = _build_trade_history_rows_html(df)
        assert "BUY" in html and "SELL" in html
        assert "$100.00" in html
        assert "£20,000" in html

    def test_build_stop_description_vol_scaled(self):
        from Strategy_Auto_Trader.output.report import _build_stop_description
        desc = _build_stop_description(0.15, 2.0, 20, 0.0, {})
        assert "Vol-scaled: 15.0%" in desc

    def test_build_stop_description_fixed(self):
        from Strategy_Auto_Trader.output.report import _build_stop_description
        desc = _build_stop_description(0.0, 0.0, 20, 0.30, {})
        assert desc == "Fixed: 30%"

    def test_build_stop_description_off(self):
        from Strategy_Auto_Trader.output.report import _build_stop_description
        desc = _build_stop_description(0.0, 0.0, 20, 0.0, {})
        assert desc == "Off"

    def test_build_stop_description_profit_scale_appended(self):
        from Strategy_Auto_Trader.output.report import _build_stop_description
        bt = {"config": {"profit_stop_scale": 0.5, "min_stop_pct": 0.05}}
        desc = _build_stop_description(0.0, 0.0, 20, 0.30, bt)
        assert "Profit scale: 0.5 (floor 5%)" in desc

    def test_build_stationary_dist_rows_html(self):
        from Strategy_Auto_Trader.output.report import _build_stationary_dist_rows_html
        html = _build_stationary_dist_rows_html({"Bull": 0.5, "Bear": 0.2, "Sideways": 0.3})
        assert "Bull" in html and "50.0%" in html

    def test_build_transition_matrix_html(self):
        from Strategy_Auto_Trader.output.report import _build_transition_matrix_html
        tm = np.array([[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.1, 0.1, 0.8]])
        html = _build_transition_matrix_html(tm, ["Bear", "Sideways", "Bull"])
        assert "&rarr; Bull" in html
        assert "80.0%" in html

    def test_compute_portfolio_pl_positive(self):
        from Strategy_Auto_Trader.output.report import _compute_portfolio_pl
        bt = {"initial_cash": 20000.0, "final_portfolio": 21000.0, "total_pl": 1000.0,
              "total_return_bh": 0.02}
        result = _compute_portfolio_pl(bt)
        assert result["pl_colour"] == "#69f0ae"
        assert result["bh_fp"] == pytest.approx(20400.0)
        assert result["bh_pl"] == pytest.approx(400.0)
        assert result["bh_pl_colour"] == "#69f0ae"

    def test_compute_portfolio_pl_negative(self):
        from Strategy_Auto_Trader.output.report import _compute_portfolio_pl
        bt = {"initial_cash": 20000.0, "final_portfolio": 19000.0, "total_pl": -1000.0,
              "total_return_bh": -0.02}
        result = _compute_portfolio_pl(bt)
        assert result["pl_colour"] == "#ef9a9a"
        assert result["bh_pl_colour"] == "#ef9a9a"

    def test_compute_portfolio_pl_missing_bh_return(self):
        from Strategy_Auto_Trader.output.report import _compute_portfolio_pl
        result = _compute_portfolio_pl({"initial_cash": 20000.0})
        assert not np.isfinite(result["bh_fp"])
        assert not np.isfinite(result["bh_pl"])

    def test_build_hmm_section_html_none_returns_empty(self):
        from Strategy_Auto_Trader.output.report import _build_hmm_section_html
        assert _build_hmm_section_html(None) == ""

    def test_build_hmm_section_html_with_data(self):
        from Strategy_Auto_Trader.output.report import _build_hmm_section_html
        hmm = {
            "current_regime": "Bull",
            "regime_names": ["Bear", "Sideways", "Bull"],
            "regime_means": np.array([-0.01, 0.0, 0.01]),
            "regime_vols": np.array([0.02, 0.01, 0.015]),
            "state_counts": {"Bear": 10, "Sideways": 20, "Bull": 30},
            "stationary_distribution": np.array([0.2, 0.3, 0.5]),
            "transition_matrix": np.eye(3),
            "n_seeds": 10,
            "n_converged": 8,
        }
        html = _build_hmm_section_html(hmm)
        assert "Hidden Markov Model" in html
        assert "Current HMM state" in html
        assert "Bull</span>" in html

    def test_build_exit_indicators_section_html_none_returns_empty(self):
        from Strategy_Auto_Trader.output.report import _build_exit_indicators_section_html
        assert _build_exit_indicators_section_html(None) == ""

    def test_build_exit_indicators_section_html_with_data(self):
        from Strategy_Auto_Trader.output.report import _build_exit_indicators_section_html
        ei = {
            "macd_status": "green", "rsi_status": "amber", "bb_status": "red", "atr_status": "grey",
            "warnings": ["RSI overbought"],
            "bb_width_avg": 0.1, "bb_width": 0.08, "atr_ratio": 1.2,
            "macd_histogram": 0.5, "macd_hist_pct": 0.1, "macd_label": "Bullish",
            "macd_bearish_cross": False, "macd_bullish_cross": True,
            "rsi_status_label": "OK", "rsi_exit_overbought": False, "rsi_momentum_loss": False,
            "bb_label": "Squeeze", "atr_label": "Normal", "consolidating": False,
        }
        html = _build_exit_indicators_section_html(ei)
        assert "Exit indicators" in html
        assert "RSI overbought" in html

    def test_write_daily_summary_writes_file(self, tmp_path):
        from Strategy_Auto_Trader.output.report import write_daily_summary
        from datetime import date as date_cls
        close = _rising_prices(60)
        sig = {"flag": "BUY", "score": 2.0, "max_score": 4, "votes": {"markov": 1, "rsi": 1}}
        mom = {"cur_rsi": 60.0, "rsi_label": "neutral", "cur_close": float(close.iloc[-1]),
               "above_sma20": True, "cur_sma20": float(close.iloc[-1]) * 0.98,
               "pct_from_sma20": 0.02, "above_sma50": True,
               "cur_sma50": float(close.iloc[-1]) * 0.95, "pct_from_sma50": 0.05}
        bt = {"detail": pd.DataFrame(), "initial_cash": 20000.0, "final_portfolio": 20000.0,
              "total_pl": 0.0, "total_return_bh": float("nan")}
        out_path = tmp_path / "daily_summary.html"
        write_daily_summary(
            ticker="TEST", run_date=date_cls(2024, 1, 1), close=close,
            current_state_name="Bull", markov_sig=0.5, sig=sig, mom=mom, bt=bt,
            stationary={"Bull": 0.5, "Bear": 0.2, "Sideways": 0.3},
            transition_matrix=np.eye(3), state_names=["Bear", "Sideways", "Bull"],
            eff_stop_today=0.0, vol_stop_mult=0.0, vol_stop_window=20, trailing_stop=0.30,
            out_path=out_path,
        )
        assert out_path.exists()
        html = out_path.read_text(encoding="utf-8")
        assert "TEST" in html
        assert "BUY" in html

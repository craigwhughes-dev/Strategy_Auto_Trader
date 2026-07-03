from __future__ import annotations

import pytest

from tests.conftest import _rising_prices


class TestIntegration:

    def test_momentum_to_composite_signal(self):
        """Compute momentum indicators and pass to composite_signal."""
        from Strategy_Auto_Trader.core.momentum import momentum_signals, composite_signal
        close = _rising_prices(300, daily_pct=0.005)
        mom = momentum_signals(close)
        sig = composite_signal(0.3, mom)
        assert sig["flag"] in ("BUY", "HOLD", "SELL")
        # Weighted voting (markov/rsi/trend/sma200 weights are floats), so
        # score is always a float, not an int — even for whole-number totals.
        assert isinstance(sig["score"], float)

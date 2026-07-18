"""Transaction cost models — pluggable per-side cost for the P&L cash sim.

The engine's cash simulation (`quant_engine._simulate_portfolio_value`)
deducts a cost on every BUY/SELL event. Historically that was a flat
GBP 10/side, which is ~10x real IBKR fees at this project's Kelly-sized
stakes (~GBP 2k) and distorted cross-strategy comparisons (high-frequency
strategies were over-penalised — see BACKTEST_LOG 2026-07-18 correction).

Models here implement `cost(trade_value, is_buy) -> float` (GBP). Currency
routing (UK vs US fee schedule) happens at construction via the ticker —
the engine itself stays currency-agnostic.

Fee schedule (IBKR UK, Tiered plan, from the 2026-07 pricing pages):
  GBP-denominated: 0.05% of trade value, min GBP 1.00 per order
  USD-denominated: 0.05% of trade value, min USD 1.70, max USD 39.00
UK extras on BUY orders only: 0.5% stamp duty (SDRT) and the GBP 1.00
PTM levy on trades over GBP 10,000.
The Fixed plan (min GBP 3 / USD 4) is deliberately not modelled yet —
add it here if the account turns out to be on Fixed.

The optional spread term is a crude half-spread estimate per side
(defaults: 15 bps FTSE, 5 bps US large-cap) — commission-only backtests
overstate the edge, so scans should prefer the spread variant unless
isolating commission effects.

At current stake sizes (~GBP 2k) the percentage clauses mostly resolve to
the per-order minimums; the tiered percentages matter only if stakes grow.
"""

from __future__ import annotations

#: Static USD->GBP conversion for the USD per-order min/max. The min/max
#: differ by pennies at current stakes, so a live FX feed is not warranted;
#: revisit if stakes grow to where the USD 39 cap can bind (~USD 78k).
USD_GBP = 0.79

_STAMP_DUTY_PCT = 0.005      # SDRT on UK share purchases
_PTM_LEVY_GBP = 1.00         # PTM levy, UK trades > GBP 10k
_PTM_THRESHOLD_GBP = 10_000.0

_TIER_PCT = 0.0005           # 0.05% of trade value (both currencies)
_UK_MIN_GBP = 1.00
_US_MIN_GBP = 1.70 * USD_GBP
_US_MAX_GBP = 39.00 * USD_GBP

_UK_SPREAD = 0.0015          # ~15 bps half-spread per side, FTSE
_US_SPREAD = 0.0005          # ~5 bps half-spread per side, US large-cap


class FlatCost:
    """Fixed cost per side — the engine's historical GBP 10/side behaviour."""

    def __init__(self, cost_per_side: float = 10.0) -> None:
        self._cost = float(cost_per_side)

    def cost(self, trade_value: float, is_buy: bool) -> float:
        return self._cost


class IbkrTieredCost:
    """IBKR UK Tiered commission + UK stamp duty/PTM levy, optional spread."""

    def __init__(self, ticker: str, include_spread: bool = False) -> None:
        self._uk = ticker.upper().endswith(".L")
        self._include_spread = include_spread

    def cost(self, trade_value: float, is_buy: bool) -> float:
        v = max(0.0, float(trade_value))
        if self._uk:
            c = max(_UK_MIN_GBP, _TIER_PCT * v)
            if is_buy:
                c += _STAMP_DUTY_PCT * v
                if v > _PTM_THRESHOLD_GBP:
                    c += _PTM_LEVY_GBP
            if self._include_spread:
                c += _UK_SPREAD * v
        else:
            c = min(max(_US_MIN_GBP, _TIER_PCT * v), _US_MAX_GBP)
            if self._include_spread:
                c += _US_SPREAD * v
        return c


#: CLI names for --cost-model flags.
COST_MODEL_CHOICES = ("flat", "ibkr_tiered", "ibkr_tiered_spread")


def make_cost_model(name: str, ticker: str, trade_cost: float = 10.0):
    """Build a cost model from its CLI name.

    "flat" reproduces the historical fixed trade_cost/side exactly (the
    engine's parity baseline); the ibkr variants need the ticker for
    UK/US fee routing.
    """
    if name == "flat":
        return FlatCost(trade_cost)
    if name == "ibkr_tiered":
        return IbkrTieredCost(ticker, include_spread=False)
    if name == "ibkr_tiered_spread":
        return IbkrTieredCost(ticker, include_spread=True)
    raise ValueError(f"Unknown cost model '{name}'. Choices: {COST_MODEL_CHOICES}")

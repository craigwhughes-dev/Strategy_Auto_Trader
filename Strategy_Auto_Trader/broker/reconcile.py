"""Position reconciliation — compare internal execution state against the broker.

Both internal and broker positions are keyed by yfinance ticker ("HSBA.L", "SPY").
get_open_positions() returns broker positions already keyed by yfinance ticker,
matching the internal positions dict key space directly.
"""

from __future__ import annotations

import math

from .types import QUANTITY_DECIMALS, round_quantity


def _fmt_qty(quantity: float) -> str:
    """5 dp without trailing zeros, so 5.0 reads as 5 and 27.017 as 27.017 (discrepancy text is compared for alert dedup)."""
    return f"{quantity:.{QUANTITY_DECIMALS}f}".rstrip("0").rstrip(".")


def _expected_quantities(internal_positions: dict[str, dict]) -> dict[str, float]:
    """Internal holding per ticker, summed across any duplicate entries."""
    expected: dict[str, float] = {}
    for ticker, pos in internal_positions.items():
        expected[ticker] = round_quantity(expected.get(ticker, 0) + pos.get("quantity", 0))
    return expected


def reconcile_positions(
    internal_positions: dict[str, dict],
    broker_positions: dict[str, int],
) -> list[str]:
    """Compare internal positions against broker positions.

    Both dicts are keyed by yfinance ticker. Returns a list of human-readable
    discrepancy strings; empty means the two states agree. Never mutates either
    side — surfacing is the caller's job.
    """
    expected = _expected_quantities(internal_positions)

    discrepancies: list[str] = []
    for ticker in sorted(expected):
        exp_qty = expected[ticker]
        held = broker_positions.get(ticker)
        if held is None:
            discrepancies.append(
                f"{ticker}: internal state shows {_fmt_qty(exp_qty)} shares, "
                f"broker shows no position"
            )
        elif round_quantity(held) != exp_qty:
            discrepancies.append(
                f"{ticker}: internal state shows {_fmt_qty(exp_qty)} shares, "
                f"broker shows {_fmt_qty(round_quantity(held))}"
            )

    return discrepancies


def discrepancy_values_gbp(
    internal_positions: dict[str, dict],
    broker_positions: dict[str, int],
    broker: object,
    logger: object | None = None,
) -> dict[str, float]:
    """£ value of each discrepant ticker's quantity drift, for materiality filtering.

    A position missing entirely at the broker (internal thinks we hold shares, broker
    shows none) is always math.inf — that signals a position vanished or was closed
    without us knowing, a far more serious case than a small quantity drift, and must
    never be filtered out by a £ threshold. Same for any pricing failure: if a
    discrepancy can't be priced, it can't be judged safe to ignore.
    """
    from .symbols import sizing_price

    expected = _expected_quantities(internal_positions)

    values: dict[str, float] = {}
    for ticker in sorted(expected):
        exp_qty = expected[ticker]
        held = broker_positions.get(ticker)
        if held is None:
            values[ticker] = math.inf
            continue
        held = round_quantity(held)
        if held == exp_qty:
            continue
        try:
            price = sizing_price(ticker, broker.get_last_price(ticker))
            values[ticker] = abs(held - exp_qty) * price
        except Exception as e:
            if logger is not None:
                logger.warning(f"Reconciliation: could not price {ticker} discrepancy for materiality check: {e}")
            values[ticker] = math.inf
    return values


def check_stop_fills_for_missing_positions(
    internal_positions: dict[str, dict],
    broker_positions: dict[str, int],
    broker: object,
    portfolio: object,
) -> list[str]:
    """Check for protective stop fills on positions missing at broker.

    For each internal position missing at the broker, if its protective stop
    is no longer open, try to fetch the execution. If found, record as stop_loss.
    If not found, record as reconciled_stop_loss at the estimated stop price.

    Returns list of explanations for positions resolved via stop fill (these
    should be removed from reconciliation discrepancies).
    """
    resolved: list[str] = []
    expected: dict[str, dict] = {}
    for ticker, pos in internal_positions.items():
        entry = expected.setdefault(ticker, {"quantity": 0, "pos": pos})
        entry["quantity"] = round_quantity(entry["quantity"] + pos.get("quantity", 0))

    # Fetch open stops once at the start; if this fails, return empty (cannot safely resolve).
    try:
        open_stops = broker.get_open_stop_orders()
    except Exception:
        return resolved

    for ticker in sorted(expected):
        exp = expected[ticker]
        pos = exp["pos"]
        held = broker_positions.get(ticker)

        if held is not None:
            continue

        perm_id = pos.get("stop_perm_id")
        if not perm_id:
            continue

        if perm_id in open_stops:
            continue

        try:
            fill = broker.get_stop_fill(perm_id)
            if fill is not None:
                from .symbols import PENCE_PER_POUND, normalize_fill_price
                from .types import FillResult
                # Stop fills for LSE arrive in ambiguous units (pence or
                # pounds) — disambiguate against the stored stop price
                # (pot currency, so x100 gives the pence reference).
                ref_pence = float(pos.get("stop_price") or 0.0) * PENCE_PER_POUND
                fill = FillResult(
                    ticker=fill.ticker, action=fill.action,
                    fill_price=normalize_fill_price(ticker, fill.fill_price, ref_pence),
                    quantity=fill.quantity, timestamp=fill.timestamp,
                )
                portfolio.record_exit(ticker, fill, exit_type="stop_loss")
                resolved.append(
                    f"{ticker}: protective stop filled @ {fill.fill_price}, "
                    f"position closed by backstop"
                )
                continue
        except Exception:
            pass

        stop_price = pos.get("stop_price") or 0
        if stop_price > 0:
            from .types import FillResult
            estimated_fill = FillResult(
                ticker=ticker, action="SELL", fill_price=stop_price,
                quantity=round_quantity(pos.get("quantity", 0)),
                timestamp="",
            )
            portfolio.record_exit(ticker, estimated_fill, exit_type="reconciled_stop_loss")
            resolved.append(
                f"{ticker}: protective stop vanished, position closed at "
                f"estimated stop price {stop_price} (multi-day outage)"
            )

    return resolved

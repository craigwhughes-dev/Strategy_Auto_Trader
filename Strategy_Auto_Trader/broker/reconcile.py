"""Position reconciliation — compare internal execution state against the broker.

Both internal and broker positions are keyed by yfinance ticker ("HSBA.L", "SPY").
get_open_positions() returns broker positions already keyed by yfinance ticker,
matching the internal positions dict key space directly.
"""

from __future__ import annotations


def reconcile_positions(
    internal_positions: dict[str, dict],
    broker_positions: dict[str, int],
) -> list[str]:
    """Compare internal positions against broker positions.

    Both dicts are keyed by yfinance ticker. Returns a list of human-readable
    discrepancy strings; empty means the two states agree. Never mutates either
    side — surfacing is the caller's job.
    """
    expected: dict[str, dict] = {}
    for ticker, pos in internal_positions.items():
        entry = expected.setdefault(ticker, {"quantity": 0})
        entry["quantity"] += int(pos.get("quantity", 0))

    discrepancies: list[str] = []
    for ticker in sorted(expected):
        exp = expected[ticker]
        held = broker_positions.get(ticker)
        if held is None:
            discrepancies.append(
                f"{ticker}: internal state shows {exp['quantity']} shares, "
                f"broker shows no position"
            )
        elif held != exp["quantity"]:
            discrepancies.append(
                f"{ticker}: internal state shows {exp['quantity']} shares, "
                f"broker shows {held}"
            )

    for ticker in sorted(broker_positions):
        if ticker not in expected:
            discrepancies.append(
                f"{ticker}: broker shows {broker_positions[ticker]} shares, "
                f"no internal position"
            )

    return discrepancies


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
        entry["quantity"] += int(pos.get("quantity", 0))

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
                quantity=int(pos.get("quantity", 0)),
                timestamp="",
            )
            portfolio.record_exit(ticker, estimated_fill, exit_type="reconciled_stop_loss")
            resolved.append(
                f"{ticker}: protective stop vanished, position closed at "
                f"estimated stop price {stop_price} (multi-day outage)"
            )

    return resolved

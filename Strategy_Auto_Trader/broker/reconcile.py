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

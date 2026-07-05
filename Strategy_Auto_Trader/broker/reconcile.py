"""Position reconciliation — compare internal execution state against the broker.

Internal positions are keyed by yfinance ticker ("HSBA.L"); IBKR reports bare
contract symbols ("HSBA", "BT.A"), so internal tickers are mapped through
ibkr_contract_params before comparing.
"""

from __future__ import annotations

from .symbols import ibkr_contract_params


def reconcile_positions(
    internal_positions: dict[str, dict],
    broker_positions: dict[str, int],
) -> list[str]:
    """Compare internal positions against broker positions.

    Returns a list of human-readable discrepancy strings; empty means the two
    states agree. Never mutates either side — surfacing is the caller's job.
    """
    expected: dict[str, dict] = {}
    for ticker, pos in internal_positions.items():
        symbol = ibkr_contract_params(ticker)[0]
        entry = expected.setdefault(symbol, {"tickers": [], "quantity": 0})
        entry["tickers"].append(ticker)
        entry["quantity"] += int(pos.get("quantity", 0))

    discrepancies: list[str] = []
    for symbol in sorted(expected):
        exp = expected[symbol]
        label = "/".join(sorted(exp["tickers"]))
        held = broker_positions.get(symbol)
        if held is None:
            discrepancies.append(
                f"{label}: internal state shows {exp['quantity']} shares, "
                f"broker shows no position"
            )
        elif held != exp["quantity"]:
            discrepancies.append(
                f"{label}: internal state shows {exp['quantity']} shares, "
                f"broker shows {held}"
            )

    for symbol in sorted(broker_positions):
        if symbol not in expected:
            discrepancies.append(
                f"{symbol}: broker shows {broker_positions[symbol]} shares, "
                f"no internal position"
            )

    return discrepancies

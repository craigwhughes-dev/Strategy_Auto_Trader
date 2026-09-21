import logging
from datetime import datetime

import pytest

from Strategy_Auto_Trader.core.cli_logging import (
    _OvernightGatewayFilter,
    install_overnight_gateway_filter,
)


def _record(level: int, msg: str) -> logging.LogRecord:
    return logging.LogRecord("t", level, __file__, 1, msg, None, None)


def _filter_at(hour: int, minute: int = 0) -> _OvernightGatewayFilter:
    return _OvernightGatewayFilter(now_fn=lambda: datetime(2026, 9, 21, hour, minute))


GATEWAY_MSGS = [
    "Startup reconciliation could not reach broker — will retry next poll",
    "Reconciliation: broker connect failed: [WinError 10061] refused",
    "Error in IBKR data reconciliation (unreachable for ~5m so far): boom",
    "IBKR data reconciliation giving up for today after 3 failures",
    "Auto-bouncing IBC after 10 consecutive failures — running bounce_ibc.ps1",
    "API connection failed: ConnectionRefusedError(1225, 'remote refused')",
]


@pytest.mark.parametrize("msg", GATEWAY_MSGS)
@pytest.mark.parametrize("level", [logging.WARNING, logging.ERROR, logging.CRITICAL])
def test_gateway_records_dropped_inside_window(msg, level):
    assert _filter_at(2).filter(_record(level, msg)) is False


@pytest.mark.parametrize("hour,minute,dropped", [
    (0, 0, True),    # window opens at midnight
    (3, 59, True),
    (4, 0, False),   # end is exclusive
    (23, 59, False),
    (12, 0, False),
])
def test_window_boundaries(hour, minute, dropped):
    keep = _filter_at(hour, minute).filter(_record(logging.ERROR, GATEWAY_MSGS[0]))
    assert keep is (not dropped)


def test_gateway_records_kept_outside_window():
    assert _filter_at(10).filter(_record(logging.ERROR, GATEWAY_MSGS[1])) is True


def test_non_gateway_errors_kept_inside_window():
    assert _filter_at(2).filter(_record(logging.ERROR, "RECONCILIATION MISMATCH (1 discrepancies)")) is True


def test_info_and_debug_kept_inside_window():
    f = _filter_at(2)
    assert f.filter(_record(logging.INFO, GATEWAY_MSGS[0])) is True
    assert f.filter(_record(logging.DEBUG, GATEWAY_MSGS[0])) is True


def test_install_is_idempotent():
    h = logging.StreamHandler()
    install_overnight_gateway_filter(h)
    install_overnight_gateway_filter(h)
    assert sum(isinstance(x, _OvernightGatewayFilter) for x in h.filters) == 1

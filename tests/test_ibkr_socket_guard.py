"""Tests for the conftest socket guard that keeps tests off the real IBKR Gateway."""

from __future__ import annotations

import socket

import pytest

from tests.conftest import IBKRConnectBlocked


@pytest.mark.parametrize("port", [4001, 4002, 7496, 7497])
def test_connect_to_ibkr_port_is_blocked(port):
    s = socket.socket()
    try:
        with pytest.raises(IBKRConnectBlocked):
            s.connect(("127.0.0.1", port))
    finally:
        s.close()


def test_connect_ex_to_ibkr_port_is_blocked():
    s = socket.socket()
    try:
        with pytest.raises(IBKRConnectBlocked):
            s.connect_ex(("127.0.0.1", 4002))
    finally:
        s.close()


def test_create_connection_to_ibkr_port_is_blocked():
    with pytest.raises(IBKRConnectBlocked):
        socket.create_connection(("127.0.0.1", 4002), timeout=1)


def test_non_ibkr_port_passes_through_to_real_connect():
    # Bind a throwaway local listener so the connect succeeds without touching the network.
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    client = socket.socket()
    try:
        client.connect(server.getsockname())
    finally:
        client.close()
        server.close()


def test_guard_active_by_default():
    assert socket.socket.connect.__name__ == "guarded_connect"


@pytest.mark.allow_ibkr_connect
def test_marker_disables_guard():
    assert socket.socket.connect.__name__ == "connect"

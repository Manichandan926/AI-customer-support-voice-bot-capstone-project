"""Shared test setup: the network is simulated as reachable by default, so no
test ever depends on real connectivity. Tests about going offline override
network._connect themselves."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import network


class _Conn:
    def close(self):
        pass


@pytest.fixture(autouse=True)
def simulated_network(monkeypatch):
    network.reset()
    monkeypatch.setattr(network, "_connect", lambda *a, **k: _Conn())
    yield
    network.reset()

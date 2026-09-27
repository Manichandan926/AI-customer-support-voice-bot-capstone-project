"""Is the internet reachable? Decides between online services (Google speech
recognition, neural Edge voices) and offline models (Vosk, Piper).

The answer is cached for a short while: checking before every utterance
would add latency, and a dead network must cost at most one short timeout,
not one per reply. When an online call fails anyway, callers report it with
mark_offline() so the next turn goes straight to the offline path.
"""

import socket
import time

import config

_state = {"checked_at": float("-inf"), "online": False}
# Module-level handle so tests can simulate connectivity without replacing
# socket.create_connection for everything else (urllib, HTTP servers).
_connect = socket.create_connection


def is_online(force: bool = False) -> bool:
    if config.OFFLINE_MODE == "always":
        return False
    if not force and time.monotonic() - _state["checked_at"] < config.NET_CHECK_TTL:
        return _state["online"]
    try:
        _connect((config.NET_CHECK_HOST, 443), timeout=config.NET_CHECK_TIMEOUT).close()
        online = True
    except OSError:
        online = False
    _state.update(checked_at=time.monotonic(), online=online)
    return online


def mark_offline() -> None:
    _state.update(checked_at=time.monotonic(), online=False)


def reset() -> None:
    """Forget the cached answer (used by tests)."""
    _state.update(checked_at=float("-inf"), online=False)

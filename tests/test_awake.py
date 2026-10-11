"""keep_awake never breaks a run, on any platform."""
import sys

from notbydefault import awake


def test_keep_awake_is_a_safe_context_manager():
    with awake.keep_awake() as active:
        assert isinstance(active, bool)


def test_keep_awake_is_a_no_op_off_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    with awake.keep_awake() as active:
        assert active is False


def test_keep_awake_survives_a_broken_power_api(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    import ctypes
    monkeypatch.setattr(ctypes, "windll", None, raising=False)   # attribute access fails
    with awake.keep_awake() as active:
        assert active is False

"""Keep the machine from idling into sleep while a long run (eval, check) is working.

A laptop that enters (Modern) Standby freezes or throttles the process: a run "hangs" for as
long as the machine sleeps. On Windows this keeps the system and display awake (on Modern
Standby machines the screen turning off is what starts standby) and registers an
"execution required" power request, until the run ends. Elsewhere it does nothing. It never
blocks the run: any failure is ignored. Closing the lid or pressing the power button still
sleeps the machine.
"""
from __future__ import annotations

import contextlib
import sys

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002
POWER_REQUEST_EXECUTION_REQUIRED = 3
POWER_REQUEST_CONTEXT_VERSION = 0
POWER_REQUEST_CONTEXT_SIMPLE_STRING = 0x1
REASON = "NotByDefault: eval/check run in progress"


def _power_request(kernel32):
    """A PowerCreateRequest handle with ExecutionRequired set, or None."""
    import ctypes
    from ctypes import wintypes

    class REASON_CONTEXT(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("Flags", wintypes.DWORD),
                    ("SimpleReasonString", wintypes.LPWSTR)]
    kernel32.PowerCreateRequest.restype = wintypes.HANDLE
    kernel32.PowerCreateRequest.argtypes = [ctypes.POINTER(REASON_CONTEXT)]
    kernel32.PowerSetRequest.argtypes = [wintypes.HANDLE, ctypes.c_int]
    ctx = REASON_CONTEXT(POWER_REQUEST_CONTEXT_VERSION, POWER_REQUEST_CONTEXT_SIMPLE_STRING,
                         REASON)
    handle = kernel32.PowerCreateRequest(ctypes.byref(ctx))
    if not handle or handle == wintypes.HANDLE(-1).value:
        return None
    if not kernel32.PowerSetRequest(handle, POWER_REQUEST_EXECUTION_REQUIRED):
        kernel32.CloseHandle(handle)
        return None
    return handle


@contextlib.contextmanager
def keep_awake():
    active, handle, kernel32 = False, None, None
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            active = bool(kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED))
            handle = _power_request(kernel32)
        except Exception:
            pass
    try:
        yield active or handle is not None
    finally:
        if kernel32 is not None:
            try:
                if handle:
                    from ctypes import wintypes
                    kernel32.PowerClearRequest.argtypes = [wintypes.HANDLE, ctypes.c_int]
                    kernel32.PowerClearRequest(handle, POWER_REQUEST_EXECUTION_REQUIRED)
                    kernel32.CloseHandle(handle)
                if active:
                    kernel32.SetThreadExecutionState(ES_CONTINUOUS)
            except Exception:
                pass

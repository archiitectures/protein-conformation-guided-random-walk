"""Safe release helpers for hosted Google Colab runtimes."""

from __future__ import annotations

from typing import Any


def release_colab_runtime(runtime_module: Any | None = None) -> bool:
    """Disconnect and delete the current managed Colab runtime.

    Call this only after durable output verification. Outside hosted Colab the
    function is a no-op and returns ``False``.
    """
    if runtime_module is None:
        try:
            from google.colab import runtime as runtime_module
        except ImportError:
            return False
    runtime_module.unassign()
    return True

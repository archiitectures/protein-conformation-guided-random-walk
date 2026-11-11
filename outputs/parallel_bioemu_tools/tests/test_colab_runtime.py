from __future__ import annotations

from parallel_bioemu_tools.colab_runtime import release_colab_runtime


class FakeRuntime:
    def __init__(self) -> None:
        self.released = False

    def unassign(self) -> None:
        self.released = True


def test_release_colab_runtime_calls_unassign() -> None:
    runtime = FakeRuntime()
    assert release_colab_runtime(runtime) is True
    assert runtime.released is True

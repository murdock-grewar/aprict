
# A do-nothing context manager
class EmptyContextManager:
    """A context manager that does nothing."""

    def __enter__(self) -> None:  # noqa: D105
        pass

    def __exit__(self, exc_type, exc_value, exc_tb) -> None:  # noqa: ANN001, D105
        pass

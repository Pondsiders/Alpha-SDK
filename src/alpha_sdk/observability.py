"""Observability setup - placeholder.

Previously configured Logfire for tracing and logging.
Now a no-op retained for import compatibility.
"""


def configure(service_name: str = "alpha_sdk", debug: bool = False) -> None:
    """Configure observability (no-op).

    Args:
        service_name: Name to identify this service in traces.
        debug: If True, also log to console. Default False (quiet mode).
    """
    pass

"""Observability setup - Logfire configuration.

Centralizes logging and tracing configuration so all
alpha_sdk consumers get consistent observability.

We use logfire.info/warn/error/debug directly instead of
Python's logging module. This ensures all logs are properly
associated with the current span context.
"""

import logfire


def configure(service_name: str = "alpha_sdk", debug: bool = False) -> None:
    """Configure Logfire for observability.

    Args:
        service_name: Name to identify this service in traces.
        debug: If True, also log to console. Default False (quiet mode).
    """
    logfire.configure(
        service_name=service_name,
        distributed_tracing=True,
        scrubbing=False,  # Too aggressive, redacts normal words
        send_to_logfire="if-token-present",
        console=debug,  # Only show console output in debug mode
    )

    # Instrument httpx for trace propagation
    logfire.instrument_httpx()

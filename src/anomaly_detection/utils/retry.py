"""Retry utilities with exponential backoff."""

import asyncio
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import ParamSpec, TypeVar

from anomaly_detection.utils.logging import get_logger

log = get_logger("retry")

P = ParamSpec("P")
T = TypeVar("T")


def retry_async(
    max_attempts: int = 3,
    delay_seconds: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorator for retrying async functions with exponential backoff."""

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Exception | None = None
            delay = delay_seconds

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        log.warning(
                            "retrying",
                            func=func.__name__,
                            attempt=attempt,
                            max_attempts=max_attempts,
                            delay=delay,
                            error=str(e),
                        )
                        await asyncio.sleep(delay)
                        delay *= backoff_factor
                    else:
                        log.error(
                            "all retries failed",
                            func=func.__name__,
                            error=str(e),
                        )

            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected retry state")  # pragma: no cover

        return wrapper

    return decorator

"""Tests for retry utilities."""


import pytest

from anomaly_detection.utils.retry import retry_async


class TestRetryAsync:
    """Tests for retry_async decorator."""

    @pytest.mark.asyncio
    async def test_success_no_retry(self) -> None:
        """Should return immediately on success."""
        call_count = 0

        @retry_async(max_attempts=3)
        async def successful_func() -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        result = await successful_func()

        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_then_success(self) -> None:
        """Should retry and succeed after failures."""
        call_count = 0

        @retry_async(max_attempts=3, delay_seconds=0.01)
        async def failing_then_success() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("temporary failure")
            return "success"

        result = await failing_then_success()

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_all_retries_fail(self) -> None:
        """Should raise after all retries exhausted."""
        call_count = 0

        @retry_async(max_attempts=3, delay_seconds=0.01)
        async def always_fails() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("permanent failure")

        with pytest.raises(ValueError, match="permanent failure"):
            await always_fails()

        assert call_count == 3

    @pytest.mark.asyncio
    async def test_specific_exception_type(self) -> None:
        """Should only retry specified exception types."""
        call_count = 0

        @retry_async(max_attempts=3, delay_seconds=0.01, exceptions=(ValueError,))
        async def wrong_exception() -> str:
            nonlocal call_count
            call_count += 1
            raise TypeError("not retried")

        with pytest.raises(TypeError):
            await wrong_exception()

        # Should not retry TypeError
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_backoff_factor(self) -> None:
        """Should apply exponential backoff."""
        import time

        call_times: list[float] = []

        @retry_async(max_attempts=3, delay_seconds=0.05, backoff_factor=2.0)
        async def track_timing() -> str:
            call_times.append(time.time())
            if len(call_times) < 3:
                raise ValueError("retry me")
            return "done"

        await track_timing()

        # First retry delay: 0.05s, second: 0.1s
        # Allow some tolerance for timing
        if len(call_times) >= 2:
            first_delay = call_times[1] - call_times[0]
            assert first_delay >= 0.04  # ~0.05s

        if len(call_times) >= 3:
            second_delay = call_times[2] - call_times[1]
            assert second_delay >= 0.08  # ~0.1s (backoff applied)

    @pytest.mark.asyncio
    async def test_preserves_function_metadata(self) -> None:
        """Should preserve the wrapped function's name and docstring."""

        @retry_async(max_attempts=2)
        async def my_function() -> str:
            """My docstring."""
            return "value"

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    @pytest.mark.asyncio
    async def test_with_args_and_kwargs(self) -> None:
        """Should pass through args and kwargs correctly."""

        @retry_async(max_attempts=2)
        async def add(a: int, b: int, multiplier: int = 1) -> int:
            return (a + b) * multiplier

        result = await add(2, 3, multiplier=2)
        assert result == 10

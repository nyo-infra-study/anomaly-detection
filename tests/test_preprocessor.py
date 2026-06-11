"""Tests for data preprocessing."""

import numpy as np

from anomaly_detection.detector.preprocessor import (
    normalize,
    pad_or_truncate,
    prometheus_to_array,
)


class TestPrometheusToArray:
    def test_empty_result(self) -> None:
        data, names = prometheus_to_array([])
        assert data.shape == (0,)
        assert names == []

    def test_single_series(self) -> None:
        result = [
            {
                "metric": {"__name__": "http_requests", "job": "api"},
                "values": [[1000, "1.0"], [1060, "2.0"], [1120, "3.0"]],
            }
        ]
        data, names = prometheus_to_array(result)
        assert data.shape == (1, 3)
        assert names == ['http_requests{job="api"}']
        np.testing.assert_array_equal(data[0], [1.0, 2.0, 3.0])

    def test_multiple_series(self) -> None:
        result = [
            {"metric": {"__name__": "m1"}, "values": [[1, "1"], [2, "2"]]},
            {"metric": {"__name__": "m2"}, "values": [[1, "3"], [2, "4"]]},
        ]
        data, names = prometheus_to_array(result)
        assert data.shape == (2, 2)
        assert names == ["m1", "m2"]

    def test_with_fixture(self, prometheus_response: list[dict]) -> None:
        data, names = prometheus_to_array(prometheus_response)
        assert data.shape == (1, 96)
        assert len(names) == 1


class TestNormalize:
    def test_basic_normalization(self) -> None:
        data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        norm, means, stds = normalize(data)

        # Mean should be ~0, std should be ~1
        np.testing.assert_almost_equal(norm.mean(), 0, decimal=5)
        np.testing.assert_almost_equal(norm.std(), 1, decimal=5)

    def test_constant_series(self) -> None:
        # Edge case: all values the same
        data = np.array([[5.0, 5.0, 5.0, 5.0]])
        norm, means, stds = normalize(data)

        # Should not produce NaN
        assert not np.isnan(norm).any()
        np.testing.assert_array_equal(norm, [[0, 0, 0, 0]])

    def test_multiple_series(self) -> None:
        data = np.array(
            [
                [1.0, 2.0, 3.0],
                [10.0, 20.0, 30.0],
            ]
        )
        norm, means, stds = normalize(data)

        assert norm.shape == data.shape
        # Each row should be normalized independently
        np.testing.assert_almost_equal(norm[0].mean(), 0, decimal=5)
        np.testing.assert_almost_equal(norm[1].mean(), 0, decimal=5)


class TestPadOrTruncate:
    def test_exact_length(self) -> None:
        data = np.array([[1, 2, 3]])
        result = pad_or_truncate(data, 3)
        np.testing.assert_array_equal(result, data)

    def test_truncate(self) -> None:
        data = np.array([[1, 2, 3, 4, 5]])
        result = pad_or_truncate(data, 3)
        np.testing.assert_array_equal(result, [[3, 4, 5]])  # Keep most recent

    def test_pad(self) -> None:
        data = np.array([[1, 2, 3]])
        result = pad_or_truncate(data, 5)
        np.testing.assert_array_equal(result, [[0, 0, 1, 2, 3]])  # Pad at start

    def test_multiple_series(self) -> None:
        data = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
            ]
        )
        result = pad_or_truncate(data, 5)
        expected = np.array(
            [
                [0, 0, 1, 2, 3],
                [0, 0, 4, 5, 6],
            ]
        )
        np.testing.assert_array_equal(result, expected)

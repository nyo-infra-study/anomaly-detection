"""Data preprocessing for TimesNet input."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from anomaly_detection.utils.logging import get_logger

log = get_logger("preprocessor")


def prometheus_to_array(
    result: list[dict[str, Any]],
) -> tuple[NDArray[np.floating[Any]], list[str]]:
    """
    Convert Prometheus query_range result to numpy array.

    Args:
        result: List of {"metric": {...}, "values": [[ts, val], ...]}

    Returns:
        (data, metric_names)
        - data: shape (num_metrics, num_timestamps)
        - metric_names: list of metric identifiers
    """
    if not result:
        return np.array([]), []

    # Extract metric names (for labeling output)
    metric_names = []
    for series in result:
        labels = series["metric"]
        # Create readable name from labels
        name = labels.get("__name__", "unknown")
        extra = ",".join(f'{k}="{v}"' for k, v in labels.items() if k != "__name__")
        metric_names.append(f"{name}{{{extra}}}" if extra else name)

    # Convert to numpy
    # Assume all series have same timestamps (Prometheus guarantees this for range queries)
    arrays = []
    for series in result:
        values = [float(v[1]) for v in series["values"]]
        arrays.append(values)

    data = np.array(arrays, dtype=np.float32)
    log.debug("preprocessed", shape=data.shape, metrics=len(metric_names))

    return data, metric_names


def normalize(
    data: NDArray[np.floating[Any]],
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """
    Instance normalization (per-metric, per-window).

    Returns:
        (normalized, means, stds)
    """
    means = data.mean(axis=1, keepdims=True)
    stds = data.std(axis=1, keepdims=True)
    stds = np.where(stds == 0, 1.0, stds)  # Avoid division by zero

    normalized = (data - means) / stds
    return normalized, means.squeeze(), stds.squeeze()


def pad_or_truncate(
    data: NDArray[np.floating[Any]], target_length: int
) -> NDArray[np.floating[Any]]:
    """Ensure data has exactly target_length timestamps."""
    current_length = data.shape[1]

    if current_length == target_length:
        return data
    elif current_length > target_length:
        # Truncate (keep most recent)
        return data[:, -target_length:]
    else:
        # Pad with zeros at the start
        padding = np.zeros((data.shape[0], target_length - current_length), dtype=data.dtype)
        return np.concatenate([padding, data], axis=1)

"""Detector modules."""

from pathlib import Path
from typing import TYPE_CHECKING, Union

from anomaly_detection.config import settings
from anomaly_detection.utils.logging import get_logger

if TYPE_CHECKING:
    from anomaly_detection.detector.hybrid import HybridDetector
    from anomaly_detection.detector.mock import MockDetector
    from anomaly_detection.detector.statistical import (
        RollingStatisticalDetector,
        StatisticalDetector,
    )
    from anomaly_detection.detector.timesnet import TimesNetDetector

log = get_logger("detector")

DetectorType = Union[
    "MockDetector",
    "TimesNetDetector",
    "StatisticalDetector",
    "RollingStatisticalDetector",
    "HybridDetector",
]


def get_detector(detector_type: str | None = None) -> DetectorType:
    """
    Factory: return detector based on config or explicit type.

    Args:
        detector_type: One of "timesnet", "statistical", "rolling", "mock", or None.
                      If None, auto-detect based on model file existence.

    Returns:
        Detector instance (call .load() before use)
    """
    # Use env var or argument
    dtype = detector_type or settings.detector_type

    if dtype == "statistical":
        from anomaly_detection.detector.statistical import StatisticalDetector

        log.info("using statistical detector (z-score based)")
        return StatisticalDetector(z_threshold=settings.z_threshold)

    elif dtype == "rolling":
        from anomaly_detection.detector.statistical import RollingStatisticalDetector

        log.info("using rolling statistical detector (EMA-based)")
        return RollingStatisticalDetector(
            alpha=settings.ema_alpha,
            z_threshold=settings.z_threshold,
        )

    elif dtype == "timesnet":
        from anomaly_detection.detector.timesnet import TimesNetDetector

        log.info("using TimesNet detector", model_path=settings.model_path)
        return TimesNetDetector()

    elif dtype == "hybrid":
        from anomaly_detection.detector.hybrid import HybridDetector

        log.info(
            "using hybrid detector",
            critical_services=settings.critical_services or "(none)",
        )
        return HybridDetector()

    elif dtype == "mock":
        from anomaly_detection.detector.mock import MockDetector

        log.info("using mock detector")
        return MockDetector()

    else:
        # Auto-detect: use TimesNet if model exists, else statistical
        model_path = Path(settings.model_path)

        if model_path.exists():
            from anomaly_detection.detector.timesnet import TimesNetDetector

            log.info("auto-detected TimesNet model", model_path=str(model_path))
            return TimesNetDetector()
        else:
            from anomaly_detection.detector.statistical import StatisticalDetector

            log.warning(
                "no model found, falling back to statistical detector",
                expected_path=str(model_path),
            )
            return StatisticalDetector(z_threshold=settings.z_threshold)

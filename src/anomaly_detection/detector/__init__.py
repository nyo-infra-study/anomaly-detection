"""Detector modules."""

from pathlib import Path
from typing import TYPE_CHECKING

from anomaly_detection.config import settings
from anomaly_detection.utils.logging import get_logger

if TYPE_CHECKING:
    from anomaly_detection.detector.mock import MockDetector
    from anomaly_detection.detector.timesnet import TimesNetDetector

log = get_logger("detector")


def get_detector() -> "MockDetector | TimesNetDetector":
    """Factory: return real TimesNet if model exists, else mock."""
    model_path = Path(settings.model_path)

    if model_path.exists():
        from anomaly_detection.detector.timesnet import TimesNetDetector

        log.info("using TimesNet detector", model_path=str(model_path))
        return TimesNetDetector()
    else:
        from anomaly_detection.detector.mock import MockDetector

        log.warning(
            "model not found, using mock detector",
            expected_path=str(model_path),
        )
        return MockDetector()
        return MockDetector()

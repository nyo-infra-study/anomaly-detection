"""Hybrid detector — TimesNet for critical services, statistical for the rest.

This implements the tiered approach from base knowledge:
- Tier 1: Critical services → TimesNet (cross-metric correlation, more accurate)
- Tier 2: Long tail → Statistical (simple, fast, no training needed)
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from anomaly_detection.config import settings
from anomaly_detection.utils.logging import get_logger

log = get_logger("hybrid_detector")


class HybridDetector:
    """
    Routes metrics to different detectors based on service criticality.
    
    Usage:
        detector = HybridDetector()
        detector.load()
        
        # Automatically uses the right detector per metric
        scores = detector.predict_with_labels(data, metric_names)
    """

    def __init__(
        self,
        critical_services: list[str] | None = None,
        critical_detector_type: str = "timesnet",
        default_detector_type: str = "statistical",
    ):
        """
        Args:
            critical_services: List of service names to use TimesNet for.
                              If None, reads from settings.critical_services
            critical_detector_type: Detector for critical services
            default_detector_type: Detector for everything else
        """
        # Parse critical services from config or argument
        if critical_services is not None:
            self.critical_services = set(critical_services)
        elif settings.critical_services:
            # Parse comma-separated list from env var
            self.critical_services = {
                s.strip() for s in settings.critical_services.split(",") if s.strip()
            }
        else:
            self.critical_services = set()

        self.critical_detector_type = critical_detector_type
        self.default_detector_type = default_detector_type

        self.critical_detector: Any = None
        self.default_detector: Any = None

    def load(self) -> None:
        """Load both detectors."""
        from anomaly_detection.detector import get_detector

        # Load critical detector (TimesNet)
        if self.critical_services:
            log.info(
                "loading critical detector",
                detector=self.critical_detector_type,
                services=list(self.critical_services),
            )
            self.critical_detector = get_detector(self.critical_detector_type)
            self.critical_detector.load()
        else:
            log.info("no critical services configured, skipping TimesNet")

        # Load default detector (statistical)
        log.info("loading default detector", detector=self.default_detector_type)
        self.default_detector = get_detector(self.default_detector_type)
        self.default_detector.load()

    def _extract_service_name(self, metric_name: str) -> str:
        """
        Extract service name from metric name.
        
        Metric names look like:
          http_requests_total{service="checkout",pod="checkout-abc123"}
          
        Returns: "checkout"
        """
        # Try to find service= label
        if 'service="' in metric_name:
            start = metric_name.index('service="') + len('service="')
            end = metric_name.index('"', start)
            return metric_name[start:end]

        # Try job= label as fallback
        if 'job="' in metric_name:
            start = metric_name.index('job="') + len('job="')
            end = metric_name.index('"', start)
            return metric_name[start:end]

        # No service label found — use full metric name
        return metric_name

    def _is_critical(self, metric_name: str) -> bool:
        """Check if this metric belongs to a critical service."""
        service = self._extract_service_name(metric_name)
        return service in self.critical_services

    def predict_with_labels(
        self,
        data: NDArray[Any],
        metric_names: list[str],
    ) -> dict[str, dict[str, Any]]:
        """
        Run inference on all metrics, routing to appropriate detector.
        
        Returns:
            {
                "metric_name": {
                    "score": 0.85,
                    "detector": "timesnet" | "statistical",
                    "service": "checkout"
                },
                ...
            }
        """
        results: dict[str, dict[str, Any]] = {}

        # Separate metrics by criticality
        critical_indices: list[int] = []
        critical_names: list[str] = []
        default_indices: list[int] = []
        default_names: list[str] = []

        for i, name in enumerate(metric_names):
            if self._is_critical(name):
                critical_indices.append(i)
                critical_names.append(name)
            else:
                default_indices.append(i)
                default_names.append(name)

        # Run critical detector
        if critical_indices and self.critical_detector:
            critical_data = data[critical_indices]
            critical_scores = self.critical_detector.predict(critical_data)

            for i, name in enumerate(critical_names):
                results[name] = {
                    "score": float(critical_scores[i]),
                    "detector": self.critical_detector_type,
                    "service": self._extract_service_name(name),
                }

        # Run default detector
        if default_indices and self.default_detector:
            default_data = data[default_indices]
            default_scores = self.default_detector.predict(default_data)

            for i, name in enumerate(default_names):
                results[name] = {
                    "score": float(default_scores[i]),
                    "detector": self.default_detector_type,
                    "service": self._extract_service_name(name),
                }

        log.debug(
            "hybrid prediction complete",
            critical_count=len(critical_indices),
            default_count=len(default_indices),
        )

        return results

    def predict(self, data: NDArray[Any]) -> NDArray[np.floating[Any]]:
        """
        Simple predict interface — uses default detector for all.
        
        Note: For hybrid routing, use predict_with_labels() instead.
        """
        if self.default_detector is None:
            raise RuntimeError("Detector not loaded. Call load() first.")
        return self.default_detector.predict(data)

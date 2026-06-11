"""Configuration management via environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All config comes from environment variables."""

    model_config = SettingsConfigDict(env_prefix="")

    # Data source
    prometheus_url: str = "http://localhost:9090"
    prometheus_query: str = "up"
    fetch_interval_seconds: int = 60

    # Model / Detector
    model_path: str = "models/timesnet.pt"
    window_size: int = 96
    anomaly_threshold: float = 0.75

    # Detector type: "auto", "timesnet", "statistical", "rolling", "hybrid", "mock"
    detector_type: str = "auto"

    # Statistical detector settings
    z_threshold: float = 2.5  # Z-score threshold for anomaly
    ema_alpha: float = 0.1  # EMA smoothing factor for rolling detector

    # Hybrid detector settings
    # Comma-separated list of service names to use TimesNet for
    # e.g., "checkout,payments,auth" — these get the full CNN-based detector
    # All other services use the lightweight statistical detector
    critical_services: str = ""

    # Output
    pushgateway_url: str | None = None
    grafana_url: str | None = None
    grafana_api_token: str | None = None
    grafana_dashboard_uid: str | None = None

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # or "console"

    # Health server
    health_port: int = 8080


settings = Settings()

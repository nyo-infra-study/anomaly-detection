"""Configuration management via environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All config comes from environment variables."""

    model_config = SettingsConfigDict(env_prefix="")

    # Data source
    prometheus_url: str = "http://localhost:9090"
    prometheus_query: str = "up"
    fetch_interval_seconds: int = 60

    # Model
    model_path: str = "models/timesnet.pt"
    window_size: int = 96
    anomaly_threshold: float = 0.75

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

"""Application configuration loaded from environment variables.

Uses pydantic-settings for typed, validated configuration.
No hardcoded secrets. Every configurable value is an environment variable.
"""

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global application settings.

    Attributes:
        edgar_user_agent: Required. Format: "Name email@example.com".
            SEC requires a valid user agent for API access.
        edgar_base_url: EDGAR full-text search API base URL.
        edgar_archive_url: EDGAR filing archive base URL.
        data_dir: Local directory for storing downloaded filings.
        log_level: Logging verbosity. One of DEBUG, INFO, WARNING, ERROR.
        edgar_max_rps: Maximum requests per second to EDGAR API.
            SEC enforces 10 req/s. Violating this gets you banned.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore undeclared env vars (e.g. GOOGLE_API_KEY, LANGFUSE_*)
    )

    edgar_user_agent: str = Field(
        ...,
        description="SEC EDGAR user agent. Format: 'Name email@example.com'",
    )
    edgar_base_url: str = Field(
        default="https://efts.sec.gov/LATEST",
        description="EDGAR full-text search API base URL",
    )
    edgar_archive_url: str = Field(
        default="https://www.sec.gov/Archives/edgar/data",
        description="EDGAR filing archive base URL",
    )
    data_dir: Path = Field(
        default=Path("./data/raw"),
        description="Local directory for downloaded filings",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR",
    )
    edgar_max_rps: int = Field(
        default=10,
        description="Max requests per second to EDGAR. SEC limit is 10.",
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL for caching",
    )
    cache_ttl_seconds: int = Field(
        default=3600,
        description="Cache Time-To-Live in seconds",
    )

    @field_validator("edgar_user_agent")
    @classmethod
    def validate_user_agent(cls, v: str) -> str:
        """Validate user agent contains name and email.

        Args:
            v: The user agent string to validate.

        Returns:
            The validated user agent string.

        Raises:
            ValueError: If user agent doesn't contain an email-like pattern.
        """
        if "@" not in v:
            msg = "EDGAR_USER_AGENT must include an email address. Format: 'Your Name your.email@example.com'"
            raise ValueError(msg)
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is a recognized value.

        Args:
            v: The log level string to validate.

        Returns:
            The uppercased log level string.

        Raises:
            ValueError: If log level is not one of the recognized values.
        """
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid_levels:
            msg = f"LOG_LEVEL must be one of {valid_levels}, got '{v}'"
            raise ValueError(msg)
        return upper


def get_settings() -> Settings:
    """Create and return application settings.

    Returns:
        Validated Settings instance loaded from environment.
    """
    return Settings()

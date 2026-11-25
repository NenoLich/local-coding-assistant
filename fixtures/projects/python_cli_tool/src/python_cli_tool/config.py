"""Configuration management for the CLI tool."""

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = {
    "api_url": "https://api.example.com",
    "timeout": 30,
    "retries": 3,
    "output_format": "json",
}


class Config:
    """Configuration loader with file and environment variable support."""

    def __init__(self, config_path: str | None = None):
        self.config_path = config_path or self._find_config_file()
        self._config = DEFAULT_CONFIG.copy()
        self.load()

    def _find_config_file(self) -> str | None:
        """Find config file in standard locations."""
        candidates = [
            Path.cwd() / "config.yaml",
            Path.cwd() / ".python-cli-tool.yaml",
            Path.home() / ".config" / "python-cli-tool" / "config.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return None

    def load(self) -> None:
        """Load configuration from file."""
        if self.config_path and Path(self.config_path).exists():
            with open(self.config_path) as f:
                file_config = yaml.safe_load(f) or {}
                self._config.update(file_config)

        # Environment variables override (prefix: PYTHON_CLI_TOOL_)
        for key in self._config:
            env_key = f"PYTHON_CLI_TOOL_{key.upper()}"
            if env_key in os.environ:
                self._config[key] = self._parse_value(os.environ[env_key])

    def _parse_value(self, value: str) -> Any:
        """Parse string value to appropriate type."""
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self._config[key] = value

    def save(self, path: str | None = None) -> None:
        """Save configuration to file."""
        save_path = path or self.config_path
        if not save_path:
            save_path = str(Path.cwd() / "config.yaml")

        with open(save_path, "w") as f:
            yaml.dump(self._config, f, default_flow_style=False)

    @property
    def all(self) -> dict[str, Any]:
        """Get all configuration."""
        return self._config.copy()


def load_config(config_path: str | None = None) -> Config:
    """Factory function to load configuration."""
    return Config(config_path)
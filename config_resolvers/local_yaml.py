from pathlib import Path
from typing import Any

import yaml

from config_resolvers.base import BaseConfigResolver


LOCAL = 'local'


class LocalConfigResolver(BaseConfigResolver):
    """Config Resolver for local configs."""

    def _load(self, path: str) -> dict[str, Any]:
        """Loads the config from yaml in local at path."""
        p = Path(path)
        with p.open('r') as config:
            content = config.read()
        return yaml.safe_load(content)

    def load(self, path: str) -> dict[str, Any]:
        """Loads the local config."""
        return self._load(path)

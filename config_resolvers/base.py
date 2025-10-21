from abc import ABC, abstractmethod
from typing import Any


class BaseConfigResolver(ABC):
    """Base abstract class for Config Resolver."""

    @abstractmethod
    async def load(self, path: str) -> dict[str, Any]: ...

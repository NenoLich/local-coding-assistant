"""Field-level caching system for configuration validation.

This module provides a caching system for validation results to improve
performance when validating the same fields repeatedly.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Any

from local_coding_assistant.core.system_registry import ValidationResult
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("config.cache")


class FieldValidationCache:
    """Cache for field validation results.

    This cache stores validation results for specific field/value combinations
    to avoid repeated validation operations.
    """

    def __init__(self, maxsize: int = 16):
        """Initialize the validation cache.

        Args:
            maxsize: Maximum number of cached results
        """
        self.maxsize = maxsize
        self._cache = {}
        self._hits = 0
        self._misses = 0

    def _generate_cache_key(
        self, field_path: str, value: Any, capabilities_hash: str
    ) -> str:
        """Generate a cache key for a field validation.

        Args:
            field_path: The field path
            value: The field value
            capabilities_hash: Hash of current capabilities

        Returns:
            Cache key string
        """
        # Create a stable representation of the value
        try:
            value_str = json.dumps(value, sort_keys=True)
        except (TypeError, ValueError):
            # Fallback for non-serializable values
            value_str = str(value)

        # Combine all components
        key_data = f"{field_path}:{value_str}:{capabilities_hash}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    def get(
        self, field_path: str, value: Any, capabilities_hash: str
    ) -> ValidationResult | None:
        """Get a cached validation result.

        Args:
            field_path: The field path
            value: The field value
            capabilities_hash: Hash of current capabilities

        Returns:
            Cached ValidationResult or None if not found
        """
        cache_key = self._generate_cache_key(field_path, value, capabilities_hash)

        if cache_key in self._cache:
            self._hits += 1
            return self._cache[cache_key]

        self._misses += 1
        return None

    def set(
        self,
        field_path: str,
        value: Any,
        capabilities_hash: str,
        result: ValidationResult,
    ) -> None:
        """Cache a validation result.

        Args:
            field_path: The field path
            value: The field value
            capabilities_hash: Hash of current capabilities
            result: The validation result to cache
        """
        cache_key = self._generate_cache_key(field_path, value, capabilities_hash)

        # Implement simple LRU eviction if cache is full
        if len(self._cache) >= self.maxsize and cache_key not in self._cache:
            # Remove the oldest entry (simple FIFO for now)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[cache_key] = result

    def invalidate(self, field_path: str | None = None) -> None:
        """Invalidate cache entries.

        Args:
            field_path: If provided, only invalidate entries for this field.
                       If None, invalidate all entries.
        """
        if field_path is None:
            self._cache.clear()
            logger.debug("Cleared all cache entries")
        else:
            # Remove entries for specific field
            keys_to_remove = [
                key
                for key in self._cache.keys()
                if key.startswith(
                    hashlib.sha256(f"{field_path}:".encode()).hexdigest()[:16]
                )
            ]
            for key in keys_to_remove:
                del self._cache[key]
            logger.debug(f"Cleared cache entries for field: {field_path}")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary containing cache statistics
        """
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0

        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "size": len(self._cache),
            "maxsize": self.maxsize,
        }

    def clear(self) -> None:
        """Clear all cache entries and reset statistics."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        logger.debug("Cache cleared and statistics reset")


@lru_cache(maxsize=16)
def get_capabilities_hash(capabilities: frozenset[str]) -> str:
    """Get a hash of the current capabilities set.

    Args:
        capabilities: Frozen set of capability names

    Returns:
        Hash string representing the capabilities
    """
    capabilities_str = json.dumps(sorted(capabilities), sort_keys=True)
    return hashlib.sha256(capabilities_str.encode()).hexdigest()[:16]

"""Language grammar loading for tree-sitter parsing."""

from typing import ClassVar

from tree_sitter import Language
from tree_sitter_language_pack import get_language


class LanguageRegistry:
    """Manages language grammar loading using tree-sitter-language-pack."""

    _cache: ClassVar[dict[str, Language]] = {}

    @classmethod
    def get_language(cls, language_name: str) -> Language:
        """Get a tree-sitter Language object for the given language.

        Args:
            language_name: Name of the language (e.g., 'python', 'javascript').

        Returns:
            Tree-sitter Language object.

        Raises:
            ValueError: If the language is not supported.
        """
        if language_name not in cls._cache:
            try:
                cls._cache[language_name] = get_language(language_name)
            except Exception as e:
                msg = f"Failed to load language '{language_name}': {e}"
                raise ValueError(msg) from e
        return cls._cache[language_name]

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the language cache."""
        cls._cache.clear()

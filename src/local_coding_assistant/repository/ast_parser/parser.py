"""Main AST parser interface using tree-sitter-language-pack."""

import hashlib
import time
from pathlib import Path

from tree_sitter import Parser

from local_coding_assistant.repository.models import (
    ASTNode,
    CallRelationship,
    FileMetadata,
)

from .registry import LanguageRegistry
from .relationships import CallRelationshipExtractor
from .symbols import SymbolExtractor

_LANGUAGE_BY_EXTENSION = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "jsx",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".rs": "rust",
    ".go": "go",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
}


class ASTParser:
    """Main parser interface using tree-sitter-language-pack."""

    def __init__(self, language_pack: str | None = None) -> None:
        """Initialize the AST parser."""
        self.parser = Parser()

    def detect_language(self, file_path: str | Path) -> str:
        """Detect the programming language from file extension.

        Args:
            file_path: Path to the source file.

        Returns:
            Language name (e.g., 'python', 'javascript').

        Raises:
            ValueError: If the file extension is not supported.
        """
        path = Path(file_path) if isinstance(file_path, str) else file_path
        ext = path.suffix.lower()

        language = _LANGUAGE_BY_EXTENSION.get(ext)
        if language is None:
            msg = f"Unsupported file extension: {ext}"
            raise ValueError(msg)
        return language

    def parse(
        self,
        file_path: str | Path,
        content: str | None = None,
        language: str | None = None,
    ) -> tuple[list[ASTNode], FileMetadata, list[CallRelationship]]:
        """Parse a source file and extract symbols and call relationships.

        Args:
            file_path: Path to the source file.
            content: File content. If None, reads from file_path.
            language: Language of the file. If None, detected from file extension.

        Returns:
            Tuple of (list of ASTNode symbols, FileMetadata, list of CallRelationship).

        Raises:
            ValueError: If the file cannot be parsed or language is not supported.
            FileNotFoundError: If the file does not exist and content is not provided.
        """
        path = Path(file_path) if isinstance(file_path, str) else file_path

        if content is None:
            if not path.exists():
                msg = f"File not found: {file_path}"
                raise FileNotFoundError(msg)
            content = path.read_text(encoding="utf-8")

        if not language:
            language = self.detect_language(path)

        # Get the language object and set the parser language
        self.parser.language = LanguageRegistry.get_language(language)

        # Parse the content
        tree = self.parser.parse(content.encode("utf-8"))

        # Extract symbols and call relationships
        symbols = SymbolExtractor.extract_symbols(
            tree.root_node, language, str(file_path)
        )
        call_relationships = CallRelationshipExtractor.extract_call_relationships(
            tree.root_node, language, str(file_path)
        )

        # Create FileMetadata
        content_bytes = content.encode("utf-8")
        content_hash = hashlib.sha256(content_bytes).hexdigest()
        file_metadata = FileMetadata(
            path=str(file_path),
            language=language,
            last_modified=int(path.stat().st_mtime)
            if path.exists()
            else int(time.time()),
            last_indexed=int(time.time()),
            hash=content_hash,
            size_bytes=len(content_bytes),
            content=content,
        )

        return symbols, file_metadata, call_relationships

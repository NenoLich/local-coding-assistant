"""AST parser using tree-sitter-language-pack for code analysis.

Parses source files into symbols and call relationships and manages the
language grammar loading.
"""

from local_coding_assistant.repository.models import (
    ASTNode,
    CallRelationship,
    FileMetadata,
    SymbolType,
)

from .inheritance import InheritanceExtractor
from .parser import ASTParser
from .registry import LanguageRegistry
from .relationships import CallRelationshipExtractor
from .symbols import SymbolExtractor
from .types import TypeExtractor

__all__ = [
    "ASTNode",
    "ASTParser",
    "CallRelationship",
    "CallRelationshipExtractor",
    "FileMetadata",
    "InheritanceExtractor",
    "LanguageRegistry",
    "SymbolExtractor",
    "SymbolType",
    "TypeExtractor",
]

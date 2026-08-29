"""Repository map generation using PageRank-based symbol ranking."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_coding_assistant.repository.call_graph import CallGraph
from local_coding_assistant.repository.database import RepositoryDatabase
from local_coding_assistant.repository.models import CallRelationship


@dataclass
class RepoMapData:
    """Structured repository map data."""

    project_name: str | None
    language_distribution: dict[str, float]
    total_symbols: int
    grouped_symbols: dict[str, list[dict[str, Any]]]


class RepoMapBuilder:
    """Builds repository maps using PageRank-based symbol selection."""

    def __init__(
        self,
        database: RepositoryDatabase,
        max_symbols: int = 100,
        include_docstrings: bool = False,
        include_imports: bool = False,
        included_symbol_types: set[str] | None = None,
        relationship_weights: dict[str, dict[str, float]] | None = None,
    ) -> None:
        """Initialize the repo map builder.

        Args:
            database: Repository database instance.
            max_symbols: Maximum number of symbols to include in repo map.
            include_docstrings: Whether to include docstrings in repo map (not implemented).
            include_imports: Whether to include imports in repo map (not implemented).
            included_symbol_types: Set of symbol types to include in repo map.
            relationship_weights: Optional dictionary mapping languages to relationship type weights.
        """
        self.database = database
        self.max_symbols = max_symbols
        self.include_docstrings = include_docstrings
        self.include_imports = include_imports
        self.included_symbol_types = included_symbol_types or {
            "function",
            "method",
            "class",
            "interface",
            "type_alias",
        }
        self.relationship_weights = relationship_weights

    def build(
        self,
        project_name: str | None = None,
        target_project_root: str | Path | None = None,
        convert_to_rel_path_if_possible: bool = False,
    ) -> RepoMapData:
        """Build a repository map.

        Args:
            project_name: Optional project name for the header.
            target_project_root: Optional root directory of the project.
            convert_to_rel_path_if_possible: Whether to convert absolute paths to relative paths.

        Returns:
            RepoMapData with structured repository map information.
        """
        # Get all symbols from the database
        all_symbols = self.database.get_all_symbols()

        # Build call graph from database
        call_graph = self._build_call_graph_from_db()

        # Compute PageRank scores
        ranks = call_graph.compute_pagerank()

        # Filter symbols by type
        filtered_symbols = self._filter_symbols_by_type(all_symbols)

        # Add rank scores to symbols
        ranked_symbols = self._rank_symbols(filtered_symbols, ranks)

        # Select top N symbols
        top_symbols = self._select_top_symbols(ranked_symbols)

        # Group symbols by file
        grouped_symbols = self._group_symbols_by_file(
            top_symbols, target_project_root, convert_to_rel_path_if_possible
        )

        # Get language distribution
        language_distribution = self.database.get_language_distribution()

        return RepoMapData(
            project_name=project_name,
            language_distribution=language_distribution,
            total_symbols=len(top_symbols),
            grouped_symbols=grouped_symbols,
        )

    def _build_call_graph_from_db(self) -> CallGraph:
        """Build a call graph from database call relationships.

        Returns:
            CallGraph instance.
        """
        call_graph = CallGraph(relationship_weights=self.relationship_weights)

        # Get all call relationships from the database
        call_relationships = self.database.get_all_call_relationships()

        # Build the graph from actual call relationships
        for rel in call_relationships:
            call_graph.add_relationship(
                CallRelationship(
                    caller_name=rel["caller_name"],
                    caller_line=rel["caller_line"],
                    callee_name=rel["callee_name"],
                    callee_line=rel["callee_line"],
                    relationship_type=rel["relationship_type"],
                    language=rel.get("language", "default"),
                    file_path=rel.get("file_path", "default"),
                )
            )

        return call_graph

    def _filter_symbols_by_type(
        self, symbols: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Filter symbols by configured symbol types.

        Args:
            symbols: List of symbol dictionaries.

        Returns:
            Filtered list of symbols.
        """
        if not self.included_symbol_types:
            return symbols

        return [
            symbol
            for symbol in symbols
            if symbol["symbol_type"] in self.included_symbol_types
        ]

    def _rank_symbols(
        self, symbols: list[dict[str, Any]], ranks: dict[str, float]
    ) -> list[dict[str, Any]]:
        """Add PageRank scores to symbols.

        Args:
            symbols: List of symbol dictionaries.
            ranks: Dictionary of symbol names to rank scores.

        Returns:
            List of symbols with rank scores added.
        """
        for symbol in symbols:
            name = symbol["name"]
            if not name:
                continue
            # 1. Reconstruct the Fully Qualified Name (FQN)
            parent_scope = symbol.get("parent_scope")
            if parent_scope:
                fqn = f"{parent_scope}.{name}"
            else:
                fqn = name

            # 2. Lookup the rank using the FQN
            symbol["rank"] = ranks.get(fqn, 0.0)

        return symbols

    def _deduplicate_symbols(
        self, symbols: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Remove symbols with the same file_path, parent_scope and name from symbols.

        Args:
            symbols: List of symbol dictionaries
        Returns:
            Curated list of symbols with unique file_path + parent_scope + name.
        """
        seen_fqns = set()
        new_symbols = []

        # Assuming 'symbols' is your list of ASTNodes sorted by rank
        for symbol in symbols:
            # 1. Build the FQN
            file_path = symbol.get("file_path")
            parent_scope = symbol.get("parent_scope")
            name = symbol.get("name")
            fqn = (
                f"{file_path}.{parent_scope}.{name}"
                if parent_scope
                else f"{file_path}.{name}"
            )

            # 2. DEDUPLICATE: If we already rendered this FQN, skip it!
            if fqn in seen_fqns:
                continue

            seen_fqns.add(fqn)
            new_symbols.append(symbol)

        return new_symbols

    def _select_top_symbols(
        self, symbols: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Select top N symbols by rank score.

        Args:
            symbols: List of symbol dictionaries with rank scores.

        Returns:
            List of top N symbols sorted by rank.
        """
        # Sort by rank score descending
        sorted_symbols = sorted(symbols, key=lambda x: x["rank"], reverse=True)

        # Return top N
        return sorted_symbols[: self.max_symbols]

    def _group_symbols_by_file(
        self,
        symbols: list[dict[str, Any]],
        target_project_root: str | Path | None = None,
        convert_to_rel_path_if_possible: bool = False,
    ) -> dict[str, list[dict[str, Any]]]:
        """Group symbols by their file path.

        Args:
            symbols: List of symbol dictionaries.
            target_project_root: Optional root directory of the project.
            convert_to_rel_path_if_possible: Whether to convert absolute paths to relative paths.

        Returns:
            Dictionary mapping file paths to lists of symbols.
        """
        resolved_target_project_root = (
            Path(target_project_root) if target_project_root else Path.cwd()
        )
        grouped: dict[str, list[dict[str, Any]]] = {}
        for symbol in symbols:
            file_path = symbol["file_path"]
            if file_path not in grouped:
                if convert_to_rel_path_if_possible:
                    try:
                        file_path = str(
                            Path(file_path).relative_to(resolved_target_project_root)
                        )
                    except ValueError:
                        file_path = file_path

                grouped[file_path] = []
            grouped[file_path].append(symbol)

        return grouped


class MapFormatter:
    """Formats repository maps for display."""

    def format(self, repo_map_data: RepoMapData, include_ranks: bool = False) -> str:
        """Format the repository map from structured data.

        Args:
            repo_map_data: Structured repository map data.
            include_ranks: Whether to include rank scores in the output.

        Returns:
            Formatted repository map string.
        """
        if not repo_map_data.grouped_symbols:
            return ""

        # Note about symbol selection
        lines = [
            "The Repo Map below is a high - level architectural guide. It may be slightly out of sync with recent edits.",
            f"Note: Showing top {repo_map_data.total_symbols} symbols by usage frequency. Not all symbols are shown.",
        ]

        # Group symbols by file
        for file_path in sorted(repo_map_data.grouped_symbols.keys()):
            symbols = repo_map_data.grouped_symbols[file_path]
            # Sort symbols by line number
            symbols = sorted(symbols, key=lambda x: x["line_number"])

            # Get relative path if possible
            try:
                rel_path = str(Path(file_path).relative_to(Path.cwd()))
            except ValueError:
                rel_path = file_path

            lines.append(f"{rel_path}")

            for symbol in symbols:
                # Format each symbol
                symbol_line = self._format_symbol(symbol, include_ranks)
                lines.append(f"  {symbol_line}")

            lines.append("")

        return "\n".join(lines)

    def _format_symbol(
        self, symbol: dict[str, Any], include_ranks: bool = False
    ) -> str:
        """Format a single symbol for display.

        Args:
            symbol: Symbol dictionary.
            include_ranks: Whether to include rank scores in the output.

        Returns:
            Formatted symbol string.
        """
        name = symbol["name"]
        symbol_type = symbol["symbol_type"]
        signature = symbol.get("signature")
        line_number = symbol["line_number"]
        end_line_number = symbol.get("end_line_number")

        # Build the line range
        if end_line_number and end_line_number != line_number:
            line_range = f"[line: {line_number}-{end_line_number}]"
        else:
            line_range = f"[line: {line_number}]"

        # Build the symbol line
        signature = signature or name
        rank = symbol.get("rank")
        if include_ranks and rank:
            return f"- [{symbol_type}] {signature} {line_range} (Rank: {rank:.3f})"

        return f"- [{symbol_type}] {signature} {line_range}"

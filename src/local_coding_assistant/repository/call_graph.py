"""Call graph analysis using NetworkX for symbol ranking."""

from dataclasses import dataclass
from typing import Any

import networkx as nx

from local_coding_assistant.repository.models import CallRelationship


@dataclass
class SymbolRank:
    """Represents a symbol with its pagerank score."""

    symbol_name: str
    rank_score: float
    file_path: str | None = None
    line_number: int | None = None


class CallGraph:
    """Builds and analyzes call graphs for symbol ranking."""

    def __init__(
        self, relationship_weights: dict[str, dict[str, float]] | None = None
    ) -> None:
        """Initialize the call graph.

        Args:
            relationship_weights: Optional dictionary mapping languages to relationship type weights.
                If None, uses default weights.
        """
        self.graph: nx.DiGraph = nx.DiGraph()
        self.ranks: dict[str, float] = {}
        self.relationship_weights = relationship_weights or self._get_default_weights()

    def _get_default_weights(self) -> dict[str, dict[str, float]]:
        """Get default relationship weights.

        Returns:
            Dictionary mapping languages to relationship type weights.
        """
        return {
            "python": {
                "call": 1.5,
                "type_reference": 1.2,
                "inherits": 1.0,
                "belongs_to": 0.2,
                "contains": 0.0,
            },
            "javascript": {
                "call": 1.5,
                "type_reference": 1.2,
                "inherits": 1.0,
                "belongs_to": 0.2,
                "contains": 0.0,
            },
            "typescript": {
                "call": 1.5,
                "type_reference": 1.2,
                "inherits": 1.0,
                "belongs_to": 0.2,
                "contains": 0.0,
            },
            "rust": {
                "call": 1.5,
                "type_reference": 1.2,
                "inherits": 1.0,
                "belongs_to": 0.2,
                "contains": 0.0,
            },
            "go": {
                "call": 1.5,
                "type_reference": 1.2,
                "inherits": 1.0,
                "belongs_to": 0.2,
                "contains": 0.0,
            },
            "default": {
                "call": 1.5,
                "type_reference": 1.2,
                "inherits": 1.0,
                "belongs_to": 0.2,
                "contains": 0.0,
            },
        }

    def add_relationship(self, relationship: CallRelationship) -> None:
        """Add a caller-callee relationship to the graph.

        Args:
            relationship: Call relationship to add.
        """
        caller = relationship.caller_name or "unknown"
        callee = relationship.callee_name or "unknown"

        # Get relationship type and language
        rel_type = getattr(relationship, "relationship_type", "call")
        language = getattr(relationship, "language", "default").lower()

        # Get language-specific weights, fallback to default
        lang_weights = self.relationship_weights.get(
            language, self.relationship_weights.get("default", {})
        )
        weight = lang_weights.get(rel_type, 1.0)

        # Add nodes if they don't exist
        if caller not in self.graph:
            self.graph.add_node(caller)
        if callee not in self.graph:
            self.graph.add_node(callee)

        # Accumulate weight if the edge already exists
        if self.graph.has_edge(caller, callee):
            self.graph[caller][callee]["weight"] += weight
        else:
            self.graph.add_edge(caller, callee, weight=weight)

    def add_relationships(self, relationships: list[CallRelationship]) -> None:
        """Add multiple caller-callee relationships to the graph.

        Args:
            relationships: List of call relationships to add.
        """
        for rel in relationships:
            self.add_relationship(rel)

    def compute_pagerank(
        self, alpha: float = 0.85, max_iter: int = 100, tol: float = 1e-6
    ) -> dict[str, float]:
        """Compute PageRank scores for all symbols in the graph.

        PageRank measures the importance of symbols based on how many other symbols call them.
        Symbols that are called by many important symbols get higher scores.

        Analysis:
        - Edge direction: caller -> callee (function A calls function B)
        - For importance ranking, we reverse the graph: callee <- caller
        - A symbol is important if many important symbols call it (incoming edges in reversed graph)

        Args:
            alpha: Damping parameter for PageRank (default 0.85).
            max_iter: Maximum number of iterations.
            tol: Tolerance for convergence.

        Returns:
            Dictionary mapping symbol names to their PageRank scores.
        """
        if not self.graph:
            return {}

        # DO NOT REVERSE the graph.
        # In a directed edge (Caller -> Callee), NetworkX naturally flows the
        # PageRank "water" to the Callee. This is exactly what we want to rank
        # utilities, models, and interfaces highly.

        self.ranks = nx.pagerank(
            self.graph,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            weight="weight",
        )

        return self.ranks

    def get_top_symbols(self, n: int = 100) -> list[SymbolRank]:
        """Get the top N symbols by PageRank score.

        Args:
            n: Number of top symbols to return.

        Returns:
            List of SymbolRank objects sorted by rank score (descending).
        """
        if not self.ranks:
            self.compute_pagerank()

        # Sort by rank score descending
        sorted_symbols = sorted(self.ranks.items(), key=lambda x: x[1], reverse=True)

        # Convert to SymbolRank objects
        return [
            SymbolRank(
                symbol_name=name,
                rank_score=score,
            )
            for name, score in sorted_symbols[:n]
        ]

    def get_symbol_rank(self, symbol_name: str) -> float:
        """Get the PageRank score for a specific symbol.

        Args:
            symbol_name: Name of the symbol.

        Returns:
            PageRank score, or 0.0 if symbol not found.
        """
        if not self.ranks:
            self.compute_pagerank()

        return self.ranks.get(symbol_name, 0.0)

    def clear(self) -> None:
        """Clear the graph and computed ranks."""
        self.graph.clear()
        self.ranks.clear()

    def get_graph_stats(self) -> dict[str, Any]:
        """Get statistics about the call graph.

        Returns:
            Dictionary with graph statistics.
        """
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "is_directed": self.graph.is_directed(),
            "num_weakly_connected_components": nx.number_weakly_connected_components(
                self.graph
            ),
            "num_strongly_connected_components": nx.number_strongly_connected_components(
                self.graph
            ),
        }

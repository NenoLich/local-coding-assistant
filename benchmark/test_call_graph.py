"""Benchmarks for call graph computation and PageRank ranking."""


from local_coding_assistant.repository.call_graph import CallGraph
from local_coding_assistant.repository.models import CallRelationship


def test_add_single_relationship(benchmark):
    """Benchmark adding a single call relationship to the graph."""
    call_graph = CallGraph()
    relationship = CallRelationship(
        caller_name="caller_func",
        caller_line=10,
        callee_name="callee_func",
        callee_line=20,
        relationship_type="call",
        language="python",
        file_path="test_path",
    )

    benchmark(call_graph.add_relationship, relationship)
    assert call_graph.graph.number_of_nodes() >= 1


def test_add_batch_relationships_small(benchmark):
    """Benchmark adding a small batch of call relationships."""
    call_graph = CallGraph()
    relationships = [
        CallRelationship(
            caller_name=f"caller_{i}",
            caller_line=i * 10,
            callee_name=f"callee_{i}",
            callee_line=i * 10 + 5,
            relationship_type="call",
            language="python",
            file_path="test_path",
        )
        for i in range(10)
    ]

    benchmark(call_graph.add_relationships, relationships)
    assert call_graph.graph.number_of_edges() >= 10


def test_add_batch_relationships_medium(benchmark):
    """Benchmark adding a medium batch of call relationships."""
    call_graph = CallGraph()
    relationships = [
        CallRelationship(
            caller_name=f"caller_{i}",
            caller_line=i * 10,
            callee_name=f"callee_{i % 50}",  # Create some shared callees
            callee_line=i * 10 + 5,
            relationship_type="call",
            language="python",
            file_path="test_path",
        )
        for i in range(100)
    ]

    benchmark(call_graph.add_relationships, relationships)
    assert call_graph.graph.number_of_edges() >= 100


def test_add_batch_relationships_large(benchmark):
    """Benchmark adding a large batch of call relationships."""
    call_graph = CallGraph()
    relationships = [
        CallRelationship(
            caller_name=f"caller_{i}",
            caller_line=i * 10,
            callee_name=f"callee_{i % 100}",  # Create more shared callees
            callee_line=i * 10 + 5,
            relationship_type="call",
            language="python",
            file_path="test_path",
        )
        for i in range(500)
    ]

    benchmark(call_graph.add_relationships, relationships)
    assert call_graph.graph.number_of_edges() >= 500


def test_compute_pagerank_small_graph(benchmark):
    """Benchmark PageRank computation on a small graph."""
    call_graph = CallGraph()

    # Build a small graph
    relationships = [
        CallRelationship(
            caller_name=f"caller_{i}",
            caller_line=i * 10,
            callee_name=f"callee_{i % 10}",
            callee_line=i * 10 + 5,
            relationship_type="call",
            language="python",
            file_path="test_path",
        )
        for i in range(20)
    ]
    call_graph.add_relationships(relationships)

    ranks = benchmark(call_graph.compute_pagerank)
    assert len(ranks) > 0


def test_compute_pagerank_medium_graph(benchmark):
    """Benchmark PageRank computation on a medium graph."""
    call_graph = CallGraph()

    # Build a medium graph
    relationships = [
        CallRelationship(
            caller_name=f"caller_{i}",
            caller_line=i * 10,
            callee_name=f"callee_{i % 50}",
            callee_line=i * 10 + 5,
            relationship_type="call",
            language="python",
            file_path="test_path",
        )
        for i in range(100)
    ]
    call_graph.add_relationships(relationships)

    ranks = benchmark(call_graph.compute_pagerank)
    assert len(ranks) > 0


def test_compute_pagerank_large_graph(benchmark):
    """Benchmark PageRank computation on a large graph."""
    call_graph = CallGraph()

    # Build a large graph
    relationships = [
        CallRelationship(
            caller_name=f"caller_{i}",
            caller_line=i * 10,
            callee_name=f"callee_{i % 100}",
            callee_line=i * 10 + 5,
            relationship_type="call",
            language="python",
            file_path="test_path",
        )
        for i in range(500)
    ]
    call_graph.add_relationships(relationships)

    ranks = benchmark(call_graph.compute_pagerank)
    assert len(ranks) > 0


def test_get_top_symbols(benchmark):
    """Benchmark getting top symbols from ranked graph."""
    call_graph = CallGraph()

    # Build a graph
    relationships = [
        CallRelationship(
            caller_name=f"caller_{i}",
            caller_line=i * 10,
            callee_name=f"callee_{i % 50}",
            callee_line=i * 10 + 5,
            relationship_type="call",
            language="python",
            file_path="test_path",
        )
        for i in range(100)
    ]
    call_graph.add_relationships(relationships)
    call_graph.compute_pagerank()

    top_symbols = benchmark(call_graph.get_top_symbols, 50)
    assert len(top_symbols) <= 50


def test_end_to_end_call_graph_construction_small(benchmark):
    """Benchmark end-to-end call graph construction and ranking for small dataset."""

    def build_and_rank():
        call_graph = CallGraph()

        relationships = [
            CallRelationship(
                caller_name=f"caller_{i}",
                caller_line=i * 10,
                callee_name=f"callee_{i % 10}",
                callee_line=i * 10 + 5,
                relationship_type="call",
                language="python",
                file_path="test_path",
            )
            for i in range(20)
        ]

        call_graph.add_relationships(relationships)
        call_graph.compute_pagerank()
        return call_graph.get_top_symbols(10)

    top_symbols = benchmark(build_and_rank)
    assert len(top_symbols) <= 10


def test_end_to_end_call_graph_construction_medium(benchmark):
    """Benchmark end-to-end call graph construction and ranking for medium dataset."""

    def build_and_rank():
        call_graph = CallGraph()

        relationships = [
            CallRelationship(
                caller_name=f"caller_{i}",
                caller_line=i * 10,
                callee_name=f"callee_{i % 50}",
                callee_line=i * 10 + 5,
                relationship_type="call",
                language="python",
                file_path="test_path",
            )
            for i in range(100)
        ]

        call_graph.add_relationships(relationships)
        call_graph.compute_pagerank()
        return call_graph.get_top_symbols(50)

    top_symbols = benchmark(build_and_rank)
    assert len(top_symbols) <= 50


def test_end_to_end_call_graph_construction_large(benchmark):
    """Benchmark end-to-end call graph construction and ranking for large dataset."""

    def build_and_rank():
        call_graph = CallGraph()

        relationships = [
            CallRelationship(
                caller_name=f"caller_{i}",
                caller_line=i * 10,
                callee_name=f"callee_{i % 100}",
                callee_line=i * 10 + 5,
                relationship_type="call",
                language="python",
                file_path="test_path",
            )
            for i in range(500)
        ]

        call_graph.add_relationships(relationships)
        call_graph.compute_pagerank()
        return call_graph.get_top_symbols(100)

    top_symbols = benchmark(build_and_rank)
    assert len(top_symbols) <= 100


def test_graph_stats(benchmark):
    """Benchmark getting graph statistics."""
    call_graph = CallGraph()

    # Build a graph
    relationships = [
        CallRelationship(
            caller_name=f"caller_{i}",
            caller_line=i * 10,
            callee_name=f"callee_{i % 50}",
            callee_line=i * 10 + 5,
            relationship_type="call",
            language="python",
            file_path="test_path",
        )
        for i in range(100)
    ]
    call_graph.add_relationships(relationships)

    stats = benchmark(call_graph.get_graph_stats)
    assert stats["num_nodes"] > 0

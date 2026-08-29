"""Symbol indexing and search functionality for repository context."""

from local_coding_assistant.repository.ast_parser import ASTNode
from local_coding_assistant.repository.database import RepositoryDatabase
from local_coding_assistant.repository.models import (
    CallRelationship,
    FileMetadata,
    SymbolDetail,
    SymbolResult,
    SymbolType,
)


class SymbolIndexer:
    """Indexes symbols from parsed AST into the database."""

    def __init__(self, database: RepositoryDatabase) -> None:
        """Initialize the symbol indexer.

        Args:
            database: Repository database instance.
        """
        self.database = database

    def index_file(
        self,
        symbols: list[ASTNode],
        file_metadata: FileMetadata,
        call_relationships: list[CallRelationship] | None = None,
    ) -> int:
        """Index symbols from a file into the database.

        Args:
            symbols: List of ASTNode symbols extracted from the file.
            file_metadata: File metadata (includes content).
            call_relationships: Optional list of call relationships to store.

        Returns:
            Number of symbols indexed.
        """
        indexed_count = self.database.index_file_with_symbols_and_relationships(
            file_metadata, symbols, call_relationships
        )

        return indexed_count

    def index_files(
        self,
        files_data: list[
            tuple[list[ASTNode], FileMetadata, list[CallRelationship] | None]
        ],
    ) -> int:
        """Index multiple files into the database in bulk.

        Args:
            files_data: List of tuples (symbols, file_metadata, call_relationships) for each file.

        Returns:
            Total number of symbols indexed.
        """
        all_file_metadata = []
        all_symbol_details = []
        all_call_relationships = []

        # Collect all data
        for symbols, file_metadata, call_relationships in files_data:
            all_file_metadata.append(file_metadata)
            language = file_metadata.language
            content = file_metadata.content or ""

            symbol_details = [
                (symbol.to_symbol_detail(content, language), file_metadata.path)
                for symbol in symbols
            ]
            all_symbol_details.extend(symbol_details)

            if call_relationships:
                all_call_relationships.extend(call_relationships)

        total_indexed = (
            self.database.index_multiple_files_with_symbols_and_relationships(
                all_file_metadata, all_symbol_details, all_call_relationships
            )
        )

        return total_indexed


class SymbolSearcher:
    """Searches for symbols in the repository database."""

    def __init__(self, database: RepositoryDatabase) -> None:
        """Initialize the symbol searcher.

        Args:
            database: Repository database instance.
        """
        self.database = database

    def search_symbols(
        self,
        query: str,
        symbol_types: list[str] | None = None,
        file_path: str | None = None,
        limit: int = 20,
    ) -> list[SymbolResult]:
        """Search for symbols using FTS5.

        Args:
            query: Search query (supports FTS5 syntax).
            symbol_types: Optional list of symbol types to filter by.
            file_path: Optional file path to filter by.
            limit: Maximum number of results to return.

        Returns:
            List of SymbolResult objects.
        """
        with self.database.get_connection() as conn:
            cursor = conn.cursor()

            # Handle empty query - return all symbols without FTS5 search
            if not query or query.strip() == "":
                sql = """
                    SELECT
                        sm.symbol_id,
                        sm.name,
                        sm.symbol_type,
                        f.path as file_path,
                        sm.line_number,
                        sm.end_line_number,
                        sm.parent_scope,
                        sm.docstring,
                        sm.signature
                    FROM symbol_metadata sm
                    JOIN files f ON sm.file_id = f.id
                    WHERE 1=1
                """
                params: list[str | int] = []
            else:
                # Build the FTS5 query
                fts_query = self._build_fts_query(query, symbol_types)

                # Build the SQL query with optional file path filter
                # symbols_fts is external content table, so we join with symbol_metadata for full details
                sql = """
                    SELECT
                        sm.symbol_id,
                        sm.name,
                        sm.symbol_type,
                        f.path as file_path,
                        sm.line_number,
                        sm.end_line_number,
                        sm.parent_scope,
                        sm.docstring,
                        sm.signature
                    FROM symbols_fts
                    JOIN symbol_metadata sm ON symbols_fts.rowid = sm.symbol_id
                    JOIN files f ON sm.file_id = f.id
                    WHERE symbols_fts MATCH ?
                """
                params = [fts_query]

            # Add symbol type filter if provided
            if symbol_types:
                placeholders = ",".join(["?" for _ in symbol_types])
                sql += f" AND sm.symbol_type IN ({placeholders})"
                params.extend(symbol_types)

            # Add file path filter if provided
            if file_path:
                sql += " AND f.path = ?"
                params.append(file_path)

            # Add ordering and limit
            if query and query.strip():
                sql += " ORDER BY rank"
            else:
                sql += " ORDER BY sm.name"
            sql += " LIMIT ?"
            params.append(str(limit))

            cursor.execute(sql, params)

            results = []
            for row in cursor.fetchall():
                symbol_type = SymbolType(row["symbol_type"])
                results.append(
                    SymbolResult(
                        symbol_id=row["symbol_id"],
                        name=row["name"],
                        symbol_type=symbol_type,
                        file_path=row["file_path"],
                        line_number=row["line_number"],
                        end_line_number=row["end_line_number"],
                        parent_scope=row["parent_scope"],
                        docstring=row["docstring"],
                        signature=row["signature"],
                        rank=0.0,  # FTS5 rank is not exposed in this query
                    )
                )

            return results

    def get_symbol_by_id(self, symbol_id: int) -> SymbolDetail | None:
        """Get detailed information about a symbol by its ID.

        Args:
            symbol_id: Symbol ID.

        Returns:
            SymbolDetail if found, None otherwise.
        """
        with self.database.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    sm.symbol_id,
                    sm.file_id,
                    sm.name,
                    sm.symbol_type,
                    sm.line_number,
                    sm.end_line_number,
                    sm.parent_scope,
                    sm.docstring,
                    sm.content,
                    sm.signature,
                    f.path as file_path
                FROM symbol_metadata sm
                JOIN files f ON sm.file_id = f.id
                WHERE sm.symbol_id = ?
                """,
                (symbol_id,),
            )

            row = cursor.fetchone()
            if row:
                symbol_type = SymbolType(row["symbol_type"])
                return SymbolDetail(
                    symbol_id=row["symbol_id"],
                    file_id=row["file_id"],
                    name=row["name"],
                    symbol_type=symbol_type,
                    line_number=row["line_number"],
                    end_line_number=row["end_line_number"],
                    parent_scope=row["parent_scope"],
                    docstring=row["docstring"],
                    content=row["content"],
                    signature=row["signature"],
                    file_path=row["file_path"],
                )
            return None

    def get_symbols_in_file(self, file_path: str) -> list[SymbolDetail]:
        """Get all symbols in a specific file.

        Args:
            file_path: Path to the file.

        Returns:
            List of SymbolDetail objects.
        """
        with self.database.get_connection() as conn:
            cursor = conn.cursor()

            # Get file ID
            cursor.execute("SELECT id FROM files WHERE path = ?", (file_path,))
            row = cursor.fetchone()
            if not row:
                return []

            file_id = row["id"]

            # Get symbols for this file
            cursor.execute(
                """
                SELECT
                    sm.symbol_id,
                    sm.file_id,
                    sm.name,
                    sm.symbol_type,
                    sm.line_number,
                    sm.end_line_number,
                    sm.parent_scope,
                    sm.docstring,
                    sm.content,
                    sm.signature,
                    f.path as file_path
                FROM symbol_metadata sm
                JOIN files f ON sm.file_id = f.id
                WHERE sm.file_id = ?
                ORDER BY sm.line_number
                """,
                (file_id,),
            )

            results = []
            for row in cursor.fetchall():
                symbol_type = SymbolType(row["symbol_type"])
                results.append(
                    SymbolDetail(
                        symbol_id=row["symbol_id"],
                        file_id=row["file_id"],
                        name=row["name"],
                        symbol_type=symbol_type,
                        line_number=row["line_number"],
                        end_line_number=row["end_line_number"],
                        parent_scope=row["parent_scope"],
                        docstring=row["docstring"],
                        content=row["content"],
                        signature=row["signature"],
                        file_path=row["file_path"],
                    )
                )

            return results

    def _build_fts_query(self, query: str, symbol_types: list[str] | None) -> str:
        """Build an FTS5 query from search parameters.

        Args:
            query: Search query.
            symbol_types: Optional symbol type filters (ignored in FTS query, handled in SQL WHERE).

        Returns:
            FTS5 query string.
        """
        # Check if query contains special FTS5 characters that need escaping
        special_chars = set('"*[]{}^~')
        has_special = any(char in query for char in special_chars)

        if has_special:
            # Escape special FTS5 characters by wrapping in double quotes
            # This treats the entire query as a literal string
            fts_query = query.replace('"', '""')
            return f'"{fts_query}"'
        else:
            # For normal queries, use prefix search for fuzzy matching
            # Add * at the end to match any suffix
            return f"{query}*"

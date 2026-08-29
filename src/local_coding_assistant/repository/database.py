"""SQLite FTS5 database for storing parsed AST data."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import orjson

from local_coding_assistant.repository.models import (
    ASTNode,
    CallRelationship,
    FileMetadata,
    ImportInfo,
    SymbolDetail,
)
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("repository.database")


class RepositoryDatabase:
    """Database connection and schema management for repository context."""

    SCHEMA_SQL = """
    -- Files table
    CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY,
        path TEXT UNIQUE NOT NULL,
        language TEXT NOT NULL,
        last_modified INTEGER NOT NULL,
        last_indexed INTEGER NOT NULL,
        hash TEXT NOT NULL,
        size_bytes INTEGER DEFAULT 0
    );

    -- Symbol metadata table
    CREATE TABLE IF NOT EXISTS symbol_metadata (
        symbol_id INTEGER PRIMARY KEY,
        file_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        symbol_type TEXT NOT NULL,
        line_number INTEGER NOT NULL,
        end_line_number INTEGER,
        parent_scope TEXT,
        docstring TEXT,
        content TEXT,
        signature TEXT,
        FOREIGN KEY (file_id) REFERENCES files(id)
    );

    -- Symbols table with FTS5 using external content
    CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(
        name,
        docstring,
        content,
        signature,
        content='symbol_metadata',
        content_rowid='symbol_id',
        tokenize='porter unicode61'
    );

    -- Triggers to keep FTS table in sync with symbol_metadata
    CREATE TRIGGER IF NOT EXISTS symbol_metadata_ai AFTER INSERT ON symbol_metadata BEGIN
        INSERT INTO symbols_fts(rowid, name, docstring, content, signature)
        VALUES (NEW.symbol_id, NEW.name, NEW.docstring, NEW.content, NEW.signature);
    END;

    CREATE TRIGGER IF NOT EXISTS symbol_metadata_ad AFTER DELETE ON symbol_metadata BEGIN
        DELETE FROM symbols_fts WHERE rowid = OLD.symbol_id;
    END;

    CREATE TRIGGER IF NOT EXISTS symbol_metadata_au AFTER UPDATE ON symbol_metadata BEGIN
        DELETE FROM symbols_fts WHERE rowid = OLD.symbol_id;
        INSERT INTO symbols_fts(rowid, name, docstring, content, signature)
        VALUES (NEW.symbol_id, NEW.name, NEW.docstring, NEW.content, NEW.signature);
    END;

    -- Call relationships table
    CREATE TABLE IF NOT EXISTS call_relationships (
        id INTEGER PRIMARY KEY,
        file_id INTEGER NOT NULL,
        caller_name TEXT,
        caller_line INTEGER NOT NULL,
        callee_name TEXT,
        callee_line INTEGER NOT NULL,
        relationship_type TEXT NOT NULL,
        language TEXT NOT NULL,
        FOREIGN KEY (file_id) REFERENCES files(id)
    );

    -- Imports table
    CREATE TABLE IF NOT EXISTS imports (
        id INTEGER PRIMARY KEY,
        file_id INTEGER NOT NULL,
        import_statement TEXT NOT NULL,
        import_type TEXT NOT NULL,
        imported_symbols TEXT,
        FOREIGN KEY (file_id) REFERENCES files(id)
    );

    -- Project metadata table
    CREATE TABLE IF NOT EXISTS project_metadata (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );

    -- Indexes
    CREATE INDEX IF NOT EXISTS idx_symbols_file_id ON symbol_metadata(file_id);
    CREATE INDEX IF NOT EXISTS idx_symbols_type ON symbol_metadata(symbol_type);
    CREATE INDEX IF NOT EXISTS idx_files_path ON files(path);
    CREATE INDEX IF NOT EXISTS idx_call_relationships_file_id ON call_relationships(file_id);
    CREATE INDEX IF NOT EXISTS idx_call_relationships_caller ON call_relationships(caller_name);
    CREATE INDEX IF NOT EXISTS idx_call_relationships_callee ON call_relationships(callee_name);
    """

    def __init__(self, db_path: str | Path) -> None:
        """Initialize the database connection.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = Path(db_path) if isinstance(db_path, str) else db_path
        self._ensure_database_exists()

    def _ensure_database_exists(self) -> None:
        """Create the database file and schema if it doesn't exist."""
        if not self.db_path.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            with self.get_connection() as conn:
                conn.executescript(self.SCHEMA_SQL)
                conn.commit()

    def configure_sqlite_pragmas(self, conn: sqlite3.Connection) -> None:
        cursor = conn.cursor()

        # 1. Write-Ahead Logging: Allows concurrent reads while writing, eliminates most locks
        cursor.execute("PRAGMA journal_mode = WAL;")

        # 2. Relaxed Disk Sync: 'NORMAL' is safe in WAL mode and avoids waiting for disk spin
        cursor.execute("PRAGMA synchronous = NORMAL;")

        # 3. Increase Cache Size: Negative number means kilobytes (-64000 = ~64MB memory cache)
        cursor.execute("PRAGMA cache_size = -64000;")

        # 4. In-Memory Temp Storage: Avoids writing temp tables/indexes to disk
        cursor.execute("PRAGMA temp_store = MEMORY;")

        # 5. Memory-Mapped I/O: Maps DB file directly into RAM (256MB)
        cursor.execute("PRAGMA mmap_size = 268435456;")

        cursor.close()

    @contextmanager
    def get_connection(self) -> Any:
        """Get a database connection with context manager.

        Yields:
            SQLite connection object.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        # 1. ALWAYS configure pragmas for every connection session
        self.configure_sqlite_pragmas(conn)

        try:
            yield conn
        finally:
            conn.close()

    def add_or_update_file(self, metadata: FileMetadata) -> int:
        """Add or update a file in the database.

        Args:
            metadata: File metadata to store.

        Returns:
            The file ID.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            result = self._add_or_update_file(metadata, cursor)
            conn.commit()

            return result

    def _add_or_update_file(
        self, metadata: FileMetadata, cursor: sqlite3.Cursor
    ) -> int:
        """Add or update a file in the database.

        Args:
            metadata: File metadata to store.
            cursor: Database cursor.

        Returns:
            The file ID.
        """
        cursor.execute(
            """
            INSERT INTO files (path, language, last_modified, last_indexed, hash, size_bytes)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                language = excluded.language,
                last_modified = excluded.last_modified,
                last_indexed = excluded.last_indexed,
                hash = excluded.hash,
                size_bytes = excluded.size_bytes
            RETURNING id
            """,
            (
                metadata.path,
                metadata.language,
                metadata.last_modified,
                metadata.last_indexed,
                metadata.hash,
                metadata.size_bytes,
            ),
        )
        result = cursor.fetchone()
        file_id = result["id"] if result else cursor.lastrowid

        if not isinstance(file_id, int):
            raise ValueError(
                f"Failed to retrieve a valid integer file_id for path: {metadata.path}"
            )

        return file_id

    def add_or_update_files(self, metadata_list: list[FileMetadata]) -> dict[str, int]:
        """Add or update multiple files in the database.

        Args:
            metadata_list: List of file metadata to store.

        Returns:
            Dictionary mapping file paths to file IDs.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            path_to_id = self._add_or_update_files(metadata_list, cursor)
            conn.commit()

            return path_to_id

    def _add_or_update_files(
        self, metadata_list: list[FileMetadata], cursor: sqlite3.Cursor
    ) -> dict[str, int]:
        """Add or update multiple files in the database.

        Args:
            metadata_list: List of file metadata to store.
            cursor: Database cursor.

        Returns:
            Dictionary mapping file paths to file IDs.
        """
        path_to_id = {}
        for metadata in metadata_list:
            try:
                cursor.execute(
                    """
                    INSERT INTO files (path, language, last_modified, last_indexed, hash, size_bytes)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(path) DO UPDATE SET
                        language = excluded.language,
                        last_modified = excluded.last_modified,
                        last_indexed = excluded.last_indexed,
                        hash = excluded.hash,
                        size_bytes = excluded.size_bytes
                    RETURNING id
                    """,
                    (
                        metadata.path,
                        metadata.language,
                        metadata.last_modified,
                        metadata.last_indexed,
                        metadata.hash,
                        metadata.size_bytes,
                    ),
                )
                result = cursor.fetchone()
                file_id = result["id"] if result else cursor.lastrowid
                if not isinstance(file_id, int):
                    raise ValueError(
                        f"Failed to retrieve a valid integer file_id for path: {metadata.path}"
                    )

                path_to_id[metadata.path] = file_id

            except ValueError as e:
                logger.debug(e)
                continue

        return path_to_id

    def get_file(self, file_path: str) -> FileMetadata | None:
        """Get file metadata from the database.

        Args:
            file_path: Path to the file.

        Returns:
            FileMetadata if found, None otherwise.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM files WHERE path = ?", (file_path,))
            row = cursor.fetchone()
            if row:
                return FileMetadata(
                    path=row["path"],
                    language=row["language"],
                    last_modified=row["last_modified"],
                    last_indexed=row["last_indexed"],
                    hash=row["hash"],
                    size_bytes=row["size_bytes"],
                )
            return None

    def delete_file(self, file_path: str) -> None:
        """Delete a file and its associated symbols from the database.

        Args:
            file_path: Path to the file to delete.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM files WHERE path = ?", (file_path,))
            row = cursor.fetchone()
            if row:
                file_id = row["id"]
                cursor.execute(
                    "DELETE FROM symbol_metadata WHERE file_id = ?", (file_id,)
                )
                # FTS deletion is handled by trigger
                cursor.execute("DELETE FROM imports WHERE file_id = ?", (file_id,))
                cursor.execute(
                    "DELETE FROM call_relationships WHERE file_id = ?", (file_id,)
                )
                cursor.execute("DELETE FROM files WHERE path = ?", (file_path,))
                conn.commit()

    def add_symbol(self, symbol: SymbolDetail) -> int:
        """Add a symbol to the database.

        Args:
            symbol: Symbol detail to store.

        Returns:
            The symbol ID.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Insert into metadata table (trigger will sync to FTS)
            cursor.execute(
                """
                INSERT INTO symbol_metadata
                (file_id, name, symbol_type, line_number, end_line_number, parent_scope, docstring, content, signature)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol.file_id,
                    symbol.name,
                    symbol.symbol_type.value,
                    symbol.line_number,
                    symbol.end_line_number,
                    symbol.parent_scope,
                    symbol.docstring,
                    symbol.content,
                    symbol.signature,
                ),
            )
            symbol_id = cursor.lastrowid
            conn.commit()
            return symbol_id

    def add_symbols(self, symbols: list[SymbolDetail]) -> None:
        """Add list of symbols to the database.

        Args:
            symbols: List of symbol details to store.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            self._add_symbols(symbols, cursor)
            conn.commit()

    def _add_symbols(self, symbols: list[SymbolDetail], cursor: sqlite3.Cursor) -> None:
        """Add list of symbols to the database.

        Args:
            symbols: List of symbol details to store.
            cursor: Database cursor.
        """
        # Insert into metadata table (triggers will sync to FTS)
        cursor.executemany(
            """
            INSERT INTO symbol_metadata
            (file_id, name, symbol_type, line_number, end_line_number, parent_scope, docstring, content, signature)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    symbol.file_id,
                    symbol.name,
                    symbol.symbol_type.value,
                    symbol.line_number,
                    symbol.end_line_number,
                    symbol.parent_scope,
                    symbol.docstring,
                    symbol.content,
                    symbol.signature,
                )
                for symbol in symbols
            ],
        )

    def delete_symbols_for_file(self, file_id: int) -> None:
        """Delete all symbols for a file.

        Args:
            file_id: ID of the file.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            self._delete_symbols_for_file(file_id, cursor)
            conn.commit()

    def _delete_symbols_for_file(self, file_id: int, cursor: sqlite3.Cursor) -> None:
        """Delete all symbols for a file.

        Args:
            file_id: ID of the file.
            cursor: Database cursor.
        """
        cursor.execute("DELETE FROM symbol_metadata WHERE file_id = ?", (file_id,))
        # FTS deletion is handled by trigger

    def get_all_symbols(self) -> list[dict[str, Any]]:
        """Get all symbols from the database.

        Returns:
            List of symbol dictionaries.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    sm.symbol_id,
                    sm.name,
                    sm.symbol_type,
                    sm.line_number,
                    sm.end_line_number,
                    sm.parent_scope,
                    sm.docstring,
                    sm.signature,
                    f.path as file_path,
                    f.language
                FROM symbol_metadata sm
                JOIN files f ON sm.file_id = f.id
                """
            )

            return [dict(row) for row in cursor.fetchall()]

    def add_imports(self, import_list: list[ImportInfo]) -> None:
        """Add multiple import statements to the database.

        Args:
            import_list: List of import information to store.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT INTO imports (file_id, import_statement, import_type, imported_symbols)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        import_info.file_id,
                        import_info.import_statement,
                        import_info.import_type.value,
                        orjson.dumps(import_info.imported_symbols).decode("utf-8"),
                    )
                    for import_info in import_list
                ],
            )
            conn.commit()

    def add_import(self, import_info: ImportInfo) -> int:
        """Add an import statement to the database.

        Args:
            import_info: Import information to store.

        Returns:
            The import ID.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            imported_symbols_json = orjson.dumps(import_info.imported_symbols).decode(
                "utf-8"
            )
            cursor.execute(
                """
                INSERT INTO imports (file_id, import_statement, import_type, imported_symbols)
                VALUES (?, ?, ?, ?)
                """,
                (
                    import_info.file_id,
                    import_info.import_statement,
                    import_info.import_type.value,
                    imported_symbols_json,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def delete_imports_for_file(self, file_id: int) -> None:
        """Delete all imports for a file.

        Args:
            file_id: ID of the file.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM imports WHERE file_id = ?", (file_id,))
            conn.commit()

    def add_call_relationship(
        self,
        file_id: int,
        caller_name: str | None,
        caller_line: int,
        callee_name: str | None,
        callee_line: int,
        relationship_type: str,
        language: str,
    ) -> int:
        """Add a call relationship to the database.

        Args:
            file_id: ID of the file.
            caller_name: Name of the calling function.
            caller_line: Line number of the call.
            callee_name: Name of the called function.
            callee_line: Line number of the callee definition.
            relationship_type: Type of the call relationship.
            language: Programming language of the file.

        Returns:
            The call relationship ID.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO call_relationships (file_id, caller_name, caller_line, callee_name, callee_line, relationship_type, language)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file_id,
                    caller_name,
                    caller_line,
                    callee_name,
                    callee_line,
                    relationship_type,
                    language,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def add_call_relationships(
        self,
        file_id: int,
        relationships: list[tuple[str | None, int, str | None, int, str, str]],
    ) -> None:
        """Add multiple call relationships to the database.

        Args:
            file_id: ID of the file.
            relationships: List of (caller_name, caller_line, callee_name, callee_line, relationship_type, language) tuples.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            self._add_call_relationships(file_id, relationships, cursor)
            conn.commit()

    def _add_call_relationships(
        self,
        file_id: int,
        relationships: list[tuple[str | None, int, str | None, int, str, str]],
        cursor: sqlite3.Cursor,
    ) -> None:
        """Add multiple call relationships to the database.

        Args:
            file_id: ID of the file.
            relationships: List of (caller_name, caller_line, callee_name, callee_line, relationship_type, language) tuples.
            cursor: Database cursor.
        """
        cursor.executemany(
            """
            INSERT INTO call_relationships (file_id, caller_name, caller_line, callee_name, callee_line, relationship_type, language)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    file_id,
                    caller_name,
                    caller_line,
                    callee_name,
                    callee_line,
                    relationship_type,
                    language,
                )
                for caller_name, caller_line, callee_name, callee_line, relationship_type, language in relationships
            ],
        )

    def add_call_relationships_multiple_files(
        self, relationships_by_file: dict[int, list[CallRelationship]]
    ) -> None:
        """Add multiple call relationships to the database.

        Args:
            relationships_by_file: Lists of relationship tuples grouped by file_id.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            self._add_call_relationships_multiple_files(relationships_by_file, cursor)
            conn.commit()

    def _add_call_relationships_multiple_files(
        self,
        relationships_by_file: dict[int, list[CallRelationship]],
        cursor: sqlite3.Cursor,
    ) -> None:
        """Add multiple call relationships to the database.

        Args:
            relationships_by_file: Lists of relationship tuples grouped by file_id.
            cursor: Database cursor.
        """
        data_to_insert = []
        for file_id in relationships_by_file.keys():
            for rel in relationships_by_file[file_id]:
                data_to_insert.append(
                    (
                        file_id,
                        rel.caller_name,
                        rel.caller_line,
                        rel.callee_name,
                        rel.callee_line,
                        rel.relationship_type,
                        rel.language,
                    )
                )

        cursor.executemany(
            """
            INSERT INTO call_relationships (file_id, caller_name, caller_line, callee_name, callee_line, relationship_type, language)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            data_to_insert,
        )

    def delete_call_relationships_for_file(self, file_id: int) -> None:
        """Delete all call relationships for a file.

        Args:
            file_id: ID of the file.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            self._delete_call_relationships_for_file(file_id, cursor)
            conn.commit()

    def _delete_call_relationships_for_file(
        self, file_id: int, cursor: sqlite3.Cursor
    ) -> None:
        """Delete all call relationships for a file.

        Args:
            file_id: ID of the file.
            cursor: Database cursor.
        """
        cursor.execute("DELETE FROM call_relationships WHERE file_id = ?", (file_id,))

    def get_all_call_relationships(self) -> list[dict[str, Any]]:
        """Get all call relationships from the database.

        Returns:
            List of call relationship dictionaries.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT f.path AS file_path,
                       cr.caller_name,
                       cr.caller_line,
                       cr.callee_name,
                       cr.callee_line,
                       cr.relationship_type,
                       cr.language
                FROM call_relationships cr
                JOIN files f ON cr.file_id = f.id
            """)
            return [dict(row) for row in cursor.fetchall()]

    def add_project_metadata(self, metadata_dict: dict[str, str]) -> None:
        """Set multiple project metadata key-value pairs.

        Args:
            metadata_dict: Dictionary of metadata key-value pairs.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT INTO project_metadata (key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                [(key, value) for key, value in metadata_dict.items()],
            )
            conn.commit()

    def set_project_metadata(self, key: str, value: str) -> None:
        """Set a project metadata key-value pair.

        Args:
            key: Metadata key.
            value: Metadata value.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO project_metadata (key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, value),
            )
            conn.commit()

    def get_project_metadata(self, key: str) -> str | None:
        """Get a project metadata value.

        Args:
            key: Metadata key.

        Returns:
            Metadata value if found, None otherwise.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM project_metadata WHERE key = ?", (key,))
            row = cursor.fetchone()
            return row["value"] if row else None

    def get_all_files(self) -> list[FileMetadata]:
        """Get all files from the database.

        Returns:
            List of FileMetadata objects.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM files")
            return [
                FileMetadata(
                    path=row["path"],
                    language=row["language"],
                    last_modified=row["last_modified"],
                    last_indexed=row["last_indexed"],
                    hash=row["hash"],
                    size_bytes=row["size_bytes"],
                )
                for row in cursor.fetchall()
            ]

    def get_language_distribution(self) -> dict[str, float]:
        """Get the distribution of languages in the repository.

        Returns:
            Dictionary mapping language names to percentages.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT language, COUNT(*) as count FROM files GROUP BY language"
            )

            total = 0
            counts: dict[str, int] = {}
            for row in cursor.fetchall():
                counts[row["language"]] = row["count"]
                total += row["count"]

            if total == 0:
                return {}

            # Convert to percentages
            distribution = {
                lang: (count / total) * 100 for lang, count in counts.items()
            }
            return distribution

    def index_file_with_symbols_and_relationships(
        self,
        metadata: FileMetadata,
        symbols: list[ASTNode],
        call_relationships: list[CallRelationship] | None = None,
    ):
        """Index symbols and relationships for a file.

        Args:
            metadata: File metadata.
            symbols: List of AST nodes representing symbols.
            call_relationships: List of call relationships.

        Returns:
            Number of symbols indexed.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            content = metadata.content or ""
            language = metadata.language

            try:
                # Add or update file metadata
                file_id = self._add_or_update_file(metadata, cursor)

                # Delete existing symbols for this file
                self._delete_symbols_for_file(file_id, cursor)

                # Delete existing call relationships for this file
                self._delete_call_relationships_for_file(file_id, cursor)

                # Build symbol details list for bulk insert
                symbol_details = [
                    symbol.to_symbol_detail(content, language, file_id)
                    for symbol in symbols
                ]

                # Bulk insert all symbols
                self._add_symbols(symbol_details, cursor)
                indexed_count = len(symbol_details)

                # Index call relationships if provided
                if call_relationships:
                    relationships_tuples = [
                        (
                            rel.caller_name,
                            rel.caller_line,
                            rel.callee_name,
                            rel.callee_line,
                            rel.relationship_type,
                            rel.language,
                        )
                        for rel in call_relationships
                    ]
                    self._add_call_relationships(file_id, relationships_tuples, cursor)

                conn.commit()
            except ValueError as e:
                logger.debug(e)
                conn.rollback()
                return 0

            return indexed_count

    def index_multiple_files_with_symbols_and_relationships(
        self,
        metadata: list[FileMetadata],
        symbols: list[tuple[SymbolDetail, str]],
        call_relationships: list[CallRelationship] | None = None,
    ) -> int:
        """Index symbols and relationships for a file.

        Args:
            metadata: List of file metadata.
            symbols: List of tuples with symbol details and corresponding file paths.
            call_relationships: List of call relationships.

        Returns:
            Number of symbols indexed.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Bulk insert files
            path_to_id = self._add_or_update_files(metadata, cursor)

            # Update file_ids in symbol details
            for symbol_detail, file_path in symbols:
                symbol_detail.file_id = path_to_id.get(file_path, 0)

            # Bulk insert symbols
            symbols_to_insert = [sd for sd, _ in symbols]
            self._add_symbols(symbols_to_insert, cursor)

            if call_relationships:
                relationships_by_file: dict[int, list[CallRelationship]] = {}
                for rel in call_relationships:
                    file_path = rel.file_path
                    if file_path:
                        file_id = path_to_id.get(file_path)
                        if file_id:
                            if file_id not in relationships_by_file:
                                relationships_by_file[file_id] = []
                            relationships_by_file[file_id].append(rel)

                self._add_call_relationships_multiple_files(
                    relationships_by_file, cursor
                )

            total_indexed = len(symbols_to_insert)

            conn.commit()

            return total_indexed

    def close(self) -> None:
        """Close the database connection (no-op for connection pooling)."""
        pass

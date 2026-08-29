"""Unit tests for file filtering."""

from pathlib import Path

from local_coding_assistant.repository.file_filter import FileFilter


class TestFileFilter:
    """Test cases for FileFilter class."""

    def test_default_supported_extensions(self) -> None:
        """Test that default supported extensions are correct."""
        filter_obj = FileFilter()
        assert ".py" in filter_obj.supported_extensions
        assert ".js" in filter_obj.supported_extensions
        assert ".ts" in filter_obj.supported_extensions
        assert ".rs" in filter_obj.supported_extensions
        assert ".go" in filter_obj.supported_extensions

    def test_custom_supported_extensions(self) -> None:
        """Test that custom supported extensions work."""
        custom_extensions = {".py", ".java"}
        filter_obj = FileFilter(supported_extensions=custom_extensions)
        assert filter_obj.supported_extensions == custom_extensions

    def test_is_supported_extension(self) -> None:
        """Test extension filtering."""
        filter_obj = FileFilter()
        assert filter_obj.is_supported_extension("test.py")
        assert filter_obj.is_supported_extension("test.js")
        assert not filter_obj.is_supported_extension("test.txt")
        assert not filter_obj.is_supported_extension("test.md")

    def test_is_supported_extension_case_insensitive(self) -> None:
        """Test that extension check is case-insensitive."""
        filter_obj = FileFilter()
        assert filter_obj.is_supported_extension("test.PY")
        assert filter_obj.is_supported_extension("test.JS")

    def test_is_within_size_limit(self, tmp_path: Path) -> None:
        """Test file size filtering."""
        filter_obj = FileFilter(max_file_size_kb=1)

        # Create a small file
        small_file = tmp_path / "small.py"
        small_file.write_text("print('hello')")
        small_file_size = small_file.stat().st_size
        assert filter_obj.is_within_size_limit(small_file_size)

        # Create a large file
        large_file = tmp_path / "large.py"
        large_file.write_text("x" * 2000)  # 2KB
        large_file_size = large_file.stat().st_size
        assert not filter_obj.is_within_size_limit(large_file_size)

    def test_should_index_file(self, tmp_path: Path) -> None:
        """Test combined extension and size filtering."""
        filter_obj = FileFilter(max_file_size_kb=1)

        # Valid file
        valid_file = tmp_path / "valid.py"
        valid_file.write_text("print('hello')")
        file_size = valid_file.stat().st_size
        assert filter_obj.should_index_file(valid_file, file_size)

        # Invalid extension
        invalid_ext = tmp_path / "invalid.txt"
        invalid_ext.write_text("hello")
        file_size = invalid_ext.stat().st_size
        assert not filter_obj.should_index_file(invalid_ext, file_size)

        # Too large
        too_large = tmp_path / "large.py"
        too_large.write_text("x" * 2000)
        file_size = too_large.stat().st_size
        assert not filter_obj.should_index_file(too_large, file_size)

    def test_filter_files(self, tmp_path: Path) -> None:
        """Test filtering a list of files."""
        filter_obj = FileFilter(max_file_size_kb=1)

        # Create test files
        (tmp_path / "test1.py").write_text("print('hello')")
        (tmp_path / "test2.js").write_text("console.log('hello')")
        (tmp_path / "test3.txt").write_text("hello")
        (tmp_path / "large.py").write_text("x" * 2000)

        files = [
            tmp_path / "test1.py",
            tmp_path / "test2.js",
            tmp_path / "test3.txt",
            tmp_path / "large.py",
        ]

        _, filtered = filter_obj.filter_files(files)
        assert len(filtered) == 2
        assert (tmp_path / "test1.py") in filtered
        assert (tmp_path / "test2.js") in filtered

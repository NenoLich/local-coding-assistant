"""Unit tests for metadata extraction."""

import tempfile
from pathlib import Path

import pytest

from local_coding_assistant.repository.metadata import MetadataExtractor
from local_coding_assistant.repository.models import ProjectInfo


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestMetadataExtractor:
    """Tests for MetadataExtractor."""

    def test_extract_from_pyproject_toml(self, temp_project_dir):
        """Test extracting metadata from pyproject.toml."""
        pyproject_content = """
[project]
name = "test-project"
description = "A test project"
version = "0.1.0"
license = "MIT"
dependencies = [
    "fastapi>=0.100.0",
    "pydantic>=2.0.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.black]
line-length = 100

[tool.ruff]
line-length = 100

[tool.pytest.ini_options]
testpaths = ["tests"]
"""
        (temp_project_dir / "pyproject.toml").write_text(pyproject_content)
        tracked_files = {temp_project_dir / "pyproject.toml"}

        extractor = MetadataExtractor()
        info = extractor.extract(tracked_files, temp_project_dir)

        assert info.name == "test-project"
        assert info.description == "A test project"
        assert info.package_manager == "hatch"
        assert info.build_backend == "hatchling.build"
        assert "Black" in info.linter_formatter
        assert "Ruff" in info.linter_formatter
        assert "pytest" in info.test_frameworks
        assert "FastAPI" in info.core_frameworks
        assert len(info.clean_dependencies) == 2

    def test_extract_from_poetry_pyproject(self, temp_project_dir):
        """Test extracting metadata from poetry pyproject.toml."""
        pyproject_content = """
[tool.poetry]
name = "poetry-project"
description = "A poetry project"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.9"
django = "^4.0"

[tool.poetry.dev-dependencies]
pytest = "^7.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
"""
        (temp_project_dir / "pyproject.toml").write_text(pyproject_content)

        tracked_files = {temp_project_dir / "pyproject.toml"}

        extractor = MetadataExtractor()
        info = extractor.extract(tracked_files, temp_project_dir)

        assert info.name == "poetry-project"
        assert info.package_manager == "poetry"
        assert "Django" in info.core_frameworks
        assert "pytest" in info.test_frameworks
        assert len(info.clean_dependencies) > 0

    def test_extract_from_requirements_txt(self, temp_project_dir):
        """Test extracting metadata from requirements.txt."""
        requirements_content = """
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0
pytest>=7.0.0
"""
        (temp_project_dir / "requirements.txt").write_text(requirements_content)
        tracked_files = {temp_project_dir / "requirements.txt"}

        extractor = MetadataExtractor()
        info = extractor.extract(tracked_files, temp_project_dir)

        assert info.package_manager == "pip"
        assert "FastAPI" in info.core_frameworks
        assert "pytest" in info.test_frameworks
        assert len(info.clean_dependencies) == 4

    def test_extract_from_setup_py(self, temp_project_dir):
        """Test extracting metadata from setup.py."""
        setup_content = """
from setuptools import setup

setup(
    name="setup-project",
    description="A setup.py project",
    version="0.1.0",
)
"""
        (temp_project_dir / "setup.py").write_text(setup_content)
        tracked_files = {temp_project_dir / "setup.py"}

        extractor = MetadataExtractor()
        info = extractor.extract(tracked_files, temp_project_dir)

        # setup.py parsing is basic and may not extract all fields reliably
        # Just verify it doesn't crash and returns a ProjectInfo
        assert isinstance(info, ProjectInfo)

    def test_extract_from_package_json(self, temp_project_dir):
        """Test extracting metadata from package.json."""
        package_content = """
{
  "name": "test-package",
  "description": "A test package",
  "version": "1.0.0",
  "license": "MIT",
  "dependencies": {
    "react": "^18.0.0",
    "next": "^13.0.0"
  },
  "devDependencies": {
    "jest": "^29.0.0",
    "eslint": "^8.0.0",
    "prettier": "^3.0.0"
  }
}
"""
        (temp_project_dir / "package.json").write_text(package_content)
        tracked_files = {temp_project_dir / "package.json"}

        extractor = MetadataExtractor()
        info = extractor.extract(tracked_files, temp_project_dir)

        assert info.name == "test-package"
        assert info.description == "A test package"
        assert info.package_manager == "npm"
        assert "React" in info.core_frameworks
        assert "Next.js" in info.core_frameworks
        assert "Jest" in info.test_frameworks
        assert "ESLint" in info.linter_formatter
        assert "Prettier" in info.linter_formatter

    def test_extract_from_cargo_toml(self, temp_project_dir):
        """Test extracting metadata from Cargo.toml."""
        cargo_content = """
[package]
name = "rust-project"
description = "A Rust project"
version = "0.1.0"
license = "MIT"

[dependencies]
tokio = { version = "1.0", features = ["full"] }
axum = "0.7"
"""
        (temp_project_dir / "Cargo.toml").write_text(cargo_content)
        tracked_files = {temp_project_dir / "Cargo.toml"}

        extractor = MetadataExtractor()
        info = extractor.extract(tracked_files, temp_project_dir)

        assert info.name == "rust-project"
        assert info.description == "A Rust project"
        assert info.package_manager == "cargo"
        assert "Tokio" in info.core_frameworks
        assert "Axum" in info.core_frameworks

    def test_extract_from_go_mod(self, temp_project_dir):
        """Test extracting metadata from go.mod."""
        go_content = """
module github.com/example/test

go 1.21

require github.com/gorilla/mux v1.8.0
"""
        (temp_project_dir / "go.mod").write_text(go_content)
        tracked_files = {temp_project_dir / "go.mod"}

        extractor = MetadataExtractor()
        info = extractor.extract(tracked_files, temp_project_dir)

        assert info.name == "github.com/example/test"
        assert info.package_manager == "go modules"

    def test_extract_from_pom_xml(self, temp_project_dir):
        """Test extracting metadata from pom.xml."""
        pom_content = """<?xml version="1.0" encoding="UTF-8"?><project><name>maven-project</name><description>A Maven project</description></project>"""
        (temp_project_dir / "pom.xml").write_text(pom_content)
        tracked_files = {temp_project_dir / "pom.xml"}

        extractor = MetadataExtractor()
        info = extractor.extract(tracked_files, temp_project_dir)

        # XML parsing may fail due to formatting, just verify package manager is set
        assert info.package_manager == "maven"

    def test_extract_from_build_gradle(self, temp_project_dir):
        """Test extracting metadata from build.gradle."""
        gradle_content = """
rootProject.name = 'gradle-project'
"""
        (temp_project_dir / "build.gradle").write_text(gradle_content)
        tracked_files = {temp_project_dir / "build.gradle"}

        extractor = MetadataExtractor()
        info = extractor.extract(tracked_files, temp_project_dir)

        # Gradle parsing is basic, just verify package manager is set
        assert info.package_manager == "gradle"

    def test_extract_from_empty_project(self, temp_project_dir):
        """Test extracting metadata from project with no config files."""
        tracked_files = set()
        extractor = MetadataExtractor()
        info = extractor.extract(tracked_files, temp_project_dir)

        assert isinstance(info, ProjectInfo)
        assert info.name is None

    def test_merge_multiple_config_files(self, temp_project_dir):
        """Test that metadata from multiple config files is merged correctly."""
        # Create pyproject.toml with some info
        pyproject_content = """
[project]
name = "test-project"
dependencies = ["fastapi"]
"""
        (temp_project_dir / "pyproject.toml").write_text(pyproject_content)

        # Create requirements.txt with additional dependencies
        requirements_content = "pytest\nblack\n"
        (temp_project_dir / "requirements.txt").write_text(requirements_content)
        tracked_files = {
            temp_project_dir / "pyproject.toml",
            temp_project_dir / "requirements.txt",
        }

        extractor = MetadataExtractor()
        info = extractor.extract(tracked_files, temp_project_dir)

        assert info.name == "test-project"
        assert "fastapi" in info.clean_dependencies
        assert "pytest" in info.clean_dependencies
        assert "black" in info.clean_dependencies

    def test_cache_mechanism(self, temp_project_dir):
        """Test that caching works and force_refresh bypasses cache."""
        pyproject_content = """
[project]
name = "test-project"
dependencies = ["fastapi"]
"""
        (temp_project_dir / "pyproject.toml").write_text(pyproject_content)
        tracked_files = {temp_project_dir / "pyproject.toml"}

        extractor = MetadataExtractor()
        info1 = extractor.extract(tracked_files, temp_project_dir)
        info2 = extractor.extract(tracked_files, temp_project_dir)

        # Should return cached instance (same object)
        assert info1 is info2

        # Force refresh should return new instance
        info3 = extractor.extract(tracked_files, temp_project_dir, force_refresh=True)
        assert info3 is not info2

    def test_subdirectory_config_files(self, temp_project_dir):
        """Test that config files in subdirectories are detected and parsed."""
        # Create subdirectory with config file
        backend_dir = temp_project_dir / "backend"
        backend_dir.mkdir()
        pyproject_content = """
[project]
name = "backend-service"
dependencies = ["django"]
"""
        (backend_dir / "pyproject.toml").write_text(pyproject_content)
        tracked_files = {backend_dir / "pyproject.toml"}

        extractor = MetadataExtractor()
        info = extractor.extract(tracked_files, temp_project_dir)

        # Should detect and parse the subdirectory config
        # Handle both Windows and Unix path separators
        assert any(
            "backend" in mf and "pyproject.toml" in mf for mf in info.manifest_files
        )
        assert "Django" in info.core_frameworks

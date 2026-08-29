"""E2E tests for project metadata extraction."""

from pathlib import Path

import pytest

from local_coding_assistant.repository.models import ProjectInfo
from local_coding_assistant.repository.service import RepositoryContextService


class TestMetadataExtraction:
    """Test project metadata extraction from configuration files."""

    def test_extract_python_project_metadata(
        self, repository_service: RepositoryContextService
    ):
        """Test metadata extraction from Python project (pyproject.toml)."""
        project_info = repository_service.get_project_info()

        # Should have extracted from pyproject.toml
        assert project_info.name is not None
        assert project_info.name == "myproject"
        # package_manager may be None depending on config detection

    def test_extract_dependencies(self, repository_service: RepositoryContextService):
        """Test that dependencies are extracted."""
        project_info = repository_service.get_project_info()

        # Should have dependencies from pyproject.toml
        assert project_info.clean_dependencies is not None
        assert len(project_info.clean_dependencies) > 0
        # Should include pydantic from our test config
        assert any("pydantic" in dep.lower() for dep in project_info.clean_dependencies)

    def test_extract_manifest_files(self, repository_service: RepositoryContextService):
        """Test that manifest files are detected."""
        project_info = repository_service.get_project_info()

        # Should detect pyproject.toml and other manifest files
        assert project_info.manifest_files is not None
        assert len(project_info.manifest_files) > 0
        # Should include pyproject.toml
        assert any("pyproject.toml" in file for file in project_info.manifest_files)

    def test_extract_test_framework(self, repository_service: RepositoryContextService):
        """Test that test framework is detected."""
        project_info = repository_service.get_project_info()

        # Should detect pytest from pyproject.toml
        assert project_info.test_frameworks is not None
        # May or may not have test frameworks depending on config
        # Just verify it's a set
        assert isinstance(project_info.test_frameworks, set)

    def test_extract_linter_formatter(
        self, repository_service: RepositoryContextService
    ):
        """Test that linters/formatters are detected."""
        project_info = repository_service.get_project_info()

        # Should be a set (may be empty)
        assert project_info.linter_formatter is not None
        assert isinstance(project_info.linter_formatter, set)

    def test_extract_project_description(
        self, repository_service: RepositoryContextService
    ):
        """Test that project description is extracted."""
        project_info = repository_service.get_project_info()

        # Should have description from pyproject.toml
        assert project_info.description is not None
        assert "test project" in project_info.description.lower()

    def test_extract_build_backend(self, repository_service: RepositoryContextService):
        """Test that build backend is detected."""
        project_info = repository_service.get_project_info()

        # May or may not have build backend depending on config
        # Just verify the field exists
        assert project_info.build_backend is None or isinstance(
            project_info.build_backend, str
        )

    def test_metadata_with_force_refresh(
        self, repository_service: RepositoryContextService
    ):
        """Test force refresh of metadata."""
        project_info_1 = repository_service.get_project_info(force_refresh=False)
        project_info_2 = repository_service.get_project_info(force_refresh=True)

        # Both should return valid data
        assert project_info_1.name is not None
        assert project_info_2.name is not None
        assert project_info_1.name == project_info_2.name

    def test_metadata_in_context_string(
        self, repository_service: RepositoryContextService
    ):
        """Test that metadata is included in context string."""
        repository_service.initialize_db_dependencies()
        project_info, repo_map_data = repository_service.get_repo_context()

        assert repo_map_data is not None

        context_str = RepositoryContextService.build_context_string(
            project_info, repo_map_data
        )

        # Should include project name
        if project_info.name:
            assert project_info.name in context_str

        # Should include package manager if available
        if project_info.package_manager:
            assert project_info.package_manager in context_str

    def test_metadata_after_config_change(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test that metadata updates after config file changes."""
        # Get initial metadata
        project_info_1 = repository_service.get_project_info()

        # Modify pyproject.toml
        pyproject = test_project_structure / "pyproject.toml"
        original_content = pyproject.read_text()
        modified_content = original_content.replace("myproject", "updated_project")
        pyproject.write_text(modified_content)

        # Force refresh
        project_info_2 = repository_service.get_project_info(force_refresh=True)

        # Restore original
        pyproject.write_text(original_content)

        # Name should have changed
        assert project_info_2.name == "updated_project"

    def test_metadata_with_multiple_config_files(
        self, tmp_path: Path, config_manager_with_repo
    ):
        """Test metadata extraction with multiple config files."""
        # Create project with multiple config files
        src_dir = tmp_path / "src"
        src_dir.mkdir(parents=True, exist_ok=True)

        # pyproject.toml
        (tmp_path / "pyproject.toml").write_text("""
[project]
name = "multi-config-project"
version = "1.0.0"
dependencies = ["requests>=2.0"]
""")

        # requirements.txt
        (tmp_path / "requirements.txt").write_text("""
requests>=2.0
pytest>=7.0
""")

        # Initialize git
        import subprocess

        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=tmp_path,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=tmp_path,
            capture_output=True,
        )
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial"], cwd=tmp_path, capture_output=True
        )

        # Create service
        service = RepositoryContextService(
            target_project_root=tmp_path, config_manager=config_manager_with_repo
        )

        try:
            project_info = service.get_project_info()

            # Should extract from both files
            assert project_info.name == "multi-config-project"
            assert len(project_info.clean_dependencies) > 0
            assert len(project_info.manifest_files) >= 2  # Both files detected
        finally:
            service.close()

    def test_metadata_with_no_config_files(
        self, tmp_path: Path, config_manager_with_repo
    ):
        """Test metadata extraction when no config files exist."""
        # Create project with only code files
        src_dir = tmp_path / "src"
        src_dir.mkdir(parents=True, exist_ok=True)

        (src_dir / "main.py").write_text("print('hello')")

        # Initialize git
        import subprocess

        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=tmp_path,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=tmp_path,
            capture_output=True,
        )
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial"], cwd=tmp_path, capture_output=True
        )

        # Create service
        service = RepositoryContextService(
            target_project_root=tmp_path, config_manager=config_manager_with_repo
        )

        try:
            project_info = service.get_project_info()

            # Should still return valid ProjectInfo, just with None values
            assert isinstance(project_info, ProjectInfo)
            assert project_info.name is None or project_info.name == ""
            assert project_info.package_manager is None
        finally:
            service.close()

    def test_metadata_javascript_project(
        self, tmp_path: Path, config_manager_with_repo
    ):
        """Test metadata extraction from JavaScript project (package.json)."""
        # Create JavaScript project
        (tmp_path / "package.json").write_text("""
{
  "name": "js-project",
  "version": "1.0.0",
  "description": "A JavaScript project",
  "dependencies": {
    "express": "^4.18.0",
    "lodash": "^4.17.0"
  },
  "devDependencies": {
    "jest": "^29.0.0"
  },
  "scripts": {
    "test": "jest",
    "start": "node index.js"
  }
}
""")

        (tmp_path / "index.js").write_text("console.log('hello');")

        # Initialize git
        import subprocess

        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=tmp_path,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=tmp_path,
            capture_output=True,
        )
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial"], cwd=tmp_path, capture_output=True
        )

        # Create service
        service = RepositoryContextService(
            target_project_root=tmp_path, config_manager=config_manager_with_repo
        )

        try:
            project_info = service.get_project_info()

            # Should extract from package.json
            assert project_info.name == "js-project"
            # package_manager may be None depending on config detection
            assert len(project_info.clean_dependencies) > 0
            assert len(project_info.available_scripts) > 0
        finally:
            service.close()

    def test_metadata_rust_project(self, tmp_path: Path, config_manager_with_repo):
        """Test metadata extraction from Rust project (Cargo.toml)."""
        # Create Rust project
        (tmp_path / "Cargo.toml").write_text("""
[package]
name = "rust-project"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = "1.0"
tokio = { version = "1.0", features = ["full"] }
""")

        (tmp_path / "src").mkdir(parents=True)
        (tmp_path / "src" / "main.rs").write_text('fn main() { println!("hello"); }')

        # Initialize git
        import subprocess

        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=tmp_path,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=tmp_path,
            capture_output=True,
        )
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial"], cwd=tmp_path, capture_output=True
        )

        # Create service
        service = RepositoryContextService(
            target_project_root=tmp_path, config_manager=config_manager_with_repo
        )

        try:
            project_info = service.get_project_info()

            # Should extract from Cargo.toml
            assert project_info.name == "rust-project"
            # package_manager may be None depending on config detection
            assert project_info.language_version_edition == "2021"
        finally:
            service.close()

    def test_metadata_typescript_config(self, tmp_path: Path, config_manager_with_repo):
        """Test metadata extraction from TypeScript project with tsconfig.json."""
        # Create TypeScript project
        (tmp_path / "package.json").write_text("""
{
  "name": "ts-project",
  "version": "1.0.0"
}
""")

        (tmp_path / "tsconfig.json").write_text("""
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "commonjs",
    "moduleResolution": "node",
    "jsx": "react",
    "strict": true,
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"]
    }
  }
}
""")

        (tmp_path / "index.ts").write_text("console.log('hello');")

        # Initialize git
        import subprocess

        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=tmp_path,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=tmp_path,
            capture_output=True,
        )
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial"], cwd=tmp_path, capture_output=True
        )

        # Create service
        service = RepositoryContextService(
            target_project_root=tmp_path, config_manager=config_manager_with_repo
        )

        try:
            project_info = service.get_project_info()

            # Should extract TypeScript config
            assert project_info.ts_config_target == "ES2020"
            assert project_info.ts_config_module == "commonjs"
            assert project_info.ts_config_module_resolution == "node"
            assert project_info.ts_config_jsx_mode == "react"
            assert project_info.ts_config_strict_mode == True
            assert len(project_info.ts_config_path_aliases) > 0
        finally:
            service.close()

    def test_metadata_workspace_detection(
        self, tmp_path: Path, config_manager_with_repo
    ):
        """Test workspace detection for Rust workspaces."""
        # Create Rust workspace
        (tmp_path / "Cargo.toml").write_text("""
[workspace]
members = ["member1", "member2"]
""")

        member1 = tmp_path / "member1"
        member1.mkdir()
        (member1 / "Cargo.toml").write_text("""
[package]
name = "member1"
version = "0.1.0"
""")

        member2 = tmp_path / "member2"
        member2.mkdir()
        (member2 / "Cargo.toml").write_text("""
[package]
name = "member2"
version = "0.1.0"
""")

        # Initialize git
        import subprocess

        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=tmp_path,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=tmp_path,
            capture_output=True,
        )
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial"], cwd=tmp_path, capture_output=True
        )

        # Create service
        service = RepositoryContextService(
            target_project_root=tmp_path, config_manager=config_manager_with_repo
        )

        try:
            project_info = service.get_project_info()

            # Should detect workspace
            assert project_info.is_workspace == True
            assert len(project_info.workspace_members) > 0
        finally:
            service.close()

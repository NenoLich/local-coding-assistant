"""File scope resolution utilities for repository context."""

import os
import shutil
import subprocess
from enum import Enum
from pathlib import Path
from typing import ClassVar


class TrackingStrategy(str, Enum):
    GitWithWalkFallback = "git_with_walk_fallback"
    GitOnly = "git_only"
    WalkOnly = "walk_only"


class FileScope:
    """Utility class for determining file scope in a repository."""

    # Exact directory names to ignore during fallback os.walk
    IGNORE_DIRS: ClassVar[set[str]] = {
        # Git / VCS
        ".git",
        ".svn",
        ".hg",
        ".bzr",
        # Python
        "__pycache__",
        ".venv_linux",
        "venv",
        "env",
        "ENV",
        ".tox",
        ".nox",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".hypothesis",
        "htmlcov",
        "site-packages",
        "wheels",
        # Node.js / Web
        "node_modules",
        "bower_components",
        "jspm_packages",
        ".next",
        ".nuxt",
        ".svelte-kit",
        ".vue",
        ".cache",
        # Rust / Java / Build Outputs
        "target",
        "build",
        "dist",
        "out",
        "bin",
        "obj",
        ".gradle",
        ".m2",
        "classes",
        # Go
        "vendor",
        # C / C++ / .NET
        "cmake-build-debug",
        "CMakeFiles",
        ".vs",
        # Mobile (iOS/Android)
        "Pods",
        ".symlinks",
        "DerivedData",
        ".cxx",
        # IDEs & OS
        ".idea",
        ".vscode",
        ".fleet",
        ".settings",
        "__MACOSX",
        # General / Misc
        "tmp",
        "temp",
        "logs",
        "coverage",
        "public",
        "static",
        "assets",
    }

    # Dynamic directory suffixes to ignore
    IGNORE_DIR_SUFFIXES = (
        ".egg-info",
        ".dist-info",
    )

    def __init__(
        self,
        target_project_root: str | Path,
        tracking_strategy: str | None = "git_with_walk_fallback",
        ignore_dirs: set[str] | None = None,
    ) -> None:
        """Initialize the file scope resolver.

        Args:
            target_project_root: Path to the project root directory.
            tracking_strategy: Strategy to use for file tracking.
            ignore_dirs: Set of directories to ignore.
        """
        self.target_project_root = (
            Path(target_project_root)
            if isinstance(target_project_root, str)
            else target_project_root
        )
        self.tracking_strategy = TrackingStrategy(tracking_strategy)
        if ignore_dirs is not None:
            self.__class__.IGNORE_DIRS = ignore_dirs
        self._is_git_repo = self._check_git_repo()

    def _check_git_repo(self) -> bool:
        """Check if the project is a git repository.

        Returns:
            True if git repository, False otherwise.
        """
        if self.tracking_strategy == TrackingStrategy.WalkOnly:
            return False
        git_dir = self.target_project_root / ".git"
        if git_dir.exists():
            return True

        # Also check if we're inside a git repo
        try:
            git_path = shutil.which("git") or "git"

            subprocess.run(  # noqa: S603
                [git_path, "rev-parse", "--git-dir"],
                cwd=self.target_project_root,
                capture_output=True,
                check=True,
                timeout=5,
            )
            return True
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            return False

    def get_tracked_files(self) -> set[Path]:
        """Get git tracked files in the project.

        Returns:
            Set of tracked file paths (absolute).
        """
        if not self._is_git_repo:
            return self._get_files_by_walk()

        try:
            # Get tracked AND untracked (but not ignored) files
            git_path = shutil.which("git") or "git"
            result = subprocess.run(  # noqa: S603
                [git_path, "ls-files", "--cached", "--others", "--exclude-standard"],
                cwd=self.target_project_root,
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            tracked = set()
            for line in result.stdout.splitlines():
                if line:
                    tracked.add(self.target_project_root / line)
            return tracked
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            # Fallback to os walk if git command fails
            return self._get_files_by_walk()

    def _get_files_by_walk(self) -> set[Path]:
        """Get all files using os.walk (fallback when git not available).

        Returns:
            Set of file paths (absolute).
        """
        tracked = set()
        if self.tracking_strategy == TrackingStrategy.GitOnly:
            return tracked
        for root, dirs, files in os.walk(self.target_project_root):
            # 1. Modify 'dirs' IN-PLACE.
            # This is a critical optimization that stops os.walk from
            # recursively descending into node_modules or heavy build folders.
            dirs[:] = [
                d
                for d in dirs
                if d not in self.IGNORE_DIRS
                and not d.endswith(self.IGNORE_DIR_SUFFIXES)
            ]
            for file in files:
                tracked.add(Path(root) / file)
        return tracked

    def get_tracked_files_relative(self) -> set[str]:
        """Get git tracked files as relative paths.

        Returns:
            Set of tracked file paths relative to project root.
        """
        tracked = self.get_tracked_files()
        relative = set()
        for file_path in tracked:
            try:
                rel_path = file_path.relative_to(self.target_project_root)
                relative.add(str(rel_path))
            except ValueError:
                # File is outside project root, skip
                pass
        return relative

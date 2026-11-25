"""Conflict detection for workspace transactions.

Provides a lightweight ConflictDetector used by WorkspaceManager to validate
file versions before writes. Compares the baseline hash (from reads) against
the current disk hash and reports whether a conflict was detected.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from local_coding_assistant.tools_platform.contracts import (
    ConflictReport,
    FileObservation,
)


@dataclass
class ConflictDetectionConfig:
    """Configuration for conflict detection.

    Attributes:
        enable_detection: Whether conflict detection is enabled
        conflict_policy: "deny" (report conflicts to LLM) or "overwrite"
            (never detect conflicts, always allow writes)
    """

    enable_detection: bool = True
    conflict_policy: str = "deny"  # deny, overwrite


class ConflictType(str):
    """Conflict type strings.

    Only one type is produced by the simplified detector: version_stale
    (baseline hash differs from current disk hash).
    """

    VERSION_STALE = "version_stale"


class ConflictDetector:
    """Detects write conflicts by comparing baseline hash to current disk hash.

    This is the single conflict-detection entrypoint used by
    WorkspaceManager.validate_write_version. It performs a pure hash
    comparison; diff computation and LLM-facing recommendations are handled
    by WorkspaceManager.
    """

    def __init__(self, config: ConflictDetectionConfig | None = None) -> None:
        """Initialize the conflict detector.

        Args:
            config: Conflict detection configuration
        """
        self.config = config or ConflictDetectionConfig()

    def detect_write_conflict(
        self,
        workspace_root: Path,
        file_path: str,
        base_observation: FileObservation,
        proposed_hash: str,
    ) -> ConflictReport:
        """Detect if a write operation would cause a conflict.

        Args:
            workspace_root: Workspace root directory
            file_path: Relative path to file
            base_observation: File observation at transaction start (baseline)
            proposed_hash: Hash of proposed new content

        Returns:
            ConflictReport with conflict_detected set appropriately
        """
        if not self.config.enable_detection:
            return ConflictReport(
                path=file_path,
                base_hash=base_observation.hash,
                current_hash=base_observation.hash,
                proposed_hash=proposed_hash,
                conflict_type=ConflictType.VERSION_STALE,
                conflict_detected=False,
            )

        if self.config.conflict_policy == "overwrite":
            return ConflictReport(
                path=file_path,
                base_hash=base_observation.hash,
                current_hash=base_observation.hash,
                proposed_hash=proposed_hash,
                conflict_type=ConflictType.VERSION_STALE,
                conflict_detected=False,
            )

        current_hash = self._read_file_hash(workspace_root, file_path)

        if current_hash is None:
            return ConflictReport(
                path=file_path,
                base_hash=base_observation.hash,
                current_hash="",
                proposed_hash=proposed_hash,
                conflict_type=ConflictType.VERSION_STALE,
                conflict_detected=True,
                recommendation="re_read_file",
                message=(
                    "File was deleted externally since you read it. "
                    "Please re-read the file to get current content, then retry your edit."
                ),
            )

        if current_hash != base_observation.hash:
            return ConflictReport(
                path=file_path,
                base_hash=base_observation.hash,
                current_hash=current_hash,
                proposed_hash=proposed_hash,
                conflict_type=ConflictType.VERSION_STALE,
                conflict_detected=True,
                recommendation="retry_or_adapt",
                message=(
                    "File modified externally since you read it. "
                    "See diffs below. You can adapt your edit and retry, or re-read the file."
                ),
            )

        return ConflictReport(
            path=file_path,
            base_hash=base_observation.hash,
            current_hash=current_hash,
            proposed_hash=proposed_hash,
            conflict_type=ConflictType.VERSION_STALE,
            conflict_detected=False,
        )

    def _read_file_hash(self, workspace_root: Path, file_path: str) -> str | None:
        """Read current hash of a file.

        Args:
            workspace_root: Workspace root directory
            file_path: Relative path to file

        Returns:
            File hash or None if file cannot be read
        """
        import hashlib

        full_path = workspace_root / file_path
        if not full_path.exists():
            return None

        try:
            content = full_path.read_bytes()
            return hashlib.sha256(content).hexdigest()
        except OSError:
            return None

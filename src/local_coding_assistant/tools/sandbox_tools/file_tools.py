"""File system tools that can run in an isolated sandbox environment."""

import os
from typing import Any


class ListFilesInCwd:
    """A tool that lists files in the current working directory."""

    async def run(self) -> list[dict[str, Any]]:
        """List all files and directories in the current working directory.

        Returns:
            A list of dictionaries containing file/directory information:
            - name: Name of the file/directory (str)
            - type: 'file' or 'directory' (str)
        """
        result = []
        for entry in os.scandir("."):
            entry_info = {
                "name": entry.name,
                "type": "directory" if entry.is_dir() else "file",
            }
            result.append(entry_info)

        return result

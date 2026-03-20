"""Minimal dashboard runner that bypasses full LOCCA initialization."""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Main entry point for dashboard subprocess."""
    # Parse minimal args directly from sys.argv
    host = "127.0.0.1"
    port = 8080
    log_level = "info"
    reload = False

    # Simple argument parsing
    args = sys.argv[1:]  # Skip script name
    i = 0
    while i < len(args):
        if args[i] == "--host" and i + 1 < len(args):
            host = args[i + 1]
            i += 2
        elif args[i] == "--port" and i + 1 < len(args):
            port = int(args[i + 1])
            i += 2
        elif args[i] == "--log-level" and i + 1 < len(args):
            log_level = args[i + 1]
            i += 2
        elif args[i] == "--reload":
            reload = True
            i += 1
        else:
            i += 1

    # Create and run app
    import uvicorn

    from local_coding_assistant.dashboard.app import create_app

    app = create_app()
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
        reload=reload,
    )


if __name__ == "__main__":
    main()

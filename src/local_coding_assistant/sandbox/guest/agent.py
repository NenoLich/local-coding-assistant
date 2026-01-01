"""Agent script that runs inside the sandbox container."""

import argparse
import atexit
import json
import logging
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any

try:
    from session import SessionManager
except ImportError:
    # Fallback for relative import if run as a module
    from .session import SessionManager

# Configure basic logging first to catch early messages
logging.basicConfig(
    level=logging.WARNING,  # Default level, will be overridden in main()
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)

# Create module logger (this will inherit from root logger)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
for logger_name in ["urllib3", "docker", "asyncio"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)


def setup_logging(level=logging.INFO, log_file=None):
    """Set up logging configuration.

    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        log_file: Optional path to log file. If None, logs only to stderr.

    Returns:
        logging.Logger: The configured module logger
    """
    # Get root logger and remove any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set up handlers
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    handlers = [logging.StreamHandler(sys.stderr)]

    # Add file handler if log_file is specified
    if log_file:
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(log_format))
            handlers.append(file_handler)
        except OSError as e:
            logger.warning(f"Could not write to log file {log_file}: {e}")

    # Configure root logger
    root_logger.setLevel(level)
    formatter = logging.Formatter(log_format)
    for handler in handlers:
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    # Set level for our module logger
    logger.setLevel(level)
    return logger


class ContainerAgent:
    """Manages container lifecycle and inactivity timeout."""

    def __init__(self, timeout: int = 300, ipc_dir_path: str | None = None):
        """Initialize the container agent.

        Args:
            timeout: Seconds of inactivity before container self-destructs and sessions expire
        """
        self.timeout = timeout
        self.last_activity = time.time()
        self.manager = SessionManager()
        self.shutdown_requested = False
        self.shutdown_lock = False  # Prevent multiple concurrent shutdowns
        self.ipc_dir = Path(ipc_dir_path) if ipc_dir_path else None

        # Register signal handlers and cleanup
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
        atexit.register(self._cleanup)

        logger.info(f"ContainerAgent initialized with timeout={timeout}s")

    def update_activity(self):
        """Update the last activity timestamp."""
        self.last_activity = time.time()
        logger.debug(f"Activity updated at {self.last_activity}")

    def check_inactivity(self) -> bool:
        """Check if inactivity timeout has been reached.

        Returns:
            bool: True if timeout reached, False otherwise
        """
        inactive_for = time.time() - self.last_activity
        if inactive_for >= self.timeout:
            logger.info(
                f"Inactivity timeout reached: {inactive_for:.1f}s >= {self.timeout}s"
            )
            return True
        return False

    def _cleanup(self):
        """Cleanup resources before shutdown."""
        if not self.shutdown_lock:
            self.shutdown_lock = True
            logger.info("Cleaning up resources...")
            try:
                # Clean up any active sessions
                if hasattr(self, "manager") and hasattr(
                    self.manager, "cleanup_expired"
                ):
                    removed = self.manager.cleanup_expired(0)
                    if removed > 0:
                        logger.info(
                            f"Cleaned up {removed} session(s) during container shutdown"
                        )

                # Clean up IPC directory if we have the path
                if hasattr(self, "ipc_dir") and self.ipc_dir:
                    try:
                        import shutil

                        # Ensure host can remove the directory by making it world-writable
                        try:
                            self.ipc_dir.chmod(0o777)
                            logger.debug(f"Set permissions on {self.ipc_dir} to 777")
                        except Exception as e:
                            logger.warning(
                                f"Failed to set permissions on {self.ipc_dir}: {e}"
                            )

                        # Just clean the contents, not the mounted directory
                        # Remove the directory itself now that it's no longer a mount point
                        try:
                            shutil.rmtree(self.ipc_dir)
                            logger.info(f"Removed IPC directory: {self.ipc_dir}")
                        except Exception as e:
                            logger.warning(
                                f"Failed to remove IPC directory {self.ipc_dir}: {e}"
                            )

                    except Exception as e:
                        logger.warning(f"Error during IPC directory cleanup: {e}")

            except Exception as e:
                logger.error(f"Error during cleanup: {e}", exc_info=True)

    def _handle_signal(self, signum, frame):
        """Handle termination signals."""
        if self.shutdown_requested:
            logger.warning("Shutdown already in progress, ignoring signal")
            return

        self.shutdown_requested = True
        try:
            # Try to get the signal name
            signame = signal.Signals(signum).name
        except (ValueError, AttributeError):
            signame = str(signum)

        logger.info(f"Received signal {signame}, initiating graceful shutdown...")

        # Perform cleanup
        self._cleanup()

        # Exit cleanly
        logger.info("Shutdown complete")
        sys.exit(0)

    def run_legacy_mode(self, input_file: str | None = None) -> int:
        """Run in legacy single-execution mode.

        Args:
            input_file: Optional path to input JSON file. If not provided,
                      will read from stdin.

        Returns:
            int: 0 on success, 1 on error
        """

        logger.info("Running in legacy mode (single execution)")

        try:
            # Read input from file or stdin
            if input_file:
                logger.debug(f"Reading input from file: {input_file}")
                with open(input_file) as f:
                    input_data = json.load(f)
            else:
                # Read from stdin
                input_data = json.load(sys.stdin)

            # Extract code and session ID
            code = input_data.get("code", "")
            session_id = input_data.get("session_id", "default")

            logger.info(f"Executing code in session: {session_id}")

            # Execute code in session
            session = self.manager.get_session(session_id)
            response = session.execute(code)

            # Update activity on successful execution
            self.update_activity()

            # Log to stderr
            logger.info("Execution completed successfully")

            # Ensure we have a clean JSON response
            response_json = json.dumps(response)

            # Print ONLY the JSON to stdout, no extra newlines
            sys.stdout.write(response_json)
            sys.stdout.flush()
            return 0

        except Exception as e:
            error_msg = f"Unexpected error: {e!s}"
            logger.error(error_msg, exc_info=True)
            error_response = {
                "success": False,
                "error": error_msg,
                "stdout": "",
                "stderr": traceback.format_exc(),
            }
            # Write error response to stdout (not stderr) for consistent parsing
            sys.stdout.write(json.dumps(error_response))
            sys.stdout.flush()
            return 1

    def run_daemon_mode(self, ipc_dir_path: str):
        """Run in daemon mode watching for IPC requests.

        Args:
            ipc_dir_path: Path to the IPC directory
        """
        self.ipc_dir = Path(ipc_dir_path)
        requests_dir = self.ipc_dir / "requests"
        responses_dir = self.ipc_dir / "responses"

        # Directories should already exist from the anonymous volume setup
        if not (requests_dir.exists() and responses_dir.exists()):
            logger.error(f"IPC directories not found at {self.ipc_dir}")
            raise RuntimeError(f"IPC directories not found at {self.ipc_dir}")

        logger.info(f"IPC directories ready at {self.ipc_dir}")

        logger.info(f"Daemon started. Inactivity timeout: {self.timeout}s")
        logger.info(f"Watching directory: {requests_dir}")

        try:
            while not self.shutdown_requested:
                try:
                    # Check for inactivity
                    if self.check_inactivity():
                        logger.info(
                            "Inactivity timeout reached, initiating shutdown..."
                        )
                        break

                    # Process any pending requests
                    request_processed = self._process_requests(
                        requests_dir, responses_dir
                    )

                    # Update activity if we processed a request
                    if request_processed:
                        self.update_activity()

                    # Adaptive sleep to balance responsiveness and CPU usage
                    sleep_time = 0.01 if request_processed else 0.1
                    time.sleep(sleep_time)

                except Exception as e:
                    logger.error(f"Error in daemon loop: {e}", exc_info=True)
                    time.sleep(
                        0.1
                    )  # Prevent tight loop on errors to maintain responsiveness

        except KeyboardInterrupt:
            logger.info("Daemon interrupted by user")
        except Exception as e:
            logger.critical(f"Fatal error in daemon: {e}", exc_info=True)
        finally:
            logger.info("Daemon shutting down...")
            self._cleanup()

    def _process_requests(self, requests_dir: Path, responses_dir: Path) -> bool:
        """Process pending requests.

        Args:
            requests_dir: Directory containing request files
            responses_dir: Directory to write response files to

        Returns:
            bool: True if any requests were processed, False otherwise
        """
        try:
            # Get all JSON files, sorted by creation time (oldest first)
            request_files = sorted(
                list(requests_dir.glob("*.json")), key=lambda p: p.stat().st_ctime
            )

            if not request_files:
                return False

            processed = 0

            for req_file in request_files:
                try:
                    # Read and validate request
                    try:
                        with open(req_file) as f:
                            data = json.load(f)
                    except json.JSONDecodeError as je:
                        logger.error(f"Invalid JSON in {req_file}: {je}")
                        self._write_error_response(
                            responses_dir, req_file.stem, f"Invalid JSON: {je}"
                        )
                        req_file.unlink(missing_ok=True)
                        continue

                    req_id = req_file.stem
                    session_id = data.get("session_id", "default")
                    code = data.get("code", "")

                    if not code.strip():
                        raise ValueError("No code provided in request")

                    logger.info(f"Processing request {req_id} for session {session_id}")

                    # Execute code in session
                    session = self.manager.get_session(session_id)
                    result = session.execute(code)

                    # Add request ID to response
                    result["request_id"] = req_id

                    # Write response
                    self._write_response(responses_dir, req_id, result)

                    # Remove processed request
                    req_file.unlink(missing_ok=True)
                    processed += 1

                except Exception as e:
                    logger.error(f"Error processing {req_file}: {e}", exc_info=True)
                    self._write_error_response(responses_dir, req_file.stem, str(e))
                    req_file.unlink(missing_ok=True)

            if processed > 0:
                logger.debug(f"Processed {processed} requests")
                return True

        except Exception as e:
            logger.error(f"Unexpected error in request processing: {e}", exc_info=True)

        return False

    def _write_response(
        self, responses_dir: Path, req_id: str, result: dict[str, Any]
    ) -> None:
        """Write a response to a file.

        Args:
            responses_dir: Directory to write the response to
            req_id: Request ID (used for filename)
            result: Response data to write
        """
        try:
            resp_file = responses_dir / f"{req_id}.json"
            temp_file = responses_dir / f".{req_id}.tmp"

            # Write to temp file first, then rename atomically
            with open(temp_file, "w") as f:
                json.dump(result, f)
                f.flush()
                os.fsync(f.fileno())

            # Atomic rename
            temp_file.replace(resp_file)
            logger.debug(f"Wrote response to {resp_file}")

        except Exception as e:
            logger.error(f"Failed to write response for {req_id}: {e}", exc_info=True)
            raise

    def _write_error_response(
        self, responses_dir: Path, req_id: str, error: str
    ) -> None:
        """Write an error response to a file.

        Args:
            responses_dir: Directory to write the response to
            req_id: Request ID (used for filename)
            error: Error message
        """
        try:
            self._write_response(
                responses_dir,
                req_id,
                {
                    "success": False,
                    "error": error,
                    "stdout": "",
                    "stderr": traceback.format_exc(),
                    "request_id": req_id,
                },
            )
        except Exception as e:
            logger.error(
                f"Failed to write error response for {req_id}: {e}", exc_info=True
            )


# Legacy functions for backward compatibility
def run_legacy_mode():
    """Legacy mode entry point."""
    agent = ContainerAgent()
    agent.run_legacy_mode()


def run_daemon_mode(ipc_dir_path: str, timeout: int):
    """Daemon mode entry point.

    Args:
        ipc_dir_path: Path to the IPC directory
        timeout: Timeout in seconds for both container and session inactivity
    """
    # Configure logging level
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.getLogger().setLevel(log_level)

    logger.info(f"Starting daemon with timeout={timeout}s")

    try:
        agent = ContainerAgent(timeout=timeout, ipc_dir_path=ipc_dir_path)
        agent.run_daemon_mode(ipc_dir_path)
    except Exception as e:
        logger.critical(f"Fatal error in daemon: {e}", exc_info=True)
        sys.exit(1)


def main():
    """Main entry point for the container agent."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Sandbox Container Agent")

    # Daemon mode options
    parser.add_argument("--daemon", action="store_true", help="Run in daemon mode")
    parser.add_argument(
        "--ipc-dir",
        type=str,
        default="/workspace/.ipc",
        help="Directory for IPC files (default: /workspace/.ipc)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Inactivity timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "input_file", nargs="?", help="Input JSON file for single execution mode"
    )

    args = parser.parse_args()

    # Configure logging
    log_level = getattr(logging, args.log_level, logging.INFO)
    log_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "container_agent.log"
    )
    setup_logging(level=log_level, log_file=log_file)

    logger.info(f"Starting container agent with args: {vars(args)}")

    # Initialize the agent
    agent = ContainerAgent(timeout=args.timeout, ipc_dir_path=args.ipc_dir)

    try:
        if args.daemon:
            logger.info("Running in daemon mode")
            agent.run_daemon_mode(ipc_dir_path=args.ipc_dir)
        elif args.input_file:
            logger.info(f"Running in legacy mode with input file: {args.input_file}")
            agent.run_legacy_mode(input_file=args.input_file)
        else:
            logger.info("Running in legacy mode (single execution)")
            agent.run_legacy_mode()

    except Exception as e:
        logger.error(f"Container agent error: {e}", exc_info=True)
        return 1

    logger.info("Container agent shutdown complete")
    return 0


if __name__ == "__main__":
    # Add a NullHandler to prevent "No handlers could be found" warnings
    # before proper logging is configured in main()
    logger.addHandler(logging.NullHandler())

    # Import here to avoid circular imports
    import sys

    # Run the main function
    sys.exit(main() or 0)

"""Start the dashboard server."""

import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil
import typer

from local_coding_assistant.core.error_handler import safe_entrypoint
from local_coding_assistant.utils.logging import get_logger

app = typer.Typer(name="dashboard", help="Start the dashboard server")
log = get_logger("cli.dashboard")


@app.command()
@safe_entrypoint("cli.dashboard.serve")
def serve(
    host: str = typer.Option("127.0.0.1", help="Host to bind to"),
    port: int = typer.Option(8080, help="Port to listen on"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
    log_level: str = typer.Option(
        "INFO", "--log-level", help="Logging level (e.g., DEBUG, INFO, WARNING)"
    ),
    detach: bool = typer.Option(False, help="Run server in background (detached)"),
) -> None:
    """Start the dashboard web server."""
    if detach:
        _start_detached_server(host, port, reload, log_level)
    else:
        _start_foreground_server(host, port, reload, log_level)


def _start_detached_server(host: str, port: int, reload: bool, log_level: str) -> None:
    """Start the dashboard server in detached/background mode."""
    typer.echo(f"Starting dashboard on {host}:{port} in background...")

    cmd = _build_server_command(host, port, reload, log_level)
    process = _launch_subprocess(cmd)

    # Give it a moment to start and verify
    time.sleep(1)

    if process.poll() is None:
        typer.echo(f"Dashboard started with PID: {process.pid}")
        typer.echo(f"Access at: http://{host}:{port}")
        typer.echo(f"To stop: uv run locca dashboard stop --port {port}")
    else:
        typer.echo("Failed to start dashboard server")
        typer.echo("Try running with --no-detach to see error details")
        typer.echo(f"Return code: {process.returncode}")


def _start_foreground_server(
    host: str, port: int, reload: bool, log_level: str
) -> None:
    """Start the dashboard server in foreground mode."""
    typer.echo(f"Starting dashboard on {host}:{port}")
    if reload:
        typer.echo("Auto-reload enabled")

    # Import here to avoid circular imports
    import uvicorn

    from local_coding_assistant.dashboard.app import create_app

    app_instance = create_app()

    uvicorn.run(
        app_instance,
        host=host,
        port=port,
        reload=reload,
        log_level=log_level.lower(),
    )


def _build_server_command(
    host: str, port: int, reload: bool, log_level: str
) -> list[str]:
    """Build the command list for starting the dashboard server."""
    runner_path = Path(__file__).parent.parent.parent / "dashboard" / "run_dashboard.py"
    cmd = [
        sys.executable,
        str(runner_path),
        "--host",
        host,
        "--port",
        str(port),
        "--log-level",
        log_level.lower(),
    ]

    if reload:
        typer.echo("Auto-reload enabled")
        cmd.append("--reload")

    return cmd


def _launch_subprocess(cmd: list[str]) -> subprocess.Popen:
    """Launch the dashboard server as a subprocess."""
    # Set creation flags for Windows
    kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    return subprocess.Popen(  # noqa: S603
        cmd,
        cwd=Path(__file__).parent.parent.parent.parent,
        shell=False,
        **kwargs,
    )


@app.command()
@safe_entrypoint("cli.dashboard.stop")
def stop(
    port: int = typer.Option(
        8080, help="Port of dashboard to stop (default: all ports)"
    ),
    all_ports: bool = typer.Option(
        False, "--all", "-a", help="Stop dashboard servers on all ports"
    ),
) -> None:
    """Stop dashboard server(s)."""
    if all_ports:
        _stop_all_dashboard_servers()
    else:
        _stop_dashboard_server_on_port(port)


def _stop_all_dashboard_servers() -> None:
    """Stop all dashboard servers regardless of port."""
    typer.echo("Looking for all dashboard servers...")
    stopped_count = _stop_all_dashboards()

    if stopped_count == 0:
        typer.echo("No dashboard servers found")
    else:
        typer.echo(f"Stopped {stopped_count} dashboard server(s)")


def _stop_dashboard_server_on_port(port: int) -> None:
    """Stop dashboard servers on a specific port."""
    typer.echo(f"Looking for dashboard server on port {port}...")
    stopped_count = _stop_dashboard_on_port(port)

    if stopped_count == 0:
        typer.echo(f"No dashboard server found running on port {port}")
    else:
        typer.echo(f"Stopped {stopped_count} dashboard server(s) on port {port}")


class DashboardProcessManager:
    """Manages finding and stopping dashboard server processes."""

    @staticmethod
    def is_dashboard_process(cmdline_str: str) -> bool:
        """Check if a process command line indicates it's a dashboard server."""
        return (
            "run_dashboard.py" in cmdline_str
            or "uvicorn" in cmdline_str
            or ("dashboard" in cmdline_str and "serve" in cmdline_str)
        )

    @staticmethod
    def find_all_dashboard_processes() -> list[psutil.Process]:
        """Find all dashboard server processes."""
        dashboard_processes = []
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = proc.info.get("cmdline", [])
                if cmdline:
                    cmdline_str = " ".join(cmdline)
                    if DashboardProcessManager.is_dashboard_process(cmdline_str):
                        dashboard_processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return dashboard_processes

    @staticmethod
    def find_dashboard_processes_on_port(port: int) -> list[psutil.Process]:
        """Find dashboard server processes running on a specific port."""
        dashboard_processes = []
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = proc.info.get("cmdline", [])
                if cmdline:
                    cmdline_str = " ".join(cmdline)
                    if DashboardProcessManager.is_dashboard_process(cmdline_str):
                        # Check if this process is using our port
                        try:
                            connections = proc.net_connections()
                            if any(
                                conn.laddr and conn.laddr.port == port
                                for conn in connections
                            ):
                                dashboard_processes.append(proc)
                        except (
                            psutil.NoSuchProcess,
                            psutil.AccessDenied,
                            psutil.TimeoutExpired,
                        ):
                            continue
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return dashboard_processes

    @staticmethod
    def find_any_processes_on_port(port: int) -> list[psutil.Process]:
        """Find any processes using a specific port (fallback method)."""
        processes_on_port = []

        try:
            for proc in psutil.process_iter(["pid", "connections"]):
                try:
                    connections = proc.info.get("connections", [])
                    for conn in connections:
                        if conn.laddr and conn.laddr.port == port:
                            processes_on_port.append(proc)
                            break
                except (
                    psutil.NoSuchProcess,
                    psutil.AccessDenied,
                    psutil.TimeoutExpired,
                ):
                    # Process may have ended or we don't have permission
                    continue
        except psutil.Error:
            # psutil-specific errors (e.g., insufficient permissions)
            log.warning("Insufficient permissions to scan all processes")
        except Exception as e:
            # Unexpected errors - log but don't crash
            log.warning(f"Unexpected error while scanning processes: {e}")

        return processes_on_port


class ProcessTerminator:
    """Handles process termination with graceful fallback."""

    @staticmethod
    def terminate_process(proc: psutil.Process) -> bool:
        """Terminate a process gracefully, then forcefully if needed."""
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
                return True
            except psutil.TimeoutExpired:
                typer.echo("Process didn't terminate gracefully, forcing...")
                proc.kill()
                proc.wait(timeout=2)
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
            return False


def _stop_all_dashboards() -> int:
    """Stop all dashboard servers regardless of port."""
    stopped_count = 0
    dashboard_processes = DashboardProcessManager.find_all_dashboard_processes()

    for proc in dashboard_processes:
        typer.echo(f"Stopping dashboard server PID: {proc.pid}")
        if ProcessTerminator.terminate_process(proc):
            stopped_count += 1

    return stopped_count


def _stop_dashboard_on_port(port: int) -> int:
    """Stop dashboard servers on a specific port."""
    stopped_count = 0

    # First try to find dashboard-specific processes
    dashboard_processes = DashboardProcessManager.find_dashboard_processes_on_port(port)
    for proc in dashboard_processes:
        typer.echo(f"Stopping dashboard server PID: {proc.pid} on port {port}")
        if ProcessTerminator.terminate_process(proc):
            stopped_count += 1

    # If no dashboard-specific processes found, try any process using the port
    if stopped_count == 0:
        other_processes = DashboardProcessManager.find_any_processes_on_port(port)
        for proc in other_processes:
            typer.echo(f"Stopping process {proc.pid} using port {port}")
            if ProcessTerminator.terminate_process(proc):
                stopped_count += 1

    return stopped_count


def _terminate_process(proc) -> bool:
    """Deprecated: Use ProcessTerminator.terminate_process instead."""
    return ProcessTerminator.terminate_process(proc)

"""
Unit tests for dashboard CLI commands.
"""

import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch

from typer.testing import CliRunner

from local_coding_assistant.cli.commands.dashboard import (
    DashboardProcessManager,
    ProcessTerminator,
    _build_server_command,
    _launch_subprocess,
    _start_detached_server,
    _start_foreground_server,
    _stop_all_dashboard_servers,
    _stop_dashboard_on_port,
    app,
    serve,
    stop,
)


class TestDashboardServeCommand:
    """Test dashboard serve command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch("local_coding_assistant.cli.commands.dashboard._start_foreground_server")
    def test_serve_foreground_default_params(self, mock_start_foreground):
        """Test serve command with default parameters in foreground mode."""
        result = self.runner.invoke(app, ["serve"])

        assert result.exit_code == 0
        mock_start_foreground.assert_called_once_with("127.0.0.1", 8080, False, "INFO")

    @patch("local_coding_assistant.cli.commands.dashboard._start_foreground_server")
    def test_serve_foreground_custom_params(self, mock_start_foreground):
        """Test serve command with custom parameters."""
        result = self.runner.invoke(
            app,
            [
                "serve",
                "--host",
                "0.0.0.0",
                "--port",
                "9000",
                "--reload",
                "--log-level",
                "DEBUG",
            ],
        )

        assert result.exit_code == 0
        mock_start_foreground.assert_called_once_with("0.0.0.0", 9000, True, "DEBUG")

    @patch("local_coding_assistant.cli.commands.dashboard._start_detached_server")
    def test_serve_detached_mode(self, mock_start_detached):
        """Test serve command in detached mode."""
        result = self.runner.invoke(app, ["serve", "--detach"])

        assert result.exit_code == 0
        mock_start_detached.assert_called_once_with("127.0.0.1", 8080, False, "INFO")

    @patch("local_coding_assistant.cli.commands.dashboard._start_foreground_server")
    def test_serve_function_direct_call(self, mock_start_foreground):
        """Test serve function called directly."""
        serve(
            host="192.168.1.100",
            port=3000,
            reload=True,
            log_level="WARNING",
            detach=False,
        )

        mock_start_foreground.assert_called_once_with(
            "192.168.1.100", 3000, True, "WARNING"
        )


class TestDashboardStopCommand:
    """Test dashboard stop command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch("local_coding_assistant.cli.commands.dashboard._stop_dashboard_on_port")
    def test_stop_default_port(self, mock_stop_port):
        """Test stop command with default port."""
        mock_stop_port.return_value = 1

        result = self.runner.invoke(app, ["stop"])

        assert result.exit_code == 0
        mock_stop_port.assert_called_once_with(8080)

    @patch("local_coding_assistant.cli.commands.dashboard._stop_dashboard_on_port")
    def test_stop_custom_port(self, mock_stop_port):
        """Test stop command with custom port."""
        mock_stop_port.return_value = 0

        result = self.runner.invoke(app, ["stop", "--port", "9000"])

        assert result.exit_code == 0
        mock_stop_port.assert_called_once_with(9000)

    @patch("local_coding_assistant.cli.commands.dashboard._stop_all_dashboard_servers")
    def test_stop_all_ports(self, mock_stop_all):
        """Test stop command with --all flag."""
        mock_stop_all.return_value = 3

        result = self.runner.invoke(app, ["stop", "--all"])

        assert result.exit_code == 0
        mock_stop_all.assert_called_once()

    @patch("local_coding_assistant.cli.commands.dashboard._stop_all_dashboard_servers")
    def test_stop_function_direct_call(self, mock_stop_all):
        """Test stop function called directly."""
        stop(port=3000, all_ports=False)

        mock_stop_all.assert_not_called()
        # This would call _stop_dashboard_on_port in real implementation


class TestServerCommandBuilder:
    """Test server command building functions."""

    def test_build_server_command_default(self):
        """Test building server command with default parameters."""
        cmd = _build_server_command(
            host="127.0.0.1", port=8080, reload=False, log_level="info"
        )

        expected = [
            sys.executable,
            str(
                Path(__file__).parent.parent.parent.parent
                / "src"
                / "local_coding_assistant"
                / "dashboard"
                / "run_dashboard.py"
            ),
            "--host",
            "127.0.0.1",
            "--port",
            "8080",
            "--log-level",
            "info",
        ]

        assert cmd == expected

    def test_build_server_command_with_reload(self):
        """Test building server command with reload enabled."""
        cmd = _build_server_command(
            host="0.0.0.0", port=9000, reload=True, log_level="debug"
        )

        assert "--reload" in cmd
        assert cmd[cmd.index("--host") + 1] == "0.0.0.0"
        assert cmd[cmd.index("--port") + 1] == "9000"
        assert cmd[cmd.index("--log-level") + 1] == "debug"

    def test_build_server_command_custom_values(self):
        """Test building server command with custom values."""
        cmd = _build_server_command(
            host="192.168.1.1", port=3000, reload=False, log_level="warning"
        )

        assert cmd[cmd.index("--host") + 1] == "192.168.1.1"
        assert cmd[cmd.index("--port") + 1] == "3000"
        assert cmd[cmd.index("--log-level") + 1] == "warning"


class TestSubprocessManagement:
    """Test subprocess management functions."""

    @patch("subprocess.Popen")
    @patch("sys.platform", "win32")
    def test_launch_subprocess_windows(self, mock_popen):
        """Test launching subprocess on Windows."""
        mock_process = Mock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process

        cmd = ["python", "script.py", "--port", "8080"]
        process = _launch_subprocess(cmd)

        assert process == mock_process
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        assert call_args[0][0] == cmd
        assert call_args[1]["shell"] is False
        assert call_args[1]["creationflags"] == subprocess.CREATE_NEW_PROCESS_GROUP

    @patch("subprocess.Popen")
    @patch("sys.platform", "linux")
    def test_launch_subprocess_linux(self, mock_popen):
        """Test launching subprocess on Linux."""
        mock_process = Mock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process

        cmd = ["python", "script.py", "--port", "8080"]
        process = _launch_subprocess(cmd)

        assert process == mock_process
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        assert call_args[0][0] == cmd
        assert call_args[1]["shell"] is False
        # No creationflags on Linux
        assert "creationflags" not in call_args[1]

    @patch("local_coding_assistant.cli.commands.dashboard._build_server_command")
    @patch("local_coding_assistant.cli.commands.dashboard._launch_subprocess")
    @patch("time.sleep")
    @patch("typer.echo")
    def test_start_detached_server_success(
        self, mock_echo, mock_sleep, mock_launch, mock_build
    ):
        """Test successful detached server start."""
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_process.pid = 12345
        mock_launch.return_value = mock_process
        mock_build.return_value = ["python", "dashboard.py"]

        _start_detached_server(
            host="127.0.0.1", port=8080, reload=False, log_level="INFO"
        )

        mock_build.assert_called_once_with("127.0.0.1", 8080, False, "INFO")
        mock_launch.assert_called_once_with(["python", "dashboard.py"])
        mock_sleep.assert_called_once_with(1)

        # Check echo calls
        echo_calls = [call[0][0] for call in mock_echo.call_args_list]
        assert any("Starting dashboard" in call for call in echo_calls)
        assert any("PID: 12345" in call for call in echo_calls)
        assert any("http://127.0.0.1:8080" in call for call in echo_calls)

    @patch("local_coding_assistant.cli.commands.dashboard._build_server_command")
    @patch("local_coding_assistant.cli.commands.dashboard._launch_subprocess")
    @patch("time.sleep")
    @patch("typer.echo")
    def test_start_detached_server_failure(
        self, mock_echo, mock_sleep, mock_launch, mock_build
    ):
        """Test detached server start failure."""
        mock_process = Mock()
        mock_process.poll.return_value = 1  # Non-zero return code
        mock_process.returncode = 1
        mock_launch.return_value = mock_process
        mock_build.return_value = ["python", "dashboard.py"]

        _start_detached_server(
            host="127.0.0.1", port=8080, reload=False, log_level="INFO"
        )

        # Check error messages
        echo_calls = [call[0][0] for call in mock_echo.call_args_list]
        assert any("Failed to start dashboard" in call for call in echo_calls)
        assert any("Return code: 1" in call for call in echo_calls)

    @patch.dict(
        "sys.modules",
        {"uvicorn": Mock(), "local_coding_assistant.dashboard.app": Mock()},
    )
    @patch("typer.echo")
    def test_start_foreground_server(self, mock_echo):
        """Test foreground server start."""
        import sys

        mock_uvicorn = sys.modules["uvicorn"]
        mock_dashboard_app = sys.modules["local_coding_assistant.dashboard.app"]
        mock_app_instance = Mock()
        mock_dashboard_app.create_app.return_value = mock_app_instance

        _start_foreground_server(
            host="0.0.0.0", port=9000, reload=True, log_level="DEBUG"
        )

        mock_dashboard_app.create_app.assert_called_once()
        mock_uvicorn.run.assert_called_once()
        call_args = mock_uvicorn.run.call_args[1]
        assert call_args["host"] == "0.0.0.0"
        assert call_args["port"] == 9000
        assert call_args["reload"] is True
        assert call_args["log_level"] == "debug"


class TestDashboardProcessManager:
    """Test DashboardProcessManager class."""

    def test_is_dashboard_process_run_dashboard(self):
        """Test identifying dashboard process with run_dashboard.py."""
        cmdline = "python run_dashboard.py --port 8080"
        assert DashboardProcessManager.is_dashboard_process(cmdline) is True

    def test_is_dashboard_process_uvicorn(self):
        """Test identifying dashboard process with uvicorn."""
        cmdline = "uvicorn dashboard.app:create_app --host 127.0.0.1"
        assert DashboardProcessManager.is_dashboard_process(cmdline) is True

    def test_is_dashboard_process_dashboard_serve(self):
        """Test identifying dashboard process with dashboard serve."""
        cmdline = "locca dashboard serve --port 8080"
        assert DashboardProcessManager.is_dashboard_process(cmdline) is True

    def test_is_dashboard_process_false(self):
        """Test that non-dashboard processes are not identified."""
        cmdline = "python some_other_script.py"
        assert DashboardProcessManager.is_dashboard_process(cmdline) is False

    @patch("psutil.process_iter")
    def test_find_all_dashboard_processes(self, mock_iter):
        """Test finding all dashboard processes."""
        # Mock processes
        mock_dashboard_proc = Mock()
        mock_dashboard_proc.info = {
            "pid": 123,
            "name": "python",
            "cmdline": ["python", "run_dashboard.py", "--port", "8080"],
        }
        mock_dashboard_proc.pid = 123

        mock_other_proc = Mock()
        mock_other_proc.info = {
            "pid": 456,
            "name": "python",
            "cmdline": ["python", "other_script.py"],
        }
        mock_other_proc.pid = 456

        mock_iter.return_value = [mock_dashboard_proc, mock_other_proc]

        with patch.object(
            DashboardProcessManager,
            "is_dashboard_process",
            side_effect=lambda x: "run_dashboard.py" in x,
        ):
            processes = DashboardProcessManager.find_all_dashboard_processes()

        assert len(processes) == 1
        assert processes[0].pid == 123

    @patch("psutil.process_iter")
    def test_find_dashboard_processes_on_port(self, mock_iter):
        """Test finding dashboard processes on specific port."""
        # Mock process with network connection
        mock_proc = Mock()
        mock_proc.info = {
            "pid": 123,
            "name": "python",
            "cmdline": ["python", "run_dashboard.py", "--port", "8080"],
        }
        mock_proc.pid = 123

        mock_connection = Mock()
        mock_connection.laddr.port = 8080
        mock_proc.net_connections.return_value = [mock_connection]

        mock_iter.return_value = [mock_proc]

        with patch.object(
            DashboardProcessManager, "is_dashboard_process", return_value=True
        ):
            processes = DashboardProcessManager.find_dashboard_processes_on_port(8080)

        assert len(processes) == 1
        assert processes[0].pid == 123

    @patch("psutil.process_iter")
    def test_find_any_processes_on_port(self, mock_iter):
        """Test finding any processes on specific port."""
        mock_proc = Mock()
        mock_proc.info = {"pid": 456, "connections": [Mock(laddr=Mock(port=8080))]}
        mock_proc.pid = 456

        mock_iter.return_value = [mock_proc]

        processes = DashboardProcessManager.find_any_processes_on_port(8080)

        assert len(processes) == 1
        assert processes[0].pid == 456


class TestProcessTerminator:
    """Test ProcessTerminator class."""

    @patch("psutil.Process")
    def test_terminate_process_graceful(self, mock_process_class):
        """Test graceful process termination."""
        mock_proc = Mock()
        mock_process_class.return_value = mock_proc

        # Simulate graceful termination
        mock_proc.wait.return_value = None

        result = ProcessTerminator.terminate_process(mock_proc)

        assert result is True
        mock_proc.terminate.assert_called_once()
        mock_proc.wait.assert_called_once_with(timeout=5)

    @patch("psutil.Process")
    def test_terminate_process_force_kill(self, mock_process_class):
        """Test force kill when graceful termination fails."""
        mock_proc = Mock()
        mock_process_class.return_value = mock_proc

        # Simulate timeout on graceful termination
        from psutil import TimeoutExpired

        mock_proc.wait.side_effect = [TimeoutExpired(123, 5), None]

        result = ProcessTerminator.terminate_process(mock_proc)

        assert result is True
        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()
        # Should be called twice - once for terminate wait, once for kill wait
        assert mock_proc.wait.call_count == 2

    def test_terminate_process_no_such_process(self):
        """Test handling of non-existent process."""
        mock_proc = Mock()

        # Simulate process not found - the exception should be caught
        from psutil import NoSuchProcess

        mock_proc.terminate.side_effect = NoSuchProcess(123)

        result = ProcessTerminator.terminate_process(mock_proc)

        # The current implementation catches the exception and returns False
        assert result is False


class TestStopFunctions:
    """Test stop functions."""

    @patch("typer.echo")
    def test_stop_all_dashboard_servers(self, mock_echo):
        """Test stopping all dashboard servers."""
        with patch(
            "local_coding_assistant.cli.commands.dashboard._stop_all_dashboards"
        ) as mock_stop:
            mock_stop.return_value = 2

            _stop_all_dashboard_servers()

            mock_stop.assert_called_once()

    @patch("typer.echo")
    def test_stop_dashboard_on_port(self, mock_echo):
        """Test stopping dashboard servers on specific port."""
        with patch.object(
            DashboardProcessManager, "find_dashboard_processes_on_port"
        ) as mock_find:
            with patch.object(
                ProcessTerminator, "terminate_process", return_value=True
            ) as mock_terminate:
                mock_proc = Mock()
                mock_proc.pid = 789
                mock_find.return_value = [mock_proc]

                count = _stop_dashboard_on_port(8080)

                assert count == 1
                mock_terminate.assert_called_once_with(mock_proc)

    @patch("typer.echo")
    def test_stop_dashboard_on_port_fallback(self, mock_echo):
        """Test stopping any process on port when no dashboard processes found."""
        with patch.object(
            DashboardProcessManager, "find_dashboard_processes_on_port", return_value=[]
        ):
            with patch.object(
                DashboardProcessManager, "find_any_processes_on_port"
            ) as mock_find_any:
                with patch.object(
                    ProcessTerminator, "terminate_process", return_value=True
                ) as mock_terminate:
                    mock_proc = Mock()
                    mock_proc.pid = 999
                    mock_find_any.return_value = [mock_proc]

                    count = _stop_dashboard_on_port(8080)

                    assert count == 1
                    mock_terminate.assert_called_once_with(mock_proc)


class TestIntegrationScenarios:
    """Integration test scenarios for dashboard CLI."""

    @patch("typer.echo")
    def test_full_lifecycle_start_stop(self, mock_echo):
        """Test full lifecycle of starting and stopping dashboard."""
        with patch.object(
            DashboardProcessManager, "find_dashboard_processes_on_port"
        ) as mock_find:
            with patch.object(
                ProcessTerminator, "terminate_process", return_value=True
            ) as mock_terminate:
                # Simulate starting dashboard (would normally involve subprocess)
                mock_proc = Mock()
                mock_proc.pid = 12345

                # Simulate stopping dashboard
                mock_find.return_value = [mock_proc]

                count = _stop_dashboard_on_port(8080)

                assert count == 1
                mock_terminate.assert_called_once_with(mock_proc)

    def test_command_line_interface_integration(self):
        """Test CLI command integration."""
        runner = CliRunner()

        # Test help
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "dashboard" in result.stdout.lower()

        # Test serve help
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--host" in result.stdout
        assert "--port" in result.stdout
        assert "--detach" in result.stdout

        # Test stop help
        result = runner.invoke(app, ["stop", "--help"])
        assert result.exit_code == 0
        assert "--port" in result.stdout
        assert "--all" in result.stdout

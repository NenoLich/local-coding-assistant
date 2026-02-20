# Sandbox Subsystem

The sandbox subsystem provides a secure, isolated environment for executing user-provided code within Docker containers. It ensures safe execution by validating code against security policies, managing container lifecycles, and tracking resource usage.

## Architecture Overview

The sandbox subsystem is organized into several key components:

### Core Components

- **`base.py`**: Defines the abstract `ISandbox` interface that all sandbox implementations must follow. It includes methods for starting/stopping the sandbox, executing code, and checking availability.

- **`manager.py`**: `SandboxManager` handles the lifecycle and configuration of sandbox instances. It creates and manages `DockerSandbox` instances based on configuration settings.

- **`docker_sandbox.py`**: The main Docker-based implementation of the sandbox. It manages Docker containers, handles execution requests, and collects resource metrics.

- **`exceptions.py`**: Defines custom exception classes for sandbox-specific errors, including security violations, runtime errors, and timeout errors.

- **`sandbox_types.py`**: Contains type definitions for execution requests and responses, including detailed resource metrics.

- **`security.py`**: `SecurityManager` class that validates code against security policies, blocking dangerous imports and patterns.

### Container Runtime (guest/)

The `guest/` directory contains code that runs inside the Docker containers:

- **`agent.py`**: `ContainerAgent` manages the container lifecycle, handles IPC communication in daemon mode, and processes execution requests.

- **`session.py`**: `Session` and `SessionManager` classes manage execution contexts. Each session maintains its own global namespace and tracks execution state.

- **`resource_tracker.py`**: `ResourceTracker` monitors resource usage (CPU, memory, I/O) during tool executions using psutil.

- **`tools_api.py`**: Provides a standardized API for sandbox tools (math operations, file listing, final answers) with automatic resource tracking.

### Infrastructure Files

- **`Dockerfile`**: Defines the Docker image used for sandbox containers.

- **`docker-entrypoint.sh`**: Entrypoint script that sets up the container environment and permissions.

## Setup and Initialization

The sandbox container setup is designed with security, flexibility, and reliability as core principles, implemented through sophisticated logic in the Dockerfile and entrypoint script.

### Dockerfile: Secure Base Image and Dependencies

The Dockerfile establishes a minimal, secure foundation for sandbox containers:

- **Minimal Base Image**: Uses `python:3.12-slim` to reduce attack surface and image size, containing only essential Python runtime components.

- **Security-Focused User Management**: Creates a dedicated non-root user `locca` (UID/GID 1000) to prevent privilege escalation. All operations run under this restricted user account.

- **Essential Tooling**: Installs `git` for potential code retrieval and `gosu` for safe privilege dropping in the entrypoint.

- **Workspace Preparation**: Establishes `/workspace` as the working directory, owned by the `locca` user, ensuring proper file permissions.

- **IPC Infrastructure**: Pre-creates `/workspace/ipc` directory structure for inter-process communication between host and container.

- **Dependency Management**: Installs critical Python packages (`pyyaml`, `pydantic`, `python-dotenv`, `psutil`) with `--no-cache-dir` to minimize image size.

- **Environment Optimization**: Sets `PYTHONPATH=/app` for module discovery and `PYTHONUNBUFFERED=1` to ensure immediate output flushing for real-time monitoring.

- **Daemon-Ready Configuration**: Uses `tail -f /dev/null` as default command to keep containers running indefinitely for persistent sessions.

### Entrypoint Script: Dynamic Permission Management

The `docker-entrypoint.sh` script handles runtime setup with sophisticated permission logic:

- **Flexible IPC Configuration**: Accepts `IPC_DIR` environment variable (defaults to `/workspace/ipc`) for customizable communication paths.

- **Robust Directory Creation**: Ensures parent directories exist and are accessible, with fallback error handling to prevent container startup failures.

- **Hierarchical Permission Strategy**: Applies ownership (`locca:locca`) and permissions recursively:
  - Directories: `750` (owner read/write/execute, group read/execute)
  - Files: `640` (owner read/write, group read)
  - This allows the `locca` user to operate while enabling host cleanup of session directories.

- **Atomic IPC Setup**: Creates `requests` and `responses` subdirectories unconditionally, ensuring consistent communication channels.

- **Privilege Dropping**: Uses `gosu` to switch from root (required for setup) to the `locca` user before executing the main command, following principle of least privilege.

### Key Concepts Highlighted

1. **Security by Design**: Non-root execution, minimal dependencies, and strict permission controls prevent container escape and privilege escalation.

2. **Inter-Process Communication (IPC)**: Shared volume-based communication enables efficient host-container interaction without network dependencies.

3. **Lifecycle Management**: Entrypoint handles both initial setup and cleanup, supporting both ephemeral and persistent execution modes.

4. **Resource Efficiency**: Minimal base image and cached dependency removal optimize container startup time and resource usage.

5. **Error Resilience**: Graceful handling of permission failures (`|| true`) ensures containers start even in restrictive environments.

This setup enables secure, isolated code execution while maintaining flexibility for different deployment scenarios and execution patterns.

## Execution Modes

The sandbox supports two execution modes:

### Ephemeral Mode
- Single execution per container
- Container is created, code is executed, and container is destroyed
- No session persistence between executions
- Suitable for one-off code evaluations

### Persistent (Daemon) Mode
- Long-running container with session persistence
- Multiple executions can share the same session context
- Uses IPC (inter-process communication) via shared files
- Containers auto-terminate after inactivity timeout
- Supports resource tracking and metrics collection

## Security Model

Security is enforced through multiple layers:

1. **Code Validation**: The `SecurityManager` scans code for blocked patterns (exec, eval, subprocess calls, etc.) and restricted imports.

2. **Container Isolation**: All code executes in Docker containers with limited network access and resource constraints.

3. **Import Restrictions**: Configurable allowlists for Python imports prevent access to potentially dangerous modules.

4. **Command Filtering**: Shell commands are validated against blocked command lists.

5. **Resource Limits**: CPU, memory, and other resources are capped at container level.

## Resource Tracking

The subsystem tracks comprehensive metrics:

- **CPU Usage**: Percentage and deltas
- **Memory Usage**: RSS and limits
- **I/O Operations**: Read/write rates
- **Tool Execution Metrics**: Duration, success/failure, arguments
- **Session Management**: Creation, access times, expiration

Metrics are collected using psutil and integrated into execution responses.

## Configuration

The sandbox is configured through the main application config:

- `enabled`: Whether sandbox is active
- `image`: Docker image to use
- `memory_limit`: Container memory limit
- `cpu_limit`: CPU quota (0-100)
- `network_enabled`: Whether containers can access network
- `allowed_imports`: List of permitted Python modules
- `blocked_patterns`: Code patterns to block
- `blocked_shell_commands`: Shell commands to prohibit
- `session_timeout`: Inactivity timeout for persistent sessions
- `max_sessions`: Maximum concurrent persistent sessions

## Usage

### Basic Execution

```python
from local_coding_assistant.sandbox.manager import SandboxManager

# Get sandbox instance
manager = SandboxManager(config_manager)
sandbox = manager.get_sandbox()

# Execute code
request = SandboxExecutionRequest(
    code="print('Hello, sandbox!')",
    session_id="test_session"
)
response = await sandbox.execute(request)

print(response.stdout)  # "Hello, sandbox!"
```

### Shell Commands

```python
response = await sandbox.execute_shell(
    command="ls -la",
    session_id="shell_session"
)
```

## Error Handling

The subsystem provides comprehensive error handling:

- `SandboxSecurityError`: Security policy violations
- `SandboxRuntimeError`: Infrastructure failures
- `SandboxTimeoutError`: Execution timeouts
- `SandboxOutputFormatError`: Response parsing failures

All errors are logged and include relevant context for debugging.

## Logging

The sandbox uses centralized logging through `utils/logging`. Key loggers:

- `sandbox.manager`: Manager operations
- `sandbox.docker`: Docker container management
- Container logs: Captured in configured log directories

## Dependencies

- `docker`: Docker SDK for Python
- `psutil`: System and process utilities (inside containers)
- `pydantic`: Data validation and serialization
- Standard library modules for execution and security

## Development Notes

- All code follows PEP 8 standards
- Type hints are used throughout for clarity
- Async/await patterns for non-blocking operations
- Thread-safe resource tracking with locks
- Graceful shutdown handling with signal traps

The sandbox subsystem is designed for reliability, security, and performance, providing a robust foundation for safe code execution in the Local Coding Assistant.

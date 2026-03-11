# Installation

This guide will help you install and set up LOCCA on your system.

## Prerequisites

- **Python 3.12+** - LOCCA requires Python 3.12 or higher
- **[uv](https://github.com/astral-sh/uv)** - Fast Python package manager and project manager

## Install uv

If you haven't installed uv yet, you can install it using the following command:

=== "macOS/Linux"
    ```bash
    curl -sSf https://astral.sh/uv/install.sh | sh
    ```

=== "Windows"
    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

## Clone the Repository

Clone the LOCCA repository to your local machine:

```bash
git clone https://github.com/NenoLich/local-coding-assistant.git
cd local-coding-assistant
```

## Create Virtual Environment

Create and activate a virtual environment using uv:

```bash
uv venv
```

Activate the virtual environment:

=== "macOS/Linux"
    ```bash
    source .venv/bin/activate
    ```

=== "Windows"
    ```cmd
    .venv\Scripts\activate
    ```

## Install Dependencies

Install the project and its development dependencies:

```bash
uv pip install -e ".[dev]"
```

## Install Pre-commit Hooks

Install pre-commit hooks for code quality:

```bash
pre-commit install
```

## Verify Installation

Verify that LOCCA is installed correctly:

```bash
locca --version
```

You should see the version information if the installation was successful.

## Next Steps

Now that you have LOCCA installed, you can:

- [Configure your environment](configuration.md)
- [Run your first query](quick-start.md)
- [Learn about CLI usage](../user-guide/cli-usage.md)

import pytest
from typer.testing import CliRunner

from local_coding_assistant.core.bootstrap import bootstrap
from local_coding_assistant.core.assistant import Assistant
from local_coding_assistant.cli.main import app as cli_app


@pytest.fixture(scope="function")
def ctx():
    # Fresh context per test to avoid state bleed
    return bootstrap()


@pytest.fixture(scope="function")
def assistant(ctx):
    return Assistant(ctx)


@pytest.fixture(scope="session")
def app():
    # CLI app is static; session scope is fine
    return cli_app


@pytest.fixture(scope="function")
def cli_runner():
    return CliRunner()

import typer

from .commands import config, provider, run, sandbox, serve, tool

app = typer.Typer(help="Local Coding Assistant CLI")

# Include sub-commands
app.add_typer(run.app, name="run", help="Run a single LLM or tool request")
app.add_typer(serve.app, name="serve", help="Start the assistant server")
app.add_typer(tool.app, name="tool", help="Manage tools")
app.add_typer(config.app, name="config", help="Configure system settings")
app.add_typer(provider.app, name="provider", help="Manage LLM providers")
app.add_typer(sandbox.app, name="sandbox", help="Manage and interact with the sandbox")


def main():
    app()


if __name__ == "__main__":
    main()

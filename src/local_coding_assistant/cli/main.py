import typer

from .commands import config, list_tools, run, serve

app = typer.Typer(help="Local Coding Assistant CLI")

# Include sub-commands
app.add_typer(run.app, name="run", help="Run a single LLM or tool request")
app.add_typer(serve.app, name="serve", help="Start the assistant server")
app.add_typer(list_tools.app, name="list-tools", help="List available tools")
app.add_typer(config.app, name="config", help="Configure system settings")


def main():
    app()


if __name__ == "__main__":
    main()

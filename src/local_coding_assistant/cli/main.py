import typer

from .commands import (
    chat,
    config,
    dashboard,
    provider,
    replay,
    run,
    sandbox,
    serve,
    sessions,
    tool,
)

app = typer.Typer(help="Local Coding Assistant CLI")

app.add_typer(run.app, name="run", help="Run a single LLM or tool request")
app.add_typer(chat.app, name="chat", help="Interactive multi-turn chat session")
app.add_typer(
    sessions.app,
    name="sessions",
    help="List, inspect and resume stored sessions",
)
app.add_typer(replay.app, name="replay", help="Replay a stored session or turn")
app.add_typer(serve.app, name="serve", help="Start the assistant server")
app.add_typer(tool.app, name="tool", help="Manage tools")
app.add_typer(config.app, name="config", help="Configure system settings")
app.add_typer(provider.app, name="provider", help="Manage LLM providers")
app.add_typer(sandbox.app, name="sandbox", help="Manage and interact with the sandbox")
app.add_typer(dashboard.app, name="dashboard", help="Start the dashboard server")


def main():
    app()


if __name__ == "__main__":
    main()

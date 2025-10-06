class Assistant:
    """High-level interface to interact with the system."""

    def __init__(self, ctx):
        self.ctx = ctx

    def run_query(self, text: str, verbose: bool = False) -> str:
        llm = self.ctx.get("llm")
        tools = self.ctx.get("tools")

        if verbose:
            print(f"[Assistant] Running query with LLM and {len(tools)} tools")

        return llm.ask(text, tools=tools)

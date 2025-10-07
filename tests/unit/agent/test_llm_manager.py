from local_coding_assistant.agent.llm_manager import LLMManager


def test_llm_manager_ask_without_tools():
    llm = LLMManager()
    out = llm.ask("hello")
    assert "[LLMManager] Echo: hello" in out
    assert "tools available: 0" in out


def test_llm_manager_ask_with_tools_iterable():
    llm = LLMManager()
    tools = [object(), object()]
    out = llm.ask("test", tools=tools)
    assert "[LLMManager] Echo: test" in out
    assert "tools available: 2" in out

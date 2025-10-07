from local_coding_assistant.core.assistant import Assistant


def test_assistant_run_query_integration(assistant: Assistant):
    text = "What can you do?"
    out = assistant.run_query(text, verbose=False)

    # Should echo text and include tool count from ToolRegistry (initially empty)
    assert "[LLMManager] Echo: What can you do?" in out
    assert "tools available: 0" in out


def test_assistant_verbose_prints(assistant: Assistant, capsys):
    out = assistant.run_query("ping", verbose=True)

    # Verify LLM echo
    assert "[LLMManager] Echo: ping" in out

    # Verify verbose message printed to stdout by Assistant
    captured = capsys.readouterr()
    assert "[Assistant] Running query with LLM and" in captured.out

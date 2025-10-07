from local_coding_assistant.tools.tool_registry import ToolRegistry


def test_tool_registry_register_and_len():
    reg = ToolRegistry()
    assert len(reg) == 0

    reg.register("t1")
    reg.register({"name": "t2"})

    assert len(reg) == 2


def test_tool_registry_iteration_order():
    reg = ToolRegistry()
    items = ["a", "b", "c"]
    for it in items:
        reg.register(it)

    assert list(iter(reg)) == items

from local_coding_assistant.runtime.runtime_manager import RuntimeManager


def test_runtime_manager_start_stop_noop():
    rm = RuntimeManager()
    # Should not raise
    rm.start()
    rm.stop()

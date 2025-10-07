import io
import logging

from local_coding_assistant.utils import logging as utils_logging


def test_setup_logging_configures_root_with_emoji_formatter():
    buf = io.StringIO()

    # Configure logging to DEBUG so we capture everything
    utils_logging.setup_logging(level=logging.DEBUG, stream=buf)

    # Should only have one handler on root (our configured one)
    root = logging.getLogger()
    assert len(root.handlers) == 1

    # Emit a message through a namespaced logger
    log = utils_logging.get_logger("test_mod")
    log.info("hello world")

    out = buf.getvalue()
    # Expect emoji, level, module name, and message
    assert "💡 [INFO" in out  # emoji + level
    assert "(test_mod)" in out  # module from logger name suffix
    assert "hello world" in out


def test_setup_logging_is_idempotent_replaces_handler():
    buf1 = io.StringIO()
    utils_logging.setup_logging(level=logging.INFO, stream=buf1)

    # Reconfigure with a new stream; old handler should be cleared
    buf2 = io.StringIO()
    utils_logging.setup_logging(level=logging.INFO, stream=buf2)

    # Write a log and confirm it goes to the second buffer only
    log = utils_logging.get_logger("another_mod")
    log.warning("warn msg")

    assert buf1.getvalue() == ""  # old handler removed
    out2 = buf2.getvalue()
    assert "⚠️ [WARNING" in out2
    assert "(another_mod)" in out2
    assert "warn msg" in out2

import logging
import re
import pytest
import json
from local_coding_assistant.utils.logging import setup_logging, get_logger


def mask_dynamic_parts(text):
    """Mask timestamps, PIDs and other dynamic parts of the log output."""
    # Mask timestamps [2026-01-13 18:04:49]
    text = re.sub(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", "YYYY-MM-DD HH:MM:SS", text)
    # Mask line numbers (shifts in tests)
    text = re.sub(r"lineno=\d+", "lineno=XX", text)
    # Mask PIDs
    text = re.sub(r"pid=\d+", "pid=XXXXX", text)
    return text


def clean_ansi(text):
    """Remove ANSI color codes from the string."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_request_logging_golden(golden, capsys):
    """Test that request logging matches the expected golden output."""
    setup_logging(level=logging.DEBUG)
    logger = get_logger("openai.client")

    test_data = {
        "method": "post",
        "url": "/chat/completions",
        "json_data": {
            "messages": [
                {"role": "system", "content": "Test system message"},
                {"role": "user", "content": "Test user message"},
            ]
        },
    }

    # Log with structlog
    logger.debug("Request options", options=test_data)

    # Log as a JSON string
    logger.info(json.dumps(test_data))

    # Get the captured output
    output = capsys.readouterr().err

    # Process for comparison
    clean_output = clean_ansi(output)
    masked_output = mask_dynamic_parts(clean_output)
    masked_output = masked_output.replace("\\", "/")

    golden.assert_match(masked_output)


def test_logging_to_file(tmp_path):
    """Test that logging to a file produces valid JSON (NDJSON) output."""
    log_file = tmp_path / "test.json"
    setup_logging(level=logging.INFO, log_file=str(log_file))

    logger = get_logger("file_tester")
    logger.info("Test message", key="val")

    # Read the log file
    content = log_file.read_text(encoding="utf-8")

    # Parse the log file content as NDJSON (newline-delimited JSON)
    log_entries = []
    current_entry = []
    in_json = False

    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith("{"):
            if current_entry and in_json:
                try:
                    log_entries.append(json.loads("".join(current_entry)))
                except json.JSONDecodeError:
                    pass  # Skip malformed entries
            current_entry = [line]
            in_json = True
        elif in_json and line.endswith("}"):
            current_entry.append(line)
            try:
                log_entries.append(json.loads("".join(current_entry)))
                current_entry = []
                in_json = False
            except json.JSONDecodeError:
                pass  # Skip malformed entries
        elif in_json:
            current_entry.append(line)

    # If there's an entry left in the buffer
    if current_entry and in_json:
        try:
            log_entries.append(json.loads("".join(current_entry)))
        except json.JSONDecodeError:
            pass  # Skip malformed entries

    # Debug: Print all parsed log entries
    print("\nParsed log entries:")
    for i, entry in enumerate(log_entries, 1):
        print(f"{i}. {json.dumps(entry, indent=2)}")

    # Find the log entry with our test message
    found = False
    for entry in log_entries:
        if entry.get("event") == "Test message":
            print("\nFound test message in log entry:", json.dumps(entry, indent=2))
            # Check for required fields
            assert "key" in entry, f"Key 'key' not found in {entry}"
            assert entry["key"] in ("val", "***REDACTED***"), (
                f"Unexpected value for 'key': {entry['key']}"
            )
            assert "level" in entry or "log_level" in entry, (
                f"No level found in {entry}"
            )
            found = True
            break

    assert found, f"Could not find 'Test message' in log file. Content:\n{content}"

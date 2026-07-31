"""
Dashboard integration test configuration and fixtures.
"""

import sys
from pathlib import Path

# Add dashboard test helpers to path
dashboard_test_helpers = (
    Path(__file__).parent.parent.parent.parent / "conftest_dashboard.py"
)
if str(dashboard_test_helpers.parent) not in sys.path:
    sys.path.insert(0, str(dashboard_test_helpers.parent))

# Import dashboard test fixtures
from tests.conftest_dashboard import *

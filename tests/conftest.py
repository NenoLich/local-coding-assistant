from __future__ import annotations

import sys
from pathlib import Path

# Add project src/ to sys.path for imports like `local_coding_assistant.*`
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

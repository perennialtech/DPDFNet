from __future__ import annotations

import os
import sys
from pathlib import Path


# Test the source tree, unless CI is testing an installed wheel.
if not os.environ.get("DPDFNET_TEST_INSTALLED"):
    SRC_DIR = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(SRC_DIR))

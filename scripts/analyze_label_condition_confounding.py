#!/usr/bin/env python3
"""Run label–condition confounding analysis on Phase 1C balanced development sample."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.probes.confounding import main

if __name__ == "__main__":
    raise SystemExit(main())

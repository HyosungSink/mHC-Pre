#!/usr/bin/env python3
"""Compatibility entry for the system-test pipeline."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.st.scripts.run_tests import main  # noqa: E402


if __name__ == "__main__":
    main()

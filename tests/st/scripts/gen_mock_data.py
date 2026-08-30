#!/usr/bin/env python3
"""Materialize the fixed local Mock fixtures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TESTS_ROOT.parent))

from tests.common.mock_cases import MOCK_CASES_PATH, load_cases, materialize_case  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=MOCK_CASES_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = load_cases(args.cases.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    for case in document["cases"]:
        materialize_case(case, args.output / case["case_id"])
    print(f"generated {len(document['cases'])} mock fixtures")


if __name__ == "__main__":
    main()

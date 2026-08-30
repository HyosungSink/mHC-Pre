#!/usr/bin/env python3
"""Run mHcPre test-asset UTs and the Ascend-style ST asset pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


TESTS_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = TESTS_ROOT.parent
SCRIPTS_ROOT = TESTS_ROOT / "st" / "scripts"


def run(*arguments: str) -> None:
    subprocess.run(arguments, cwd=REPO_ROOT, check=True)


def execute_pipeline(data_root: Path, coverage_report: Path, suite: str = "full") -> None:
    run(
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests/ut",
        "-t",
        ".",
        "-v",
    )
    run(
        sys.executable,
        str(SCRIPTS_ROOT / "gen_data.py"),
        "--suite",
        suite,
        "--output",
        str(data_root),
        "--coverage-report",
        str(coverage_report),
    )
    run(
        sys.executable,
        str(SCRIPTS_ROOT / "audit_data.py"),
        "--data-root",
        str(data_root),
    )
    run(sys.executable, str(SCRIPTS_ROOT / "verify_result.py"), "--self-test")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("coverage", "correctness", "full"), default="full")
    parser.add_argument("--output", type=Path, help="persist generated ST manifest here")
    parser.add_argument("--coverage-report", type=Path, help="persist the coverage report here")
    args = parser.parse_args()
    if (args.output is None) != (args.coverage_report is None):
        parser.error("--output and --coverage-report must be supplied together")

    if args.output is not None:
        execute_pipeline(args.output.resolve(), args.coverage_report.resolve(), args.suite)
    else:
        with tempfile.TemporaryDirectory(prefix="mhc_pre_tests_") as temporary:
            root = Path(temporary)
            execute_pipeline(root / "data", root / "reports" / "COVERAGE.md", args.suite)
    print("PASS: mHcPre UT and ST test assets are generated, audited, and verifiable")


if __name__ == "__main__":
    main()

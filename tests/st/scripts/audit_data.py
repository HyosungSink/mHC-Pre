#!/usr/bin/env python3
"""Audit a generated mHcPre ST manifest against the shared case source."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TESTS_ROOT.parent))

from tests.st.scripts.gen_data import build_manifest  # noqa: E402


def audit(data_root: Path) -> dict:
    manifest_path = data_root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing manifest: {manifest_path}")
    actual = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = build_manifest(actual.get("suite", "full"))
    if actual != expected:
        raise ValueError("manifest differs from Coverage/Correctness case definitions; regenerate it")
    summary = actual.get("case_summary", {})
    if not summary.get("passed"):
        raise ValueError("manifest contains a failed coverage audit")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
    )
    args = parser.parse_args()
    try:
        summary = audit(args.data_root.resolve())
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"FAIL: {error}")
        raise SystemExit(1) from error
    print(
        f"PASS: audited {summary['valid_case_count']} valid and "
        f"{summary['invalid_case_count']} invalid cases"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit Coverage, Correctness, and formal-contract rejection cases."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TESTS_ROOT.parent))

from tests.common.cases import (  # noqa: E402
    BFLOAT16, FLOAT16, MAX_D, OPTIONAL_OUTPUT_NAMES,
    SUPPORTED_B, SUPPORTED_N, SUPPORTED_S,
    optional_outputs_present, validate_call,
)
from tests.st.cases import (  # noqa: E402
    CORRECTNESS_CASES, COVERAGE_INVALID_CASES, COVERAGE_VALID_CASES,
    INVALID_CASES, VALID_CASES,
)


def audit_catalog() -> dict:
    failures: list[str] = []
    valid_errors = {
        case.name: validate_call(case.call)
        for case in VALID_CASES
        if validate_call(case.call)
    }
    if valid_errors:
        failures.append(f"formal-valid cases failed validation: {valid_errors}")
    for case in INVALID_CASES:
        errors = validate_call(case.call)
        if case.expected_error not in errors:
            failures.append(f"{case.name}: expected {case.expected_error}, got {errors}")

    matrix = [case for case in VALID_CASES if "shape_matrix" in case.focus]
    actual_matrix = {
        (case.call.x.shape[0], case.call.x.shape[1], case.call.x.shape[2])
        for case in matrix
    }
    expected_matrix = set(product(SUPPORTED_B, SUPPORTED_S, SUPPORTED_N))
    d_values = {case.call.x.shape[3] for case in VALID_CASES}
    dtypes = {case.call.x.dtype for case in VALID_CASES}
    gamma_modes = {case.call.gamma is not None for case in VALID_CASES}
    optional_modes = {optional_outputs_present(case.call) for case in VALID_CASES}
    focuses = {focus for case in VALID_CASES for focus in case.focus}
    numerical_focuses = {
        "zero",
        "all_tensors_zero",
        "large_values",
        "tiny_values",
        "ND_flatten_order",
        "split_boundaries",
        "asymmetric_phi",
        "sigmoid_positive",
    }
    required_d = {16, 2560, 4096, 8176, 16368, MAX_D}
    rules = (
        ("B/S/n 全组合", len(expected_matrix), len(actual_matrix), actual_matrix == expected_matrix),
        ("D 边界与典型值", len(required_d), len(required_d & d_values), required_d.issubset(d_values)),
        ("x/hIn DType", 2, len(dtypes), dtypes == {FLOAT16, BFLOAT16}),
        ("gamma 有/无", 2, len(gamma_modes), gamma_modes == {False, True}),
        (
            "可选输出 全有/全无",
            2,
            len(optional_modes),
            optional_modes == {frozenset(), OPTIONAL_OUTPUT_NAMES},
        ),
        (
            "数值与排布重点",
            len(numerical_focuses),
            len(numerical_focuses & focuses),
            numerical_focuses.issubset(focuses),
        ),
    )
    for category, minimum, actual, passed in rules:
        if not passed:
            failures.append(f"coverage rule failed: {category}: {actual} < {minimum}")
    return {
        "passed": not failures,
        "failures": failures,
        "valid_case_count": len(VALID_CASES),
        "invalid_case_count": len(INVALID_CASES),
        "category_counts": {
            "coverage_valid": len(COVERAGE_VALID_CASES),
            "coverage_reject": len(COVERAGE_INVALID_CASES),
            "correctness": len(CORRECTNESS_CASES),
        },
        "rules": [
            {
                "category": category,
                "minimum": minimum,
                "actual": actual,
                "passed": passed,
            }
            for category, minimum, actual, passed in rules
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    report = audit_catalog()
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

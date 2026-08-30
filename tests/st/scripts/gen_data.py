#!/usr/bin/env python3
"""Generate the lazy mHcPre ST manifest and its coverage report."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TESTS_ROOT.parent))

from tests.common.cases import (  # noqa: E402
    PATTERN_RECIPE_VERSION,
    estimated_tensor_bytes,
    optional_outputs_present,
)
from tests.st.cases import (  # noqa: E402
    CORRECTNESS_CASES,
    COVERAGE_VALID_CASES,
    accepted_cases,
    rejected_cases,
)
from tests.st.scripts.audit_coverage import audit_catalog  # noqa: E402


def build_manifest(suite: str = "full") -> dict:
    selected_valid = accepted_cases(suite)
    selected_invalid = rejected_cases(suite)
    audit = dict(audit_catalog())
    if not audit["passed"]:
        raise RuntimeError("case catalog audit failed:\n" + "\n".join(audit["failures"]))
    audit.update(
        valid_case_count=len(selected_valid),
        invalid_case_count=len(selected_invalid),
        category_counts={
            "coverage_valid": len(COVERAGE_VALID_CASES) if suite in ("coverage", "full") else 0,
            "coverage_reject": len(selected_invalid),
            "correctness": len(CORRECTNESS_CASES) if suite in ("correctness", "full") else 0,
        },
    )
    valid = []
    for category, cases in (("coverage", COVERAGE_VALID_CASES),
                            ("correctness", CORRECTNESS_CASES)):
        if suite not in (category, "full"):
            continue
        for case in cases:
            record = asdict(case)
            record.update(
                category=category,
                estimated_tensor_bytes=estimated_tensor_bytes(case.call),
                abi_profile="formal",
                current_schema_runnable=not optional_outputs_present(case.call),
            )
            valid.append(record)
    document = {
        "schema": "mhc_pre_st_manifest/v3",
        "suite": suite,
        "endianness": "little",
        "pattern_recipe": PATTERN_RECIPE_VERSION,
        "reference": (
            "FP16/BF16 input quantization; FP32 RMS, projection, activations and accumulation"
        ),
        "case_summary": audit,
        "valid_cases": valid,
        "invalid_cases": [dict(asdict(case), category="coverage") for case in selected_invalid],
    }
    # Normalize tuples to JSON arrays so in-memory and reloaded documents compare exactly.
    return json.loads(json.dumps(document, ensure_ascii=False))


def render_coverage(manifest: dict) -> str:
    summary = manifest["case_summary"]
    valid = manifest["valid_cases"]
    levels = Counter(case["level"] for case in valid)
    lines = [
        f"# mHcPre 测试覆盖审计（{manifest['suite']}）",
        "",
        "此文件由 `python3 -m tests.st.scripts.gen_data` 生成。覆盖门槛审计完整案例目录。",
        "",
        "## 汇总",
        "",
        f"- 合法正式契约样例：{summary['valid_case_count']}",
        f"- 非法输入/拒绝样例：{summary['invalid_case_count']}",
        f"- Coverage：{summary['category_counts']['coverage_valid']} 合法、"
        f"{summary['category_counts']['coverage_reject']} 拒绝样例",
        f"- Correctness：{summary['category_counts']['correctness']} 合法样例",
        f"- 分级：{dict(sorted(levels.items()))}",
        f"- 自动审计：{'通过' if summary['passed'] else '失败'}",
        "",
        "## 覆盖门槛",
        "",
        "| 类别 | 最少 | 实际 | 结果 |",
        "|---|---:|---:|---|",
    ]
    for rule in summary["rules"]:
        lines.append(
            f"| {rule['category']} | {rule['minimum']} | {rule['actual']} | "
            f"{'通过' if rule['passed'] else '失败'} |"
        )
    lines.extend(
        [
            "",
            "## 逐例清单",
            "",
            "| 样例 | 分类 | 级别 | Shape | DType | gamma | 可选中间输出 | 模式 |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for case in valid:
        call = case["call"]
        shape = "x".join(str(value) for value in call["x"]["shape"])
        lines.append(
            f"| {case['name']} | {case['category']} | {case['level']} | {shape} | {call['x']['dtype']} | "
            f"{'有' if call['gamma'] is not None else '无'} | "
            f"{'有' if call['inv_rms'] is not None else '无'} | {case['pattern']} |"
        )
    lines.extend(("", "## 结论", ""))
    if manifest["suite"] in ("coverage", "full"):
        lines.append("- Coverage 包含全部 36 个 `(B,S,n)` 组合、D 边界及拒绝样例；全维最大样例仅做 Shape/规划。")
    if manifest["suite"] in ("correctness", "full"):
        lines.append("- Correctness 包含零值、极值、gamma、eps、展平和输出切分等数值样例。")
    lines.extend((
        "- 不请求可选中间输出的混合精度样例可由当前 ACLNN Schema 运行；请求中间输出的样例仍只做正式契约覆盖。",
        "",
    ))
    return "\n".join(lines)


def write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def generate(output: Path, coverage_report: Path, suite: str = "full") -> dict:
    manifest = build_manifest(suite)
    write_atomic(
        output / "manifest.json",
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    )
    write_atomic(coverage_report, render_coverage(manifest))
    return manifest["case_summary"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("coverage", "correctness", "full"), default="full")
    parser.add_argument("--list", action="store_true", help="list selected cases without writing data")
    parser.add_argument("--output", type=Path, help="persist generated ST manifest here")
    parser.add_argument("--coverage-report", type=Path, help="persist the coverage report here")
    args = parser.parse_args()
    if args.list:
        for case in accepted_cases(args.suite):
            print(f"{case.name}\tvalid\t{case.call.x.shape}\t{case.pattern}")
        for case in rejected_cases(args.suite):
            print(f"{case.name}\treject\t{case.expected_error}")
        return
    if (args.output is None) != (args.coverage_report is None):
        parser.error("--output and --coverage-report must be supplied together")
    if args.output is not None:
        summary = generate(args.output.resolve(), args.coverage_report.resolve(), args.suite)
    else:
        with tempfile.TemporaryDirectory(prefix="mhc_pre_st_") as temporary:
            root = Path(temporary)
            summary = generate(root / "data", root / "COVERAGE.md", args.suite)
    print(
        f"generated {summary['valid_case_count']} valid and "
        f"{summary['invalid_case_count']} invalid mHcPre cases"
    )


if __name__ == "__main__":
    main()

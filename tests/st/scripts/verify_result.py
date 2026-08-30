#!/usr/bin/env python3
"""Compare all declared mHcPre device outputs with golden binary files."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path


DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4}
TOLERANCES = {
    "float16": (2e-3, 2e-3),
    "bfloat16": (2e-2, 2e-2),
    "float32": (1e-4, 1e-3),
}
OUTPUT_NAMES = ("h_in", "h_post", "h_res", "inv_rms", "h_mix", "h_pre")


def decode_tensor(content: bytes, dtype: str) -> list[float]:
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"unsupported dtype: {dtype}")
    width = DTYPE_BYTES[dtype]
    if len(content) % width:
        raise ValueError(f"{dtype} content size {len(content)} is not divisible by {width}")
    count = len(content) // width
    if dtype == "float32":
        return list(struct.unpack(f"<{count}f", content))
    if dtype == "float16":
        return list(struct.unpack(f"<{count}e", content))
    bits = struct.unpack(f"<{count}H", content)
    return [struct.unpack("<f", struct.pack("<I", value << 16))[0] for value in bits]


def compare_tensors(
    actual: list[float], golden: list[float], atol: float, rtol: float
) -> tuple[int, float, float]:
    if len(actual) != len(golden):
        raise ValueError(f"element count mismatch: {len(actual)} != {len(golden)}")
    mismatches = 0
    max_abs = 0.0
    max_rel = 0.0
    for actual_value, golden_value in zip(actual, golden):
        if not math.isfinite(actual_value) or not math.isfinite(golden_value):
            if actual_value != golden_value:
                mismatches += 1
            continue
        abs_error = abs(actual_value - golden_value)
        rel_error = abs_error / max(abs(golden_value), 1e-30)
        max_abs = max(max_abs, abs_error)
        max_rel = max(max_rel, rel_error)
        if abs_error > atol + rtol * abs(golden_value):
            mismatches += 1
    return mismatches, max_abs, max_rel


def element_count(shape: list[int]) -> int:
    count = 1
    for dim in shape:
        count *= dim
    return count


def verify_case(data_root: Path, case_name: str, actual_dir: Path, golden_dir: Path) -> None:
    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    case = next(
        (item for item in manifest["valid_cases"] if item["name"] == case_name),
        None,
    )
    if case is None:
        raise ValueError(f"case not found: {case_name}")
    for output_name in OUTPUT_NAMES:
        meta = case["call"].get(output_name)
        if meta is None:
            continue
        actual_path = actual_dir / f"{output_name}.bin"
        golden_path = golden_dir / f"{output_name}.bin"
        if not actual_path.is_file() or not golden_path.is_file():
            raise ValueError(f"missing {output_name} actual/golden file")
        expected_bytes = element_count(meta["shape"]) * DTYPE_BYTES[meta["dtype"]]
        actual_content = actual_path.read_bytes()
        golden_content = golden_path.read_bytes()
        if len(actual_content) != expected_bytes or len(golden_content) != expected_bytes:
            raise ValueError(
                f"{output_name} byte size mismatch: actual={len(actual_content)}, "
                f"golden={len(golden_content)}, expected={expected_bytes}"
            )
        actual = decode_tensor(actual_content, meta["dtype"])
        golden = decode_tensor(golden_content, meta["dtype"])
        atol, rtol = TOLERANCES[meta["dtype"]]
        mismatches, max_abs, max_rel = compare_tensors(actual, golden, atol, rtol)
        if mismatches:
            raise ValueError(
                f"{output_name}: {mismatches}/{len(actual)} values exceed "
                f"atol={atol}, rtol={rtol}"
            )
        print(f"PASS: {output_name}: max_abs={max_abs:.9g}, max_rel={max_rel:.9g}")


def self_test() -> None:
    fp32 = struct.pack("<3f", 0.0, 1.0, -2.5)
    fp16 = struct.pack("<3e", 0.0, 1.0, -2.5)
    bf16 = struct.pack("<3H", 0x0000, 0x3F80, 0xC020)
    expected = [0.0, 1.0, -2.5]
    for content, dtype in (
        (fp32, "float32"),
        (fp16, "float16"),
        (bf16, "bfloat16"),
    ):
        decoded = decode_tensor(content, dtype)
        mismatches, _, _ = compare_tensors(decoded, expected, 0.0, 0.0)
        if mismatches:
            raise ValueError(f"{dtype} verifier self-test failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--case")
    parser.add_argument("--actual-dir", type=Path)
    parser.add_argument("--golden-dir", type=Path)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
    )
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            print("PASS: mHcPre multi-output verifier self-test")
            return
        if not args.case or args.actual_dir is None or args.golden_dir is None:
            parser.error(
                "--case, --actual-dir and --golden-dir are required without --self-test"
            )
        verify_case(
            args.data_root.resolve(),
            args.case,
            args.actual_dir.resolve(),
            args.golden_dir.resolve(),
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"FAIL: {error}")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()

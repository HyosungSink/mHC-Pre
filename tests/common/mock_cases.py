"""Fixed Mock case loading, deterministic inputs, and FP32 golden files."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

from .reference import bfloat16, f32, fp16, mhc_pre_reference

MOCK_CASES_PATH = Path(__file__).resolve().parents[1] / "st" / "cases" / "mock.json"


def quantized(value: float, dtype: str) -> float:
    if dtype == "float16":
        return fp16(value)
    if dtype == "bfloat16":
        return bfloat16(value)
    if dtype == "float32":
        return f32(value)
    raise ValueError(f"unsupported dtype: {dtype}")


def encode(values: list[float] | tuple[float, ...], dtype: str) -> bytes:
    if dtype == "float16":
        return b"".join(struct.pack("<e", value) for value in values)
    if dtype == "bfloat16":
        words = []
        for value in values:
            bits = struct.unpack("<I", struct.pack("<f", bfloat16(value)))[0]
            words.append(struct.pack("<H", bits >> 16))
        return b"".join(words)
    if dtype == "float32":
        return b"".join(struct.pack("<f", f32(value)) for value in values)
    raise ValueError(f"unsupported dtype: {dtype}")


def _pattern(length: int, multiplier: int, modulus: int, center: int, scale: float) -> list[float]:
    return [((index * multiplier) % modulus - center) / scale for index in range(length)]


def _repeat_positions(values: list[float] | tuple[float, ...], values_per_position: int,
                      positions: int) -> list[float]:
    period_positions = len(values) // values_per_position
    repeated = list(values) * (positions // period_positions)
    repeated.extend(values[: (positions % period_positions) * values_per_position])
    return repeated


def materialize_case(case: dict, output: Path) -> None:
    b, s, n, d = case["shape"]
    positions = b * s
    width = n * d
    rows = n * n + 2 * n
    x_dtype = case["x_dtype"]
    param_dtype = case["param_dtype"]

    x_period_positions = 13 // math.gcd(width * 17, 13)
    input_profile = case.get("input_profile", "nominal")
    if input_profile == "nominal":
        x_period = _pattern(x_period_positions * width, 17, 13, 6, 8.0)
        phi = _pattern(rows * width, 29, 17, 8, 64.0)
        alpha = [0.75, -0.5, 1.25]
        bias = _pattern(rows, 7, 9, 4, 16.0)
        gamma = [0.5 + ((index * 5) % 7) / 8.0 for index in range(width)]
    elif input_profile == "all_zeros":
        x_period = [0.0] * (x_period_positions * width)
        phi = [0.0] * (rows * width)
        alpha = [1.0, 1.0, 1.0]
        bias = [0.0] * rows
        gamma = [0.0] * width
    elif input_profile == "sigmoid_extremes":
        x_period = [0.0] * (x_period_positions * width)
        phi = [0.0] * (rows * width)
        alpha = [1.0, 1.0, 1.0]
        bias = [100.0 if index % 2 == 0 else -100.0 for index in range(2 * n)]
        bias.extend((index - 2 * n) / 16.0 for index in range(2 * n, rows))
        gamma = [1.0] * width
    elif input_profile == "negative_overflow_logits":
        x_period = [1.0] * (x_period_positions * width)
        phi = [1.0] * (rows * width)
        alpha = [-4.0, -4.0, 1.0]
        bias = [0.0] * rows
        gamma = [1.0] * width
    elif input_profile == "public_example":
        # A-hard-mHcPre.md publishes this exact B=1, S=1, n=4, D=16
        # example.
        if (b, s, n, d) != (1, 1, 4, 16):
            raise ValueError("public_example requires shape [1, 1, 4, 16]")
        x_period = [1.0] * (x_period_positions * width)
        phi = [1.0] * (rows * width)
        alpha = [1.0, 1.0, 1.0]
        bias = [0.0] * rows
        gamma = [1.0] * width
    elif input_profile == "all_ones_unit_parameters":
        # Deterministic extension of the public all-ones construction.
        x_period = [1.0] * (x_period_positions * width)
        phi = [1.0] * (rows * width)
        alpha = [1.0, 1.0, 1.0]
        bias = [0.0] * rows
        gamma = [1.0] * width
    elif input_profile == "large_magnitude":
        x_period = [60000.0 if (index // d) % 2 == 0 else -60000.0
                    for index in range(x_period_positions * width)]
        phi = [0.0] * (rows * width)
        alpha = [1.0, 1.0, 1.0]
        bias = [0.0] * rows
        gamma = [1.0] * width
    elif input_profile == "alternating_positions":
        # Amplify cross-position UB reuse hazards: the expected hIn differs by
        # three orders of magnitude between adjacent positions while the zero
        # projection keeps hPre deterministic and finite.
        x_period = [
            1.0 if position == 0 else 1000.0
            for position in range(2)
            for _ in range(width)
        ]
        phi = [0.0] * (rows * width)
        alpha = [1.0, 1.0, 1.0]
        bias = [0.0] * rows
        gamma = [1.0] * width
    else:
        raise ValueError(f"unsupported input_profile: {input_profile}")
    x_period_positions = len(x_period) // width
    x = _repeat_positions(x_period, width, positions)

    phi_q = [quantized(value, param_dtype) for value in phi]
    alpha_q = [quantized(value, param_dtype) for value in alpha]
    bias_q = [quantized(value, param_dtype) for value in bias]
    gamma_q = [quantized(value, param_dtype) for value in gamma]
    golden = mhc_pre_reference(
        x=x_period,
        shape=(1, x_period_positions, n, d),
        phi=phi_q,
        alpha=alpha_q,
        bias=bias_q,
        gamma=gamma_q if case["has_gamma"] else None,
        x_dtype=x_dtype,
        norm_eps=case["norm_eps"],
        hc_eps=case["hc_eps"],
        include_intermediates=False,
    )
    h_in = _repeat_positions(golden.h_in, d, positions)
    h_post = _repeat_positions(golden.h_post, n, positions)
    h_res = _repeat_positions(golden.h_res, n * n, positions)

    output.mkdir(parents=True, exist_ok=False)
    (output / "x.bin").write_bytes(encode(x, x_dtype))
    (output / "phi.bin").write_bytes(encode(phi_q, param_dtype))
    (output / "alpha.bin").write_bytes(encode(alpha_q, param_dtype))
    (output / "bias.bin").write_bytes(encode(bias_q, param_dtype))
    if case["has_gamma"]:
        (output / "gamma.bin").write_bytes(encode(gamma_q, param_dtype))
    (output / "h_in_golden.bin").write_bytes(encode(h_in, x_dtype))
    (output / "h_post_golden.bin").write_bytes(encode(h_post, param_dtype))
    (output / "h_res_golden.bin").write_bytes(encode(h_res, param_dtype))

    metadata = [
        f"case_id {case['case_id']}",
        f"b {b}",
        f"s {s}",
        f"n {n}",
        f"d {d}",
        f"x_dtype {x_dtype}",
        f"param_dtype {param_dtype}",
        f"has_gamma {1 if case['has_gamma'] else 0}",
        f"norm_eps {case['norm_eps']:.17g}",
        f"hc_eps {case['hc_eps']:.17g}",
        f"abs_tolerance {case['abs_tolerance']:.17g}",
        f"rel_tolerance {case['rel_tolerance']:.17g}",
    ]
    (output / "case.meta").write_text("\n".join(metadata) + "\n", encoding="utf-8")


def load_cases(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "mhc_pre_mock_cases/v1" or document.get("category") != "mock":
        raise ValueError("unsupported Mock case schema or category")
    identifiers = [case["case_id"] for case in document["cases"]]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate mock case_id")
    return document

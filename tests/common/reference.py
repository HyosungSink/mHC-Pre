"""Dependency-free FP32 golden implementation for small mHcPre semantic cases."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Sequence

from .cases import BFLOAT16, FLOAT16, FLOAT32


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def fp16(value: float) -> float:
    try:
        return struct.unpack("<e", struct.pack("<e", float(value)))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def bfloat16(value: float) -> float:
    """Round a Python float through IEEE FP32 to BF16, ties-to-even."""

    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    exponent = bits & 0x7F800000
    mantissa = bits & 0x007FFFFF
    if exponent == 0x7F800000:
        if mantissa:
            bits |= 0x00400000  # Keep NaNs quiet while truncating the payload.
        return struct.unpack("<f", struct.pack("<I", bits & 0xFFFF0000))[0]
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)
    rounded = (bits + rounding_bias) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", rounded))[0]


def quantize(value: float, dtype: str) -> float:
    if dtype == FLOAT16:
        return fp16(value)
    if dtype == BFLOAT16:
        return bfloat16(value)
    if dtype == FLOAT32:
        return f32(value)
    raise ValueError(f"unsupported x dtype: {dtype}")


def stable_sigmoid(value: float) -> float:
    value = f32(value)
    if value >= 0.0:
        z = math.exp(-value)
        return f32(1.0 / (1.0 + z))
    z = math.exp(value)
    return f32(z / (1.0 + z))


@dataclass(frozen=True)
class ReferenceResult:
    h_in: tuple[float, ...]
    h_in_shape: tuple[int, ...]
    h_post: tuple[float, ...]
    h_post_shape: tuple[int, ...]
    h_res: tuple[float, ...]
    h_res_shape: tuple[int, ...]
    inv_rms: tuple[float, ...] | None
    inv_rms_shape: tuple[int, ...] | None
    h_mix: tuple[float, ...] | None
    h_mix_shape: tuple[int, ...] | None
    h_pre: tuple[float, ...] | None
    h_pre_shape: tuple[int, ...] | None


def _check_length(name: str, values: Sequence[float], expected: int) -> None:
    if len(values) != expected:
        raise ValueError(f"{name} has {len(values)} elements, expected {expected}")


def mhc_pre_reference(
    *,
    x: Sequence[float],
    shape: tuple[int, int, int, int],
    phi: Sequence[float],
    alpha: Sequence[float],
    bias: Sequence[float],
    gamma: Sequence[float] | None,
    x_dtype: str = FLOAT16,
    norm_eps: float = 1e-6,
    hc_eps: float = 1e-6,
    include_intermediates: bool = True,
) -> ReferenceResult:
    """Evaluate mHcPre with explicit row-major buffers and FP32 accumulations.

    This implementation is intentionally dense and intended for small semantic
    oracles.  Legal long-sequence cases live in tests/st/cases as lazy specs.
    """

    b, s, n, d = shape
    positions = b * s
    width = n * d
    rows = n * n + 2 * n
    _check_length("x", x, positions * width)
    _check_length("phi", phi, rows * width)
    _check_length("alpha", alpha, 3)
    _check_length("bias", bias, rows)
    if gamma is not None:
        _check_length("gamma", gamma, width)

    x_values = tuple(quantize(value, x_dtype) for value in x)
    phi_values = tuple(f32(value) for value in phi)
    alpha_values = tuple(f32(value) for value in alpha)
    bias_values = tuple(f32(value) for value in bias)
    gamma_values = None if gamma is None else tuple(f32(value) for value in gamma)
    norm_eps = f32(norm_eps)
    hc_eps = f32(hc_eps)

    out_h_in: list[float] = []
    out_h_post: list[float] = []
    out_h_res: list[float] = []
    out_inv_rms: list[float] = []
    out_h_mix: list[float] = []
    out_h_pre: list[float] = []

    for position in range(positions):
        x_offset = position * width
        x_flat = x_values[x_offset : x_offset + width]

        square_sum = f32(0.0)
        for value in x_flat:
            square_sum = f32(square_sum + f32(value * value))
        mean_square = f32(square_sum / width)
        inv_rms = f32(1.0 / math.sqrt(f32(mean_square + norm_eps)))

        h_mix: list[float] = []
        for row in range(rows):
            acc = f32(0.0)
            phi_offset = row * width
            for column, x_value in enumerate(x_flat):
                scaled = x_value
                if gamma_values is not None:
                    scaled = f32(scaled * gamma_values[column])
                product_value = f32(scaled * phi_values[phi_offset + column])
                acc = f32(acc + product_value)
            h_mix.append(acc)

        w = [f32(value * inv_rms) for value in h_mix]
        h_pre: list[float] = []
        for index in range(n):
            logit = f32(f32(w[index] * alpha_values[0]) + bias_values[index])
            h_pre.append(f32(stable_sigmoid(logit) + hc_eps))
        for index in range(n):
            source = n + index
            logit = f32(f32(w[source] * alpha_values[1]) + bias_values[source])
            out_h_post.append(f32(2.0 * stable_sigmoid(logit)))
        for index in range(n * n):
            source = 2 * n + index
            out_h_res.append(f32(f32(w[source] * alpha_values[2]) + bias_values[source]))

        for column in range(d):
            acc = f32(0.0)
            for lane in range(n):
                product_value = f32(h_pre[lane] * x_flat[lane * d + column])
                acc = f32(acc + product_value)
            out_h_in.append(quantize(acc, x_dtype))

        out_inv_rms.append(inv_rms)
        out_h_mix.extend(h_mix)
        out_h_pre.extend(h_pre)

    return ReferenceResult(
        h_in=tuple(out_h_in),
        h_in_shape=(b, s, d),
        h_post=tuple(out_h_post),
        h_post_shape=(b, s, n),
        h_res=tuple(out_h_res),
        h_res_shape=(b, s, n, n),
        inv_rms=tuple(out_inv_rms) if include_intermediates else None,
        inv_rms_shape=(b, s) if include_intermediates else None,
        h_mix=tuple(out_h_mix) if include_intermediates else None,
        h_mix_shape=(b, s, rows) if include_intermediates else None,
        h_pre=tuple(out_h_pre) if include_intermediates else None,
        h_pre_shape=(b, s, n) if include_intermediates else None,
    )

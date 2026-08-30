"""Shared mHcPre call contract and deterministic input recipes.

This module follows A-hard-mHcPre.md. Concrete Coverage and Correctness cases
are defined separately in tests/st/cases.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


FLOAT16 = "float16"
BFLOAT16 = "bfloat16"
FLOAT32 = "float32"

SUPPORTED_B = (1, 2, 4)
SUPPORTED_S = (4096, 8192, 16384, 32768)
SUPPORTED_N = (4, 6, 8)
MIN_D = 16  # D >= 1 and D % 16 == 0 makes 16 the smallest legal value.
MAX_D = 16384
OPTIONAL_OUTPUT_NAMES = frozenset(("invRms", "hMix", "hPre"))


@dataclass(frozen=True)
class TensorMeta:
    shape: tuple[int, ...]
    dtype: str
    layout: str = "ND"
    contiguous: bool = True


@dataclass(frozen=True)
class MhcPreCall:
    x: TensorMeta | None
    phi: TensorMeta | None
    alpha: TensorMeta | None
    bias: TensorMeta | None
    gamma: TensorMeta | None
    h_in: TensorMeta | None
    h_post: TensorMeta | None
    h_res: TensorMeta | None
    inv_rms: TensorMeta | None = None
    h_mix: TensorMeta | None = None
    h_pre: TensorMeta | None = None
    norm_eps: object = 1e-6
    hc_eps: object = 1e-6


@dataclass(frozen=True)
class ValidCase:
    name: str
    call: MhcPreCall
    pattern: str
    level: str
    focus: tuple[str, ...]
    seed: int


@dataclass(frozen=True)
class InvalidCase:
    name: str
    call: MhcPreCall
    expected_error: str


def tensor(
    shape: Iterable[int],
    dtype: str,
    *,
    layout: str = "ND",
    contiguous: bool = True,
) -> TensorMeta:
    return TensorMeta(tuple(shape), dtype, layout, contiguous)


def expected_output_metas(
    b: int,
    s: int,
    n: int,
    d: int,
    x_dtype: str,
    include_intermediates: bool,
) -> dict[str, TensorMeta | None]:
    outputs: dict[str, TensorMeta | None] = {
        "h_in": tensor((b, s, d), x_dtype),
        "h_post": tensor((b, s, n), FLOAT32),
        "h_res": tensor((b, s, n, n), FLOAT32),
        "inv_rms": None,
        "h_mix": None,
        "h_pre": None,
    }
    if include_intermediates:
        outputs.update(
            inv_rms=tensor((b, s), FLOAT32),
            h_mix=tensor((b, s, n * n + 2 * n), FLOAT32),
            h_pre=tensor((b, s, n), FLOAT32),
        )
    return outputs


def make_valid_call(
    b: int,
    s: int,
    n: int,
    d: int,
    *,
    x_dtype: str,
    gamma: bool,
    include_intermediates: bool,
    norm_eps: float = 1e-6,
    hc_eps: float = 1e-6,
) -> MhcPreCall:
    rows = n * n + 2 * n
    outputs = expected_output_metas(b, s, n, d, x_dtype, include_intermediates)
    return MhcPreCall(
        x=tensor((b, s, n, d), x_dtype),
        phi=tensor((rows, n * d), FLOAT32),
        alpha=tensor((3,), FLOAT32),
        bias=tensor((rows,), FLOAT32),
        gamma=tensor((n, d), FLOAT32) if gamma else None,
        norm_eps=norm_eps,
        hc_eps=hc_eps,
        **outputs,
    )


def _validate_tensor_common(name: str, value: TensorMeta, errors: list[str]) -> None:
    if value.layout != "ND":
        errors.append(f"{name}.layout")
    if not value.contiguous:
        errors.append(f"{name}.contiguous")


def _expect_tensor(
    name: str,
    value: TensorMeta | None,
    shape: tuple[int, ...],
    dtype: str,
    errors: list[str],
) -> None:
    if value is None:
        errors.append(f"{name}.required")
        return
    _validate_tensor_common(name, value, errors)
    if value.shape != shape:
        errors.append(f"{name}.shape")
    if value.dtype != dtype:
        errors.append(f"{name}.dtype")


def validate_call(call: MhcPreCall) -> tuple[str, ...]:
    """Return formal-contract violations as stable, assertion-friendly codes."""

    errors: list[str] = []
    for name in ("x", "phi", "alpha", "bias", "h_in", "h_post", "h_res"):
        if getattr(call, name) is None:
            errors.append(f"{name}.required")

    x = call.x
    if x is None:
        return tuple(dict.fromkeys(errors))

    _validate_tensor_common("x", x, errors)
    if len(x.shape) != 4:
        errors.append("x.rank")
        return tuple(dict.fromkeys(errors))
    if x.dtype not in (FLOAT16, BFLOAT16):
        errors.append("x.dtype")

    b, s, n, d = x.shape
    if b not in SUPPORTED_B:
        errors.append("x.B")
    if s not in SUPPORTED_S:
        errors.append("x.S")
    if n not in SUPPORTED_N:
        errors.append("x.n")
    if d < 1 or d > MAX_D or d % 16 != 0:
        errors.append("x.D")

    rows = n * n + 2 * n
    _expect_tensor("phi", call.phi, (rows, n * d), FLOAT32, errors)
    _expect_tensor("alpha", call.alpha, (3,), FLOAT32, errors)
    _expect_tensor("bias", call.bias, (rows,), FLOAT32, errors)
    if call.gamma is not None:
        _expect_tensor("gamma", call.gamma, (n, d), FLOAT32, errors)

    _expect_tensor("h_in", call.h_in, (b, s, d), x.dtype, errors)
    _expect_tensor("h_post", call.h_post, (b, s, n), FLOAT32, errors)
    _expect_tensor("h_res", call.h_res, (b, s, n, n), FLOAT32, errors)

    intermediate_values = (call.inv_rms, call.h_mix, call.h_pre)
    present_count = sum(value is not None for value in intermediate_values)
    if present_count not in (0, 3):
        errors.append("optional_outputs.group")
    if present_count == 3:
        _expect_tensor("inv_rms", call.inv_rms, (b, s), FLOAT32, errors)
        _expect_tensor("h_mix", call.h_mix, (b, s, rows), FLOAT32, errors)
        _expect_tensor("h_pre", call.h_pre, (b, s, n), FLOAT32, errors)

    if type(call.norm_eps) is not float:  # bool/int are not the documented double attr.
        errors.append("norm_eps.type")
    if type(call.hc_eps) is not float:
        errors.append("hc_eps.type")
    return tuple(dict.fromkeys(errors))


def make_case(
    name: str,
    b: int,
    s: int,
    n: int,
    d: int,
    dtype: str,
    gamma: bool,
    intermediates: bool,
    pattern: str,
    level: str,
    focus: tuple[str, ...],
    seed: int,
    norm_eps: float = 1e-6,
    hc_eps: float = 1e-6,
) -> ValidCase:
    return ValidCase(
        name,
        make_valid_call(
            b, s, n, d,
            x_dtype=dtype,
            gamma=gamma,
            include_intermediates=intermediates,
            norm_eps=norm_eps,
            hc_eps=hc_eps,
        ),
        pattern,
        level,
        focus,
        seed,
    )


def optional_outputs_present(call: MhcPreCall) -> frozenset[str]:
    names = set()
    if call.inv_rms is not None:
        names.add("invRms")
    if call.h_mix is not None:
        names.add("hMix")
    if call.h_pre is not None:
        names.add("hPre")
    return frozenset(names)


def estimated_tensor_bytes(call: MhcPreCall) -> int:
    """Estimate resident bytes if every declared formal tensor is materialized."""

    dtype_bytes = {FLOAT16: 2, BFLOAT16: 2, FLOAT32: 4}
    total = 0
    for value in asdict(call).values():
        if not isinstance(value, dict) or "shape" not in value:
            continue
        elements = 1
        for dim in value["shape"]:
            elements *= dim
        total += elements * dtype_bytes[value["dtype"]]
    return total


PATTERN_RECIPE_VERSION = "mhc_pre_pattern/v1"


def tensor_element_count(value: TensorMeta) -> int:
    count = 1
    for dim in value.shape:
        count *= dim
    return count


def _counter_uniform(seed: int, stream: int, index: int) -> float:
    """Stateless uint32 mixer, stable across Python versions and platforms."""

    value = (seed ^ (stream * 0x9E3779B9) ^ index) & 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    value ^= value >> 16
    return value / 0xFFFFFFFF


def pattern_value(case: ValidCase, tensor_name: str, index: int) -> float:
    """Generate one deterministic tensor value for a lazy valid case.

    The returned value is a real number before the declared tensor DType is
    applied.  A harness may call this function in chunks to avoid materializing
    maximum-shape inputs in host memory.
    """

    attribute_name = {"hIn": "h_in", "hPost": "h_post", "hRes": "h_res"}.get(
        tensor_name, tensor_name
    )
    meta = getattr(case.call, attribute_name, None)
    if tensor_name not in ("x", "phi", "alpha", "bias", "gamma") or meta is None:
        raise ValueError(f"{tensor_name} is not a present input tensor in {case.name}")
    count = tensor_element_count(meta)
    if index < 0 or index >= count:
        raise IndexError(f"{tensor_name}[{index}] outside [0, {count})")

    assert case.call.x is not None
    _, _, n, d = case.call.x.shape
    width = n * d
    pattern = case.pattern
    stream = {"x": 1, "phi": 2, "alpha": 3, "bias": 4, "gamma": 5}[tensor_name]
    signed_random = 2.0 * _counter_uniform(case.seed, stream, index) - 1.0

    if tensor_name == "alpha":
        if pattern == "all_tensors_zero":
            return 0.0
        if pattern == "all_ones":
            return 1.0
        if pattern == "all_zeros":
            return (1.0, -2.0, 3.0)[index]
        if pattern == "sigmoid_extremes":
            return 1.0
        if pattern == "alternating_sign":
            return (0.5, -1.25, 2.0)[index]
        return (0.75, -1.25, 1.5)[index]

    if tensor_name == "bias":
        if pattern == "all_tensors_zero":
            return 0.0
        if pattern == "all_ones":
            return 0.0
        if pattern == "sigmoid_extremes":
            return (100.0 if index % 2 == 0 else -100.0) if index < 2 * n else (index - 2 * n) / 16.0
        if pattern in ("asymmetric_phi_bias", "all_zeros"):
            return ((index * 7) % 29 - 14) / 16.0
        return ((index * 5) % 17 - 8) / 32.0

    if tensor_name == "gamma":
        if pattern == "all_tensors_zero":
            return 0.0
        if pattern == "all_ones":
            return 1.0
        if pattern == "nonuniform_gamma":
            return 0.5 + (index % 17) / 16.0
        if pattern == "flatten_sentinel":
            return 0.75 + (index % d) / max(1, d - 1)
        if pattern == "large_magnitude":
            return 0.5 if index % 2 == 0 else 1.5
        return 0.75 + _counter_uniform(case.seed, stream, index) * 0.5

    if tensor_name == "x":
        if pattern == "all_tensors_zero":
            return 0.0
        if pattern == "all_ones":
            return 1.0
        if pattern in ("all_zeros", "sigmoid_extremes"):
            return 0.0
        if pattern == "flatten_sentinel":
            return float((index % width) + 1)
        if pattern == "large_magnitude":
            return 60000.0 if index % 2 == 0 else -60000.0
        if pattern == "tiny_magnitude":
            return (1e-4 if index % 2 == 0 else -1e-4) * (1 + index % 7)
        if pattern == "alternating_sign":
            return (1.0 if index % 2 == 0 else -1.0) * (1 + index % 13) / 8.0
        return signed_random * 2.0

    # phi is row-major (n^2 + 2n, nD).
    row, column = divmod(index, width)
    if pattern == "all_tensors_zero":
        return 0.0
    if pattern == "all_ones":
        return 1.0
    if pattern in ("all_zeros", "sigmoid_extremes"):
        return 0.0
    if pattern == "flatten_sentinel":
        selected_column = (row * 37 + (row % n) * d) % width
        return 1.0 if column == selected_column else 0.0
    if pattern == "asymmetric_phi_bias":
        return (((row + 1) * 11 + (column + 1) * 3) % 31 - 15) / 64.0
    if pattern == "large_magnitude":
        return (1e-5 if index % 3 else -1e-5) * (1 + row % 3)
    if pattern == "tiny_magnitude":
        return ((row % 5) - 2) / 16.0
    if pattern == "alternating_sign":
        return (1.0 if (row + column) % 2 == 0 else -1.0) / 32.0
    return signed_random / 8.0


def iter_input_values(case: ValidCase, tensor_name: str):
    """Yield a complete input tensor lazily according to PATTERN_RECIPE_VERSION."""

    meta = getattr(case.call, tensor_name)
    if meta is None:
        raise ValueError(f"{tensor_name} is absent in {case.name}")
    for index in range(tensor_element_count(meta)):
        yield pattern_value(case, tensor_name, index)

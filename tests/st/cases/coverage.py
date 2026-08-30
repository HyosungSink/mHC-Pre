"""Coverage cases for documented dimensions, boundaries, and rejected calls."""
from __future__ import annotations

from dataclasses import replace

from tests.common.cases import (
    BFLOAT16, FLOAT16, FLOAT32, InvalidCase, MhcPreCall, ValidCase,
    SUPPORTED_B, SUPPORTED_N, SUPPORTED_S, make_case, make_valid_call, tensor,
)


def _shape_matrix_cases() -> tuple[ValidCase, ...]:
    """Cover every documented (B, S, n) tuple without a huge Cartesian D sweep."""

    patterns = (
        "random_signed",
        "asymmetric_phi_bias",
        "alternating_sign",
        "nonuniform_gamma",
        "all_zeros",
        "large_magnitude",
    )
    d_choices = (16, 32, 48, 80, 256)
    cases: list[ValidCase] = []
    index = 0
    for ib, b in enumerate(SUPPORTED_B):
        for iseq, s in enumerate(SUPPORTED_S):
            for inode, n in enumerate(SUPPORTED_N):
                dtype = FLOAT16 if (ib + iseq + inode) % 2 == 0 else BFLOAT16
                has_gamma = (ib + inode) % 2 == 0
                include_intermediates = (iseq + inode) % 2 == 0
                d = d_choices[index % len(d_choices)]
                pattern = patterns[index % len(patterns)]
                cases.append(
                    ValidCase(
                        name=f"matrix_b{b}_s{s}_n{n}_d{d}_{dtype}_{'gamma' if has_gamma else 'nogamma'}",
                        call=make_valid_call(
                            b,
                            s,
                            n,
                            d,
                            x_dtype=dtype,
                            gamma=has_gamma,
                            include_intermediates=include_intermediates,
                        ),
                        pattern=pattern,
                        level="standard",
                        focus=("shape_matrix", "long_sequence", f"B={b}", f"S={s}", f"n={n}"),
                        seed=0x4D484300 + index,
                    )
                )
                index += 1
    return tuple(cases)


def _boundary_cases() -> tuple[ValidCase, ...]:
    return (
        make_case("d_typical_2560", 1, 4096, 4, 2560, FLOAT16, False, False, "random_signed", "extended",
             ("typical_D", "long_reduction", "gamma_absent"), 201),
        make_case("d_typical_4096", 1, 4096, 6, 4096, BFLOAT16, True, True, "nonuniform_gamma", "extended",
             ("typical_D", "long_reduction", "gamma"), 202),
        make_case("d_tile_tail_8176", 1, 4096, 8, 8176, FLOAT16, False, False, "random_signed", "extended",
             ("D_tile_tail", "long_reduction"), 203),
        make_case("d_near_max_tail_16368", 1, 4096, 4, 16368, BFLOAT16, True, False, "asymmetric_phi_bias", "stress",
             ("D_tile_tail", "near_max_D", "gamma"), 301),
        make_case("d_max_16384", 1, 4096, 4, 16384, FLOAT16, False, True, "random_signed", "stress",
             ("max_D", "long_reduction", "optional_outputs"), 302),
        make_case("max_bs_positions", 4, 32768, 8, 16, BFLOAT16, True, False, "alternating_sign", "stress",
             ("max_B", "max_S", "parallel_axis", "gamma"), 303),
        make_case("max_all_dimensions_shape_only", 4, 32768, 8, 16384, FLOAT16, False, False, "random_signed", "shape_only",
             ("max_B", "max_S", "max_n", "max_D", "shape_only"), 401),
    )


VALID_CASES: tuple[ValidCase, ...] = _shape_matrix_cases() + _boundary_cases()


def _invalid_cases() -> tuple[InvalidCase, ...]:
    base = make_valid_call(
        1,
        4096,
        4,
        16,
        x_dtype=FLOAT16,
        gamma=True,
        include_intermediates=True,
    )
    no_optional = make_valid_call(
        1,
        4096,
        4,
        16,
        x_dtype=FLOAT16,
        gamma=False,
        include_intermediates=False,
    )
    invalid: list[InvalidCase] = []

    def add(name: str, call: MhcPreCall, error: str) -> None:
        invalid.append(InvalidCase(name, call, error))

    add("missing_x", replace(base, x=None), "x.required")
    add("missing_phi", replace(base, phi=None), "phi.required")
    add("missing_alpha", replace(base, alpha=None), "alpha.required")
    add("missing_bias", replace(base, bias=None), "bias.required")
    add("missing_hIn", replace(base, h_in=None), "h_in.required")
    add("missing_hPost", replace(base, h_post=None), "h_post.required")
    add("missing_hRes", replace(base, h_res=None), "h_res.required")
    add("x_rank_3", replace(base, x=tensor((1, 4096, 64), FLOAT16)), "x.rank")
    add("unsupported_B", replace(base, x=tensor((3, 4096, 4, 16), FLOAT16)), "x.B")
    add("zero_B", replace(base, x=tensor((0, 4096, 4, 16), FLOAT16)), "x.B")
    add("B_above_max", replace(base, x=tensor((8, 4096, 4, 16), FLOAT16)), "x.B")
    add("unsupported_S", replace(base, x=tensor((1, 4097, 4, 16), FLOAT16)), "x.S")
    add("S_below_min", replace(base, x=tensor((1, 2048, 4, 16), FLOAT16)), "x.S")
    add("S_between_supported", replace(base, x=tensor((1, 12288, 4, 16), FLOAT16)), "x.S")
    add("S_above_max", replace(base, x=tensor((1, 65536, 4, 16), FLOAT16)), "x.S")
    add("unsupported_n", replace(base, x=tensor((1, 4096, 5, 16), FLOAT16)), "x.n")
    add("n_below_min", replace(base, x=tensor((1, 4096, 2, 16), FLOAT16)), "x.n")
    add("n_above_max", replace(base, x=tensor((1, 4096, 10, 16), FLOAT16)), "x.n")
    add("D_one_unaligned", replace(base, x=tensor((1, 4096, 4, 1), FLOAT16)), "x.D")
    add("D_not_aligned", replace(base, x=tensor((1, 4096, 4, 17), FLOAT16)), "x.D")
    add("D_zero", replace(base, x=tensor((1, 4096, 4, 0), FLOAT16)), "x.D")
    add("D_above_max", replace(base, x=tensor((1, 4096, 4, 16400), FLOAT16)), "x.D")
    add("x_wrong_dtype", replace(base, x=tensor((1, 4096, 4, 16), FLOAT32)), "x.dtype")
    add("x_wrong_layout", replace(base, x=tensor((1, 4096, 4, 16), FLOAT16, layout="TND")), "x.layout")
    add("x_non_contiguous", replace(base, x=tensor((1, 4096, 4, 16), FLOAT16, contiguous=False)), "x.contiguous")
    add("phi_wrong_rows", replace(base, phi=tensor((23, 64), FLOAT32)), "phi.shape")
    add("phi_wrong_columns", replace(base, phi=tensor((24, 63), FLOAT32)), "phi.shape")
    add("phi_wrong_rank", replace(base, phi=tensor((24, 4, 16), FLOAT32)), "phi.shape")
    add("phi_wrong_dtype", replace(base, phi=tensor((24, 64), FLOAT16)), "phi.dtype")
    add("phi_wrong_layout", replace(base, phi=tensor((24, 64), FLOAT32, layout="NZ")), "phi.layout")
    add("alpha_wrong_shape", replace(base, alpha=tensor((4,), FLOAT32)), "alpha.shape")
    add("alpha_wrong_dtype", replace(base, alpha=tensor((3,), FLOAT16)), "alpha.dtype")
    add("bias_wrong_shape", replace(base, bias=tensor((23,), FLOAT32)), "bias.shape")
    add("bias_wrong_dtype", replace(base, bias=tensor((24,), FLOAT16)), "bias.dtype")
    add("gamma_wrong_n", replace(base, gamma=tensor((5, 16), FLOAT32)), "gamma.shape")
    add("gamma_wrong_D", replace(base, gamma=tensor((4, 32), FLOAT32)), "gamma.shape")
    add("gamma_wrong_dtype", replace(base, gamma=tensor((4, 16), FLOAT16)), "gamma.dtype")
    add("gamma_non_contiguous", replace(base, gamma=tensor((4, 16), FLOAT32, contiguous=False)), "gamma.contiguous")
    add("hIn_wrong_shape", replace(base, h_in=tensor((1, 4096, 4, 16), FLOAT16)), "h_in.shape")
    add("hIn_wrong_dtype", replace(base, h_in=tensor((1, 4096, 16), BFLOAT16)), "h_in.dtype")
    add("hPost_wrong_shape", replace(base, h_post=tensor((1, 4096, 5), FLOAT32)), "h_post.shape")
    add("hPost_wrong_dtype", replace(base, h_post=tensor((1, 4096, 4), FLOAT16)), "h_post.dtype")
    add("hRes_wrong_shape", replace(base, h_res=tensor((1, 4096, 16), FLOAT32)), "h_res.shape")
    add("hRes_wrong_dtype", replace(base, h_res=tensor((1, 4096, 4, 4), FLOAT16)), "h_res.dtype")
    add("optional_only_invRms", replace(no_optional, inv_rms=tensor((1, 4096), FLOAT32)), "optional_outputs.group")
    add(
        "optional_missing_hPre",
        replace(base, h_pre=None),
        "optional_outputs.group",
    )
    add("invRms_wrong_shape", replace(base, inv_rms=tensor((1, 4096, 1), FLOAT32)), "inv_rms.shape")
    add("invRms_wrong_dtype", replace(base, inv_rms=tensor((1, 4096), FLOAT16)), "inv_rms.dtype")
    add("hMix_wrong_shape", replace(base, h_mix=tensor((1, 4096, 23), FLOAT32)), "h_mix.shape")
    add("hMix_wrong_dtype", replace(base, h_mix=tensor((1, 4096, 24), FLOAT16)), "h_mix.dtype")
    add("hPre_wrong_shape", replace(base, h_pre=tensor((1, 4096, 5), FLOAT32)), "h_pre.shape")
    add("hPre_wrong_dtype", replace(base, h_pre=tensor((1, 4096, 4), FLOAT16)), "h_pre.dtype")
    add("normEps_wrong_type", replace(base, norm_eps="1e-6"), "norm_eps.type")
    add("hcEps_wrong_type", replace(base, hc_eps=1), "hc_eps.type")
    return tuple(invalid)


INVALID_CASES: tuple[InvalidCase, ...] = _invalid_cases()

"""Correctness cases for mHcPre math, layout, and numerical extremes."""
from __future__ import annotations

from tests.common.cases import BFLOAT16, FLOAT16, ValidCase, make_case


CASES: tuple[ValidCase, ...] = (
    make_case("doc_ones", 1, 4096, 4, 16, FLOAT16, True, True, "all_ones", "focused",
         ("document_example", "closed_form", "gamma", "optional_outputs"), 101),
    make_case("zero_no_gamma", 2, 4096, 6, 16, FLOAT16, False, False, "all_zeros", "focused",
         ("zero", "norm_eps", "gamma_absent", "no_optional_outputs"), 102),
    make_case("all_tensors_zero", 1, 4096, 4, 16, BFLOAT16, True, True, "all_tensors_zero", "focused",
         ("all_tensors_zero", "zero_gamma", "closed_form", "finite_outputs"), 109),
    make_case("flatten_order", 1, 8192, 8, 48, BFLOAT16, True, True, "flatten_sentinel", "focused",
         ("ND_flatten_order", "phi_column_order", "split_boundaries"), 103),
    make_case("asymmetric_parameters", 4, 4096, 6, 80, FLOAT16, True, True, "asymmetric_phi_bias", "focused",
         ("asymmetric_phi", "asymmetric_bias", "hRes_row_major"), 104),
    make_case("sigmoid_saturation", 1, 16384, 4, 32, BFLOAT16, False, False, "sigmoid_extremes", "focused",
         ("sigmoid_positive", "sigmoid_negative", "finite_outputs"), 105),
    make_case("large_values", 1, 32768, 8, 16, FLOAT16, True, False, "large_magnitude", "focused",
         ("large_values", "fp32_accumulation", "rms_stability"), 106),
    make_case("custom_eps", 2, 8192, 6, 64, BFLOAT16, False, True, "tiny_magnitude", "focused",
         ("custom_norm_eps", "custom_hc_eps", "tiny_values"), 107, 1e-4, 2e-3),
    make_case("signed_alpha", 4, 4096, 8, 96, FLOAT16, True, False, "alternating_sign", "focused",
         ("negative_alpha", "signed_inputs", "hIn_weighting"), 108),
)

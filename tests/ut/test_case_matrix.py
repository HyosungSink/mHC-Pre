from __future__ import annotations

import unittest
from itertools import product

from tests.common.cases import (
    BFLOAT16,
    FLOAT16,
    FLOAT32,
    MAX_D,
    OPTIONAL_OUTPUT_NAMES,
    PATTERN_RECIPE_VERSION,
    SUPPORTED_B,
    SUPPORTED_N,
    SUPPORTED_S,
    estimated_tensor_bytes,
    pattern_value,
    optional_outputs_present,
    validate_call,
)
from tests.st.cases import (
    CORRECTNESS_CASES,
    COVERAGE_INVALID_CASES,
    COVERAGE_VALID_CASES,
    INVALID_CASES,
    VALID_CASES,
)
from tests.st.scripts.gen_data import build_manifest
from tests.st.cases import accepted_cases, rejected_cases


class TestValidCaseCatalog(unittest.TestCase):
    def test_named_suite_selection(self) -> None:
        expected = {"coverage": (43, 54), "correctness": (9, 0), "full": (52, 54)}
        for suite, counts in expected.items():
            with self.subTest(suite=suite):
                self.assertEqual((len(accepted_cases(suite)), len(rejected_cases(suite))), counts)
                manifest = build_manifest(suite)
                self.assertEqual(manifest["suite"], suite)
                self.assertEqual((len(manifest["valid_cases"]), len(manifest["invalid_cases"])), counts)

    def test_categories_are_separate_and_complete(self) -> None:
        self.assertEqual((len(COVERAGE_VALID_CASES), len(CORRECTNESS_CASES)), (43, 9))
        self.assertEqual(len(COVERAGE_INVALID_CASES), 54)
        self.assertEqual(set(VALID_CASES), set(COVERAGE_VALID_CASES) | set(CORRECTNESS_CASES))
        self.assertFalse(set(COVERAGE_VALID_CASES) & set(CORRECTNESS_CASES))
        self.assertTrue(all(case.level != "focused" for case in COVERAGE_VALID_CASES))
        self.assertTrue(all(case.level == "focused" for case in CORRECTNESS_CASES))
        manifest = build_manifest()
        self.assertEqual(
            {case["name"] for case in manifest["valid_cases"] if case["category"] == "coverage"},
            {case.name for case in COVERAGE_VALID_CASES},
        )
        self.assertEqual(
            {case["name"] for case in manifest["valid_cases"] if case["category"] == "correctness"},
            {case.name for case in CORRECTNESS_CASES},
        )
        self.assertTrue(all(case["category"] == "coverage" for case in manifest["invalid_cases"]))

    def test_names_are_unique(self) -> None:
        names = [case.name for case in VALID_CASES]
        self.assertEqual(len(names), len(set(names)))

    def test_every_valid_case_satisfies_formal_contract(self) -> None:
        failures = {case.name: validate_call(case.call) for case in VALID_CASES if validate_call(case.call)}
        self.assertEqual(failures, {})

    def test_complete_B_S_n_cartesian_coverage(self) -> None:
        matrix_cases = [case for case in VALID_CASES if "shape_matrix" in case.focus]
        actual = {(case.call.x.shape[0], case.call.x.shape[1], case.call.x.shape[2]) for case in matrix_cases}
        expected = set(product(SUPPORTED_B, SUPPORTED_S, SUPPORTED_N))
        self.assertEqual(actual, expected)
        self.assertEqual(len(matrix_cases), len(expected))

    def test_D_boundaries_typical_values_and_tail_cases(self) -> None:
        dimensions = {case.call.x.shape[3] for case in VALID_CASES}
        self.assertTrue({16, 2560, 4096, 8176, 16368, MAX_D}.issubset(dimensions))
        self.assertGreaterEqual(sum("D_tile_tail" in case.focus for case in VALID_CASES), 2)

    def test_gamma_branches_are_covered_for_each_n_and_dtype(self) -> None:
        for n in SUPPORTED_N:
            self.assertEqual({case.call.gamma is not None for case in VALID_CASES if case.call.x.shape[2] == n}, {False, True})
        for dtype in (FLOAT16, BFLOAT16):
            self.assertEqual({case.call.gamma is not None for case in VALID_CASES if case.call.x.dtype == dtype}, {False, True})

    def test_mixed_precision_contract_is_encoded(self) -> None:
        self.assertEqual({case.call.x.dtype for case in VALID_CASES}, {FLOAT16, BFLOAT16})
        for case in VALID_CASES:
            self.assertEqual(case.call.phi.dtype, FLOAT32)
            self.assertEqual(case.call.alpha.dtype, FLOAT32)
            self.assertEqual(case.call.bias.dtype, FLOAT32)
            if case.call.gamma is not None:
                self.assertEqual(case.call.gamma.dtype, FLOAT32)
            self.assertEqual(case.call.h_in.dtype, case.call.x.dtype)
            self.assertEqual(case.call.h_post.dtype, FLOAT32)
            self.assertEqual(case.call.h_res.dtype, FLOAT32)

    def test_optional_outputs_are_all_or_none_and_both_modes_exist(self) -> None:
        modes = {optional_outputs_present(case.call) for case in VALID_CASES}
        self.assertIn(frozenset(), modes)
        self.assertIn(OPTIONAL_OUTPUT_NAMES, modes)
        self.assertEqual(modes, {frozenset(), OPTIONAL_OUTPUT_NAMES})

    def test_numerical_and_layout_focuses_exist(self) -> None:
        focuses = {focus for case in VALID_CASES for focus in case.focus}
        required = {
            "zero",
            "large_values",
            "tiny_values",
            "sigmoid_positive",
            "sigmoid_negative",
            "asymmetric_phi",
            "asymmetric_bias",
            "ND_flatten_order",
            "phi_column_order",
            "split_boundaries",
            "hRes_row_major",
            "hIn_weighting",
            "custom_norm_eps",
            "custom_hc_eps",
            "fp32_accumulation",
        }
        self.assertTrue(required.issubset(focuses), required - focuses)

    def test_long_sequence_and_parallel_extremes_exist(self) -> None:
        focuses = {focus for case in VALID_CASES for focus in case.focus}
        self.assertTrue({"max_B", "max_S", "max_D", "parallel_axis"}.issubset(focuses))

    def test_shape_only_case_is_explicitly_separated(self) -> None:
        shape_only = [case for case in VALID_CASES if case.level == "shape_only"]
        self.assertEqual(len(shape_only), 1)
        self.assertIn("shape_only", shape_only[0].focus)
        self.assertGreater(estimated_tensor_bytes(shape_only[0].call), 30 * 1024**3)

    def test_all_lazy_input_patterns_are_deterministic_and_materializable(self) -> None:
        self.assertEqual(PATTERN_RECIPE_VERSION, "mhc_pre_pattern/v1")
        for case in VALID_CASES:
            for tensor_name in ("x", "phi", "alpha", "bias", "gamma"):
                meta = getattr(case.call, tensor_name)
                if meta is None:
                    continue
                last = 1
                for dim in meta.shape:
                    last *= dim
                indices = sorted({0, min(1, last - 1), last - 1})
                first = [pattern_value(case, tensor_name, index) for index in indices]
                second = [pattern_value(case, tensor_name, index) for index in indices]
                self.assertEqual(first, second, (case.name, tensor_name))

    def test_focused_pattern_recipes_have_the_intended_signatures(self) -> None:
        by_name = {case.name: case for case in VALID_CASES}
        doc = by_name["doc_ones"]
        self.assertEqual([pattern_value(doc, "x", i) for i in (0, 63)], [1.0, 1.0])
        self.assertEqual([pattern_value(doc, "phi", i) for i in (0, 1535)], [1.0, 1.0])
        zero = by_name["zero_no_gamma"]
        self.assertEqual([pattern_value(zero, "x", i) for i in (0, 95)], [0.0, 0.0])
        all_zero = by_name["all_tensors_zero"]
        for tensor_name in ("x", "phi", "alpha", "bias", "gamma"):
            self.assertEqual(pattern_value(all_zero, tensor_name, 0), 0.0)
        large = by_name["large_values"]
        self.assertEqual([pattern_value(large, "x", i) for i in (0, 1)], [60000.0, -60000.0])
        flatten = by_name["flatten_order"]
        width = flatten.call.x.shape[2] * flatten.call.x.shape[3]
        self.assertEqual(pattern_value(flatten, "phi", 0), 1.0)
        self.assertEqual(sum(pattern_value(flatten, "phi", i) for i in range(width)), 1.0)


class TestInvalidCaseCatalog(unittest.TestCase):
    def test_names_are_unique(self) -> None:
        names = [case.name for case in INVALID_CASES]
        self.assertEqual(len(names), len(set(names)))

    def test_each_invalid_case_triggers_its_expected_rejection(self) -> None:
        failures = {}
        for case in INVALID_CASES:
            errors = validate_call(case.call)
            if case.expected_error not in errors:
                failures[case.name] = errors
        self.assertEqual(failures, {})

    def test_rejection_categories_cover_documented_constraints(self) -> None:
        expected_errors = {case.expected_error for case in INVALID_CASES}
        required = {
            "x.required",
            "phi.required",
            "alpha.required",
            "bias.required",
            "x.rank",
            "x.B",
            "x.S",
            "x.n",
            "x.D",
            "x.dtype",
            "x.layout",
            "x.contiguous",
            "phi.shape",
            "phi.dtype",
            "alpha.shape",
            "alpha.dtype",
            "bias.shape",
            "bias.dtype",
            "gamma.shape",
            "gamma.dtype",
            "h_in.shape",
            "h_post.shape",
            "h_res.shape",
            "optional_outputs.group",
        }
        self.assertTrue(required.issubset(expected_errors), required - expected_errors)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from tests.common.mock_cases import MOCK_CASES_PATH, load_cases, materialize_case


class TestMockCases(unittest.TestCase):
    def test_catalog_covers_small_mixed_precision_paths(self) -> None:
        document = load_cases(MOCK_CASES_PATH)
        self.assertEqual(document["category"], "mock")
        cases = document["cases"]
        self.assertEqual({tuple(case["shape"][2:]) for case in cases}, {(4, 16), (8, 16)})
        self.assertTrue(all(case["x_dtype"] == "float16" for case in cases))
        self.assertTrue(all(case["param_dtype"] == "float32" for case in cases))

    def test_fixtures_have_exact_tensor_sizes_and_nontrivial_golden(self) -> None:
        cases = load_cases(MOCK_CASES_PATH)["cases"]
        with tempfile.TemporaryDirectory(prefix="mhcpre_mock_fixture_") as temporary:
            root = Path(temporary)
            for case in cases:
                directory = root / case["case_id"]
                materialize_case(case, directory)
                b, s, n, d = case["shape"]
                rows = n * n + 2 * n
                self.assertEqual((directory / "x.bin").stat().st_size, b * s * n * d * 2)
                self.assertEqual((directory / "phi.bin").stat().st_size, rows * n * d * 4)
                self.assertEqual((directory / "h_in_golden.bin").stat().st_size, b * s * d * 2)
                self.assertEqual((directory / "h_post_golden.bin").stat().st_size, b * s * n * 4)
                h_res = (directory / "h_res_golden.bin").read_bytes()
                self.assertEqual(len(h_res), b * s * n * n * 4)
                self.assertNotEqual(h_res, b"\0" * len(h_res))

    def test_absent_gamma_fixture_has_no_gamma_file(self) -> None:
        case = copy.deepcopy(load_cases(MOCK_CASES_PATH)["cases"][0])
        case["case_id"] = "mock_no_gamma"
        case["has_gamma"] = False
        case["input_profile"] = "all_zeros"
        with tempfile.TemporaryDirectory(prefix="mhcpre_mock_no_gamma_") as temporary:
            directory = Path(temporary) / case["case_id"]
            materialize_case(case, directory)
            self.assertFalse((directory / "gamma.bin").exists())
            self.assertIn("has_gamma 0", (directory / "case.meta").read_text())
            self.assertEqual((directory / "x.bin").read_bytes(),
                             b"\0" * (directory / "x.bin").stat().st_size)


if __name__ == "__main__":
    unittest.main()

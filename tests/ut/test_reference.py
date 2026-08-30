from __future__ import annotations

import math
import unittest

from tests.common.reference import bfloat16, fp16, mhc_pre_reference, stable_sigmoid
from tests.common.cases import BFLOAT16, FLOAT16


def zeros(count: int) -> list[float]:
    return [0.0] * count


class ReferenceAssertions(unittest.TestCase):
    def assertSequenceClose(self, actual, expected, *, rel_tol=1e-5, abs_tol=1e-6):
        self.assertEqual(len(actual), len(expected))
        for index, (left, right) in enumerate(zip(actual, expected)):
            self.assertTrue(
                math.isclose(left, right, rel_tol=rel_tol, abs_tol=abs_tol),
                f"element {index}: {left} != {right}",
            )


class TestMhcPreReference(ReferenceAssertions):
    def test_document_all_ones_example(self) -> None:
        n, d = 4, 16
        width = n * d
        rows = n * n + 2 * n
        result = mhc_pre_reference(
            x=[1.0] * width,
            shape=(1, 1, n, d),
            phi=[1.0] * (rows * width),
            alpha=[1.0, 1.0, 1.0],
            bias=[0.0] * rows,
            gamma=[1.0] * width,
        )
        self.assertSequenceClose(result.h_in, [4.0] * d, abs_tol=2e-3)
        self.assertSequenceClose(result.h_post, [2.0] * n, abs_tol=2e-6)
        self.assertSequenceClose(result.h_res, [64.0 / math.sqrt(1.0 + 1e-6)] * (n * n), abs_tol=1e-4)
        self.assertSequenceClose(result.h_mix, [64.0] * rows)

    def test_zero_input_uses_bias_and_never_reads_gamma(self) -> None:
        n, d = 6, 16
        width = n * d
        rows = n * n + 2 * n
        bias = [(-2.5 + index * 0.1) for index in range(rows)]
        result = mhc_pre_reference(
            x=zeros(width),
            shape=(1, 1, n, d),
            phi=[0.25] * (rows * width),
            alpha=[3.0, -2.0, 4.0],
            bias=bias,
            gamma=None,
        )
        self.assertEqual(result.h_in, tuple(zeros(d)))
        self.assertSequenceClose(result.h_mix, zeros(rows))
        self.assertSequenceClose(result.h_res, bias[2 * n :])
        self.assertSequenceClose(result.h_pre, [stable_sigmoid(value) + 1e-6 for value in bias[:n]])
        self.assertSequenceClose(result.h_post, [2.0 * stable_sigmoid(value) for value in bias[n : 2 * n]])

    def test_gamma_and_ND_flatten_order(self) -> None:
        n, d = 4, 16
        width = n * d
        rows = n * n + 2 * n
        x = [float(index + 1) for index in range(width)]
        gamma = [1.0 + index / 64.0 for index in range(width)]
        selected = (0, d - 1, d, 2 * d + 3, width - 1)
        phi = zeros(rows * width)
        for row, column in enumerate(selected):
            phi[row * width + column] = 1.0
        with_gamma = mhc_pre_reference(
            x=x,
            shape=(1, 1, n, d),
            phi=phi,
            alpha=[0.0, 0.0, 0.0],
            bias=zeros(rows),
            gamma=gamma,
        )
        without_gamma = mhc_pre_reference(
            x=x,
            shape=(1, 1, n, d),
            phi=phi,
            alpha=[0.0, 0.0, 0.0],
            bias=zeros(rows),
            gamma=None,
        )
        self.assertSequenceClose(without_gamma.h_mix[: len(selected)], [x[index] for index in selected])
        self.assertSequenceClose(
            with_gamma.h_mix[: len(selected)],
            [x[index] * gamma[index] for index in selected],
            abs_tol=2e-5,
        )

    def test_rms_covers_the_complete_nD_and_scales_projection(self) -> None:
        n, d = 4, 16
        width = n * d
        rows = n * n + 2 * n
        x = zeros(width)
        x[-1] = 2.0  # Last element of the last n lane catches partial reductions.
        phi = zeros(rows * width)
        phi[(2 * n) * width + (width - 1)] = 3.0
        result = mhc_pre_reference(
            x=x,
            shape=(1, 1, n, d),
            phi=phi,
            alpha=[0.0, 0.0, 1.0],
            bias=zeros(rows),
            gamma=None,
        )
        expected_inv_rms = 1.0 / math.sqrt((2.0 * 2.0) / width + 1e-6)
        self.assertSequenceClose(result.inv_rms, [expected_inv_rms], abs_tol=2e-6)
        self.assertSequenceClose(result.h_mix[2 * n : 2 * n + 1], [6.0])
        self.assertSequenceClose(result.h_res[:1], [6.0 * expected_inv_rms], abs_tol=2e-5)

    def test_three_alpha_values_apply_to_their_own_segments(self) -> None:
        n, d = 4, 16
        width = n * d
        rows = n * n + 2 * n
        result = mhc_pre_reference(
            x=[1.0] * width,
            shape=(1, 1, n, d),
            phi=[1.0 / width] * (rows * width),
            alpha=[0.5, -1.25, 2.0],
            bias=zeros(rows),
            gamma=None,
        )
        scaled = result.inv_rms[0]
        self.assertSequenceClose(result.h_pre, [stable_sigmoid(0.5 * scaled) + 1e-6] * n)
        self.assertSequenceClose(result.h_post, [2.0 * stable_sigmoid(-1.25 * scaled)] * n)
        self.assertSequenceClose(result.h_res, [2.0 * scaled] * (n * n))

    def test_split_boundaries_and_hRes_row_major_order(self) -> None:
        for n in (4, 6, 8):
            with self.subTest(n=n):
                d = 16
                width = n * d
                rows = n * n + 2 * n
                bias = [float(index) / 10.0 for index in range(rows)]
                result = mhc_pre_reference(
                    x=zeros(width),
                    shape=(1, 1, n, d),
                    phi=zeros(rows * width),
                    alpha=[0.0, 0.0, 0.0],
                    bias=bias,
                    gamma=None,
                )
                self.assertSequenceClose(result.h_pre, [stable_sigmoid(value) + 1e-6 for value in bias[:n]])
                self.assertSequenceClose(result.h_post, [2.0 * stable_sigmoid(value) for value in bias[n : 2 * n]])
                self.assertSequenceClose(result.h_res, bias[2 * n :])
                self.assertEqual(result.h_res_shape, (1, 1, n, n))

    def test_hIn_weights_all_n_lanes(self) -> None:
        for n in (4, 6, 8):
            with self.subTest(n=n):
                d = 16
                width = n * d
                rows = n * n + 2 * n
                x = [float(lane + 1) for lane in range(n) for _ in range(d)]
                result = mhc_pre_reference(
                    x=x,
                    shape=(1, 1, n, d),
                    phi=zeros(rows * width),
                    alpha=[0.0, 0.0, 0.0],
                    bias=zeros(rows),
                    gamma=None,
                    x_dtype=FLOAT16,
                )
                expected = fp16((0.5 + 1e-6) * sum(range(1, n + 1)))
                self.assertEqual(result.h_in, tuple([expected] * d))

    def test_hIn_uses_original_x_instead_of_xGamma(self) -> None:
        n, d = 4, 16
        width = n * d
        rows = n * n + 2 * n
        x = [float(lane + 1) for lane in range(n) for _ in range(d)]
        common = dict(
            x=x,
            shape=(1, 1, n, d),
            phi=zeros(rows * width),
            alpha=zeros(3),
            bias=zeros(rows),
        )
        unscaled = mhc_pre_reference(**common, gamma=None)
        scaled = mhc_pre_reference(**common, gamma=[0.0, 10.0] * (width // 2))
        self.assertEqual(unscaled.h_pre, scaled.h_pre)
        self.assertEqual(unscaled.h_in, scaled.h_in)

    def test_sigmoid_saturation_is_finite(self) -> None:
        n, d = 4, 16
        width = n * d
        rows = n * n + 2 * n
        bias = [1000.0, -1000.0, 100.0, -100.0] * 2 + zeros(n * n)
        result = mhc_pre_reference(
            x=zeros(width),
            shape=(1, 1, n, d),
            phi=zeros(rows * width),
            alpha=[1.0, 1.0, 1.0],
            bias=bias,
            gamma=None,
        )
        self.assertTrue(all(math.isfinite(value) for value in result.h_pre + result.h_post))
        self.assertSequenceClose(result.h_pre, [1.000001, 0.000001, 1.000001, 0.000001], abs_tol=2e-7)
        self.assertSequenceClose(result.h_post, [2.0, 0.0, 2.0, 0.0], abs_tol=1e-7)

    def test_large_magnitude_square_and_accumulation_stay_finite(self) -> None:
        n, d = 4, 16
        width = n * d
        rows = n * n + 2 * n
        result = mhc_pre_reference(
            x=[60000.0 if index % 2 == 0 else -60000.0 for index in range(width)],
            shape=(1, 1, n, d),
            phi=[1e-5 if index % 3 else -1e-5 for index in range(rows * width)],
            alpha=[0.25, -0.5, 1.5],
            bias=zeros(rows),
            gamma=[1.0] * width,
        )
        values = result.inv_rms + result.h_mix + result.h_pre + result.h_post + result.h_res
        self.assertTrue(all(math.isfinite(value) for value in values))
        self.assertGreater(result.inv_rms[0], 0.0)

    def test_normEps_and_hcEps_affect_independent_stages(self) -> None:
        n, d = 4, 16
        width = n * d
        rows = n * n + 2 * n
        common = dict(
            x=[0.01] * width,
            shape=(1, 1, n, d),
            phi=[1.0] * (rows * width),
            alpha=[1.0, 1.0, 1.0],
            bias=zeros(rows),
            gamma=None,
        )
        baseline = mhc_pre_reference(**common, norm_eps=1e-6, hc_eps=1e-6)
        changed_norm = mhc_pre_reference(**common, norm_eps=1e-2, hc_eps=1e-6)
        changed_hc = mhc_pre_reference(**common, norm_eps=1e-6, hc_eps=0.1)
        self.assertNotEqual(baseline.inv_rms, changed_norm.inv_rms)
        self.assertNotEqual(baseline.h_res, changed_norm.h_res)
        self.assertEqual(baseline.inv_rms, changed_hc.inv_rms)
        self.assertSequenceClose(
            changed_hc.h_pre,
            [value + (0.1 - 1e-6) for value in baseline.h_pre],
            abs_tol=2e-6,
        )

    def test_optional_intermediates_are_all_present_or_all_absent(self) -> None:
        n, d = 4, 16
        width = n * d
        rows = n * n + 2 * n
        common = dict(
            x=zeros(width),
            shape=(1, 1, n, d),
            phi=zeros(rows * width),
            alpha=zeros(3),
            bias=zeros(rows),
            gamma=None,
        )
        present = mhc_pre_reference(**common, include_intermediates=True)
        absent = mhc_pre_reference(**common, include_intermediates=False)
        self.assertTrue(all(value is not None for value in (present.inv_rms, present.h_mix, present.h_pre)))
        self.assertTrue(all(value is None for value in (absent.inv_rms, absent.h_mix, absent.h_pre)))
        self.assertEqual(present.h_in, absent.h_in)
        self.assertEqual(present.h_post, absent.h_post)
        self.assertEqual(present.h_res, absent.h_res)

    def test_output_shapes_and_determinism(self) -> None:
        b, s, n, d = 1, 2, 4, 16
        width = n * d
        rows = n * n + 2 * n
        kwargs = dict(
            x=[((index % 11) - 5) / 8.0 for index in range(b * s * width)],
            shape=(b, s, n, d),
            phi=[((index % 7) - 3) / 16.0 for index in range(rows * width)],
            alpha=[0.5, -1.25, 2.0],
            bias=[((index % 5) - 2) / 10.0 for index in range(rows)],
            gamma=[1.0 + (index % 3) / 8.0 for index in range(width)],
            x_dtype=BFLOAT16,
        )
        first = mhc_pre_reference(**kwargs)
        second = mhc_pre_reference(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first.h_in_shape, (b, s, d))
        self.assertEqual(first.h_post_shape, (b, s, n))
        self.assertEqual(first.h_res_shape, (b, s, n, n))
        self.assertEqual(first.inv_rms_shape, (b, s))
        self.assertEqual(first.h_mix_shape, (b, s, rows))
        self.assertEqual(first.h_pre_shape, (b, s, n))

    def test_fp16_and_bfloat16_rounding(self) -> None:
        self.assertEqual(fp16(1.0001), 1.0)
        self.assertEqual(bfloat16(1.00390625), 1.0)  # exact halfway, ties to even
        self.assertEqual(bfloat16(1.004), 1.0078125)

    def test_bad_buffer_lengths_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "x has"):
            mhc_pre_reference(
                x=[0.0],
                shape=(1, 1, 4, 16),
                phi=zeros(24 * 64),
                alpha=zeros(3),
                bias=zeros(24),
                gamma=None,
            )


if __name__ == "__main__":
    unittest.main()

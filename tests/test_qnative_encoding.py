from __future__ import annotations

import math
import unittest

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qnative_encoding import (  # noqa: E402
    QuantumNativeTokenizer,
    RelationPhaseOperator,
    born_overlap,
    givens_mix,
    norm,
    phase_aligned_state,
)


class QuantumNativeEncodingTests(unittest.TestCase):
    def test_tokenizer_outputs_normalized_complex_state(self) -> None:
        tokenizer = QuantumNativeTokenizer(dimension=16)
        state = tokenizer.encode("Transformer", tags=("noun",))

        self.assertEqual(len(state), 16)
        self.assertAlmostEqual(norm(state), 1.0)
        self.assertTrue(any(abs(z.imag) > 1e-9 for z in state))

    def test_born_overlap_bounds(self) -> None:
        left = phase_aligned_state(4, {0: 0.0, 1: 0.0})
        right = phase_aligned_state(4, {0: 0.0, 1: math.pi})

        self.assertGreaterEqual(born_overlap(left, right), 0.0)
        self.assertLessEqual(born_overlap(left, right), 1.0)
        self.assertAlmostEqual(born_overlap(left, right), 0.0)

    def test_diagonal_relation_can_separate_phase_compatible_pairs(self) -> None:
        cat = phase_aligned_state(4, {0: 0.0, 1: 0.0})
        purrs = phase_aligned_state(4, {0: 0.0, 1: math.pi})
        stone = phase_aligned_state(4, {0: 0.0, 1: math.pi})

        subject_verb = RelationPhaseOperator((0.0, math.pi, 0.0, 0.0))

        self.assertGreater(
            subject_verb.compatibility(cat, purrs),
            subject_verb.compatibility(stone, purrs),
        )
        self.assertLess(
            born_overlap(cat, purrs),
            born_overlap(stone, purrs),
        )

    def test_givens_mix_preserves_joint_norm(self) -> None:
        left = phase_aligned_state(4, {0: 0.0, 1: math.pi / 3})
        right = phase_aligned_state(4, {2: math.pi / 5, 3: math.pi / 7})

        before = math.sqrt(norm(left) ** 2 + norm(right) ** 2)
        mixed_left, mixed_right = givens_mix(left, right, theta=0.7, phi=0.2)
        after = math.sqrt(norm(mixed_left) ** 2 + norm(mixed_right) ** 2)

        self.assertAlmostEqual(before, after)


if __name__ == "__main__":
    unittest.main()

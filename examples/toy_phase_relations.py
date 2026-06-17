"""Toy demonstration of relation-as-relative-phase scoring."""

from __future__ import annotations

import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qnative_encoding import (  # noqa: E402
    RelationPhaseOperator,
    born_overlap,
    givens_mix,
    norm,
    phase_aligned_state,
)


def main() -> None:
    dim = 4

    # Basis sketch:
    # 0 = entity support
    # 1 = role/selection support
    # 2,3 = unused in this tiny probe
    #
    # Before the relation operator, "purrs" is phase-aligned with "stone" and
    # phase-opposed to "cat". The subject_verb operator flips basis 1, making
    # the animate subject compatible and the mismatched subject incompatible.
    cat = phase_aligned_state(dim, {0: 0.0, 1: 0.0})
    purrs = phase_aligned_state(dim, {0: 0.0, 1: math.pi})
    stone = phase_aligned_state(dim, {0: 0.0, 1: math.pi})

    subject_verb = RelationPhaseOperator((0.0, math.pi, 0.0, 0.0))

    compatible = subject_verb.compatibility(cat, purrs)
    incompatible = subject_verb.compatibility(stone, purrs)

    mixed_cat, mixed_purrs = givens_mix(cat, purrs, theta=math.pi / 8, phi=math.pi / 6)

    print("Born-rule relation scores")
    print(f"  subject_verb(cat, purrs):   {compatible:.3f}")
    print(f"  subject_verb(stone, purrs): {incompatible:.3f}")
    print()
    print("Unitary token mixing preserves joint norm")
    print(f"  before: {math.sqrt(norm(cat) ** 2 + norm(purrs) ** 2):.6f}")
    print(f"  after:  {math.sqrt(norm(mixed_cat) ** 2 + norm(mixed_purrs) ** 2):.6f}")
    print()
    print("Plain overlap, before relation operator")
    print(f"  overlap(cat, purrs):        {born_overlap(cat, purrs):.3f}")
    print(f"  overlap(stone, purrs):      {born_overlap(stone, purrs):.3f}")


if __name__ == "__main__":
    main()

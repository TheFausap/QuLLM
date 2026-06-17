"""Dependency-free primitives for quantum-native token encoding experiments."""

from __future__ import annotations

from dataclasses import dataclass
import cmath
import hashlib
import math
from typing import Iterable, Mapping, Sequence


ComplexVector = tuple[complex, ...]


def norm(state: Sequence[complex]) -> float:
    return math.sqrt(sum((z.real * z.real) + (z.imag * z.imag) for z in state))


def normalize(state: Sequence[complex]) -> ComplexVector:
    scale = norm(state)
    if scale == 0:
        raise ValueError("cannot normalize the zero state")
    return tuple(z / scale for z in state)


def inner(left: Sequence[complex], right: Sequence[complex]) -> complex:
    if len(left) != len(right):
        raise ValueError("states must have the same dimension")
    return sum(a.conjugate() * b for a, b in zip(left, right))


def born_overlap(left: Sequence[complex], right: Sequence[complex]) -> float:
    amplitude = inner(left, right)
    return (amplitude.real * amplitude.real) + (amplitude.imag * amplitude.imag)


def apply_diagonal_unitary(state: Sequence[complex], phases: Sequence[float]) -> ComplexVector:
    if len(state) != len(phases):
        raise ValueError("state and phase operator dimensions must match")
    return tuple(z * cmath.exp(1j * phase) for z, phase in zip(state, phases))


def givens_mix(
    left: Sequence[complex],
    right: Sequence[complex],
    theta: float,
    phi: float = 0.0,
) -> tuple[ComplexVector, ComplexVector]:
    """Mix two same-sized token states with a 2x2 complex unitary rotation."""

    if len(left) != len(right):
        raise ValueError("states must have the same dimension")

    c = math.cos(theta)
    s = math.sin(theta)
    phase = cmath.exp(1j * phi)
    phase_conj = phase.conjugate()

    mixed_left = tuple(c * a + s * phase * b for a, b in zip(left, right))
    mixed_right = tuple(-s * phase_conj * a + c * b for a, b in zip(left, right))
    return mixed_left, mixed_right


def _stable_hash(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _angle(text: str) -> float:
    return (_stable_hash(text) % 1_000_000) / 1_000_000 * math.tau


def hashed_basis_index(feature: str, dimension: int) -> int:
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    return _stable_hash(feature) % dimension


@dataclass(frozen=True)
class QuantumNativeTokenizer:
    """Feature tokenizer that prepares normalized complex token states.

    This is intentionally not BPE. A token becomes a superposition over reusable
    features such as character n-grams, affixes, shape, and optional lexical tags.
    """

    dimension: int = 16
    ngram_min: int = 2
    ngram_max: int = 4

    def features(self, token: str, tags: Iterable[str] = ()) -> tuple[str, ...]:
        cleaned = token.strip()
        lowered = cleaned.lower()
        padded = f"^{lowered}$"

        features: list[str] = []
        features.append(f"word:{lowered}")
        features.append(f"shape:{self._shape(cleaned)}")

        for n in range(self.ngram_min, self.ngram_max + 1):
            if len(padded) >= n:
                features.extend(f"ngram:{padded[i:i+n]}" for i in range(len(padded) - n + 1))

        for length in (2, 3, 4):
            if len(lowered) > length:
                features.append(f"prefix:{lowered[:length]}")
                features.append(f"suffix:{lowered[-length:]}")

        features.extend(f"tag:{tag}" for tag in tags)
        return tuple(features)

    def encode(
        self,
        token: str,
        tags: Iterable[str] = (),
        phase_biases: Mapping[str, float] | None = None,
    ) -> ComplexVector:
        state = [0j] * self.dimension
        biases = phase_biases or {}

        for feature in self.features(token, tags):
            idx = hashed_basis_index(feature, self.dimension)
            amplitude = 1.0 + 0.25 * feature.count(":")
            phase = _angle(feature) + biases.get(feature, 0.0)
            state[idx] += amplitude * cmath.exp(1j * phase)

        return normalize(state)

    @staticmethod
    def _shape(token: str) -> str:
        chars: list[str] = []
        for char in token:
            if char.isupper():
                chars.append("A")
            elif char.islower():
                chars.append("a")
            elif char.isdigit():
                chars.append("0")
            else:
                chars.append(char)
        return "".join(chars)


@dataclass(frozen=True)
class RelationPhaseOperator:
    """A diagonal unitary used to express a linguistic relation as phase shifts."""

    phases: tuple[float, ...]

    @classmethod
    def from_feature_targets(
        cls,
        dimension: int,
        constructive: Iterable[str] = (),
        destructive: Iterable[str] = (),
        angle: float = math.pi,
    ) -> "RelationPhaseOperator":
        phases = [0.0] * dimension
        for feature in constructive:
            phases[hashed_basis_index(feature, dimension)] += 0.0
        for feature in destructive:
            phases[hashed_basis_index(feature, dimension)] += angle
        return cls(tuple(phases))

    def apply(self, state: Sequence[complex]) -> ComplexVector:
        return apply_diagonal_unitary(state, self.phases)

    def compatibility(self, left: Sequence[complex], right: Sequence[complex]) -> float:
        return born_overlap(left, self.apply(right))


def phase_aligned_state(dimension: int, active: Mapping[int, float]) -> ComplexVector:
    """Build a normalized state from explicit basis index -> phase mappings."""

    state = [0j] * dimension
    for idx, phase in active.items():
        if idx < 0 or idx >= dimension:
            raise ValueError(f"basis index {idx} outside dimension {dimension}")
        state[idx] = cmath.exp(1j * phase)
    return normalize(state)

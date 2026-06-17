"""TinyStories word-relation probe.

This is a less controlled follow-up to structured_phase_scaling.py. It streams
TinyStories from Hugging Face, turns natural text into noisy relation triples,
and compares:

    frozen_phase:     fixed quantum-native token states + learned relation phase
    frozen_amplitude: same token amplitudes with phase removed
    token_phase:      feature-initialized states + learned token phase residuals
    token_phase_lowrank: lower-parameter phase residuals
    token_complex:    learned amplitude and phase residuals
    token_complex_role: separate left/right amplitude and phase residuals
    token_complex_signed: signed complex inner-product readout
    token_complex_role_signed: role-specific signed complex readout
    complex_diag:     unconstrained complex bilinear diagnostic baseline
    complex_diag_halfdim: complex baseline with half dimension, roughly real_diag-sized
    real_diag:        learned real token embeddings + relation diagonal
    real_diag_wide:   real baseline with doubled dimension, roughly complex_diag-sized

The task is binary: distinguish true local word pairs from random negatives.
Relations are signed relative-position buckets, such as -3, -2, -1, +1, +2, +3.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re
import time
from typing import Any, Iterable

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("This experiment requires PyTorch. Install requirements.txt first.") from exc

try:
    from datasets import load_dataset
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("This experiment requires Hugging Face datasets. Install requirements.txt first.") from exc


ROOT = Path(__file__).resolve().parents[1]
TOKEN_RE = re.compile(r"[A-Za-z]+|[0-9]+|[.,!?;:]")
RUN_VERSION = "tinystories_pair_probe_v2_token_phase"


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def stable_hash(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def feature_list(token: str) -> list[str]:
    padded = f"^{token}$"
    features = [f"word:{token}", f"shape:{shape(token)}"]
    for n in (2, 3, 4):
        if len(padded) >= n:
            features.extend(f"ngram:{padded[i:i+n]}" for i in range(len(padded) - n + 1))
    for length in (2, 3, 4):
        if len(token) > length:
            features.append(f"prefix:{token[:length]}")
            features.append(f"suffix:{token[-length:]}")
    return features


def shape(token: str) -> str:
    chars: list[str] = []
    for char in token:
        if char.isalpha():
            chars.append("a")
        elif char.isdigit():
            chars.append("0")
        else:
            chars.append(char)
    return "".join(chars)


def angle(text: str) -> float:
    return (stable_hash(text) % 1_000_000) / 1_000_000 * math.tau


def build_vocab(dataset_name: str, split: str, vocab_size: int, vocab_stories: int) -> list[str]:
    counts: Counter[str] = Counter()
    stream = load_dataset(dataset_name, split=split, streaming=True)
    for idx, row in enumerate(stream):
        if idx >= vocab_stories:
            break
        counts.update(tokenize(row["text"]))
    return ["<unk>"] + [token for token, _count in counts.most_common(vocab_size - 1)]


def encode_tokens(tokens: Iterable[str], vocab: dict[str, int]) -> list[int]:
    unk = vocab["<unk>"]
    return [vocab.get(token, unk) for token in tokens]


def relation_index(offset: int, window: int) -> int:
    if offset == 0 or abs(offset) > window:
        raise ValueError(f"invalid offset {offset}")
    return offset + window if offset < 0 else offset + window - 1


def relation_count(window: int) -> int:
    return 2 * window


def sample_negative(unigram: torch.Tensor, forbidden: int) -> int:
    while True:
        candidate = int(torch.multinomial(unigram, 1).item())
        if candidate != forbidden:
            return candidate


def build_examples(
    dataset_name: str,
    split: str,
    vocab: dict[str, int],
    window: int,
    max_train_examples: int,
    test_examples: int,
    seed: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    rng = random.Random(seed)
    counts = torch.ones(len(vocab), dtype=torch.float32)
    train_rows: list[tuple[int, int, int, int]] = []
    test_rows: list[tuple[int, int, int, int]] = []
    target_total = max_train_examples + test_examples
    stream = load_dataset(dataset_name, split=split, streaming=True)

    for row in stream:
        ids = encode_tokens(tokenize(row["text"]), vocab)
        if len(ids) < 2:
            continue
        for token_id in ids:
            counts[token_id] += 1
        unigram = counts / counts.sum()

        for i, left in enumerate(ids):
            offsets = [offset for offset in range(-window, window + 1) if offset != 0 and 0 <= i + offset < len(ids)]
            rng.shuffle(offsets)
            for offset in offsets[:2]:
                right = ids[i + offset]
                rel = relation_index(offset, window)
                rows = test_rows if len(test_rows) < test_examples else train_rows
                if len(test_rows) >= test_examples and len(train_rows) >= max_train_examples:
                    return tensorize(train_rows), tensorize(test_rows)

                rows.append((left, rel, right, 1))
                negative = sample_negative(unigram, right)
                rows.append((left, rel, negative, 0))

                if len(train_rows) + len(test_rows) >= target_total:
                    return tensorize(train_rows[:max_train_examples]), tensorize(test_rows[:test_examples])

    return tensorize(train_rows[:max_train_examples]), tensorize(test_rows[:test_examples])


def cache_metadata(
    dataset_name: str,
    split: str,
    vocab_size: int,
    vocab_stories: int,
    window: int,
    max_train_examples: int,
    test_examples: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "run_version": RUN_VERSION,
        "dataset": dataset_name,
        "split": split,
        "vocab_size": vocab_size,
        "vocab_stories": vocab_stories,
        "window": window,
        "max_train_examples": max_train_examples,
        "test_examples": test_examples,
        "seed": seed,
    }


def load_or_build_cached_data(
    cache_path: str | None,
    dataset_name: str,
    split: str,
    vocab_size: int,
    vocab_stories: int,
    window: int,
    max_train_examples: int,
    test_examples: int,
    seed: int,
) -> tuple[list[str], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    expected = cache_metadata(
        dataset_name=dataset_name,
        split=split,
        vocab_size=vocab_size,
        vocab_stories=vocab_stories,
        window=window,
        max_train_examples=max_train_examples,
        test_examples=test_examples,
        seed=seed,
    )

    if cache_path:
        path = ROOT / cache_path
        if path.exists():
            print(f"loading cached examples from {path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("metadata") != expected:
                print("cache metadata mismatch; rebuilding examples")
            else:
                return payload["vocab"], payload["train"], payload["test"]

    print("building vocabulary...")
    vocab_list = build_vocab(dataset_name, split, vocab_size, vocab_stories)
    vocab = {token: idx for idx, token in enumerate(vocab_list)}
    print(f"vocab_size={len(vocab_list)}")

    print("building examples...")
    train_cpu, test_cpu = build_examples(
        dataset_name=dataset_name,
        split=split,
        vocab=vocab,
        window=window,
        max_train_examples=max_train_examples,
        test_examples=test_examples,
        seed=seed,
    )

    if cache_path:
        path = ROOT / cache_path
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"metadata": expected, "vocab": vocab_list, "train": train_cpu, "test": test_cpu}, path)
        print(f"saved cached examples to {path}")

    return vocab_list, train_cpu, test_cpu


def tensorize(rows: list[tuple[int, int, int, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not rows:
        raise ValueError("no rows generated")
    left, rel, right, label = zip(*rows)
    return (
        torch.tensor(left, dtype=torch.long),
        torch.tensor(rel, dtype=torch.long),
        torch.tensor(right, dtype=torch.long),
        torch.tensor(label, dtype=torch.float32),
    )


def make_feature_states(vocab: list[str], dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    real = torch.zeros(len(vocab), dim)
    imag = torch.zeros(len(vocab), dim)
    amp = torch.zeros(len(vocab), dim)

    for token_id, token in enumerate(vocab):
        for feature in feature_list(token):
            idx = stable_hash(feature) % dim
            magnitude = 1.0 + 0.1 * feature.count(":")
            phase = angle(feature)
            real[token_id, idx] += magnitude * math.cos(phase)
            imag[token_id, idx] += magnitude * math.sin(phase)
            amp[token_id, idx] += magnitude

    complex_scale = torch.sqrt((real * real + imag * imag).sum(dim=-1, keepdim=True).clamp_min(1e-8))
    amp_scale = torch.sqrt((amp * amp).sum(dim=-1, keepdim=True).clamp_min(1e-8))
    return real / complex_scale, imag / complex_scale, amp / amp_scale


def phase_overlap_logits(
    left_real: torch.Tensor,
    left_imag: torch.Tensor,
    right_real: torch.Tensor,
    right_imag: torch.Tensor,
    rel_phase: torch.Tensor,
    logit_scale: torch.Tensor,
    logit_bias: torch.Tensor,
) -> torch.Tensor:
    c = torch.cos(rel_phase)
    s = torch.sin(rel_phase)
    rotated_real = right_real * c - right_imag * s
    rotated_imag = right_real * s + right_imag * c
    inner_real = (left_real * rotated_real + left_imag * rotated_imag).sum(dim=-1)
    inner_imag = (left_real * rotated_imag - left_imag * rotated_real).sum(dim=-1)
    score = inner_real.square() + inner_imag.square()
    return logit_scale * score + logit_bias


def signed_phase_logits(
    left_real: torch.Tensor,
    left_imag: torch.Tensor,
    right_real: torch.Tensor,
    right_imag: torch.Tensor,
    rel_phase: torch.Tensor,
    real_weight: torch.Tensor,
    imag_weight: torch.Tensor,
    logit_scale: torch.Tensor,
    logit_bias: torch.Tensor,
) -> torch.Tensor:
    c = torch.cos(rel_phase)
    s = torch.sin(rel_phase)
    rotated_real = right_real * c - right_imag * s
    rotated_imag = right_real * s + right_imag * c
    inner_real = (left_real * rotated_real + left_imag * rotated_imag).sum(dim=-1)
    inner_imag = (left_real * rotated_imag - left_imag * rotated_real).sum(dim=-1)
    score = real_weight * inner_real + imag_weight * inner_imag
    return logit_scale * score + logit_bias


class FrozenPhaseModel(nn.Module):
    def __init__(self, token_real: torch.Tensor, token_imag: torch.Tensor, relations: int) -> None:
        super().__init__()
        self.register_buffer("token_real", token_real)
        self.register_buffer("token_imag", token_imag)
        self.rel_phase = nn.Embedding(relations, token_real.shape[1])
        self.logit_scale = nn.Parameter(torch.tensor(8.0))
        self.logit_bias = nn.Parameter(torch.tensor(0.0))
        nn.init.zeros_(self.rel_phase.weight)

    def forward(self, left: torch.Tensor, rel: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left_real = self.token_real[left]
        left_imag = self.token_imag[left]
        right_real = self.token_real[right]
        right_imag = self.token_imag[right]
        return phase_overlap_logits(
            left_real,
            left_imag,
            right_real,
            right_imag,
            self.rel_phase(rel),
            self.logit_scale,
            self.logit_bias,
        )


class TokenPhaseModel(nn.Module):
    def __init__(self, token_real: torch.Tensor, token_imag: torch.Tensor, relations: int) -> None:
        super().__init__()
        base_amp = torch.sqrt(token_real.square() + token_imag.square())
        base_phase = torch.atan2(token_imag, token_real)
        self.register_buffer("base_amp", base_amp)
        self.register_buffer("base_phase", base_phase)
        self.token_phase_delta = nn.Embedding(token_real.shape[0], token_real.shape[1])
        self.rel_phase = nn.Embedding(relations, token_real.shape[1])
        self.logit_scale = nn.Parameter(torch.tensor(8.0))
        self.logit_bias = nn.Parameter(torch.tensor(0.0))
        nn.init.zeros_(self.token_phase_delta.weight)
        nn.init.zeros_(self.rel_phase.weight)

    def token_state(self, token: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        amp = self.base_amp[token]
        phase = self.base_phase[token] + self.token_phase_delta(token)
        real = amp * torch.cos(phase)
        imag = amp * torch.sin(phase)
        scale = torch.sqrt((real.square() + imag.square()).sum(dim=-1, keepdim=True).clamp_min(1e-8))
        return real / scale, imag / scale

    def forward(self, left: torch.Tensor, rel: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left_real, left_imag = self.token_state(left)
        right_real, right_imag = self.token_state(right)
        return phase_overlap_logits(
            left_real,
            left_imag,
            right_real,
            right_imag,
            self.rel_phase(rel),
            self.logit_scale,
            self.logit_bias,
        )


class TokenPhaseLowRankModel(nn.Module):
    def __init__(self, token_real: torch.Tensor, token_imag: torch.Tensor, relations: int, rank: int) -> None:
        super().__init__()
        base_amp = torch.sqrt(token_real.square() + token_imag.square())
        base_phase = torch.atan2(token_imag, token_real)
        self.register_buffer("base_amp", base_amp)
        self.register_buffer("base_phase", base_phase)
        self.token_phase_code = nn.Embedding(token_real.shape[0], rank)
        self.phase_projection = nn.Parameter(torch.zeros(rank, token_real.shape[1]))
        self.rel_phase = nn.Embedding(relations, token_real.shape[1])
        self.logit_scale = nn.Parameter(torch.tensor(8.0))
        self.logit_bias = nn.Parameter(torch.tensor(0.0))
        nn.init.normal_(self.token_phase_code.weight, std=0.01)
        nn.init.normal_(self.phase_projection, std=0.01)
        nn.init.zeros_(self.rel_phase.weight)

    def token_state(self, token: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        amp = self.base_amp[token]
        phase_delta = self.token_phase_code(token) @ self.phase_projection
        phase = self.base_phase[token] + phase_delta
        real = amp * torch.cos(phase)
        imag = amp * torch.sin(phase)
        scale = torch.sqrt((real.square() + imag.square()).sum(dim=-1, keepdim=True).clamp_min(1e-8))
        return real / scale, imag / scale

    def forward(self, left: torch.Tensor, rel: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left_real, left_imag = self.token_state(left)
        right_real, right_imag = self.token_state(right)
        return phase_overlap_logits(
            left_real,
            left_imag,
            right_real,
            right_imag,
            self.rel_phase(rel),
            self.logit_scale,
            self.logit_bias,
        )


class TokenComplexModel(nn.Module):
    def __init__(self, token_real: torch.Tensor, token_imag: torch.Tensor, relations: int) -> None:
        super().__init__()
        base_amp = torch.sqrt(token_real.square() + token_imag.square())
        base_phase = torch.atan2(token_imag, token_real)
        self.register_buffer("base_amp_logits", torch.log(torch.expm1((6.0 * base_amp).clamp_min(1e-6))))
        self.register_buffer("base_phase", base_phase)
        self.amp_delta = nn.Embedding(token_real.shape[0], token_real.shape[1])
        self.phase_delta = nn.Embedding(token_real.shape[0], token_real.shape[1])
        self.rel_phase = nn.Embedding(relations, token_real.shape[1])
        self.logit_scale = nn.Parameter(torch.tensor(8.0))
        self.logit_bias = nn.Parameter(torch.tensor(0.0))
        nn.init.zeros_(self.amp_delta.weight)
        nn.init.zeros_(self.phase_delta.weight)
        nn.init.zeros_(self.rel_phase.weight)

    def token_state(self, token: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        amp = F.softplus(self.base_amp_logits[token] + self.amp_delta(token)) + 1e-6
        phase = self.base_phase[token] + self.phase_delta(token)
        real = amp * torch.cos(phase)
        imag = amp * torch.sin(phase)
        scale = torch.sqrt((real.square() + imag.square()).sum(dim=-1, keepdim=True).clamp_min(1e-8))
        return real / scale, imag / scale

    def forward(self, left: torch.Tensor, rel: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left_real, left_imag = self.token_state(left)
        right_real, right_imag = self.token_state(right)
        return phase_overlap_logits(
            left_real,
            left_imag,
            right_real,
            right_imag,
            self.rel_phase(rel),
            self.logit_scale,
            self.logit_bias,
        )


class TokenComplexRoleModel(nn.Module):
    def __init__(self, token_real: torch.Tensor, token_imag: torch.Tensor, relations: int) -> None:
        super().__init__()
        base_amp = torch.sqrt(token_real.square() + token_imag.square())
        base_phase = torch.atan2(token_imag, token_real)
        self.register_buffer("base_amp_logits", torch.log(torch.expm1((6.0 * base_amp).clamp_min(1e-6))))
        self.register_buffer("base_phase", base_phase)
        self.left_amp_delta = nn.Embedding(token_real.shape[0], token_real.shape[1])
        self.left_phase_delta = nn.Embedding(token_real.shape[0], token_real.shape[1])
        self.right_amp_delta = nn.Embedding(token_real.shape[0], token_real.shape[1])
        self.right_phase_delta = nn.Embedding(token_real.shape[0], token_real.shape[1])
        self.rel_phase = nn.Embedding(relations, token_real.shape[1])
        self.logit_scale = nn.Parameter(torch.tensor(8.0))
        self.logit_bias = nn.Parameter(torch.tensor(0.0))
        for table in (
            self.left_amp_delta,
            self.left_phase_delta,
            self.right_amp_delta,
            self.right_phase_delta,
            self.rel_phase,
        ):
            nn.init.zeros_(table.weight)

    def token_state(self, token: torch.Tensor, side: str) -> tuple[torch.Tensor, torch.Tensor]:
        if side == "left":
            amp_delta = self.left_amp_delta(token)
            phase_delta = self.left_phase_delta(token)
        elif side == "right":
            amp_delta = self.right_amp_delta(token)
            phase_delta = self.right_phase_delta(token)
        else:
            raise ValueError(f"unknown side {side}")
        amp = F.softplus(self.base_amp_logits[token] + amp_delta) + 1e-6
        phase = self.base_phase[token] + phase_delta
        real = amp * torch.cos(phase)
        imag = amp * torch.sin(phase)
        scale = torch.sqrt((real.square() + imag.square()).sum(dim=-1, keepdim=True).clamp_min(1e-8))
        return real / scale, imag / scale

    def forward(self, left: torch.Tensor, rel: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left_real, left_imag = self.token_state(left, "left")
        right_real, right_imag = self.token_state(right, "right")
        return phase_overlap_logits(
            left_real,
            left_imag,
            right_real,
            right_imag,
            self.rel_phase(rel),
            self.logit_scale,
            self.logit_bias,
        )


class TokenComplexSignedModel(TokenComplexModel):
    def __init__(self, token_real: torch.Tensor, token_imag: torch.Tensor, relations: int) -> None:
        super().__init__(token_real, token_imag, relations)
        self.real_weight = nn.Parameter(torch.tensor(1.0))
        self.imag_weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, left: torch.Tensor, rel: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left_real, left_imag = self.token_state(left)
        right_real, right_imag = self.token_state(right)
        return signed_phase_logits(
            left_real,
            left_imag,
            right_real,
            right_imag,
            self.rel_phase(rel),
            self.real_weight,
            self.imag_weight,
            self.logit_scale,
            self.logit_bias,
        )


class TokenComplexRoleSignedModel(TokenComplexRoleModel):
    def __init__(self, token_real: torch.Tensor, token_imag: torch.Tensor, relations: int) -> None:
        super().__init__(token_real, token_imag, relations)
        self.real_weight = nn.Parameter(torch.tensor(1.0))
        self.imag_weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, left: torch.Tensor, rel: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left_real, left_imag = self.token_state(left, "left")
        right_real, right_imag = self.token_state(right, "right")
        return signed_phase_logits(
            left_real,
            left_imag,
            right_real,
            right_imag,
            self.rel_phase(rel),
            self.real_weight,
            self.imag_weight,
            self.logit_scale,
            self.logit_bias,
        )


class FrozenAmplitudeModel(nn.Module):
    def __init__(self, token_amp: torch.Tensor, relations: int) -> None:
        super().__init__()
        del relations
        self.register_buffer("token_amp", token_amp)
        self.logit_scale = nn.Parameter(torch.tensor(8.0))
        self.logit_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, left: torch.Tensor, rel: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        del rel
        score = (self.token_amp[left] * self.token_amp[right]).sum(dim=-1).square()
        return self.logit_scale * score + self.logit_bias


class RealDiagModel(nn.Module):
    def __init__(self, vocab_size: int, dim: int, relations: int) -> None:
        super().__init__()
        self.left = nn.Embedding(vocab_size, dim)
        self.right = nn.Embedding(vocab_size, dim)
        self.rel = nn.Embedding(relations, dim)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.logit_bias = nn.Parameter(torch.tensor(0.0))
        nn.init.normal_(self.left.weight, std=dim**-0.5)
        nn.init.normal_(self.right.weight, std=dim**-0.5)
        nn.init.normal_(self.rel.weight, std=dim**-0.5)

    def forward(self, left: torch.Tensor, rel: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        score = (self.left(left) * self.rel(rel) * self.right(right)).sum(dim=-1)
        return self.logit_scale * score + self.logit_bias


class ComplexDiagModel(nn.Module):
    def __init__(self, vocab_size: int, dim: int, relations: int) -> None:
        super().__init__()
        self.left_real = nn.Embedding(vocab_size, dim)
        self.left_imag = nn.Embedding(vocab_size, dim)
        self.right_real = nn.Embedding(vocab_size, dim)
        self.right_imag = nn.Embedding(vocab_size, dim)
        self.rel_real = nn.Embedding(relations, dim)
        self.rel_imag = nn.Embedding(relations, dim)
        self.real_weight = nn.Parameter(torch.tensor(1.0))
        self.imag_weight = nn.Parameter(torch.tensor(0.0))
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.logit_bias = nn.Parameter(torch.tensor(0.0))
        for table in (
            self.left_real,
            self.left_imag,
            self.right_real,
            self.right_imag,
            self.rel_real,
            self.rel_imag,
        ):
            nn.init.normal_(table.weight, std=(2 * dim) ** -0.5)

    def forward(self, left: torch.Tensor, rel: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left_real = self.left_real(left)
        left_imag = self.left_imag(left)
        right_real = self.right_real(right)
        right_imag = self.right_imag(right)
        rel_real = self.rel_real(rel)
        rel_imag = self.rel_imag(rel)

        rotated_real = rel_real * right_real - rel_imag * right_imag
        rotated_imag = rel_real * right_imag + rel_imag * right_real
        inner_real = (left_real * rotated_real + left_imag * rotated_imag).sum(dim=-1)
        inner_imag = (left_real * rotated_imag - left_imag * rotated_real).sum(dim=-1)
        score = self.real_weight * inner_real + self.imag_weight * inner_imag
        return self.logit_scale * score + self.logit_bias


class RealDiagWideModel(RealDiagModel):
    def __init__(self, vocab_size: int, dim: int, relations: int) -> None:
        super().__init__(vocab_size, dim * 2, relations)


class ComplexDiagHalfDimModel(ComplexDiagModel):
    def __init__(self, vocab_size: int, dim: int, relations: int) -> None:
        super().__init__(vocab_size, max(1, dim // 2), relations)


def move_dataset(
    dataset: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(tensor.to(device) for tensor in dataset)  # type: ignore[return-value]


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    left, rel, right, labels = dataset
    total = labels.numel()
    loss_sum = 0.0
    correct = 0
    for start in range(0, total, batch_size):
        batch = slice(start, start + batch_size)
        logits = model(left[batch], rel[batch], right[batch])
        loss_sum += float(F.binary_cross_entropy_with_logits(logits, labels[batch], reduction="sum").item())
        correct += ((torch.sigmoid(logits) >= 0.5) == labels[batch].bool()).sum().item()
    return {"loss": loss_sum / total, "accuracy": correct / total}


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def train_one(
    model_name: str,
    train_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    test_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    token_real: torch.Tensor,
    token_imag: torch.Tensor,
    token_amp: torch.Tensor,
    relations: int,
    dim: int,
    phase_rank: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> dict[str, Any]:
    if model_name == "frozen_phase":
        model = FrozenPhaseModel(token_real, token_imag, relations)
    elif model_name == "frozen_amplitude":
        model = FrozenAmplitudeModel(token_amp, relations)
    elif model_name == "token_phase":
        model = TokenPhaseModel(token_real, token_imag, relations)
    elif model_name == "token_phase_lowrank":
        model = TokenPhaseLowRankModel(token_real, token_imag, relations, phase_rank)
    elif model_name == "token_complex":
        model = TokenComplexModel(token_real, token_imag, relations)
    elif model_name == "token_complex_role":
        model = TokenComplexRoleModel(token_real, token_imag, relations)
    elif model_name == "token_complex_signed":
        model = TokenComplexSignedModel(token_real, token_imag, relations)
    elif model_name == "token_complex_role_signed":
        model = TokenComplexRoleSignedModel(token_real, token_imag, relations)
    elif model_name == "complex_diag":
        model = ComplexDiagModel(token_real.shape[0], dim, relations)
    elif model_name == "complex_diag_halfdim":
        model = ComplexDiagHalfDimModel(token_real.shape[0], dim, relations)
    elif model_name == "real_diag":
        model = RealDiagModel(token_real.shape[0], dim, relations)
    elif model_name == "real_diag_wide":
        model = RealDiagWideModel(token_real.shape[0], dim, relations)
    else:
        raise ValueError(f"unknown model {model_name}")

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    left, rel, right, labels = train_data
    count = labels.numel()
    started = time.time()

    for _epoch in range(epochs):
        model.train()
        order = torch.randperm(count, device=device)
        for start in range(0, count, batch_size):
            idx = order[start : start + batch_size]
            logits = model(left[idx], rel[idx], right[idx])
            loss = F.binary_cross_entropy_with_logits(logits, labels[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    metrics = evaluate(model, test_data, batch_size)
    metrics.update(
        {
            "model": model_name,
            "train_size": count,
            "trainable_params": trainable_parameter_count(model),
            "seconds": round(time.time() - started, 3),
        }
    )
    return metrics


def parse_ints(value: str | list[int]) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def write_results(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["train_size", "model", "accuracy", "loss", "seconds", "trainable_params"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--vocab-stories", type=int, default=50000)
    parser.add_argument("--max-train-examples", type=int, default=1_000_000)
    parser.add_argument("--train-sizes", type=str, default="50000,200000,1000000")
    parser.add_argument("--test-examples", type=int, default=100000)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--phase-rank", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--cache", type=str, default=None)
    parser.add_argument(
        "--models",
        type=str,
        default=(
            "frozen_phase,frozen_amplitude,token_phase,token_phase_lowrank,"
            "token_complex,token_complex_role,token_complex_signed,"
            "token_complex_role_signed,complex_diag_halfdim,complex_diag,"
            "real_diag,real_diag_wide"
        ),
    )
    parser.add_argument("--out", type=str, default="runs/tinystories_pair_probe.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_name = str(config.get("dataset", args.dataset))
    split = str(config.get("split", args.split))
    vocab_size = int(config.get("vocab_size", args.vocab_size))
    vocab_stories = int(config.get("vocab_stories", args.vocab_stories))
    max_train_examples = int(config.get("max_train_examples", args.max_train_examples))
    train_sizes = parse_ints(config.get("train_sizes", args.train_sizes))
    test_examples = int(config.get("test_examples", args.test_examples))
    window = int(config.get("window", args.window))
    dim = int(config.get("dim", args.dim))
    phase_rank = int(config.get("phase_rank", args.phase_rank))
    epochs = int(config.get("epochs", args.epochs))
    batch_size = int(config.get("batch_size", args.batch_size))
    learning_rate = float(config.get("learning_rate", args.learning_rate))
    seed = int(config.get("seed", args.seed))
    cache_path = config.get("cache", args.cache)
    models_raw = config.get("models", args.models)
    models = models_raw if isinstance(models_raw, list) else [item.strip() for item in models_raw.split(",")]
    known_models = {
        "frozen_phase",
        "frozen_amplitude",
        "token_phase",
        "token_phase_lowrank",
        "token_complex",
        "token_complex_role",
        "token_complex_signed",
        "token_complex_role_signed",
        "complex_diag_halfdim",
        "complex_diag",
        "real_diag",
        "real_diag_wide",
    }
    unknown_models = sorted(set(models) - known_models)
    if unknown_models:
        raise SystemExit(f"unknown model(s): {', '.join(unknown_models)}")
    device = choose_device(args.device)
    set_seed(seed)

    print(f"run_version={RUN_VERSION}")
    print(f"config={args.config or '<cli/defaults>'}")
    print(f"dataset={dataset_name} split={split} device={device}")
    print(f"models={','.join(models)}")
    print(f"dim={dim} phase_rank={phase_rank} window={window} epochs={epochs} batch_size={batch_size}")
    vocab_list, train_cpu, test_cpu = load_or_build_cached_data(
        cache_path=cache_path,
        dataset_name=dataset_name,
        split=split,
        vocab_size=vocab_size,
        vocab_stories=vocab_stories,
        window=window,
        max_train_examples=max_train_examples,
        test_examples=test_examples,
        seed=seed,
    )
    print(f"train_examples={train_cpu[3].numel()} test_examples={test_cpu[3].numel()} relations={relation_count(window)}")

    print("preparing frozen token states...")
    token_real_cpu, token_imag_cpu, token_amp_cpu = make_feature_states(vocab_list, dim)
    token_real = token_real_cpu.to(device)
    token_imag = token_imag_cpu.to(device)
    token_amp = token_amp_cpu.to(device)
    test_data = move_dataset(test_cpu, device)

    rows: list[dict[str, Any]] = []
    for train_size in train_sizes:
        train_data = tuple(tensor[:train_size] for tensor in train_cpu)
        train_data = move_dataset(train_data, device)
        print(f"\ntrain_size={train_size}")
        for model_name in models:
            set_seed(seed)
            row = train_one(
                model_name=model_name,
                train_data=train_data,
                test_data=test_data,
                token_real=token_real,
                token_imag=token_imag,
                token_amp=token_amp,
                relations=relation_count(window),
                dim=dim,
                phase_rank=phase_rank,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                device=device,
            )
            rows.append(row)
            print(
                f"  {model_name:16s} "
                f"acc={row['accuracy']:.4f} loss={row['loss']:.4f} "
                f"params={row['trainable_params']} seconds={row['seconds']}"
            )

    out = ROOT / args.out
    write_results(out, rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

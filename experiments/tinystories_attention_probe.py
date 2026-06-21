"""TinyStories next-token attention probe.

This experiment moves complex numbers into the interaction component. Given a
context window and a candidate next token, models decide whether the candidate
is the real next token or a unigram negative.

The comparison is intentionally local:

    real_attention:            real candidate-to-context attention
    complex_attention:         complex candidate-to-context attention
    complex_attention_halfdim: complex model with half dimension
    real_attention_wide:       real model with doubled dimension
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import random
import time
from typing import Any

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

from tinystories_pair_probe import encode_tokens, tokenize


ROOT = Path(__file__).resolve().parents[1]
RUN_VERSION = "tinystories_attention_probe_v1"


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


def build_vocab(dataset_name: str, split: str, vocab_size: int, vocab_stories: int) -> list[str]:
    counts: Counter[str] = Counter()
    stream = load_dataset(dataset_name, split=split, streaming=True)
    for idx, row in enumerate(stream):
        if idx >= vocab_stories:
            break
        counts.update(tokenize(row["text"]))
    return ["<unk>"] + [token for token, _count in counts.most_common(vocab_size - 1)]


def sample_negative(unigram: torch.Tensor, forbidden: int) -> int:
    while True:
        candidate = int(torch.multinomial(unigram, 1).item())
        if candidate != forbidden:
            return candidate


def tensorize(
    rows: list[tuple[list[int], int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not rows:
        raise ValueError("no examples generated")
    contexts, candidates, labels = zip(*rows)
    return (
        torch.tensor(contexts, dtype=torch.long),
        torch.tensor(candidates, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float32),
    )


def build_examples(
    dataset_name: str,
    split: str,
    vocab: dict[str, int],
    context: int,
    max_train_examples: int,
    test_examples: int,
    seed: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    del seed
    counts = torch.ones(len(vocab), dtype=torch.float32)
    train_rows: list[tuple[list[int], int, int]] = []
    test_rows: list[tuple[list[int], int, int]] = []
    target_total = max_train_examples + test_examples
    stream = load_dataset(dataset_name, split=split, streaming=True)

    for row in stream:
        ids = encode_tokens(tokenize(row["text"]), vocab)
        if len(ids) <= context:
            continue

        for token_id in ids:
            counts[token_id] += 1
        unigram = counts / counts.sum()

        for pos in range(context, len(ids)):
            ctx = ids[pos - context : pos]
            target = ids[pos]
            rows = test_rows if len(test_rows) < test_examples else train_rows
            if len(test_rows) >= test_examples and len(train_rows) >= max_train_examples:
                return tensorize(train_rows), tensorize(test_rows)

            rows.append((ctx, target, 1))
            rows.append((ctx, sample_negative(unigram, target), 0))

            if len(train_rows) + len(test_rows) >= target_total:
                return tensorize(train_rows[:max_train_examples]), tensorize(test_rows[:test_examples])

    return tensorize(train_rows[:max_train_examples]), tensorize(test_rows[:test_examples])


def cache_metadata(
    dataset_name: str,
    split: str,
    vocab_size: int,
    vocab_stories: int,
    context: int,
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
        "context": context,
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
    context: int,
    max_train_examples: int,
    test_examples: int,
    seed: int,
) -> tuple[list[str], tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    expected = cache_metadata(
        dataset_name,
        split,
        vocab_size,
        vocab_stories,
        context,
        max_train_examples,
        test_examples,
        seed,
    )
    if cache_path:
        path = ROOT / cache_path
        if path.exists():
            print(f"loading cached examples from {path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("metadata") == expected:
                return payload["vocab"], payload["train"], payload["test"]
            print("cache metadata mismatch; rebuilding examples")

    print("building vocabulary...")
    vocab_list = build_vocab(dataset_name, split, vocab_size, vocab_stories)
    vocab = {token: idx for idx, token in enumerate(vocab_list)}
    print(f"vocab_size={len(vocab_list)}")

    print("building examples...")
    train_cpu, test_cpu = build_examples(
        dataset_name=dataset_name,
        split=split,
        vocab=vocab,
        context=context,
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


def move_dataset(
    dataset: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(tensor.to(device) for tensor in dataset)  # type: ignore[return-value]


class RealAttentionProbe(nn.Module):
    def __init__(self, vocab_size: int, dim: int, context: int, layers: int = 1) -> None:
        del layers
        super().__init__()
        self.token = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(context, dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.logit_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, context_ids: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(context_ids.shape[1], device=context_ids.device)
        ctx = self.token(context_ids) + self.pos(positions)
        cand = self.token(candidate)
        q = self.q(cand).unsqueeze(1)
        k = self.k(ctx)
        v = self.v(ctx)
        attn = torch.softmax((q * k).sum(dim=-1) / math.sqrt(k.shape[-1]), dim=-1)
        pooled = (attn.unsqueeze(-1) * v).sum(dim=1)
        score = (self.out(pooled) * cand).sum(dim=-1) / math.sqrt(k.shape[-1])
        return self.logit_scale * score + self.logit_bias


class ComplexAttentionProbe(nn.Module):
    def __init__(self, vocab_size: int, dim: int, context: int, layers: int = 1) -> None:
        del layers
        super().__init__()
        self.real = nn.Embedding(vocab_size, dim)
        self.imag = nn.Embedding(vocab_size, dim)
        self.pos_phase = nn.Embedding(context, dim)
        self.q_phase = nn.Parameter(torch.zeros(dim))
        self.k_phase = nn.Parameter(torch.zeros(dim))
        self.v_phase = nn.Parameter(torch.zeros(dim))
        self.out_phase = nn.Parameter(torch.zeros(dim))
        self.real_weight = nn.Parameter(torch.tensor(1.0))
        self.imag_weight = nn.Parameter(torch.tensor(0.0))
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.logit_bias = nn.Parameter(torch.tensor(0.0))
        nn.init.normal_(self.real.weight, std=(2 * dim) ** -0.5)
        nn.init.normal_(self.imag.weight, std=(2 * dim) ** -0.5)
        nn.init.zeros_(self.pos_phase.weight)

    @staticmethod
    def rotate(real: torch.Tensor, imag: torch.Tensor, phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = torch.cos(phase)
        s = torch.sin(phase)
        return real * c - imag * s, real * s + imag * c

    @staticmethod
    def normalize(real: torch.Tensor, imag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = torch.sqrt((real.square() + imag.square()).sum(dim=-1, keepdim=True).clamp_min(1e-8))
        return real / scale, imag / scale

    def forward(self, context_ids: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(context_ids.shape[1], device=context_ids.device)
        ctx_real = self.real(context_ids)
        ctx_imag = self.imag(context_ids)
        ctx_real, ctx_imag = self.rotate(ctx_real, ctx_imag, self.pos_phase(positions))
        cand_real = self.real(candidate)
        cand_imag = self.imag(candidate)

        q_real, q_imag = self.rotate(cand_real, cand_imag, self.q_phase)
        k_real, k_imag = self.rotate(ctx_real, ctx_imag, self.k_phase)
        v_real, v_imag = self.rotate(ctx_real, ctx_imag, self.v_phase)
        q_real, q_imag = self.normalize(q_real, q_imag)
        k_real, k_imag = self.normalize(k_real, k_imag)

        compat = (q_real.unsqueeze(1) * k_real + q_imag.unsqueeze(1) * k_imag).sum(dim=-1)
        attn = torch.softmax(compat * math.sqrt(k_real.shape[-1]), dim=-1)
        pooled_real = (attn.unsqueeze(-1) * v_real).sum(dim=1)
        pooled_imag = (attn.unsqueeze(-1) * v_imag).sum(dim=1)
        pooled_real, pooled_imag = self.rotate(pooled_real, pooled_imag, self.out_phase)

        inner_real = (pooled_real * cand_real + pooled_imag * cand_imag).sum(dim=-1)
        inner_imag = (pooled_real * cand_imag - pooled_imag * cand_real).sum(dim=-1)
        score = (self.real_weight * inner_real + self.imag_weight * inner_imag) / math.sqrt(k_real.shape[-1])
        return self.logit_scale * score + self.logit_bias


class ComplexAttentionNoPhaseProbe(ComplexAttentionProbe):
    """Complex attention ablation with trainable phase rotations disabled."""

    def __init__(self, vocab_size: int, dim: int, context: int, layers: int = 1) -> None:
        super().__init__(vocab_size, dim, context, layers)
        self.pos_phase.weight.requires_grad_(False)
        self.q_phase.requires_grad_(False)
        self.k_phase.requires_grad_(False)
        self.v_phase.requires_grad_(False)
        self.out_phase.requires_grad_(False)


class ComplexAttentionBornProbe(ComplexAttentionProbe):
    """Complex attention that uses Born-style squared compatibility/readout."""

    def forward(self, context_ids: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(context_ids.shape[1], device=context_ids.device)
        ctx_real = self.real(context_ids)
        ctx_imag = self.imag(context_ids)
        ctx_real, ctx_imag = self.rotate(ctx_real, ctx_imag, self.pos_phase(positions))
        cand_real = self.real(candidate)
        cand_imag = self.imag(candidate)

        q_real, q_imag = self.rotate(cand_real, cand_imag, self.q_phase)
        k_real, k_imag = self.rotate(ctx_real, ctx_imag, self.k_phase)
        v_real, v_imag = self.rotate(ctx_real, ctx_imag, self.v_phase)
        q_real, q_imag = self.normalize(q_real, q_imag)
        k_real, k_imag = self.normalize(k_real, k_imag)

        compat_real = (q_real.unsqueeze(1) * k_real + q_imag.unsqueeze(1) * k_imag).sum(dim=-1)
        compat_imag = (q_real.unsqueeze(1) * k_imag - q_imag.unsqueeze(1) * k_real).sum(dim=-1)
        attn = torch.softmax((compat_real.square() + compat_imag.square()) * math.sqrt(k_real.shape[-1]), dim=-1)
        pooled_real = (attn.unsqueeze(-1) * v_real).sum(dim=1)
        pooled_imag = (attn.unsqueeze(-1) * v_imag).sum(dim=1)
        pooled_real, pooled_imag = self.rotate(pooled_real, pooled_imag, self.out_phase)

        inner_real = (pooled_real * cand_real + pooled_imag * cand_imag).sum(dim=-1)
        inner_imag = (pooled_real * cand_imag - pooled_imag * cand_real).sum(dim=-1)
        score = (inner_real.square() + inner_imag.square()) / math.sqrt(k_real.shape[-1])
        return self.logit_scale * score + self.logit_bias


class RealAttentionWideProbe(RealAttentionProbe):
    def __init__(self, vocab_size: int, dim: int, context: int, layers: int = 1) -> None:
        super().__init__(vocab_size, dim * 2, context, layers)


class ComplexAttentionHalfDimProbe(ComplexAttentionProbe):
    def __init__(self, vocab_size: int, dim: int, context: int, layers: int = 1) -> None:
        super().__init__(vocab_size, max(1, dim // 2), context, layers)


class RealAttentionLayer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, candidate_state: torch.Tensor, context_state: torch.Tensor) -> torch.Tensor:
        q = self.q(candidate_state).unsqueeze(1)
        k = self.k(context_state)
        v = self.v(context_state)
        attn = torch.softmax((q * k).sum(dim=-1) / math.sqrt(k.shape[-1]), dim=-1)
        pooled = (attn.unsqueeze(-1) * v).sum(dim=1)
        return self.norm(candidate_state + torch.sigmoid(self.gate) * self.out(pooled))


class RealAttentionStackedProbe(nn.Module):
    def __init__(self, vocab_size: int, dim: int, context: int, layers: int = 2) -> None:
        super().__init__()
        self.token = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(context, dim)
        self.layers = nn.ModuleList(RealAttentionLayer(dim) for _ in range(layers))
        self.readout = nn.Linear(dim, dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.logit_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, context_ids: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(context_ids.shape[1], device=context_ids.device)
        context_state = self.token(context_ids) + self.pos(positions)
        candidate_embed = self.token(candidate)
        candidate_state = candidate_embed
        for layer in self.layers:
            candidate_state = layer(candidate_state, context_state)
        score = (self.readout(candidate_state) * candidate_embed).sum(dim=-1) / math.sqrt(candidate_state.shape[-1])
        return self.logit_scale * score + self.logit_bias


class ComplexAttentionLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        phase_init: float = 0.0,
        min_mix: float = 0.0,
        force_zero_phase: bool = False,
        freeze_phase: bool = False,
        mix_init: float = -2.0,
    ) -> None:
        super().__init__()
        self.q_phase = nn.Parameter(torch.zeros(dim))
        self.k_phase = nn.Parameter(torch.zeros(dim))
        self.v_phase = nn.Parameter(torch.zeros(dim))
        self.out_phase = nn.Parameter(torch.zeros(dim))
        self.mix = nn.Parameter(torch.tensor(mix_init))
        self.min_mix = min_mix
        self.force_zero_phase = force_zero_phase
        if phase_init > 0:
            for phase in (self.q_phase, self.k_phase, self.v_phase, self.out_phase):
                nn.init.normal_(phase, std=phase_init)
        if freeze_phase:
            for phase in (self.q_phase, self.k_phase, self.v_phase, self.out_phase):
                phase.requires_grad_(False)

    @staticmethod
    def rotate(real: torch.Tensor, imag: torch.Tensor, phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = torch.cos(phase)
        s = torch.sin(phase)
        return real * c - imag * s, real * s + imag * c

    @staticmethod
    def normalize(real: torch.Tensor, imag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = torch.sqrt((real.square() + imag.square()).sum(dim=-1, keepdim=True).clamp_min(1e-8))
        return real / scale, imag / scale

    def forward(
        self,
        candidate_real: torch.Tensor,
        candidate_imag: torch.Tensor,
        context_real: torch.Tensor,
        context_imag: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.force_zero_phase:
            q_real, q_imag = candidate_real, candidate_imag
            k_real, k_imag = context_real, context_imag
            v_real, v_imag = context_real, context_imag
        else:
            q_real, q_imag = self.rotate(candidate_real, candidate_imag, self.q_phase)
            k_real, k_imag = self.rotate(context_real, context_imag, self.k_phase)
            v_real, v_imag = self.rotate(context_real, context_imag, self.v_phase)
        q_real, q_imag = self.normalize(q_real, q_imag)
        k_real, k_imag = self.normalize(k_real, k_imag)
        compat = (q_real.unsqueeze(1) * k_real + q_imag.unsqueeze(1) * k_imag).sum(dim=-1)
        attn = torch.softmax(compat * math.sqrt(k_real.shape[-1]), dim=-1)
        pooled_real = (attn.unsqueeze(-1) * v_real).sum(dim=1)
        pooled_imag = (attn.unsqueeze(-1) * v_imag).sum(dim=1)
        if not self.force_zero_phase:
            pooled_real, pooled_imag = self.rotate(pooled_real, pooled_imag, self.out_phase)
        mix = self.min_mix + (1.0 - self.min_mix) * torch.sigmoid(self.mix)
        next_real = candidate_real + mix * pooled_real
        next_imag = candidate_imag + mix * pooled_imag
        return self.normalize(next_real, next_imag)


class ComplexAttentionStackedProbe(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        context: int,
        layers: int = 2,
        phase_init: float = 0.0,
        min_mix: float = 0.0,
        force_zero_phase: bool = False,
        freeze_phase: bool = False,
        mix_init: float = -2.0,
    ) -> None:
        super().__init__()
        self.real = nn.Embedding(vocab_size, dim)
        self.imag = nn.Embedding(vocab_size, dim)
        self.pos_phase = nn.Embedding(context, dim)
        self.force_zero_phase = force_zero_phase
        self.phase_trainable = not freeze_phase and not force_zero_phase
        self.layers = nn.ModuleList(
            ComplexAttentionLayer(
                dim,
                phase_init=phase_init,
                min_mix=min_mix,
                force_zero_phase=force_zero_phase,
                freeze_phase=freeze_phase or force_zero_phase,
                mix_init=mix_init,
            )
            for _ in range(layers)
        )
        self.readout_phase = nn.Parameter(torch.zeros(dim))
        self.real_weight = nn.Parameter(torch.tensor(1.0))
        self.imag_weight = nn.Parameter(torch.tensor(0.0))
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.logit_bias = nn.Parameter(torch.tensor(0.0))
        nn.init.normal_(self.real.weight, std=(2 * dim) ** -0.5)
        nn.init.normal_(self.imag.weight, std=(2 * dim) ** -0.5)
        nn.init.zeros_(self.pos_phase.weight)
        if freeze_phase or force_zero_phase:
            self.pos_phase.weight.requires_grad_(False)
            self.readout_phase.requires_grad_(False)

    @staticmethod
    def rotate(real: torch.Tensor, imag: torch.Tensor, phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = torch.cos(phase)
        s = torch.sin(phase)
        return real * c - imag * s, real * s + imag * c

    @staticmethod
    def normalize(real: torch.Tensor, imag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = torch.sqrt((real.square() + imag.square()).sum(dim=-1, keepdim=True).clamp_min(1e-8))
        return real / scale, imag / scale

    def forward(self, context_ids: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(context_ids.shape[1], device=context_ids.device)
        context_real = self.real(context_ids)
        context_imag = self.imag(context_ids)
        if not self.force_zero_phase:
            context_real, context_imag = self.rotate(context_real, context_imag, self.pos_phase(positions))
        candidate_base_real = self.real(candidate)
        candidate_base_imag = self.imag(candidate)
        candidate_real, candidate_imag = self.normalize(candidate_base_real, candidate_base_imag)
        for layer in self.layers:
            candidate_real, candidate_imag = layer(candidate_real, candidate_imag, context_real, context_imag)
        if not self.force_zero_phase:
            candidate_real, candidate_imag = self.rotate(candidate_real, candidate_imag, self.readout_phase)
        inner_real = (candidate_real * candidate_base_real + candidate_imag * candidate_base_imag).sum(dim=-1)
        inner_imag = (candidate_real * candidate_base_imag - candidate_imag * candidate_base_real).sum(dim=-1)
        score = (self.real_weight * inner_real + self.imag_weight * inner_imag) / math.sqrt(candidate_real.shape[-1])
        return self.logit_scale * score + self.logit_bias


class ComplexAttentionStackedFloorProbe(ComplexAttentionStackedProbe):
    def __init__(self, vocab_size: int, dim: int, context: int, layers: int = 2) -> None:
        super().__init__(vocab_size, dim, context, layers, phase_init=0.05, min_mix=0.2)


class ComplexAttentionStackedFloorDecohereProbe(ComplexAttentionStackedProbe):
    def __init__(self, vocab_size: int, dim: int, context: int, layers: int = 2) -> None:
        super().__init__(
            vocab_size,
            dim,
            context,
            layers,
            phase_init=0.05,
            min_mix=0.2,
            force_zero_phase=True,
        )


class ComplexAttentionStackedFloorDecohereFreeMixProbe(ComplexAttentionStackedProbe):
    def __init__(self, vocab_size: int, dim: int, context: int, layers: int = 2) -> None:
        super().__init__(
            vocab_size,
            dim,
            context,
            layers,
            phase_init=0.0,
            min_mix=0.0,
            force_zero_phase=True,
        )


class ComplexAttentionStackedFloorFrozenPhaseProbe(ComplexAttentionStackedProbe):
    def __init__(self, vocab_size: int, dim: int, context: int, layers: int = 2) -> None:
        super().__init__(
            vocab_size,
            dim,
            context,
            layers,
            phase_init=0.05,
            min_mix=0.2,
            freeze_phase=True,
        )


class ComplexAttentionStackedScheduledProbe(ComplexAttentionStackedProbe):
    def __init__(self, vocab_size: int, dim: int, context: int, layers: int = 2) -> None:
        nn.Module.__init__(self)
        self.real = nn.Embedding(vocab_size, dim)
        self.imag = nn.Embedding(vocab_size, dim)
        self.pos_phase = nn.Embedding(context, dim)
        self.force_zero_phase = False
        self.phase_trainable = True
        scheduled_layers: list[ComplexAttentionLayer] = []
        denom = max(1, layers - 1)
        for idx in range(layers):
            depth = idx / denom
            phase_init = 0.015 + 0.085 * (depth * depth)
            min_mix = 0.05 + 0.20 * (depth * depth)
            scheduled_layers.append(ComplexAttentionLayer(dim, phase_init=phase_init, min_mix=min_mix))
        self.layers = nn.ModuleList(scheduled_layers)
        self.readout_phase = nn.Parameter(torch.zeros(dim))
        self.real_weight = nn.Parameter(torch.tensor(1.0))
        self.imag_weight = nn.Parameter(torch.tensor(0.0))
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.logit_bias = nn.Parameter(torch.tensor(0.0))
        nn.init.normal_(self.real.weight, std=(2 * dim) ** -0.5)
        nn.init.normal_(self.imag.weight, std=(2 * dim) ** -0.5)
        nn.init.zeros_(self.pos_phase.weight)


MODEL_TYPES = {
    "real_attention": RealAttentionProbe,
    "complex_attention": ComplexAttentionProbe,
    "complex_attention_nophase": ComplexAttentionNoPhaseProbe,
    "complex_attention_born": ComplexAttentionBornProbe,
    "real_attention_wide": RealAttentionWideProbe,
    "complex_attention_halfdim": ComplexAttentionHalfDimProbe,
    "real_attention_stacked": RealAttentionStackedProbe,
    "complex_attention_stacked": ComplexAttentionStackedProbe,
    "complex_attention_stacked_floor": ComplexAttentionStackedFloorProbe,
    "complex_attention_stacked_floor_decohere": ComplexAttentionStackedFloorDecohereProbe,
    "complex_attention_stacked_floor_decohere_free_mix": ComplexAttentionStackedFloorDecohereFreeMixProbe,
    "complex_attention_stacked_floor_frozen_phase": ComplexAttentionStackedFloorFrozenPhaseProbe,
    "complex_attention_stacked_scheduled": ComplexAttentionStackedScheduledProbe,
}

import experiments.coherence_isolation  # noqa: F401  (registers _pos variants)

@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    contexts, candidates, labels = dataset
    total = labels.numel()
    loss_sum = 0.0
    correct = 0
    for start in range(0, total, batch_size):
        batch = slice(start, start + batch_size)
        logits = model(contexts[batch], candidates[batch])
        loss_sum += float(F.binary_cross_entropy_with_logits(logits, labels[batch], reduction="sum").item())
        correct += ((torch.sigmoid(logits) >= 0.5) == labels[batch].bool()).sum().item()
    return {"loss": loss_sum / total, "accuracy": correct / total}


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _format_trace(values: list[float]) -> str:
    return ";".join(f"{value:.3f}" for value in values)


@torch.no_grad()
def model_diagnostics(model: nn.Module) -> dict[str, float | str]:
    diagnostics: dict[str, float | str] = {}
    if isinstance(model, ComplexAttentionStackedProbe):
        mix_values = torch.stack([torch.sigmoid(layer.mix.detach()) for layer in model.layers])
        effective_mix_values = torch.stack(
            [
                layer.min_mix + (1.0 - layer.min_mix) * torch.sigmoid(layer.mix.detach())
                for layer in model.layers
            ]
        )
        phase_layer_means: list[float] = []
        phase_layer_maxes: list[float] = []
        phase_tensors = []
        for layer in model.layers:
            layer_phases = torch.cat(
                [
                    layer.q_phase.detach().flatten(),
                    layer.k_phase.detach().flatten(),
                    layer.v_phase.detach().flatten(),
                    layer.out_phase.detach().flatten(),
                ]
            )
            phase_layer_means.append(float(layer_phases.abs().mean().item()))
            phase_layer_maxes.append(float(layer_phases.abs().max().item()))
            phase_tensors.append(layer_phases)
        phases = torch.cat([tensor.flatten() for tensor in phase_tensors])
        diagnostics["phase_active"] = 0.0 if model.force_zero_phase else 1.0
        diagnostics["phase_trainable"] = 1.0 if getattr(model, "phase_trainable", True) else 0.0
        diagnostics["mix_mean"] = float(mix_values.mean().item())
        diagnostics["mix_min"] = float(mix_values.min().item())
        diagnostics["mix_max"] = float(mix_values.max().item())
        diagnostics["mix_effective_mean"] = float(effective_mix_values.mean().item())
        diagnostics["mix_effective_min"] = float(effective_mix_values.min().item())
        diagnostics["mix_effective_max"] = float(effective_mix_values.max().item())
        diagnostics["phase_abs_mean"] = float(phases.abs().mean().item())
        diagnostics["phase_abs_max"] = float(phases.abs().max().item())
        diagnostics["mix_by_layer"] = _format_trace([float(value.item()) for value in mix_values])
        diagnostics["mix_effective_by_layer"] = _format_trace(
            [float(value.item()) for value in effective_mix_values]
        )
        diagnostics["phase_abs_by_layer"] = _format_trace(phase_layer_means)
        diagnostics["phase_abs_max_by_layer"] = _format_trace(phase_layer_maxes)
    if isinstance(model, RealAttentionStackedProbe):
        gate_values = torch.stack([torch.sigmoid(layer.gate.detach()) for layer in model.layers])
        diagnostics["gate_mean"] = float(gate_values.mean().item())
        diagnostics["gate_min"] = float(gate_values.min().item())
        diagnostics["gate_max"] = float(gate_values.max().item())
        diagnostics["gate_by_layer"] = _format_trace([float(value.item()) for value in gate_values])
    return diagnostics


def train_one(
    model_name: str,
    train_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    test_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    vocab_size: int,
    dim: int,
    context: int,
    layers: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    model = MODEL_TYPES[model_name](vocab_size, dim, context, layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    contexts, candidates, labels = train_data
    count = labels.numel()
    started = time.time()

    for _epoch in range(epochs):
        model.train()
        order = torch.randperm(count, device=device)
        for start in range(0, count, batch_size):
            idx = order[start : start + batch_size]
            logits = model(contexts[idx], candidates[idx])
            loss = F.binary_cross_entropy_with_logits(logits, labels[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    metrics = evaluate(model, test_data, batch_size)
    metrics.update(model_diagnostics(model))
    metrics.update(
        {
            "model": model_name,
            "train_size": count,
            "layers": layers,
            "seed": seed,
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
    fields = [
        "train_size",
        "layers",
        "seed",
        "model",
        "accuracy",
        "loss",
        "seconds",
        "trainable_params",
        "phase_active",
        "phase_trainable",
        "mix_mean",
        "mix_min",
        "mix_max",
        "mix_effective_mean",
        "mix_effective_min",
        "mix_effective_max",
        "phase_abs_mean",
        "phase_abs_max",
        "mix_by_layer",
        "mix_effective_by_layer",
        "phase_abs_by_layer",
        "phase_abs_max_by_layer",
        "gate_mean",
        "gate_min",
        "gate_max",
        "gate_by_layer",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--vocab-stories", type=int, default=50000)
    parser.add_argument("--max-train-examples", type=int, default=500000)
    parser.add_argument("--train-sizes", type=str, default="50000,200000,500000")
    parser.add_argument("--test-examples", type=int, default=100000)
    parser.add_argument("--context", type=int, default=16)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--layer-values", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--cache", type=str, default=None)
    parser.add_argument(
        "--models",
        type=str,
        default=(
            "real_attention,complex_attention_halfdim,complex_attention,"
            "real_attention_stacked,complex_attention_stacked,real_attention_wide"
        ),
    )
    parser.add_argument("--out", type=str, default="runs/tinystories_attention_probe.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_name = str(config.get("dataset", args.dataset))
    split = str(config.get("split", args.split))
    vocab_size = int(config.get("vocab_size", args.vocab_size))
    vocab_stories = int(config.get("vocab_stories", args.vocab_stories))
    max_train_examples = int(config.get("max_train_examples", args.max_train_examples))
    train_sizes = parse_ints(config.get("train_sizes", args.train_sizes))
    test_examples = int(config.get("test_examples", args.test_examples))
    context = int(config.get("context", args.context))
    dim = int(config.get("dim", args.dim))
    layers_config = config.get("layer_values", config.get("layers", args.layers))
    layer_values = parse_ints(layers_config) if isinstance(layers_config, (list, str)) else [int(layers_config)]
    epochs = int(config.get("epochs", args.epochs))
    batch_size = int(config.get("batch_size", args.batch_size))
    learning_rate = float(config.get("learning_rate", args.learning_rate))
    seed = int(config.get("seed", args.seed))
    seeds_config = config.get("seeds", args.seeds)
    seeds = parse_ints(seeds_config) if seeds_config is not None else [seed]
    cache_path = config.get("cache", args.cache)
    out_path = str(config.get("out", args.out))
    models_raw = config.get("models", args.models)
    models = models_raw if isinstance(models_raw, list) else [item.strip() for item in models_raw.split(",")]
    unknown = sorted(set(models) - set(MODEL_TYPES))
    if unknown:
        raise SystemExit(f"unknown model(s): {', '.join(unknown)}")

    device = choose_device(args.device)
    set_seed(seed)
    print(f"run_version={RUN_VERSION}")
    print(f"config={args.config or '<cli/defaults>'}")
    print(f"dataset={dataset_name} split={split} device={device}")
    print(f"models={','.join(models)}")
    print(
        f"dim={dim} context={context} layers={','.join(str(item) for item in layer_values)} "
        f"seeds={','.join(str(item) for item in seeds)} epochs={epochs} batch_size={batch_size}"
    )

    vocab_list, train_cpu, test_cpu = load_or_build_cached_data(
        cache_path=cache_path,
        dataset_name=dataset_name,
        split=split,
        vocab_size=vocab_size,
        vocab_stories=vocab_stories,
        context=context,
        max_train_examples=max_train_examples,
        test_examples=test_examples,
        seed=seed,
    )
    print(f"train_examples={train_cpu[2].numel()} test_examples={test_cpu[2].numel()} vocab_size={len(vocab_list)}")
    test_data = move_dataset(test_cpu, device)

    rows: list[dict[str, Any]] = []
    for train_size in train_sizes:
        train_data = tuple(tensor[:train_size] for tensor in train_cpu)
        train_data = move_dataset(train_data, device)
        print(f"\ntrain_size={train_size}")
        for layers in layer_values:
            print(f"  layers={layers}")
            for run_seed in seeds:
                print(f"    seed={run_seed}")
                for model_name in models:
                    set_seed(run_seed)
                    row = train_one(
                        model_name=model_name,
                        train_data=train_data,
                        test_data=test_data,
                        vocab_size=len(vocab_list),
                        dim=dim,
                        context=context,
                        layers=layers,
                        epochs=epochs,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        device=device,
                        seed=run_seed,
                    )
                    rows.append(row)
                    diag = ""
                    if row.get("mix_mean") is not None:
                        diag = (
                            f" mix={row['mix_mean']:.3f}"
                            f" phase_abs={row['phase_abs_mean']:.3f}"
                        )
                    elif row.get("gate_mean") is not None:
                        diag = f" gate={row['gate_mean']:.3f}"
                    print(
                        f"      {model_name:38s} "
                        f"acc={row['accuracy']:.4f} loss={row['loss']:.4f} "
                        f"params={row['trainable_params']} seconds={row['seconds']}{diag}"
                    )

    out = ROOT / out_path
    write_results(out, rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

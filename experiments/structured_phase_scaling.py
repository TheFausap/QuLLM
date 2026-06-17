"""Structured phase-token scaling probe.

This experiment asks a narrower question than phase_relation_scaling.py:

    If the tokenizer prepares structured amplitude/phase states, can a small
    phase-native relation model exploit that structure and generalize to held
    out token variants?

Tokens are synthetic but compositional. Each token has:

    group:       amplitude support, analogous to a semantic or morphology basis
    phase_class: relative phase, analogous to relational role information
    variant:     surface form identity, held out at test time

The hidden teacher marks a pair positive when the two tokens share a group and
their phase classes match a relation-specific phase shift.
"""

from __future__ import annotations

import argparse
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
    raise SystemExit(
        "This experiment requires PyTorch. Install it with:\n"
        "  python3 -m pip install -r requirements.txt"
    ) from exc


ROOT = Path(__file__).resolve().parents[1]


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


def parse_ints(value: str | list[int]) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    return [int(item.strip()) for item in value.split(",") if item.strip()]


class StructuredPhaseSpace:
    def __init__(
        self,
        groups: int,
        phase_classes: int,
        variants_per_bucket: int,
        test_variants_per_bucket: int,
        device: torch.device,
    ) -> None:
        if test_variants_per_bucket <= 0:
            raise ValueError("test_variants_per_bucket must be positive")
        if test_variants_per_bucket >= variants_per_bucket:
            raise ValueError("test variants must be fewer than total variants")
        self.groups = groups
        self.phase_classes = phase_classes
        self.variants_per_bucket = variants_per_bucket
        self.test_variants_per_bucket = test_variants_per_bucket
        self.train_variants_per_bucket = variants_per_bucket - test_variants_per_bucket
        self.device = device
        self.vocab_size = groups * phase_classes * variants_per_bucket

    def token_id(self, group: torch.Tensor, phase_class: torch.Tensor, variant: torch.Tensor) -> torch.Tensor:
        return ((phase_class * self.groups + group) * self.variants_per_bucket) + variant

    def group_of(self, token: torch.Tensor) -> torch.Tensor:
        return (token // self.variants_per_bucket) % self.groups

    def phase_class_of(self, token: torch.Tensor) -> torch.Tensor:
        return (token // (self.variants_per_bucket * self.groups)) % self.phase_classes

    def phase_angle_of(self, token: torch.Tensor) -> torch.Tensor:
        return self.phase_class_of(token).float() * (math.tau / self.phase_classes)

    def sample_token(self, group: torch.Tensor, phase_class: torch.Tensor, split: str) -> torch.Tensor:
        count = group.numel()
        if split == "train":
            variant = torch.randint(0, self.train_variants_per_bucket, (count,), device=self.device)
        elif split == "test":
            variant = torch.randint(
                self.train_variants_per_bucket,
                self.variants_per_bucket,
                (count,),
                device=self.device,
            )
        else:
            raise ValueError(f"unknown split: {split}")
        return self.token_id(group, phase_class, variant)


def make_dataset(
    space: StructuredPhaseSpace,
    count: int,
    split: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pos_count = count // 2
    neg_count = count - pos_count
    same_wrong_count = neg_count // 2
    diff_group_count = neg_count - same_wrong_count

    pos_left, pos_rel, pos_right = make_positive(space, pos_count, split)
    wrong_left, wrong_rel, wrong_right = make_same_group_negative(space, same_wrong_count, split)
    diff_left, diff_rel, diff_right = make_diff_group_negative(space, diff_group_count, split)

    left = torch.cat([pos_left, wrong_left, diff_left])
    rel = torch.cat([pos_rel, wrong_rel, diff_rel])
    right = torch.cat([pos_right, wrong_right, diff_right])
    labels = torch.cat(
        [
            torch.ones(pos_count, device=space.device),
            torch.zeros(same_wrong_count + diff_group_count, device=space.device),
        ]
    )

    order = torch.randperm(count, device=space.device)
    return left[order], rel[order], right[order], labels[order]


def make_positive(
    space: StructuredPhaseSpace,
    count: int,
    split: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    group = torch.randint(0, space.groups, (count,), device=space.device)
    rel = torch.randint(0, space.phase_classes, (count,), device=space.device)
    left_phase = torch.randint(0, space.phase_classes, (count,), device=space.device)
    right_phase = (left_phase + rel) % space.phase_classes
    return space.sample_token(group, left_phase, split), rel, space.sample_token(group, right_phase, split)


def make_same_group_negative(
    space: StructuredPhaseSpace,
    count: int,
    split: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    group = torch.randint(0, space.groups, (count,), device=space.device)
    rel = torch.randint(0, space.phase_classes, (count,), device=space.device)
    left_phase = torch.randint(0, space.phase_classes, (count,), device=space.device)
    offset = torch.randint(1, space.phase_classes, (count,), device=space.device)
    right_phase = (left_phase + rel + offset) % space.phase_classes
    return space.sample_token(group, left_phase, split), rel, space.sample_token(group, right_phase, split)


def make_diff_group_negative(
    space: StructuredPhaseSpace,
    count: int,
    split: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    left_group = torch.randint(0, space.groups, (count,), device=space.device)
    offset = torch.randint(1, space.groups, (count,), device=space.device)
    right_group = (left_group + offset) % space.groups
    rel = torch.randint(0, space.phase_classes, (count,), device=space.device)
    left_phase = torch.randint(0, space.phase_classes, (count,), device=space.device)
    right_phase = torch.randint(0, space.phase_classes, (count,), device=space.device)
    return (
        space.sample_token(left_group, left_phase, split),
        rel,
        space.sample_token(right_group, right_phase, split),
    )


class PhaseFeatureModel(nn.Module):
    def __init__(self, space: StructuredPhaseSpace) -> None:
        super().__init__()
        self.space = space
        self.rel_phase = nn.Embedding(space.phase_classes, 1)
        self.logit_scale = nn.Parameter(torch.tensor(8.0))
        self.logit_bias = nn.Parameter(torch.tensor(0.0))
        nn.init.zeros_(self.rel_phase.weight)

    def relation_phase(self, rel: torch.Tensor) -> torch.Tensor:
        return self.rel_phase(rel).squeeze(-1)

    def forward(self, left: torch.Tensor, rel: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        same_group = (self.space.group_of(left) == self.space.group_of(right)).float()
        delta = self.space.phase_angle_of(right) + self.relation_phase(rel) - self.space.phase_angle_of(left)
        score = same_group * (0.5 + 0.5 * torch.cos(delta))
        return self.logit_scale * (score - 0.5) + self.logit_bias


class PhaseMarginModel(PhaseFeatureModel):
    """Sharper relation phase readout for discrete phase classes.

    PhaseFeatureModel uses a smooth Born-style cosine score. That is a useful
    probe, but with many phase classes the nearest wrong class can sit very near
    the positive class. This model keeps the same learned relation phase shift
    while making the measurement threshold explicit.
    """

    def __init__(self, space: StructuredPhaseSpace) -> None:
        super().__init__(space)
        step = math.tau / space.phase_classes
        # In cosine-score space the nearest wrong phase has score cos(step),
        # so the natural binary decision boundary is the midpoint between the
        # correct score 1.0 and that nearest wrong score.
        self.register_buffer("phase_threshold", torch.tensor((1.0 + math.cos(step)) / 2.0))
        self.logit_scale = nn.Parameter(torch.tensor(12.0))
        self.group_penalty = nn.Parameter(torch.tensor(12.0))

    def forward(self, left: torch.Tensor, rel: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        same_group = (self.space.group_of(left) == self.space.group_of(right)).float()
        delta = self.space.phase_angle_of(right) + self.relation_phase(rel) - self.space.phase_angle_of(left)
        phase_logit = self.logit_scale * (torch.cos(delta) - self.phase_threshold)
        return phase_logit - (1.0 - same_group) * F.softplus(self.group_penalty) + self.logit_bias


class PhaseMarginFixedModel(PhaseFeatureModel):
    """Phase-margin model with fixed measurement scale and no learned bias."""

    def __init__(self, space: StructuredPhaseSpace) -> None:
        super().__init__(space)
        step = math.tau / space.phase_classes
        self.register_buffer("phase_threshold", torch.tensor((1.0 + math.cos(step)) / 2.0))
        self.register_buffer("fixed_scale", torch.tensor(32.0))
        self.register_buffer("fixed_group_penalty", torch.tensor(32.0))
        self.logit_scale.requires_grad_(False)
        self.logit_bias.requires_grad_(False)

    def forward(self, left: torch.Tensor, rel: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        same_group = (self.space.group_of(left) == self.space.group_of(right)).float()
        delta = self.space.phase_angle_of(right) + self.relation_phase(rel) - self.space.phase_angle_of(left)
        phase_logit = self.fixed_scale * (torch.cos(delta) - self.phase_threshold)
        return phase_logit - (1.0 - same_group) * self.fixed_group_penalty


class NoRelationPhaseModel(PhaseFeatureModel):
    def __init__(self, space: StructuredPhaseSpace) -> None:
        super().__init__(space)
        self.rel_phase.weight.requires_grad_(False)


class AmplitudeFeatureModel(nn.Module):
    def __init__(self, space: StructuredPhaseSpace) -> None:
        super().__init__()
        self.space = space
        self.logit_scale = nn.Parameter(torch.tensor(8.0))
        self.logit_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, left: torch.Tensor, rel: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        del rel
        score = (self.space.group_of(left) == self.space.group_of(right)).float()
        return self.logit_scale * (score - 0.5) + self.logit_bias


class RealFeatureMLP(nn.Module):
    def __init__(self, space: StructuredPhaseSpace, width: int = 64) -> None:
        super().__init__()
        self.space = space
        phase_dim = min(16, max(4, space.phase_classes))
        group_dim = 16
        rel_dim = phase_dim
        self.group_embedding = nn.Embedding(space.groups, group_dim)
        self.phase_embedding = nn.Embedding(space.phase_classes, phase_dim)
        self.rel_embedding = nn.Embedding(space.phase_classes, rel_dim)
        input_dim = 2 * group_dim + 2 * phase_dim + rel_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    def forward(self, left: torch.Tensor, rel: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        features = torch.cat(
            [
                self.group_embedding(self.space.group_of(left)),
                self.phase_embedding(self.space.phase_class_of(left)),
                self.rel_embedding(rel),
                self.group_embedding(self.space.group_of(right)),
                self.phase_embedding(self.space.phase_class_of(right)),
            ],
            dim=-1,
        )
        return self.net(features).squeeze(-1)


MODEL_TYPES = {
    "phase_feature": PhaseFeatureModel,
    "phase_margin": PhaseMarginModel,
    "phase_margin_fixed": PhaseMarginFixedModel,
    "no_relation_phase": NoRelationPhaseModel,
    "amplitude_feature": AmplitudeFeatureModel,
    "real_feature_mlp": RealFeatureMLP,
}


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


@torch.no_grad()
def relation_phase_error(model: nn.Module, space: StructuredPhaseSpace) -> tuple[float | None, float | None]:
    if not hasattr(model, "rel_phase"):
        return None, None

    learned = model.rel_phase.weight.detach().flatten()
    step = math.tau / space.phase_classes
    target = -torch.arange(space.phase_classes, device=space.device, dtype=learned.dtype) * step
    error = torch.atan2(torch.sin(learned - target), torch.cos(learned - target)).abs()
    return float(error.mean().item()), float(error.max().item())


@torch.no_grad()
def deterministic_phase_rule_accuracy(
    model: nn.Module,
    space: StructuredPhaseSpace,
    dataset: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> float | None:
    if not hasattr(model, "rel_phase") or not hasattr(model, "phase_threshold"):
        return None

    left, rel, right, labels = dataset
    same_group = space.group_of(left) == space.group_of(right)
    learned_phase = model.rel_phase(rel).squeeze(-1)
    delta = space.phase_angle_of(right) + learned_phase - space.phase_angle_of(left)
    predictions = same_group & (torch.cos(delta) > model.phase_threshold)
    return float((predictions == labels.bool()).float().mean().item())


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
        loss = F.binary_cross_entropy_with_logits(logits, labels[batch], reduction="sum")
        loss_sum += float(loss.item())
        correct += ((torch.sigmoid(logits) >= 0.5) == labels[batch].bool()).sum().item()

    return {"loss": loss_sum / total, "accuracy": correct / total}


@torch.no_grad()
def evaluate_breakdown(
    model: nn.Module,
    space: StructuredPhaseSpace,
    dataset: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, float]:
    model.eval()
    left, rel, right, labels = dataset
    predictions = torch.sigmoid(model(left, rel, right)) >= 0.5
    same_group = space.group_of(left) == space.group_of(right)
    positive = labels.bool()
    same_group_negative = (~positive) & same_group
    diff_group_negative = (~positive) & (~same_group)

    def masked_accuracy(mask: torch.Tensor) -> float:
        if not bool(mask.any().item()):
            return float("nan")
        return float((predictions[mask] == labels[mask].bool()).float().mean().item())

    return {
        "positive_accuracy": masked_accuracy(positive),
        "same_group_negative_accuracy": masked_accuracy(same_group_negative),
        "diff_group_negative_accuracy": masked_accuracy(diff_group_negative),
    }


def train_one(
    model_name: str,
    space: StructuredPhaseSpace,
    train_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    test_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> dict[str, float | str | int]:
    model = MODEL_TYPES[model_name](space).to(space.device)
    trainable_params = trainable_parameter_count(model)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=learning_rate,
        weight_decay=1e-4,
    )
    left, rel, right, labels = train_data
    count = labels.numel()
    started = time.time()

    for _epoch in range(epochs):
        model.train()
        order = torch.randperm(count, device=space.device)
        for start in range(0, count, batch_size):
            idx = order[start : start + batch_size]
            logits = model(left[idx], rel[idx], right[idx])
            loss = F.binary_cross_entropy_with_logits(logits, labels[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    metrics = evaluate(model, test_data, batch_size)
    metrics.update(evaluate_breakdown(model, space, test_data))
    phase_mean_error, phase_max_error = relation_phase_error(model, space)
    phase_rule_accuracy = deterministic_phase_rule_accuracy(model, space, test_data)
    metrics.update(
        {
            "model": model_name,
            "train_size": count,
            "seconds": round(time.time() - started, 3),
            "trainable_params": trainable_params,
            "phase_mean_error": phase_mean_error,
            "phase_max_error": phase_max_error,
            "phase_rule_accuracy": phase_rule_accuracy,
        }
    )
    return metrics


def write_results(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "train_size",
        "model",
        "accuracy",
        "loss",
        "seconds",
        "trainable_params",
        "phase_mean_error",
        "phase_max_error",
        "phase_rule_accuracy",
        "positive_accuracy",
        "same_group_negative_accuracy",
        "diff_group_negative_accuracy",
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
    parser.add_argument("--groups", type=int, default=64)
    parser.add_argument("--phase-classes", type=int, default=8)
    parser.add_argument("--variants-per-bucket", type=int, default=8)
    parser.add_argument("--test-variants-per-bucket", type=int, default=2)
    parser.add_argument("--train-sizes", type=str, default="512,2048,8192,32768,131072")
    parser.add_argument("--test-size", type=int, default=32768)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--models",
        type=str,
        default="phase_feature,phase_margin,no_relation_phase,amplitude_feature,real_feature_mlp",
    )
    parser.add_argument("--out", type=str, default="runs/structured_phase_scaling.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    groups = int(config.get("groups", args.groups))
    phase_classes = int(config.get("phase_classes", args.phase_classes))
    variants_per_bucket = int(config.get("variants_per_bucket", args.variants_per_bucket))
    test_variants_per_bucket = int(config.get("test_variants_per_bucket", args.test_variants_per_bucket))
    train_sizes = parse_ints(config.get("train_sizes", args.train_sizes))
    test_size = int(config.get("test_size", args.test_size))
    epochs = int(config.get("epochs", args.epochs))
    batch_size = int(config.get("batch_size", args.batch_size))
    learning_rate = float(config.get("learning_rate", args.learning_rate))
    seed = int(config.get("seed", args.seed))
    models_raw = config.get("models", args.models)
    models = models_raw if isinstance(models_raw, list) else [item.strip() for item in models_raw.split(",")]

    unknown = sorted(set(models) - set(MODEL_TYPES))
    if unknown:
        raise SystemExit(f"unknown model(s): {', '.join(unknown)}")

    device = choose_device(args.device)
    set_seed(seed)
    space = StructuredPhaseSpace(
        groups=groups,
        phase_classes=phase_classes,
        variants_per_bucket=variants_per_bucket,
        test_variants_per_bucket=test_variants_per_bucket,
        device=device,
    )
    test_data = make_dataset(space, test_size, split="test")

    print(
        f"device={device} vocab={space.vocab_size} groups={groups} "
        f"phase_classes={phase_classes} variants={variants_per_bucket}"
    )
    print("test split uses held-out surface variants for every group/phase bucket")

    rows: list[dict[str, Any]] = []
    for train_size in train_sizes:
        train_data = make_dataset(space, train_size, split="train")
        print(f"\ntrain_size={train_size}")
        for model_name in models:
            set_seed(seed)
            row = train_one(
                model_name=model_name,
                space=space,
                train_data=train_data,
                test_data=test_data,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
            )
            rows.append(row)
            phase_diag = ""
            if row.get("phase_mean_error") is not None:
                phase_diag = (
                    f" phase_mean_err={row['phase_mean_error']:.3f}"
                    f" phase_max_err={row['phase_max_error']:.3f}"
                )
            rule_diag = ""
            if row.get("phase_rule_accuracy") is not None:
                rule_diag = f" phase_rule_acc={row['phase_rule_accuracy']:.4f}"
            print(
                f"  {model_name:18s} "
                f"acc={row['accuracy']:.4f} loss={row['loss']:.4f} "
                f"params={row['trainable_params']} seconds={row['seconds']}{phase_diag}{rule_diag}"
            )

    out = ROOT / args.out
    write_results(out, rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

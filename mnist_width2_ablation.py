#!/usr/bin/env python3
"""Member-count ablation for 784 -> 2 -> 10 MNIST ensembles.

Every member has separate parameters.  The member axis is vectorized with einsum,
which is mathematically equivalent to evaluating K independent tiny MLPs but avoids
K Python/module calls per batch.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset


class BatchedWidth2MLP(nn.Module):
    """K independent 784 -> 2 -> 10 MLPs evaluated in one batched operation."""

    def __init__(self, members: int):
        super().__init__()
        self.members = members
        self.w1 = nn.Parameter(torch.empty(members, 2, 784))
        self.b1 = nn.Parameter(torch.zeros(members, 2))
        self.w2 = nn.Parameter(torch.empty(members, 10, 2))
        self.b2 = nn.Parameter(torch.zeros(members, 10))
        nn.init.kaiming_uniform_(self.w1, a=5**0.5)
        nn.init.kaiming_uniform_(self.w2, a=5**0.5)

    def member_logits(self, images: torch.Tensor) -> torch.Tensor:
        x = images.flatten(1)
        hidden = F.relu(torch.einsum("bi,khi->bkh", x, self.w1) + self.b1)
        return torch.einsum("bkh,koh->bko", hidden, self.w2) + self.b2

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.member_logits(images).sum(1)


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def materialize(split) -> TensorDataset:
    images = torch.stack([torch.from_numpy(np.array(image, copy=True)).unsqueeze(0) for image in split["image"]])
    images = images.float().div_(255).sub_(0.1307).div_(0.3081)
    return TensorDataset(images, torch.tensor(split["label"], dtype=torch.long))


@torch.no_grad()
def logits_for(model: BatchedWidth2MLP, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval(); chunks, labels = [], []
    for images, targets in loader:
        chunks.append(model.member_logits(images.to(device, non_blocking=True)).float().cpu())
        labels.append(targets)
    return torch.cat(chunks), torch.cat(labels)


def summary(member_logits: torch.Tensor, targets: torch.Tensor, mode: str) -> dict:
    members = member_logits.shape[1]
    aggregate = member_logits.sum(1) if mode == "joint_sum" else member_logits.mean(1)
    member_accuracy = member_logits.argmax(2).eq(targets[:, None]).float().mean(0).mul(100)
    correlations = []
    for left in range(members):
        for right in range(left + 1, members):
            correlations.append(torch.corrcoef(torch.stack((member_logits[:, left].flatten(), member_logits[:, right].flatten())))[0, 1].item())
    total = member_logits.sum(1)
    leave_one_out = [(total - member_logits[:, index]).argmax(1).eq(targets).float().mean().item() * 100 for index in range(members)]
    return {
        "accuracy": aggregate.argmax(1).eq(targets).float().mean().item() * 100,
        "nll": F.cross_entropy(aggregate, targets).item(),
        "member_accuracy_mean": member_accuracy.mean().item(),
        "member_accuracy_min": member_accuracy.min().item(),
        "member_accuracy_max": member_accuracy.max().item(),
        "mean_logit_correlation": float(np.mean(correlations)) if correlations else None,
        "leave_one_out_accuracy_mean": float(np.mean(leave_one_out)) if members > 1 else None,
        "leave_one_out_accuracy_min": float(np.min(leave_one_out)) if members > 1 else None,
    }


def train_condition(members: int, mode: str, train_loader: DataLoader, test_loader: DataLoader,
                    device: torch.device, epochs: int, seed: int, compile_model: bool) -> dict:
    seed_everything(seed)
    model: nn.Module = BatchedWidth2MLP(members).to(device)
    if compile_model:
        model = torch.compile(model, mode="max-autotune-no-cudagraphs")
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        for images, targets in train_loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            member_logits = model.member_logits(images)
            if mode == "joint_sum":
                loss = F.cross_entropy(member_logits.sum(1), targets)
            else:
                loss = F.cross_entropy(member_logits.transpose(1, 2), targets[:, None].expand(-1, members))
            optimiser.zero_grad(set_to_none=True); loss.backward(); optimiser.step()
    logits, targets = logits_for(model, test_loader, device)
    return summary(logits, targets, mode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--members", nargs="+", type=int, default=[1, 2, 3, 5, 8, 16])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("mnist-width2-results"))
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("ylecun/mnist", cache_dir=str(args.data))
    train, test = materialize(dataset["train"]), materialize(dataset["test"])
    loader_args = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    train_loader = DataLoader(train, shuffle=True, **loader_args)
    test_loader = DataLoader(test, shuffle=False, **loader_args)
    records = []
    for mode in ("independent", "joint_sum"):
        for members in args.members:
            print(f"Training {mode}, K={members}", flush=True)
            result = train_condition(members, mode, train_loader, test_loader, torch.device("cuda"), args.epochs,
                                     args.seed + members + (0 if mode == "independent" else 1000), not args.no_compile)
            record = {"mode": mode, "members": members, **result}
            records.append(record); print(record, flush=True)
    with (args.output / "summary.json").open("w") as handle: json.dump({"args": vars(args), "records": records}, handle, indent=2, default=str)
    with (args.output / "summary.csv").open("w", newline="") as handle:
        fields = list(records[0]); writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(records)


if __name__ == "__main__": main()

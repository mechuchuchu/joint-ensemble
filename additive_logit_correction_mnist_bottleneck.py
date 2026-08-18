#!/usr/bin/env python3
"""MNIST memorization with staged additive corrections and a 2D bottleneck.

Each correction network is 784 -> 128 -> 64 -> 32 -> 2 -> 10.  At every
stage, earlier logits are a frozen baseline and only the new correction network
is optimized.  The script records test accuracy after every epoch and stage.
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


class BottleneckCorrection(nn.Module):
    widths = (784, 128, 64, 32, 2, 10)

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = dropout
        self.dropout_active = dropout > 0
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(self.widths, self.widths[1:]))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = images.flatten(1)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
            if self.dropout and self.dropout_active:
                x = F.dropout(x, p=self.dropout, training=True)
        return self.layers[-1](x)


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def materialize(split) -> TensorDataset:
    images = torch.stack([torch.from_numpy(np.array(image, copy=True)).unsqueeze(0) for image in split["image"]])
    images = images.float().div_(255).sub_(0.1307).div_(0.3081)
    return TensorDataset(images, torch.tensor(split["label"], dtype=torch.long))


@torch.no_grad()
def additive_logits(models: list[nn.Module], images: torch.Tensor) -> torch.Tensor:
    if not models:
        return torch.zeros(images.size(0), 10, device=images.device)
    return torch.stack([model(images) for model in models]).sum(0)


def set_dropout(models: list[nn.Module], enabled: bool) -> None:
    for model in models:
        if hasattr(model, "dropout_active"):
            model.dropout_active = enabled and model.dropout > 0


@torch.no_grad()
def evaluate(models: list[nn.Module], loader: DataLoader, device: torch.device,
             repeats: int = 1) -> tuple[float, float]:
    set_dropout(models, False)
    for model in models: model.eval()
    correct = total = 0; loss_sum = 0.0
    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        logits = torch.stack([additive_logits(models, images) for _ in range(repeats)]).mean(0)
        loss_sum += F.cross_entropy(logits, targets).item() * targets.numel()
        correct += logits.argmax(1).eq(targets).sum().item(); total += targets.numel()
    return correct * 100 / total, loss_sum / total


def train_stage(models: list[nn.Module], train_loader: DataLoader, test_loader: DataLoader,
                device: torch.device, epochs: int, seed: int, lr: float,
                dropout: float, baseline_dropout: bool, eval_repeats: int) -> tuple[dict, list[dict]]:
    seed_everything(seed)
    correction = BottleneckCorrection(dropout).to(device)
    optimiser = torch.optim.AdamW(correction.parameters(), lr=lr, weight_decay=1e-4)
    correction.train(); curve = []
    for epoch in range(1, epochs + 1):
        loss_sum = examples = 0
        for images, targets in train_loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            set_dropout(models, baseline_dropout)
            with torch.no_grad(): baseline = additive_logits(models, images)
            correction.dropout_active = dropout > 0
            optimiser.zero_grad(set_to_none=True)
            loss = F.cross_entropy(baseline + correction(images), targets)
            loss.backward(); optimiser.step()
            loss_sum += loss.item() * targets.numel(); examples += targets.numel()
        models_for_eval = models + [correction]
        test_accuracy, test_nll = evaluate(models_for_eval, test_loader, device, eval_repeats)
        curve.append({"stage": len(models) + 1, "epoch": epoch,
                      "train_loss": loss_sum / examples, "test_accuracy": test_accuracy,
                      "test_nll": test_nll})
        print(f"stage {len(models)+1:02d} epoch {epoch:02d}/{epochs}: train_loss={loss_sum/examples:.4f} "
              f"test_acc={test_accuracy:.2f}%", flush=True)
    correction.eval(); models.append(correction)
    test_accuracy, test_nll = evaluate(models, test_loader, device, eval_repeats)
    result = {"stage": len(models), "test_accuracy": test_accuracy, "test_nll": test_nll,
              "parameters_per_stage": sum(p.numel() for p in correction.parameters()),
              "total_parameters": sum(p.numel() for model in models for p in model.parameters())}
    return result, curve


def save_plot(records: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    stages = sorted({int(row["stage"]) for row in records})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for stage in stages:
        points = [row for row in records if int(row["stage"]) == stage]
        by_epoch = {}
        for point in points:
            by_epoch.setdefault(int(point["epoch"]), []).append(float(point["test_accuracy"]))
        xs = sorted(by_epoch)
        axes[0].plot(xs, [np.mean(by_epoch[x]) for x in xs], label=f"stage {stage}")
        by_epoch = {}
        for point in points:
            by_epoch.setdefault(int(point["epoch"]), []).append(float(point["test_nll"]))
        axes[1].plot(xs, [np.mean(by_epoch[x]) for x in xs], label=f"stage {stage}")
    axes[0].set_ylabel("MNIST test accuracy (%)"); axes[0].set_ylim(0, 100)
    axes[1].set_ylabel("MNIST test NLL")
    for axis in axes:
        axis.set_xlabel("epoch within correction stage"); axis.grid(alpha=0.25); axis.legend(ncol=2, fontsize=8)
    fig.suptitle("Additive Logit Correction with a 2D MNIST bottleneck")
    fig.tight_layout(); fig.savefig(output / "epoch_curves.png", dpi=160); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", type=int, default=8); parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512); parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-3); parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Training dropout probability after every hidden activation")
    parser.add_argument("--baseline-dropout", choices=("on", "off"), default="on",
                        help="Whether frozen previous models use dropout when producing training baselines")
    parser.add_argument("--eval-repeats", type=int, default=1,
                        help="Number of stochastic forward passes averaged for evaluation")
    parser.add_argument("--seed", type=int, default=20260822); parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("additive-mnist-bottleneck-results"))
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    args.output.mkdir(parents=True, exist_ok=True); dataset = load_dataset("ylecun/mnist", cache_dir=str(args.data))
    train, test = materialize(dataset["train"]), materialize(dataset["test"])
    loader_args = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    train_loader = DataLoader(train, shuffle=True, **loader_args); test_loader = DataLoader(test, shuffle=False, **loader_args)
    records, epoch_records = [], []
    for trial in range(args.trials):
        models: list[nn.Module] = []
        for stage in range(args.stages):
            result, curve = train_stage(models, train_loader, test_loader, torch.device("cuda"), args.epochs,
                                        args.seed + trial * 100 + stage, args.lr,
                                        args.dropout, args.baseline_dropout == "on", args.eval_repeats)
            row = {"trial": trial, **result}; records.append(row)
            for point in curve: epoch_records.append({"trial": trial, **point})
            print("RESULT", row, flush=True)
    payload = {"args": vars(args), "records": records, "epoch_records": epoch_records}
    with (args.output / "summary.json").open("w") as handle: json.dump(payload, handle, indent=2, default=str)
    with (args.output / "summary.csv").open("w", newline="") as handle:
        fields = list(records[0]); writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(records)
    with (args.output / "epoch_records.csv").open("w", newline="") as handle:
        fields = list(epoch_records[0]); writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(epoch_records)
    save_plot(epoch_records, args.output)


if __name__ == "__main__": main()

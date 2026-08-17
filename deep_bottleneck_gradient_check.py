#!/usr/bin/env python3
"""Gradient-flow check for a 784 -> 128 -> 64 -> 32 -> 2 -> 10 MNIST MLP."""

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


class DeepBottleneckMLP(nn.Module):
    widths = (784, 128, 64, 32, 2, 10)

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(inp, out) for inp, out in zip(self.widths, self.widths[1:]))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = images.flatten(1)
        for layer in self.layers[:-1]: x = F.relu(layer(x))
        return self.layers[-1](x)


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def materialize(split) -> TensorDataset:
    images = torch.stack([torch.from_numpy(np.array(image, copy=True)).unsqueeze(0) for image in split["image"]])
    images = images.float().div_(255).sub_(0.1307).div_(0.3081)
    return TensorDataset(images, torch.tensor(split["label"], dtype=torch.long))


@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval(); correct = total = 0
    for images, targets in loader:
        correct += model(images.to(device, non_blocking=True)).argmax(1).cpu().eq(targets).sum().item(); total += targets.numel()
    return correct * 100 / total


def layer_statistics(model: DeepBottleneckMLP) -> list[dict]:
    rows = []
    for index, layer in enumerate(model.layers):
        grad_sq, parameter_sq, count = 0.0, 0.0, 0
        for parameter in layer.parameters():
            if parameter.grad is None: continue
            grad_sq += parameter.grad.detach().float().square().sum().item()
            parameter_sq += parameter.detach().float().square().sum().item()
            count += parameter.numel()
        grad_rms = (grad_sq / count) ** 0.5
        parameter_rms = (parameter_sq / count) ** 0.5
        rows.append({"layer": index, "shape": f"{layer.in_features}->{layer.out_features}", "gradient_rms": grad_rms,
                     "parameter_rms": parameter_rms, "relative_gradient": grad_rms / (parameter_rms + 1e-12)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("deep-bottleneck-gradient-results"))
    args = parser.parse_args()
    seed_everything(args.seed); args.output.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("ylecun/mnist", cache_dir=str(args.data))
    train, test = materialize(dataset["train"]), materialize(dataset["test"])
    loader_args = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    train_loader, test_loader = DataLoader(train, shuffle=True, **loader_args), DataLoader(test, shuffle=False, **loader_args)
    device = torch.device("cuda")
    model: nn.Module = torch.compile(DeepBottleneckMLP().to(device), mode="max-autotune-no-cudagraphs")
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    records = []
    for epoch in range(1, args.epochs + 1):
        model.train(); loss_sum = examples = 0
        epoch_gradient = None
        for batch, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(images), targets); loss.backward()
            if batch == 0: epoch_gradient = layer_statistics(model._orig_mod)
            optimiser.step(); loss_sum += loss.item() * targets.numel(); examples += targets.numel()
        for row in epoch_gradient:
            row.update({"epoch": epoch, "train_loss": loss_sum / examples, "test_accuracy": accuracy(model, test_loader, device)})
            records.append(row)
        print(f"epoch {epoch:02d}: loss={loss_sum / examples:.4f}, test_accuracy={records[-1]['test_accuracy']:.2f}%, "
              f"relative-gradient first/last={epoch_gradient[0]['relative_gradient']:.2e}/{epoch_gradient[-1]['relative_gradient']:.2e}", flush=True)
    with (args.output / "gradient_flow.json").open("w") as handle: json.dump({"args": vars(args), "records": records}, handle, indent=2, default=str)
    with (args.output / "gradient_flow.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)


if __name__ == "__main__": main()

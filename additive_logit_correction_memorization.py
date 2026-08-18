#!/usr/bin/env python3
"""Random-label memorization with staged additive logit correction.

For stage t, the preceding prediction is a frozen baseline and a fresh small
MLP learns only a correction:

    z_t(x) = stopgrad(z_{t-1}(x)) + f_t(x)

There are no comparison conditions in this script.  Each row reports the
accuracy/NLL of the additive predictor after that stage.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class CorrectionMLP(nn.Module):
    def __init__(self, width: int = 16):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(64, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


@torch.no_grad()
def additive_logits(models: list[nn.Module], inputs: torch.Tensor) -> torch.Tensor:
    if not models:
        return torch.zeros(inputs.size(0), 10, device=inputs.device)
    return torch.stack([model(inputs) for model in models]).sum(0)


def accuracy_nll(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    return (logits.argmax(1).eq(labels).float().mean().item() * 100,
            F.cross_entropy(logits, labels).item())


def train_stage(models: list[nn.Module], inputs: torch.Tensor, labels: torch.Tensor,
                steps: int, epochs: int, batch_size: int, seed: int, width: int) -> tuple[dict, list[dict]]:
    torch.manual_seed(seed)
    correction = CorrectionMLP(width).cuda()
    optimiser = torch.optim.AdamW(correction.parameters(), lr=1e-2, weight_decay=0)
    baseline_models = models
    correction.train()
    steps_per_epoch = max(1, math.ceil(steps / epochs))
    curve = []
    for epoch in range(1, epochs + 1):
        for _ in range(steps_per_epoch):
            index = torch.randint(inputs.size(0), (batch_size,), device=inputs.device)
            x, y = inputs[index], labels[index]
            with torch.no_grad():
                baseline = additive_logits(baseline_models, x)
            optimiser.zero_grad(set_to_none=True)
            loss = F.cross_entropy(baseline + correction(x), y)
            loss.backward()
            optimiser.step()
        correction.eval()
        with torch.no_grad():
            epoch_logits = additive_logits(models + [correction], inputs)
        epoch_accuracy, epoch_nll = accuracy_nll(epoch_logits, labels)
        curve.append({"stage": len(models) + 1, "epoch": epoch,
                      "accuracy": epoch_accuracy, "nll": epoch_nll})
        correction.train()
    correction.eval()
    models.append(correction)
    with torch.no_grad():
        logits = additive_logits(models, inputs)
    accuracy, nll = accuracy_nll(logits, labels)
    return {"stage": len(models), "accuracy": accuracy, "nll": nll,
            "correction_parameters": sum(p.numel() for p in correction.parameters()),
            "total_parameters": sum(p.numel() for model in models for p in model.parameters()),
            "steps_per_epoch": steps_per_epoch}, curve


def save_epoch_plot(records: list[dict], output: Path) -> None:
    vectors = sorted({int(row["vectors"]) for row in records})
    columns = 2; rows = math.ceil(len(vectors) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(13, 4.8 * rows), squeeze=False)
    for axis, count in zip(axes.flat, vectors):
        subset = [row for row in records if int(row["vectors"]) == count]
        stages = sorted({int(row["stage"]) for row in subset})
        for stage in stages:
            points = [row for row in subset if int(row["stage"]) == stage]
            by_epoch = {}
            for point in points:
                by_epoch.setdefault(int(point["epoch"]), []).append(float(point["accuracy"]))
            xs = sorted(by_epoch)
            ys = [np.mean(by_epoch[x]) for x in xs]
            axis.plot(xs, ys, label=f"stage {stage}", linewidth=1.5)
        axis.set_title(f"{count:,} random vectors")
        axis.set_xlabel("epoch within correction stage")
        axis.set_ylabel("train accuracy (%)")
        axis.set_ylim(0, 100.5); axis.grid(alpha=0.25)
        axis.legend(ncol=2, fontsize=8)
    for axis in axes.flat[len(vectors):]: axis.axis("off")
    fig.suptitle("Additive Logit Correction: epoch-wise memorization", fontsize=15)
    fig.tight_layout(); fig.savefig(output / "epoch_curves.png", dpi=160); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", nargs="+", type=int, default=[128, 512, 2048, 8192])
    parser.add_argument("--stages", type=int, default=8)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--min-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("additive-logit-correction-results"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA.")
    args.output.mkdir(parents=True, exist_ok=True)
    records, epoch_records = [], []
    for count in args.vectors:
        steps = max(args.min_steps, math.ceil(args.epochs * count / args.batch_size))
        for trial in range(args.trials):
            data_generator = torch.Generator(device="cuda").manual_seed(args.seed + count * 100 + trial)
            inputs = torch.randn(count, 64, generator=data_generator, device="cuda")
            labels = torch.randint(10, (count,), generator=data_generator, device="cuda")
            models: list[nn.Module] = []
            for stage in range(args.stages):
                result, curve = train_stage(models, inputs, labels, steps, args.epochs,
                                            args.batch_size, args.seed + count * 1000 + trial * 100 + stage,
                                            args.width)
                row = {"vectors": count, "trial": trial, "steps_per_stage": steps, **result}
                for point in curve:
                    epoch_records.append({"vectors": count, "trial": trial, **point})
                records.append(row)
                print(row, flush=True)
    payload = {"args": vars(args), "records": records, "epoch_records": epoch_records}
    with (args.output / "records.json").open("w") as handle:
        json.dump(payload, handle, indent=2, default=str)
    with (args.output / "records.csv").open("w", newline="") as handle:
        fields = list(records[0]); writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(records)
    with (args.output / "epoch_records.csv").open("w", newline="") as handle:
        fields = list(epoch_records[0]); writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(epoch_records)
    save_epoch_plot(epoch_records, args.output)


if __name__ == "__main__":
    main()

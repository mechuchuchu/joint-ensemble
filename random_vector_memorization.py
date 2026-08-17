#!/usr/bin/env python3
"""Random-label memorization: dense vs. single, bagging, and joint ensembles.

The small member is 64 -> 16 -> 16 -> 10 (1,482 parameters). Eight members
therefore have 11,856 parameters. The dense reference is 64 -> 77 -> 77 -> 10
(11,791 parameters), matching the ensemble parameter budget within 0.6%.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallMLP(nn.Module):
    def __init__(self, width: int = 16):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(64, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 10))
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.layers(x)


class DenseMLP(nn.Module):
    def __init__(self, width: int = 77):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(64, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 10))
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.layers(x)


class VectorizedEnsemble(nn.Module):
    """K fully independent small MLPs; no hidden-layer communication."""
    def __init__(self, members: int = 8, width: int = 16):
        super().__init__(); self.members = members
        self.w1 = nn.Parameter(torch.empty(members, width, 64)); self.b1 = nn.Parameter(torch.zeros(members, width))
        self.w2 = nn.Parameter(torch.empty(members, width, width)); self.b2 = nn.Parameter(torch.zeros(members, width))
        self.w3 = nn.Parameter(torch.empty(members, 10, width)); self.b3 = nn.Parameter(torch.zeros(members, 10))
        for weight in (self.w1, self.w2, self.w3): nn.init.kaiming_uniform_(weight, a=5**0.5)
    def member_logits(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(torch.einsum("bi,khi->bkh", x, self.w1) + self.b1)
        x = F.relu(torch.einsum("bkh,koh->bko", x, self.w2) + self.b2)
        return torch.einsum("bkh,koh->bko", x, self.w3) + self.b3
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.member_logits(x).sum(1)


def train_accuracy(model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor, mode: str) -> tuple[float, float]:
    with torch.no_grad():
        if mode in {"bagging", "joint"}:
            logits = model.member_logits(inputs)
            logits = logits.mean(1) if mode == "bagging" else logits.sum(1)
        else: logits = model(inputs)
        return logits.argmax(1).eq(labels).float().mean().item() * 100, F.cross_entropy(logits, labels).item()


def run(mode: str, inputs: torch.Tensor, labels: torch.Tensor, steps: int, batch_size: int, seed: int,
        optimizer_name: str, lbfgs_max_iter: int) -> dict:
    torch.manual_seed(seed)
    if mode == "single": model: nn.Module = SmallMLP()
    elif mode == "dense": model = DenseMLP()
    else: model = VectorizedEnsemble()
    model = torch.compile(model.cuda(), mode="max-autotune-no-cudagraphs")
    optimizer = (torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0)
                 if optimizer_name == "adamw" else
                 torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=lbfgs_max_iter,
                                   history_size=100, line_search_fn="strong_wolfe",
                                   tolerance_grad=1e-7, tolerance_change=1e-9))
    members = 8
    def loss_for(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if mode == "joint": loss = F.cross_entropy(model.member_logits(x).sum(1), y)
        elif mode == "bagging":
            logits = model.member_logits(x)
            loss = F.cross_entropy(logits.transpose(1, 2), y[:, None].expand(-1, members))
        else: loss = F.cross_entropy(model(x), y)
        return loss
    for _ in range(steps):
        if optimizer_name == "adamw":
            index = torch.randint(inputs.size(0), (batch_size,), device="cuda")
            x, y = inputs[index], labels[index]
            optimizer.zero_grad(set_to_none=True); loss_for(x, y).backward(); optimizer.step()
        else:
            def closure() -> torch.Tensor:
                optimizer.zero_grad(set_to_none=True)
                loss = loss_for(inputs, labels); loss.backward()
                return loss
            optimizer.step(closure)
    accuracy, nll = train_accuracy(model, inputs, labels, mode)
    return {"train_accuracy": accuracy, "train_nll": nll, "parameters": sum(p.numel() for p in model.parameters())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", nargs="+", type=int, default=[128, 512, 2048, 8192])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--min-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--optimizer", choices=("adamw", "lbfgs"), default="adamw")
    parser.add_argument("--lbfgs-steps", type=int, default=30, help="Outer full-batch L-BFGS steps")
    parser.add_argument("--lbfgs-max-iter", type=int, default=10, help="Closure iterations per L-BFGS step")
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--output", type=Path, default=Path("random-vector-results"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for count in args.vectors:
        steps = (args.lbfgs_steps if args.optimizer == "lbfgs"
                 else max(args.min_steps, math.ceil(args.epochs * count / args.batch_size)))
        for trial in range(args.trials):
            generator = torch.Generator(device="cuda").manual_seed(args.seed + count * 100 + trial)
            inputs = torch.randn(count, 64, generator=generator, device="cuda")
            labels = torch.randint(10, (count,), generator=generator, device="cuda")
            for mode in ("single", "bagging", "joint", "dense"):
                result = run(mode, inputs, labels, steps, args.batch_size, args.seed + trial * 1000 + count,
                             args.optimizer, args.lbfgs_max_iter)
                row = {"vectors": count, "trial": trial, "steps": steps, "mode": mode, **result}; records.append(row)
                print(row, flush=True)
    with (args.output / "records.json").open("w") as handle: json.dump({"args": vars(args), "records": records}, handle, indent=2, default=str)
    with (args.output / "records.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)


if __name__ == "__main__": main()

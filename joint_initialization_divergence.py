#!/usr/bin/env python3
"""Does a joint logit ensemble diversify from nearly identical initial weights?"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Ensemble(nn.Module):
    def __init__(self, members: int = 8, noise: float | None = None):
        super().__init__(); self.members = members; width = 16
        self.w1 = nn.Parameter(torch.empty(members, width, 64)); self.b1 = nn.Parameter(torch.zeros(members, width))
        self.w2 = nn.Parameter(torch.empty(members, width, width)); self.b2 = nn.Parameter(torch.zeros(members, width))
        self.w3 = nn.Parameter(torch.empty(members, 10, width)); self.b3 = nn.Parameter(torch.zeros(members, 10))
        if noise is None:
            for weight in (self.w1, self.w2, self.w3): nn.init.kaiming_uniform_(weight, a=5**0.5)
        else:
            for parameter in self.parameters():
                base = torch.empty_like(parameter[:1])
                if parameter.ndim >= 2: nn.init.kaiming_uniform_(base, a=5**0.5)
                else: nn.init.zeros_(base)
                parameter.data.copy_(base.expand_as(parameter))
                if noise: parameter.data.add_(torch.randn_like(parameter).mul_(noise))

    def member_logits(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(torch.einsum("bi,khi->bkh", x, self.w1) + self.b1)
        x = F.relu(torch.einsum("bkh,koh->bko", x, self.w2) + self.b2)
        return torch.einsum("bkh,koh->bko", x, self.w3) + self.b3


def parameter_vectors(model: Ensemble) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(model.members, -1).float() for parameter in model.parameters()], dim=1)


@torch.no_grad()
def diagnostics(model: Ensemble, inputs: torch.Tensor, labels: torch.Tensor, condition: str, trial: int, step: int) -> dict:
    logits = model.member_logits(inputs).float(); members = logits.shape[1]
    vectors = parameter_vectors(model)
    centered = vectors - vectors.mean(0, keepdim=True)
    spread = centered.square().mean().sqrt() / (vectors.mean(0).square().mean().sqrt() + 1e-12)
    parameter_cosines, logit_correlations, disagreements = [], [], []
    predictions = logits.argmax(2)
    for left, right in itertools.combinations(range(members), 2):
        parameter_cosines.append(F.cosine_similarity(vectors[left], vectors[right], dim=0).item())
        logit_correlations.append(torch.corrcoef(torch.stack((logits[:, left].flatten(), logits[:, right].flatten())))[0, 1].item())
        disagreements.append(predictions[:, left].ne(predictions[:, right]).float().mean().item())
    aggregate = logits.sum(1)
    return {"condition": condition, "trial": trial, "step": step,
            "ensemble_accuracy": aggregate.argmax(1).eq(labels).float().mean().item() * 100,
            "member_accuracy_mean": predictions.eq(labels[:, None]).float().mean().item() * 100,
            "parameter_cosine_mean": float(np.mean(parameter_cosines)),
            "parameter_relative_spread": spread.item(),
            "logit_correlation_mean": float(np.mean(logit_correlations)),
            "prediction_disagreement_mean": float(np.mean(disagreements)) * 100}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", type=int, default=2048)
    parser.add_argument("--members", type=int, default=8)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--output", type=Path, default=Path("joint-init-divergence-results"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    records = []
    conditions: list[tuple[str, float | None]] = [("independent", None), ("noise_0", 0.0)] + [(f"noise_{eps:g}", eps) for eps in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)]
    for trial in range(args.trials):
        generator = torch.Generator(device="cuda").manual_seed(args.seed + trial)
        inputs = torch.randn(args.vectors, 64, generator=generator, device="cuda")
        labels = torch.randint(10, (args.vectors,), generator=generator, device="cuda")
        for condition_index, (name, noise) in enumerate(conditions):
            torch.manual_seed(args.seed + trial * 100 + condition_index)
            model: nn.Module = torch.compile(Ensemble(args.members, noise).cuda(), mode="max-autotune-no-cudagraphs")
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0)
            records.append(diagnostics(model._orig_mod, inputs, labels, name, trial, 0))
            for step in range(1, args.steps + 1):
                index = torch.randint(args.vectors, (args.batch_size,), device="cuda")
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model.member_logits(inputs[index]).sum(1), labels[index])
                loss.backward(); optimizer.step()
                if step % args.log_every == 0 or step == args.steps:
                    row = diagnostics(model._orig_mod, inputs, labels, name, trial, step); records.append(row); print(row, flush=True)
    with (args.output / "records.json").open("w") as handle: json.dump({"args": vars(args), "records": records}, handle, indent=2, default=str)
    with (args.output / "records.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)


if __name__ == "__main__": main()

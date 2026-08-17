#!/usr/bin/env python3
"""CIFAR-100 comparison for independently and jointly trained logit ensembles.

The validation split is used only for selecting a scalar temperature for reporting
calibrated NLL/ECE; all accuracy numbers are on the untouched CIFAR-100 test set.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = (
            nn.Sequential()
            if stride == 1 and in_planes == planes
            else nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class CifarResNet(nn.Module):
    """Small ResNet-20, fast enough to make several ensemble conditions practical."""

    def __init__(self, classes: int = 100):
        super().__init__()
        self.in_planes = 16
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16))
        self.layer1 = self._layer(16, 3, 1)
        self.layer2 = self._layer(32, 3, 2)
        self.layer3 = self._layer(64, 3, 2)
        self.fc = nn.Linear(64, classes)

    def _layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(self.in_planes, planes, stride)]
        self.in_planes = planes
        layers.extend(BasicBlock(self.in_planes, planes) for _ in range(blocks - 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.stem(x), inplace=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.avg_pool2d(x, 8).flatten(1)
        return self.fc(x)


class Ensemble(nn.Module):
    def __init__(self, members: int):
        super().__init__()
        self.members = nn.ModuleList(CifarResNet() for _ in range(members))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([model(x) for model in self.members]).sum(0)


class CIFAR100Tensors:
    """Materialize the Hugging Face split once, avoiding PIL decoding per epoch."""

    def __init__(self, split):
        self.images = torch.stack([torch.from_numpy(np.array(image.convert("RGB"), copy=True)).permute(2, 0, 1)
                                   for image in split["img"]])
        self.labels = torch.tensor(split["fine_label"], dtype=torch.long)


class CIFAR100View(Dataset):
    def __init__(self, base: CIFAR100Tensors, transform: nn.Module):
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.transform(self.base.images[index]), self.base.labels[index]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Metrics:
    accuracy: float
    nll: float
    ece: float


def metrics(logits: torch.Tensor, targets: torch.Tensor, temperature: float = 1.0) -> Metrics:
    probs = (logits / temperature).softmax(1)
    conf, pred = probs.max(1)
    correct = pred.eq(targets)
    ece = torch.zeros((), device=logits.device)
    for lo in torch.linspace(0, 0.9, 10, device=logits.device):
        mask = (conf > lo) & (conf <= lo + 0.1)
        if mask.any():
            ece += mask.float().mean() * (conf[mask].mean() - correct[mask].float().mean()).abs()
    return Metrics(correct.float().mean().item() * 100, F.cross_entropy(logits / temperature, targets).item(), ece.item() * 100)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits, all_targets = [], []
    for images, targets in loader:
        all_logits.append(model(images.to(device, non_blocking=True)).float().cpu())
        all_targets.append(targets)
    return torch.cat(all_logits), torch.cat(all_targets)


def tune_temperature(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Fit a scalar temperature on validation data without touching test labels."""
    log_t = torch.zeros((), requires_grad=True)
    optimiser = torch.optim.LBFGS([log_t], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimiser.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), targets)
        loss.backward()
        return loss

    optimiser.step(closure)
    return log_t.exp().detach().item()


@torch.no_grad()
def evaluate_members(models: list[nn.Module], loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [members, examples, classes] raw logits for ensemble diagnostics."""
    for model in models:
        model.eval()
    logits, targets = [[] for _ in models], []
    for images, batch_targets in loader:
        images = images.to(device, non_blocking=True)
        for index, model in enumerate(models):
            logits[index].append(model(images).float().cpu())
        targets.append(batch_targets)
    return torch.stack([torch.cat(member_logits) for member_logits in logits]), torch.cat(targets)


def member_diagnostics(member_logits: torch.Tensor, targets: torch.Tensor) -> dict:
    """Measure individual strength, redundancy, and each member's ensemble contribution."""
    members, examples, classes = member_logits.shape
    target_logits = member_logits.gather(2, targets.view(1, examples, 1).expand(members, -1, -1)).squeeze(2)
    non_target = member_logits.clone()
    non_target.scatter_(2, targets.view(1, examples, 1).expand(members, -1, -1), -torch.inf)
    best_wrong = non_target.max(2).values
    individual = []
    for index in range(members):
        individual.append({
            "accuracy": member_logits[index].argmax(1).eq(targets).float().mean().item() * 100,
            "mean_true_logit": target_logits[index].mean().item(),
            "mean_best_wrong_logit": best_wrong[index].mean().item(),
            "mean_margin": (target_logits[index] - best_wrong[index]).mean().item(),
            "mean_logit_l2": member_logits[index].norm(dim=1).mean().item(),
        })
    pairs = []
    for first in range(members):
        for second in range(first + 1, members):
            pairs.append({
                "members": [first, second],
                "all_logit_correlation": torch.corrcoef(torch.stack((member_logits[first].flatten(), member_logits[second].flatten())))[0, 1].item(),
                "true_logit_correlation": torch.corrcoef(torch.stack((target_logits[first], target_logits[second])))[0, 1].item(),
            })
    total_logits = member_logits.sum(0)
    leave_one_out_accuracy = [
        (total_logits - member_logits[index]).argmax(1).eq(targets).float().mean().item() * 100
        for index in range(members)
    ]
    return {"members": individual, "pairwise": pairs, "leave_one_out_accuracy": leave_one_out_accuracy}


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device,
          epochs: int, lr: float, logit_divisor: float, label: str) -> tuple[nn.Module, dict]:
    model.to(device)
    # CUDA Graphs reuse output buffers.  That is unsafe here because the independent
    # ensemble retains several compiled members' logits in the same Python step.
    # Keep Inductor compilation, but use freshly allocated outputs.
    model = torch.compile(model, mode="max-autotune-no-cudagraphs")
    optimiser = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, epochs)
    scaler = torch.amp.GradScaler("cuda")
    history = []
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, examples = 0.0, 0
        for images, targets in train_loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.cross_entropy(model(images) / logit_divisor, targets)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            loss_sum += loss.detach().item() * targets.size(0)
            examples += targets.size(0)
        scheduler.step()
        val_logits, val_targets = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "loss": loss_sum / examples, "val_accuracy": metrics(val_logits, val_targets).accuracy,
               "lr": optimiser.param_groups[0]["lr"]}
        history.append(row)
        print(f"{label:18s} epoch {epoch:02d}/{epochs}: loss={row['loss']:.4f}, val_acc={row['val_accuracy']:.2f}%", flush=True)
    return model, {"history": history, "seconds": time.time() - start}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"), help="Hugging Face datasets cache directory")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--members", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA.")
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    args.output.mkdir(parents=True, exist_ok=True)
    normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    dataset = load_dataset("uoft-cs/cifar100", cache_dir=str(args.data))
    print("Materializing CIFAR-100 images as CPU tensors...", flush=True)
    raw_train, raw_test = CIFAR100Tensors(dataset["train"]), CIFAR100Tensors(dataset["test"])
    to_float = transforms.ConvertImageDtype(torch.float32)
    train_set = CIFAR100View(raw_train, transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), to_float, normalize]))
    eval_train_set = CIFAR100View(raw_train, transforms.Compose([to_float, normalize]))
    test_set = CIFAR100View(raw_test, transforms.Compose([to_float, normalize]))
    indices = torch.randperm(len(train_set), generator=torch.Generator().manual_seed(args.seed)).tolist()
    train_indices, val_indices = indices[5000:], indices[:5000]
    loader_args = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    train_loader = DataLoader(Subset(train_set, train_indices), shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(Subset(eval_train_set, val_indices), shuffle=False, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)
    conditions: dict[str, dict] = {}

    # One member, showing the capacity available without ensembling.
    seed_everything(args.seed + 1)
    single, info = train(CifarResNet(), train_loader, val_loader, device, args.epochs, 0.1, 1.0, "single")
    conditions["single"] = {"model": single, "info": info}

    # Standard independent training, then average the logits at evaluation.
    independent = []
    independent_info = []
    for member in range(args.members):
        seed_everything(args.seed + 10 + member)
        model, info = train(CifarResNet(), train_loader, val_loader, device, args.epochs, 0.1, 1.0, f"independent-{member + 1}")
        independent.append(model)
        independent_info.append(info)
    class Independent(nn.Module):
        def __init__(self, models: list[nn.Module]):
            super().__init__(); self.models = nn.ModuleList(models)
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.stack([m(x) for m in self.models]).mean(0)
    conditions["independent_mean"] = {"model": Independent(independent).to(device), "info": {"members": independent_info}}

    # README objective: CE(sum_i f_i(x), y). Its effective temperature is one.
    seed_everything(args.seed + 30)
    joint_sum, info = train(Ensemble(args.members), train_loader, val_loader, device, args.epochs, 0.1, 1.0, "joint-sum")
    conditions["joint_sum"] = {"model": joint_sum, "info": info}

    # Same architecture and update rule, but CE(mean_i f_i(x), y), controlling logit scale.
    seed_everything(args.seed + 40)
    joint_mean, info = train(Ensemble(args.members), train_loader, val_loader, device, args.epochs, 0.1, float(args.members), "joint-mean")
    conditions["joint_mean"] = {"model": joint_mean, "info": info}

    summary = {"args": vars(args), "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "conditions": {}}
    for name, item in conditions.items():
        val_logits, val_targets = evaluate(item["model"], val_loader, device)
        test_logits, test_targets = evaluate(item["model"], test_loader, device)
        temperature = tune_temperature(val_logits, val_targets)
        raw, calibrated = metrics(test_logits, test_targets), metrics(test_logits, test_targets, temperature)
        result = {"raw": asdict(raw), "temperature": temperature, "temperature_scaled": asdict(calibrated), "training": item["info"]}
        if name == "independent_mean":
            diagnostic_models = independent
        elif name.startswith("joint_"):
            diagnostic_models = list(item["model"]._orig_mod.members)
        else:
            diagnostic_models = None
        if diagnostic_models is not None:
            raw_member_logits, diagnostic_targets = evaluate_members(diagnostic_models, test_loader, device)
            result["member_diagnostics"] = member_diagnostics(raw_member_logits, diagnostic_targets)
        summary["conditions"][name] = result
        print(f"RESULT {name}: accuracy={raw.accuracy:.2f}% NLL={raw.nll:.4f} ECE={raw.ece:.2f}% | T={temperature:.3f}, calibrated NLL={calibrated.nll:.4f}", flush=True)
    with (args.output / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    with (args.output / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "accuracy", "nll", "ece", "temperature", "calibrated_nll", "calibrated_ece"])
        writer.writeheader()
        for name, result in summary["conditions"].items():
            writer.writerow({"condition": name, **result["raw"], "temperature": result["temperature"],
                             "calibrated_nll": result["temperature_scaled"]["nll"], "calibrated_ece": result["temperature_scaled"]["ece"]})


if __name__ == "__main__":
    main()

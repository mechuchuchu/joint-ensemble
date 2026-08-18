#!/usr/bin/env python3
"""CIFAR-100: sum four ResNet-18 members at logits vs penultimate hidden.

The member axis is folded into the channel axis and every convolution is a
grouped convolution.  BatchNorm has one independent channel set per member.

Conditions:
  logit_sum  : sum(K independent backbone + classifier outputs)
  hidden_sum : sum(K independent penultimate features), then one shared head

Both conditions use one CE loss on their final output.  The hidden-sum model
has one shared classifier, so its parameter count is slightly smaller.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


class GroupBlock(nn.Module):
    expansion = 1

    def __init__(self, members: int, in_width: int, width: int, stride: int):
        super().__init__()
        self.members, self.in_width, self.width = members, in_width, width
        self.conv1 = nn.Conv2d(members * in_width, members * width, 3, stride, 1,
                               bias=False, groups=members)
        self.bn1 = nn.BatchNorm2d(members * width)
        self.conv2 = nn.Conv2d(members * width, members * width, 3, 1, 1,
                               bias=False, groups=members)
        self.bn2 = nn.BatchNorm2d(members * width)
        if stride != 1 or in_width != width:
            self.shortcut = nn.Sequential(
                nn.Conv2d(members * in_width, members * width, 1, stride,
                          bias=False, groups=members),
                nn.BatchNorm2d(members * width),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        return F.relu(y + self.shortcut(x), inplace=True)


class GroupResNet18(nn.Module):
    def __init__(self, members: int, width: int = 16, classes: int = 100,
                 aggregation: str = "logit_sum"):
        super().__init__()
        if aggregation not in {"logit_sum", "hidden_sum"}:
            raise ValueError(aggregation)
        self.members, self.width, self.classes, self.aggregation = members, width, classes, aggregation
        self.stem = nn.Sequential(
            nn.Conv2d(members * 3, members * width, 3, 1, 1, bias=False, groups=members),
            nn.BatchNorm2d(members * width), nn.ReLU(inplace=True),
        )
        self.layer1 = self._stage(width, width, 2, 1)
        self.layer2 = self._stage(width, width * 2, 2, 2)
        self.layer3 = self._stage(width * 2, width * 4, 2, 2)
        self.layer4 = self._stage(width * 4, width * 8, 2, 2)
        hidden = width * 8
        if aggregation == "logit_sum":
            self.head_weight = nn.Parameter(torch.empty(members, classes, hidden))
            self.head_bias = nn.Parameter(torch.zeros(members, classes))
            nn.init.kaiming_uniform_(self.head_weight, a=5 ** 0.5)
        else:
            self.head = nn.Linear(hidden, classes)

    def _stage(self, in_width: int, out_width: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [GroupBlock(self.members, in_width, out_width, stride)]
        layers.extend(GroupBlock(self.members, out_width, out_width, 1) for _ in range(blocks - 1))
        return nn.Sequential(*layers)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        # Replication is a view-like broadcast followed by contiguous storage,
        # making one grouped convolution process all members in one call.
        x = x[:, None].expand(-1, self.members, -1, -1, -1)
        x = x.reshape(x.shape[0], self.members * 3, x.shape[-2], x.shape[-1])
        x = self.layer4(self.layer3(self.layer2(self.layer1(self.stem(x)))))
        x = F.avg_pool2d(x, 4).flatten(1).reshape(x.shape[0], self.members, -1)
        return x

    def member_logits(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.hidden(x)
        if self.aggregation == "logit_sum":
            return torch.einsum("bkh,kch->bkc", hidden, self.head_weight) + self.head_bias
        return self.head(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        member_hidden = self.hidden(x)
        if self.aggregation == "logit_sum":
            return (torch.einsum("bkh,kch->bkc", member_hidden, self.head_weight)
                    + self.head_bias).sum(1)
        return self.head(member_hidden.sum(1))


class CIFAR100Tensors:
    def __init__(self, split):
        self.images = torch.stack([
            torch.from_numpy(np.array(image.convert("RGB"), copy=True)).permute(2, 0, 1)
            for image in split["img"]
        ])
        self.labels = torch.tensor(split["fine_label"], dtype=torch.long)


class CIFAR100View(Dataset):
    def __init__(self, base: CIFAR100Tensors, transform):
        self.base, self.transform = base, transform

    def __len__(self): return len(self.base.labels)

    def __getitem__(self, index):
        return self.transform(self.base.images[index]), self.base.labels[index]


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


@dataclass
class Metrics:
    accuracy: float
    nll: float
    ece: float


def metrics(logits, targets, temperature=1.0) -> Metrics:
    probs = (logits / temperature).softmax(1)
    conf, pred = probs.max(1); correct = pred.eq(targets)
    ece = torch.zeros(())
    for lo in torch.linspace(0, 0.9, 10):
        mask = (conf > lo) & (conf <= lo + 0.1)
        if mask.any(): ece += mask.float().mean() * (conf[mask].mean() - correct[mask].float().mean()).abs()
    return Metrics(correct.float().mean().item() * 100,
                   F.cross_entropy(logits / temperature, targets).item(), ece.item() * 100)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); logits, targets = [], []
    for images, labels in loader:
        logits.append(model(images.to(device, non_blocking=True)).float().cpu()); targets.append(labels)
    return torch.cat(logits), torch.cat(targets)


def tune_temperature(logits, targets) -> float:
    log_t = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); loss = F.cross_entropy(logits / log_t.exp(), targets); loss.backward(); return loss
    opt.step(closure); return log_t.exp().item()


def train(model, train_loader, val_loader, device, epochs, lr, label):
    model.to(device)
    compiled = torch.compile(model, mode="max-autotune-no-cudagraphs")
    opt = torch.optim.SGD(compiled.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    scaler = torch.amp.GradScaler("cuda")
    history, start = [], time.time()
    for epoch in range(1, epochs + 1):
        compiled.train(); loss_sum = examples = 0
        for images, targets in train_loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16): loss = F.cross_entropy(compiled(images), targets)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            loss_sum += loss.item() * targets.size(0); examples += targets.size(0)
        sched.step()
        val_logits, val_targets = evaluate(compiled, val_loader, device)
        row = {"epoch": epoch, "loss": loss_sum / examples,
               "val_accuracy": metrics(val_logits, val_targets).accuracy,
               "lr": opt.param_groups[0]["lr"]}
        history.append(row)
        print(f"{label:12s} epoch {epoch:02d}/{epochs}: loss={row['loss']:.4f} val_acc={row['val_accuracy']:.2f}%", flush=True)
    return compiled, {"history": history, "seconds": time.time() - start}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data")); p.add_argument("--output", type=Path, default=Path("resnet18-hidden-results"))
    p.add_argument("--epochs", type=int, default=20); p.add_argument("--members", type=int, default=4); p.add_argument("--width", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=192); p.add_argument("--workers", type=int, default=6); p.add_argument("--seed", type=int, default=20260818)
    a = p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    seed_everything(a.seed); torch.backends.cudnn.benchmark = True; device = torch.device("cuda"); a.output.mkdir(parents=True, exist_ok=True)
    normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)); ds = load_dataset("uoft-cs/cifar100", cache_dir=str(a.data))
    print("Materializing CIFAR-100 images...", flush=True)
    raw_train, raw_test = CIFAR100Tensors(ds["train"]), CIFAR100Tensors(ds["test"]); to_float = transforms.ConvertImageDtype(torch.float32)
    train_set = CIFAR100View(raw_train, transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), to_float, normalize]))
    eval_set = CIFAR100View(raw_train, transforms.Compose([to_float, normalize])); test_set = CIFAR100View(raw_test, transforms.Compose([to_float, normalize]))
    indices = torch.randperm(len(train_set), generator=torch.Generator().manual_seed(a.seed)).tolist(); train_idx, val_idx = indices[5000:], indices[:5000]
    loader_args = dict(batch_size=a.batch_size, num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0)
    train_loader = DataLoader(Subset(train_set, train_idx), shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(Subset(eval_set, val_idx), shuffle=False, **loader_args); test_loader = DataLoader(test_set, shuffle=False, **loader_args)
    conditions = {}
    for offset, aggregation in enumerate(("logit_sum", "hidden_sum")):
        seed_everything(a.seed + 100 + offset); model = GroupResNet18(a.members, a.width, aggregation=aggregation)
        params = sum(x.numel() for x in model.parameters()); print(f"{aggregation}: parameters={params:,}", flush=True)
        trained, info = train(model, train_loader, val_loader, device, a.epochs, 0.1, aggregation)
        val_logits, val_targets = evaluate(trained, val_loader, device); test_logits, test_targets = evaluate(trained, test_loader, device)
        temperature = tune_temperature(val_logits, val_targets); raw, calibrated = metrics(test_logits, test_targets), metrics(test_logits, test_targets, temperature)
        conditions[aggregation] = {"parameters": params, "raw": asdict(raw), "temperature": temperature, "temperature_scaled": asdict(calibrated), "training": info}
        print(f"RESULT {aggregation}: accuracy={raw.accuracy:.2f}% NLL={raw.nll:.4f} ECE={raw.ece:.2f}% T={temperature:.3f}", flush=True)
    summary = {"args": vars(a), "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "conditions": conditions}
    with (a.output / "summary.json").open("w") as f: json.dump(summary, f, indent=2, default=str)
    with (a.output / "summary.csv").open("w", newline="") as f:
        fields = ["condition", "parameters", "accuracy", "nll", "ece", "temperature", "calibrated_nll", "calibrated_ece"]; w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for name, r in conditions.items(): w.writerow({"condition": name, "parameters": r["parameters"], **r["raw"], "temperature": r["temperature"], "calibrated_nll": r["temperature_scaled"]["nll"], "calibrated_ece": r["temperature_scaled"]["ece"]})


if __name__ == "__main__": main()

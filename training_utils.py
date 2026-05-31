"""Reusable training building blocks for the Dagster pipeline.

These functions are extracted from the Block 1 ``train_model.py`` so the
Dagster assets in ``dagster_pipeline.py`` stay short and easy to read.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


@dataclass
class TrainConfig:
    """All knobs that affect a single training run."""
    epochs: int
    batch_size: int
    learning_rate: float
    model_name: str
    image_size: int
    val_fraction: float
    seed: int
    freeze_backbone: bool
    max_train_batches: int
    num_workers: int


class BiomassDataset(Dataset):
    """Image -> fresh_weight_total (grams) regression dataset."""

    def __init__(self, df: pd.DataFrame, image_dir: Path, transform: transforms.Compose) -> None:
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(self.image_dir / row["filename"]).convert("RGB")
        x = self.transform(img)
        y = torch.tensor(float(row["fresh_weight_total"]), dtype=torch.float32)
        return x, y


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_tf, val_tf


def build_model(model_name: str, freeze_backbone: bool) -> nn.Module:
    name_to_ctor = {
        "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
        "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
        "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
    }
    if model_name not in name_to_ctor:
        raise ValueError(f"Unsupported model_name: {model_name}")
    ctor, weights = name_to_ctor[model_name]
    net = ctor(weights=weights)

    if freeze_backbone:
        for p in net.parameters():
            p.requires_grad = False

    in_features = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(128, 1),
    )
    return net


def split_dataframe(
    df: pd.DataFrame, val_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop rows with missing target and split plant-disjoint train / val sets."""
    df = df.dropna(subset=["fresh_weight_total"]).copy()
    plant_ids = df["plant_number"].astype(str) + "_" + df["experiment_id"].astype(str)
    df["_plant_key"] = plant_ids
    unique_keys = df["_plant_key"].unique()
    train_keys, val_keys = train_test_split(
        unique_keys, test_size=val_fraction, random_state=seed
    )
    train_df = df[df["_plant_key"].isin(train_keys)].drop(columns=["_plant_key"])
    val_df = df[df["_plant_key"].isin(val_keys)].drop(columns=["_plant_key"])
    return train_df, val_df


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    max_batches: int = 0,
) -> tuple[float, float]:
    """One epoch. Returns (MSE, MAE). optimizer=None puts the model in eval."""
    is_train = optimizer is not None
    model.train(is_train)

    total_sq_err = 0.0
    total_abs_err = 0.0
    n_seen = 0

    for batch_idx, (images, targets) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            preds = model(images).squeeze(-1)
            loss = criterion(preds, targets)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_sq_err += (preds.detach() - targets).pow(2).sum().item()
        total_abs_err += (preds.detach() - targets).abs().sum().item()
        n_seen += targets.size(0)

    return total_sq_err / max(n_seen, 1), total_abs_err / max(n_seen, 1)


def predict_all(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[list[float], list[float]]:
    """Run the model on the loader and return (predictions, targets) in grams."""
    model.eval()
    preds_all: list[float] = []
    targets_all: list[float] = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            preds = model(images).squeeze(-1).cpu().tolist()
            preds_all.extend(preds)
            targets_all.extend(targets.tolist())
    return preds_all, targets_all

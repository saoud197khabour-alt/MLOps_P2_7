import base64
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from dagster import (
    AssetExecutionContext,
    Config,
    Definitions,
    MetadataValue,
    asset,
)
from dagster_mlflow import mlflow_tracking
from torch.utils.data import DataLoader

from training_utils import (
    BiomassDataset,
    build_model,
    build_transforms,
    pick_device,
    predict_all,
    run_epoch,
    split_dataframe,
)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("BIOMASS_DATA_DIR", PROJECT_DIR / "mlops_biomass_data"))
FIGURES_DIR = PROJECT_DIR / "figures"
RESULTS_DIR = PROJECT_DIR / "results"
SCREENSHOTS_DIR = PROJECT_DIR / "screenshots"
for d in (FIGURES_DIR, RESULTS_DIR, SCREENSHOTS_DIR):
    d.mkdir(parents=True, exist_ok=True)



class PreprocessConfig(Config):
    image_size: int = 224
    batch_size: int = 32
    val_fraction: float = 0.2
    seed: int = 42
    num_workers: int = 0


class TrainingConfig(Config):
    epochs: int = 2
    learning_rate: float = 1e-4
    model_name: str = "resnet18"
    freeze_backbone: bool = True
    max_train_batches: int = 15
    seed: int = 42


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _png_as_markdown(path: Path, alt: str) -> MetadataValue:
    
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return MetadataValue.md(f"![{alt}](data:image/png;base64,{data})")


@asset
def raw_dataset(context: AssetExecutionContext) -> dict:
    xlsx = DATA_DIR / "digital_biomass_labels.xlsx"
    image_dir = DATA_DIR / "images_med_res"
    context.log.info("Loading metadata from %s", xlsx)

    if not xlsx.exists():
        raise FileNotFoundError(
            f"Metadata not found at {xlsx}. Set BIOMASS_DATA_DIR or place the "
            f"dataset folder next to dagster_pipeline.py."
        )

    df = pd.read_excel(xlsx)
    image_paths = [str(image_dir / f) for f in df["filename"].tolist()]
    context.log.info("Rows: %d | image files referenced: %d", len(df), len(image_paths))

    target = df["fresh_weight_total"]
    context.add_output_metadata({
        "num_rows": MetadataValue.int(len(df)),
        "num_image_files": MetadataValue.int(len(image_paths)),
        "target_missing": MetadataValue.int(int(target.isna().sum())),
        "target_mean_g": MetadataValue.float(float(target.mean())),
        "target_std_g": MetadataValue.float(float(target.std())),
    })

    return {
        "metadata": df,
        "image_paths": image_paths,
        "image_dir": str(image_dir),
    }


@asset
def preprocessed_data(
    context: AssetExecutionContext, config: PreprocessConfig, raw_dataset: dict
) -> dict:
    _seed_everything(config.seed)

    df = raw_dataset["metadata"]
    image_dir = Path(raw_dataset["image_dir"])

    train_df, val_df = split_dataframe(df, config.val_fraction, config.seed)
    context.log.info("Train rows: %d | val rows: %d", len(train_df), len(val_df))

    train_tf, val_tf = build_transforms(config.image_size)
    train_ds = BiomassDataset(train_df, image_dir, train_tf)
    val_ds = BiomassDataset(val_df, image_dir, val_tf)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    context.add_output_metadata({
        "n_train": MetadataValue.int(len(train_ds)),
        "n_val": MetadataValue.int(len(val_ds)),
        "batch_size": MetadataValue.int(config.batch_size),
        "image_size": MetadataValue.int(config.image_size),
        "val_fraction": MetadataValue.float(config.val_fraction),
    })

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "batch_size": config.batch_size,
        "image_size": config.image_size,
        "val_fraction": config.val_fraction,
        "num_workers": config.num_workers,
        "seed": config.seed,
    }


@asset
def eda_plots(context: AssetExecutionContext, raw_dataset: dict) -> dict:
    df: pd.DataFrame = raw_dataset["metadata"]

    target = df["fresh_weight_total"].dropna()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(target, bins=40, color="seagreen", edgecolor="black")
    ax.set_title("fresh_weight_total distribution")
    ax.set_xlabel("Fresh weight [g]")
    ax.set_ylabel("Count")
    fig.tight_layout()
    target_path = FIGURES_DIR / "target_distribution.png"
    fig.savefig(target_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    sub = df.dropna(subset=["fresh_weight_total", "age_days"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(sub["age_days"], sub["fresh_weight_total"], s=12, alpha=0.5, color="steelblue")
    ax.set_xlabel("Age [days]")
    ax.set_ylabel("Fresh weight [g]")
    ax.set_title("Biomass vs plant age")
    fig.tight_layout()
    age_path = FIGURES_DIR / "age_vs_biomass.png"
    fig.savefig(age_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    context.log.info("EDA plots: %s, %s", target_path, age_path)
    context.add_output_metadata({
        "target_distribution": _png_as_markdown(target_path, "target distribution"),
        "age_vs_biomass": _png_as_markdown(age_path, "biomass vs age"),
        "target_distribution_path": MetadataValue.path(str(target_path)),
        "age_vs_biomass_path": MetadataValue.path(str(age_path)),
    })
    return {"target_distribution": str(target_path), "age_vs_biomass": str(age_path)}


@asset(required_resource_keys={"mlflow"})
def trained_model(
    context: AssetExecutionContext, config: TrainingConfig, preprocessed_data: dict
) -> dict:
    _seed_everything(config.seed)
    device = pick_device()
    context.log.info("Device: %s", device)

    train_loader: DataLoader = preprocessed_data["train_loader"]
    val_loader: DataLoader = preprocessed_data["val_loader"]

    # Combine the training config with the data-shape fields decided upstream so
    # everything stays consistent for logging, the checkpoint and evaluation.
    config_dict = {
        "epochs": config.epochs,
        "batch_size": preprocessed_data["batch_size"],
        "learning_rate": config.learning_rate,
        "model_name": config.model_name,
        "image_size": preprocessed_data["image_size"],
        "val_fraction": preprocessed_data["val_fraction"],
        "seed": config.seed,
        "freeze_backbone": config.freeze_backbone,
        "max_train_batches": config.max_train_batches,
        "num_workers": preprocessed_data["num_workers"],
    }

    model = build_model(config.model_name, config.freeze_backbone).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable, lr=config.learning_rate)
    criterion = nn.MSELoss()

    mlflow.log_params({
        "epochs": config.epochs,
        "batch_size": preprocessed_data["batch_size"],
        "learning_rate": config.learning_rate,
        "model_name": config.model_name,
        "image_size": preprocessed_data["image_size"],
        "freeze_backbone": config.freeze_backbone,
        "max_train_batches": config.max_train_batches,
        "n_train": preprocessed_data["n_train"],
        "n_val": preprocessed_data["n_val"],
    })

    history: dict[str, list[float]] = {
        "train_mse": [], "val_mse": [], "train_mae": [], "val_mae": []
    }
    best_val_mse = float("inf")

    for epoch in range(1, config.epochs + 1):
        train_mse, train_mae = run_epoch(
            model, train_loader, criterion, optimizer, device,
            max_batches=config.max_train_batches,
        )
        val_mse, val_mae = run_epoch(model, val_loader, criterion, None, device)
        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)

        mlflow.log_metric("train_mse", train_mse, step=epoch)
        mlflow.log_metric("val_mse", val_mse, step=epoch)
        mlflow.log_metric("train_mae", train_mae, step=epoch)
        mlflow.log_metric("val_mae", val_mae, step=epoch)

        context.log.info(
            "Epoch %d/%d | train MSE %.5f MAE %.5f | val MSE %.5f MAE %.5f",
            epoch, config.epochs, train_mse, train_mae, val_mse, val_mae,
        )
        best_val_mse = min(best_val_mse, val_mse)

    model_path = RESULTS_DIR / "model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config_dict,
        "best_val_mse": best_val_mse,
    }, model_path)
    mlflow.log_artifact(str(model_path), artifact_path="model")
    context.log.info("Model saved to %s", model_path)

    context.add_output_metadata({
        "best_val_mse": MetadataValue.float(best_val_mse),
        "last_train_mse": MetadataValue.float(history["train_mse"][-1]),
        "last_val_mse": MetadataValue.float(history["val_mse"][-1]),
        "last_val_mae": MetadataValue.float(history["val_mae"][-1]),
        "epochs": MetadataValue.int(config.epochs),
        "learning_rate": MetadataValue.float(config.learning_rate),
        "model_name": MetadataValue.text(config.model_name),
        "freeze_backbone": MetadataValue.bool(config.freeze_backbone),
        "model_path": MetadataValue.path(str(model_path)),
    })

    return {
        "model_state_dict": model.state_dict(),
        "history": history,
        "best_val_mse": best_val_mse,
        "config": config_dict,
    }


@asset(required_resource_keys={"mlflow"})
def model_evaluation(
    context: AssetExecutionContext,
    trained_model: dict,
    preprocessed_data: dict,
) -> dict:
    device = pick_device()
    val_loader: DataLoader = preprocessed_data["val_loader"]

    model = build_model(trained_model["config"]["model_name"], trained_model["config"]["freeze_backbone"]).to(device)
    model.load_state_dict(trained_model["model_state_dict"])

    preds, targets = predict_all(model, val_loader, device)
    preds_a = np.array(preds)
    targets_a = np.array(targets)
    val_mse = float(((preds_a - targets_a) ** 2).mean())
    val_mae = float(np.abs(preds_a - targets_a).mean())
    context.log.info("Final val MSE %.5f MAE %.5f", val_mse, val_mae)

    history = trained_model["history"]
    epochs = range(1, len(history["train_mse"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_mse"], marker="o", label="train")
    axes[0].plot(epochs, history["val_mse"], marker="o", label="val")
    axes[0].set_title("MSE by epoch"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE [g²]")
    axes[0].legend()
    axes[1].plot(epochs, history["train_mae"], marker="o", label="train")
    axes[1].plot(epochs, history["val_mae"], marker="o", label="val")
    axes[1].set_title("MAE by epoch"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("MAE [g]")
    axes[1].legend()
    fig.tight_layout()
    curves_path = RESULTS_DIR / "training_curves.png"
    fig.savefig(curves_path, dpi=130, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(targets_a, preds_a, s=14, alpha=0.55, color="#3b6e8f")
    lo = float(min(targets_a.min(), preds_a.min()))
    hi = float(max(targets_a.max(), preds_a.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax.set_xlabel("True fresh weight [g]"); ax.set_ylabel("Predicted fresh weight [g]")
    ax.set_title("Predicted vs actual (validation)"); ax.legend()
    fig.tight_layout()
    parity_path = RESULTS_DIR / "parity_plot.png"
    fig.savefig(parity_path, dpi=130, bbox_inches="tight"); plt.close(fig)

    metrics_path = RESULTS_DIR / "metrics.txt"
    metrics_path.write_text(
        "# Final evaluation metrics (validation set)\n"
        f"final_val_mse: {val_mse:.6f}\n"
        f"final_val_mae: {val_mae:.6f}\n"
        f"n_val: {len(targets_a)}\n"
        f"target_mean: {targets_a.mean():.6f}\n"
        f"target_std: {targets_a.std():.6f}\n"
        "\n# Config\n"
        + json.dumps(trained_model["config"], indent=2)
    )

    mlflow.log_metric("final_val_mse", val_mse)
    mlflow.log_metric("final_val_mae", val_mae)
    mlflow.log_artifact(str(curves_path), artifact_path="evaluation")
    mlflow.log_artifact(str(parity_path), artifact_path="evaluation")
    mlflow.log_artifact(str(metrics_path), artifact_path="evaluation")

    context.log.info("Saved %s, %s, %s", curves_path, parity_path, metrics_path)
    context.add_output_metadata({
        "final_val_mse": MetadataValue.float(val_mse),
        "final_val_mae": MetadataValue.float(val_mae),
        "n_val": MetadataValue.int(int(len(targets_a))),
        "target_mean_g": MetadataValue.float(float(targets_a.mean())),
        "target_std_g": MetadataValue.float(float(targets_a.std())),
        "training_curves": _png_as_markdown(curves_path, "training curves"),
        "parity_plot": _png_as_markdown(parity_path, "parity plot"),
        "metrics_file": MetadataValue.path(str(metrics_path)),
    })
    return {
        "val_mse": val_mse,
        "val_mae": val_mae,
        "training_curves": str(curves_path),
        "parity_plot": str(parity_path),
        "metrics_file": str(metrics_path),
    }


MLFLOW_TRACKING_URI = f"sqlite:///{(PROJECT_DIR / 'mlflow.db').as_posix()}"

defs = Definitions(
    assets=[raw_dataset, preprocessed_data, eda_plots, trained_model, model_evaluation],
    resources={
        "mlflow": mlflow_tracking.configured({
            "experiment_name": "plant_biomass_pipeline",
            "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        }),
    },
)


if __name__ == "__main__":
    import dagster as dg

    result = dg.materialize(
        [raw_dataset, preprocessed_data, eda_plots, trained_model, model_evaluation],
        resources={
            "mlflow": mlflow_tracking.configured({
                "experiment_name": "plant_biomass_pipeline",
                "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
            }),
        },
    )
    print("Success!" if result.success else "Pipeline failed.")

import csv
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def build_run_dir(base_dir, run_name=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = run_name or "baseline"
    run_dir = os.path.join(base_dir, f"{run_id}_{timestamp}")
    ensure_dir(run_dir)
    return run_dir


def save_history_csv(history, output_path):
    if not history:
        return

    fieldnames = list(history[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_metrics_json(metrics, output_path):
    serializable_metrics = {}
    for key, value in metrics.items():
        if hasattr(value, "item"):
            serializable_metrics[key] = value.item()
        else:
            serializable_metrics[key] = value

    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(serializable_metrics, json_file, indent=2)


def plot_training_curves(history, output_path, dataset_name, model_name):
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    val_auc = [row["val_auc"] for row in history]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)
    fig.suptitle(f"{model_name} on {dataset_name}", fontsize=14, fontweight="bold")

    axes[0].plot(epochs, train_loss, color="#1b4965", linewidth=2.2, label="Train loss")
    axes[0].plot(epochs, val_loss, color="#ca6702", linewidth=2.2, label="Validation loss")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Cross-Entropy")
    axes[0].legend(frameon=False)

    axes[1].plot(epochs, val_auc, color="#2a9d8f", linewidth=2.4, label="Validation AUC")
    axes[1].set_title("Validation AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(frameon=False)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

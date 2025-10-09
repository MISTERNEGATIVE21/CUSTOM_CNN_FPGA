"""Utilities for monitoring model training in real time and aggregating performance reports."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib

# Use a non-interactive backend to avoid display requirements inside Docker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class LiveTrainingMonitor(tf.keras.callbacks.Callback):
    """Keras callback that tracks metrics every epoch and saves live charts."""

    def __init__(
        self,
        model_name: str,
        output_dir: str | Path,
        metrics: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = list(metrics or ("loss", "val_loss", "accuracy", "val_accuracy"))
        self.history: Dict[str, List[float]] = {metric: [] for metric in self.metrics}
        self.epoch_durations: List[float] = []
        self._epoch_start: Optional[float] = None

    def on_train_begin(self, logs: Optional[Dict[str, float]] = None) -> None:
        for metric in self.metrics:
            self.history[metric] = []
        self.epoch_durations = []

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, float]] = None) -> None:
        self._epoch_start = time.perf_counter()

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None) -> None:
        logs = logs or {}
        for metric in self.metrics:
            value = logs.get(metric)
            self.history.setdefault(metric, []).append(float(value) if value is not None else np.nan)

        if self._epoch_start is not None:
            self.epoch_durations.append(time.perf_counter() - self._epoch_start)
            self._epoch_start = None

        self._save_plot(current_epoch=epoch + 1)

    def on_train_end(self, logs: Optional[Dict[str, float]] = None) -> None:
        self._save_plot(current_epoch=len(next(iter(self.history.values()), [])))

    def get_history(self) -> Dict[str, List[float]]:
        return {metric: values[:] for metric, values in self.history.items()}

    def _save_plot(self, current_epoch: int) -> None:
        if not self.history:
            return

        epochs = np.arange(1, current_epoch + 1)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        # Accuracy plot
        if "accuracy" in self.history:
            ax[0].plot(epochs[: len(self.history["accuracy"])], self.history["accuracy"], label="Train Acc")
        if "val_accuracy" in self.history:
            ax[0].plot(epochs[: len(self.history["val_accuracy"])], self.history["val_accuracy"], label="Val Acc")
        ax[0].set_title(f"{self.model_name} - Accuracy")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Accuracy")
        ax[0].legend()
        ax[0].grid(True, linestyle="--", alpha=0.3)

        # Loss plot
        if "loss" in self.history:
            ax[1].plot(epochs[: len(self.history["loss"])], self.history["loss"], label="Train Loss")
        if "val_loss" in self.history:
            ax[1].plot(epochs[: len(self.history["val_loss"])], self.history["val_loss"], label="Val Loss")
        ax[1].set_title(f"{self.model_name} - Loss")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Loss")
        ax[1].legend()
        ax[1].grid(True, linestyle="--", alpha=0.3)

        plt.tight_layout()
        plot_path = self.output_dir / f"{self.model_name}_live.png"
        fig.savefig(plot_path)
        plt.close(fig)


class PerformanceAnalyzer:
    """Persists per-model metrics and produces aggregate comparison charts."""

    def __init__(self, results_dir: str | Path = "results") -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.results_dir / "model_performance_summary.json"

    def record_model(
        self,
        model_name: str,
        history: Dict[str, List[float]],
        val_accuracy: float,
        epoch_durations: Optional[List[float]] = None,
    ) -> None:
        summary = self._load_summary()
        summary[model_name] = {
            "val_accuracy": float(val_accuracy),
            "final_train_accuracy": float(history.get("accuracy", [np.nan])[-1] if history.get("accuracy") else np.nan),
            "final_val_accuracy": float(history.get("val_accuracy", [np.nan])[-1] if history.get("val_accuracy") else np.nan),
            "final_train_loss": float(history.get("loss", [np.nan])[-1] if history.get("loss") else np.nan),
            "final_val_loss": float(history.get("val_loss", [np.nan])[-1] if history.get("val_loss") else np.nan),
            "epochs_trained": int(max(len(values) for values in history.values() if values)),
            "epoch_durations_sec": epoch_durations or [],
        }
        self._save_summary(summary)

    def generate_overview_plot(self) -> Optional[Path]:
        summary = self._load_summary()
        if not summary:
            return None

        names = list(summary.keys())
        accuracies = [summary[name].get("val_accuracy", 0.0) for name in names]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(names, accuracies, color="#4c51bf")
        ax.set_xlabel("Model")
        ax.set_ylabel("Validation Accuracy")
        ax.set_title("Model Validation Accuracy Comparison")
        ax.set_ylim(0, max(accuracies + [1.0]))
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{acc:.2f}", ha="center", va="bottom")

        plt.tight_layout()
        output_path = self.results_dir / "model_accuracy_overview.png"
        fig.savefig(output_path)
        plt.close(fig)
        return output_path

    def _load_summary(self) -> Dict[str, Dict[str, float]]:
        if self.summary_path.exists():
            try:
                with self.summary_path.open("r", encoding="utf-8") as fp:
                    return json.load(fp)
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_summary(self, summary: Dict[str, Dict[str, float]]) -> None:
        with self.summary_path.open("w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)
#!/usr/bin/env python3
"""
Generate publication-ready plots for the YOLO inference-framework paper.

Input files (place beside this script):
  - benchmark_results.csv
  - validation_results.csv

Outputs:
  figures/fig1_accuracy.png and .pdf
  figures/fig2_throughput.png and .pdf
  figures/fig3_latency.png and .pdf
  figures/fig4_throughput_vs_vram.png and .pdf
  figures/fig5_throughput_vs_gpu_utilization.png and .pdf

Batch-1 benchmark rows are excluded from all benchmark plots.
Validation metrics are plotted exactly as present in validation_results.csv.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
BENCHMARK_CSV = BASE_DIR / "benchmark_results.csv"
VALIDATION_CSV = BASE_DIR / "validation_results.csv"
OUTPUT_DIR = BASE_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Consistent publication settings; matplotlib's default color cycle is retained.
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save_figure(fig, stem):
    """Save each figure as high-resolution PNG and vector PDF."""
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem}.png and {stem}.pdf")


def load_data():
    if not BENCHMARK_CSV.exists():
        raise FileNotFoundError(f"Could not find {BENCHMARK_CSV}")
    if not VALIDATION_CSV.exists():
        raise FileNotFoundError(f"Could not find {VALIDATION_CSV}")

    benchmark = pd.read_csv(BENCHMARK_CSV)
    validation = pd.read_csv(VALIDATION_CSV)

    required_bench = {
        "Framework", "Batch", "FPS", "Latency_ms", "VRAM_MB",
        "GPU_Utilization_%"
    }
    required_val = {"Framework", "mAP50", "mAP50-95", "Precision", "Recall"}

    missing_bench = required_bench - set(benchmark.columns)
    missing_val = required_val - set(validation.columns)
    if missing_bench:
        raise ValueError(f"Missing benchmark columns: {sorted(missing_bench)}")
    if missing_val:
        raise ValueError(f"Missing validation columns: {sorted(missing_val)}")

    # The main paper comparison is Batch 8 only; remove all Batch-1 records.
    benchmark = benchmark.loc[benchmark["Batch"] == 8].copy()

    display_names = {
        "PyTorch_FP32_B8": "PyTorch FP32",
        "PyTorch_FP16_B8": "PyTorch FP16",
        "ONNX_FP32_B8": "ONNX Runtime",
        "TensorRT_FP16_B8": "TensorRT FP16",
        "PyTorch": "PyTorch FP32",
        "ONNX": "ONNX Runtime",
        "TensorRT": "TensorRT FP16",
    }
    benchmark["Framework"] = benchmark["Framework"].map(
        lambda x: display_names.get(str(x), str(x))
    )
    validation["Framework"] = validation["Framework"].map(
        lambda x: display_names.get(str(x), str(x))
    )

    order = ["PyTorch FP32", "PyTorch FP16", "ONNX Runtime", "TensorRT FP16"]
    benchmark["Framework"] = pd.Categorical(
        benchmark["Framework"], categories=order, ordered=True
    )
    benchmark = benchmark.sort_values("Framework")

    val_order = ["PyTorch FP32", "PyTorch FP16", "ONNX Runtime", "TensorRT FP16"]
    validation["Framework"] = pd.Categorical(
        validation["Framework"], categories=val_order, ordered=True
    )
    validation = validation.sort_values("Framework")

    # F1 is derived from the precision and recall values in the CSV.
    validation["F1"] = (
        2 * validation["Precision"] * validation["Recall"]
        / (validation["Precision"] + validation["Recall"])
    )
    return benchmark, validation


def fig1_accuracy(validation):
    """2x2 metric panels; each y-axis is locally scaled and clearly labelled."""
    metrics = [
        ("mAP50", "mAP@0.5"),
        ("mAP50-95", "mAP@0.5:0.95"),
        ("Precision", "Precision"),
        ("Recall", "Recall"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.7))
    axes = axes.ravel()

    for ax, (column, title) in zip(axes, metrics):
        vals = validation[column].astype(float).to_numpy()
        labels = validation["Framework"].astype(str).tolist()
        bars = ax.bar(labels, vals, width=0.66)
        ax.set_title(title)
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=20)
        # Zoom only enough to make small differences legible; label the scale.
        lo, hi = float(np.min(vals)), float(np.max(vals))
        pad = max((hi - lo) * 0.7, 0.0015)
        ax.set_ylim(max(0, lo - pad), min(1, hi + pad))
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, vals):
            ax.annotate(
                f"{value:.4f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=8
            )

    fig.suptitle("Detection accuracy across deployment frameworks", y=1.01)
    fig.tight_layout()
    save_figure(fig, "fig1_accuracy")


def fig2_throughput(benchmark):
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    labels = benchmark["Framework"].astype(str).tolist()
    vals = benchmark["FPS"].astype(float).to_numpy()
    bars = ax.bar(labels, vals, width=0.62)
    ax.set_title("Batch-8 inference throughput")
    ax.set_ylabel("Throughput (frames/s)")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=15)
    for bar, value in zip(bars, vals):
        ax.annotate(f"{value:.2f}", (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    fig.tight_layout()
    save_figure(fig, "fig2_throughput")


def fig3_latency(benchmark):
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    labels = benchmark["Framework"].astype(str).tolist()
    vals = benchmark["Latency_ms"].astype(float).to_numpy()
    bars = ax.bar(labels, vals, width=0.62)
    ax.set_title("Batch-8 inference latency")
    ax.set_ylabel("Average latency (ms)")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=15)
    for bar, value in zip(bars, vals):
        ax.annotate(f"{value:.2f}", (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    fig.tight_layout()
    save_figure(fig, "fig3_latency")


def annotate_points(ax, df, x_col, y_col, offsets=None):
    offsets = offsets or {}
    for _, row in df.iterrows():
        label = str(row["Framework"])
        dx, dy = offsets.get(label, (5, 5))
        ax.annotate(
            label, (float(row[x_col]), float(row[y_col])),
            xytext=(dx, dy), textcoords="offset points", fontsize=8
        )


def fig4_throughput_vs_vram(benchmark):
    fig, ax = plt.subplots(figsize=(7.0, 4.7))
    ax.scatter(benchmark["VRAM_MB"], benchmark["FPS"], s=75)
    annotate_points(
        ax, benchmark, "VRAM_MB", "FPS",
        offsets={
            "PyTorch FP32": (6, 6),
            "PyTorch FP16": (6, 6),
            "ONNX Runtime": (6, -12),
            "TensorRT FP16": (6, 6),
        }
    )
    ax.set_title("Throughput–memory trade-off (Batch 8)")
    ax.set_xlabel("GPU memory usage (MB)")
    ax.set_ylabel("Throughput (frames/s)")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, "fig4_throughput_vs_vram")


def fig5_throughput_vs_gpu_utilization(benchmark):
    fig, ax = plt.subplots(figsize=(7.0, 4.7))
    ax.scatter(benchmark["GPU_Utilization_%"], benchmark["FPS"], s=75)
    annotate_points(
        ax, benchmark, "GPU_Utilization_%", "FPS",
        offsets={
            "PyTorch FP32": (6, 6),
            "PyTorch FP16": (6, 6),
            "ONNX Runtime": (6, -12),
            "TensorRT FP16": (6, 6),
        }
    )
    ax.set_title("Throughput versus average GPU utilization (Batch 8)")
    ax.set_xlabel("Average GPU utilization (%)")
    ax.set_ylabel("Throughput (frames/s)")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, "fig5_throughput_vs_gpu_utilization")


def main():
    benchmark, validation = load_data()
    print(f"Benchmark rows used (Batch 8 only): {len(benchmark)}")
    print(f"Validation rows used: {len(validation)}")
    fig1_accuracy(validation)
    fig2_throughput(benchmark)
    fig3_latency(benchmark)
    fig4_throughput_vs_vram(benchmark)
    fig5_throughput_vs_gpu_utilization(benchmark)
    print(f"\nAll figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

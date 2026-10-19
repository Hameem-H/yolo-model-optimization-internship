
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================
# CREATE PLOTS DIRECTORY
# ============================================

Path("plots").mkdir(exist_ok=True)

# ============================================
# LOAD CSV FILES
# ============================================

benchmark_df = pd.read_csv("results/benchmark_results.csv")
validation_df = pd.read_csv("results/validation_results.csv")

# ============================================
# GLOBAL STYLE
# ============================================

plt.style.use("default")

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.edgecolor": "#DDDDDD",
    "axes.linewidth": 1,
    "grid.color": "#EAEAEA",
    "grid.linestyle": "--",
    "grid.linewidth": 0.8,
    "font.family": "DejaVu Sans"
})

# ============================================
# COLORS
# ============================================

BASE_COLOR = "#5E6472"
ACCENT_COLOR = "#2A9D8F"

def get_colors(experiments):

    colors = []

    for exp in experiments:

        if "TensorRT" in exp:
            colors.append(ACCENT_COLOR)
        else:
            colors.append(BASE_COLOR)

    return colors

# ============================================
# GENERIC BAR PLOT FUNCTION
# ============================================

def create_bar_plot(
    x,
    y,
    ylabel,
    title,
    filename
):

    colors = get_colors(x)

    plt.figure()

    bars = plt.bar(
        x,
        y,
        color=colors,
        edgecolor="none"
    )

    plt.grid(axis="y")
    plt.gca().set_axisbelow(True)

    plt.ylabel(ylabel)
    plt.title(title, pad=15)

    plt.xticks(rotation=18)

    # value labels
    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width()/2,
            height,
            f"{height:.2f}",
            ha='center',
            va='bottom',
            fontsize=10
        )

    plt.tight_layout()

    plt.savefig(
        f"plots/{filename}",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ============================================
# 1. FPS COMPARISON
# ============================================

create_bar_plot(
    benchmark_df["Experiment"],
    benchmark_df["FPS"],
    "Frames Per Second (FPS)",
    "Inference Throughput Comparison",
    "fps_comparison.png"
)

# ============================================
# 2. LATENCY COMPARISON
# ============================================

create_bar_plot(
    benchmark_df["Experiment"],
    benchmark_df["Latency_ms"],
    "Latency (ms)",
    "Inference Latency Comparison",
    "latency_comparison.png"
)

# ============================================
# 3. VRAM COMPARISON
# ============================================

create_bar_plot(
    benchmark_df["Experiment"],
    benchmark_df["VRAM_MB"],
    "VRAM Usage (MB)",
    "GPU Memory Usage Comparison",
    "vram_comparison.png"
)

# ============================================
# 4. GPU UTILIZATION
# ============================================

create_bar_plot(
    benchmark_df["Experiment"],
    benchmark_df["GPU_Utilization_%"],
    "GPU Utilization (%)",
    "GPU Utilization Comparison",
    "gpu_utilization.png"
)

# ============================================
# 5. mAP50 COMPARISON
# ============================================

map_colors = [
    BASE_COLOR,
    BASE_COLOR,
    ACCENT_COLOR
]

plt.figure()

bars = plt.bar(
    validation_df["Model"],
    validation_df["mAP50"],
    color=map_colors,
    edgecolor="none"
)

plt.grid(axis="y")
plt.gca().set_axisbelow(True)

plt.ylabel("mAP50")
plt.title("Detection Accuracy Preservation", pad=15)

for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{height:.4f}",
        ha='center',
        va='bottom',
        fontsize=10
    )

plt.tight_layout()

plt.savefig(
    "plots/map50_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================
# 6. SPEEDUP COMPARISON
# ============================================

baseline = benchmark_df[
    benchmark_df["Experiment"] == "PyTorch_FP32_B1"
]["FPS"].values[0]

benchmark_df["Speedup"] = (
    benchmark_df["FPS"] / baseline
)

create_bar_plot(
    benchmark_df["Experiment"],
    benchmark_df["Speedup"],
    "Relative Speedup (x)",
    "Speedup Over Baseline",
    "speedup_comparison.png"
)

# ============================================
# 7. VRAM vs FPS SCATTER
# ============================================

plt.figure()

colors = get_colors(benchmark_df["Experiment"])

plt.scatter(
    benchmark_df["VRAM_MB"],
    benchmark_df["FPS"],
    s=120,
    c=colors
)

for i, row in benchmark_df.iterrows():

    plt.text(
        row["VRAM_MB"] + 8,
        row["FPS"],
        row["Experiment"],
        fontsize=9
    )

plt.grid(True)

plt.xlabel("VRAM Usage (MB)")
plt.ylabel("FPS")

plt.title("VRAM vs Throughput Tradeoff", pad=15)

plt.tight_layout()

plt.savefig(
    "plots/vram_vs_fps.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================
# DONE
# ============================================

print("\nPublication-quality plots generated successfully!")
print("Saved inside: plots/")

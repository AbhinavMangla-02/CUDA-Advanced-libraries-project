#!/usr/bin/env python3
"""
plot_results.py — Visualize benchmark results from results/benchmark_data.csv

Produces:
  results/plot_batch_scaling.png   — CPU vs GPU time vs batch size
  results/plot_resolution.png      — CPU vs GPU time vs image resolution
  results/plot_speedup.png         — Combined speedup curves
  results/plot_transfer.png        — Memory transfer breakdown
"""

import csv, os, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

CSV_PATH    = "results/benchmark_data.csv"
OUTPUT_DIR  = "results"
STYLE = {
    "cpu":    dict(color="#E05C5C", lw=2.2, marker="o", ms=6, label="CPU (single-thread)"),
    "gpu_t":  dict(color="#4A90D9", lw=2.2, marker="s", ms=6, label="GPU (kernel + transfer)"),
    "gpu_k":  dict(color="#50C878", lw=2.2, marker="^", ms=6, linestyle="--", label="GPU (kernel only)"),
    "speedup_t": dict(color="#4A90D9", lw=2.2, marker="o", ms=6),
    "speedup_k": dict(color="#50C878", lw=2.2, marker="^", ms=6, linestyle="--"),
}

def load_csv():
    rows = []
    if not os.path.exists(CSV_PATH):
        print(f"[WARN] {CSV_PATH} not found — run the benchmark first.")
        return rows
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "mode":       r["mode"],
                "n":          int(r["n_images"]),
                "w":          int(r["width"]),
                "h":          int(r["height"]),
                "cpu":        float(r["cpu_ms"]),
                "gpu_total":  float(r["gpu_total_ms"]),
                "gpu_kernel": float(r["gpu_kernel_ms"]),
                "speedup":    float(r["speedup"]),
            })
    return rows

# ─────────────────────────────────────────────────────────────────────────────

def plot_batch_scaling(rows):
    data = [r for r in rows if r["mode"] == "batch_scaling"]
    if not data:
        return
    data.sort(key=lambda r: r["n"])

    ns  = [r["n"]          for r in data]
    cpu = [r["cpu"]        for r in data]
    gpt = [r["gpu_total"]  for r in data]
    gpk = [r["gpu_kernel"] for r in data]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Batch-Size Scaling (512×512 images)", fontsize=14, fontweight="bold")

    # Left: absolute time
    ax = axes[0]
    ax.plot(ns, cpu, **STYLE["cpu"])
    ax.plot(ns, gpt, **STYLE["gpu_t"])
    ax.plot(ns, gpk, **STYLE["gpu_k"])
    ax.set_xlabel("Batch size (# images)")
    ax.set_ylabel("Time (ms)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("Wall-clock time vs Batch Size")
    ax.grid(True, which="both", alpha=0.3)

    # Right: speedup
    ax = axes[1]
    su_t = [r["speedup"]               for r in data]
    su_k = [r["cpu"]/r["gpu_kernel"]   for r in data]
    ax.plot(ns, su_t, **STYLE["speedup_t"], label="vs GPU+Transfer")
    ax.plot(ns, su_k, **STYLE["speedup_k"], label="vs GPU Kernel")
    ax.axhline(1, color="gray", lw=1, linestyle=":")
    ax.set_xlabel("Batch size (# images)")
    ax.set_ylabel("Speedup (×)")
    ax.set_xscale("log")
    ax.legend()
    ax.set_title("CPU/GPU Speedup vs Batch Size")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "plot_batch_scaling.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)

def plot_resolution(rows):
    data = [r for r in rows if r["mode"] == "res_scaling"]
    if not data:
        return
    data.sort(key=lambda r: r["w"] * r["h"])

    labels = [f"{r['w']}×{r['h']}" for r in data]
    x      = np.arange(len(labels))
    cpu    = [r["cpu"]        for r in data]
    gpt    = [r["gpu_total"]  for r in data]
    gpk    = [r["gpu_kernel"] for r in data]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Resolution Scaling (50 images per test)", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(x, cpu, **STYLE["cpu"])
    ax.plot(x, gpt, **STYLE["gpu_t"])
    ax.plot(x, gpk, **STYLE["gpu_k"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Wall-clock Time vs Resolution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    su = [r["cpu"]/r["gpu_total"] for r in data]
    ax.bar(x, su, color="#4A90D9", alpha=0.8, width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Speedup (×)")
    ax.set_title("GPU Speedup (incl. transfer) vs Resolution")
    ax.axhline(1, color="red", lw=1, linestyle="--")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "plot_resolution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)

def plot_speedup_combined(rows):
    batch_rows = sorted([r for r in rows if r["mode"] == "batch_scaling"], key=lambda r: r["n"])
    if not batch_rows:
        return

    ns   = [r["n"]                      for r in batch_rows]
    su_t = [r["speedup"]                for r in batch_rows]
    su_k = [r["cpu"] / r["gpu_kernel"]  for r in batch_rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(ns, su_t, 1, alpha=0.12, color="#4A90D9")
    ax.fill_between(ns, su_k, su_t, alpha=0.12, color="#50C878")
    ax.plot(ns, su_t, **STYLE["speedup_t"], label="GPU total speedup (incl. transfer)")
    ax.plot(ns, su_k, **STYLE["speedup_k"], label="GPU kernel speedup (no transfer)")
    ax.axhline(1, color="gray", lw=1.2, linestyle=":", label="Break-even (1×)")
    ax.set_xlabel("Batch size (# images)", fontsize=12)
    ax.set_ylabel("Speedup over single-thread CPU", fontsize=12)
    ax.set_xscale("log")
    ax.set_title("GPU Speedup vs Batch Size\n(512×512 images, shared-mem Sobel kernel)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    # Annotate peak speedup
    peak_k = max(zip(su_k, ns), key=lambda t: t[0])
    ax.annotate(f"Peak kernel\nspeedup: {peak_k[0]:.1f}×",
                xy=(peak_k[1], peak_k[0]),
                xytext=(peak_k[1]*0.3, peak_k[0]*0.9),
                arrowprops=dict(arrowstyle="->", color="green"),
                fontsize=9, color="green")

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "plot_speedup.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = load_csv()

    if not rows:
        print("No data to plot.  Run the benchmark first:")
        print("  ./edge_detect --benchmark")
        return

    print("Generating plots...")
    plot_batch_scaling(rows)
    plot_resolution(rows)
    plot_speedup_combined(rows)
    print("Done.")

if __name__ == "__main__":
    main()

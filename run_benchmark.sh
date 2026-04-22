#!/usr/bin/env bash
# =============================================================================
# run_benchmark.sh — End-to-end benchmark pipeline
# =============================================================================
set -euo pipefail

BINARY=./edge_detect
INPUT=input_images
OUTPUT=output_edges
RESULTS=results

echo "╔══════════════════════════════════════════════════════╗"
echo "║  CUDA Batch Edge Detection — Full Benchmark Pipeline  ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Build ─────────────────────────────────────────────────────────────
echo "[1/4] Building..."
make -j"$(nproc)" 2>&1
echo ""

# ── Step 2: Generate test images ──────────────────────────────────────────────
if ls $INPUT/*.pgm &>/dev/null; then
    echo "[2/4] Test images already present ($(ls $INPUT/*.pgm | wc -l) files) — skipping generation"
else
    echo "[2/4] Generating 200 synthetic test images..."
    python3 scripts/generate_images.py
fi
echo ""

# ── Step 3: Full benchmark suite ──────────────────────────────────────────────
echo "[3/4] Running benchmark suite..."
mkdir -p $RESULTS
$BINARY --benchmark 2>&1 | tee $RESULTS/benchmark_results.txt
echo ""

# ── Step 4: Process real image batch ──────────────────────────────────────────
echo "[4/4] Processing 200 real images (CPU + GPU, saving outputs)..."
$BINARY --process $INPUT $OUTPUT 200
echo ""

# ── Summary ───────────────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════╗"
echo "║  DONE                                                 ║"
echo "║  Benchmark log : results/benchmark_results.txt       ║"
echo "║  CSV data      : results/benchmark_data.csv          ║"
echo "║  Edge images   : output_edges/                       ║"
echo "║                                                       ║"
echo "║  Plot results : python3 scripts/plot_results.py      ║"
echo "╚══════════════════════════════════════════════════════╝"

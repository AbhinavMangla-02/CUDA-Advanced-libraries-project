# =============================================================================
# Makefile — CUDA Batch Sobel Edge Detection
# =============================================================================
#
# Requirements:
#   - CUDA Toolkit ≥ 11.0  (nvcc in PATH)
#   - GCC ≥ 9 / Clang ≥ 10 with C++17 support
#
# Usage:
#   make              # build
#   make benchmark    # build + run full benchmark suite
#   make process      # process images in input_images/ → output_edges/
#   make clean        # remove build artifacts
#   make generate     # generate synthetic test images (needs Python + pillow)
# =============================================================================

NVCC     := nvcc
CXX      := g++
TARGET   := edge_detect
BUILD    := build

# ── CUDA architecture ─────────────────────────────────────────────────────────
# Adjust sm_XX to match your GPU:
#   RTX 30xx / A-series → sm_86    RTX 20xx → sm_75
#   RTX 40xx            → sm_89    GTX 10xx → sm_61
#   V100                → sm_70    A100     → sm_80
GPU_ARCH := sm_75

# ── Source files ───────────────────────────────────────────────────────────────
CUDA_SRCS := src/main.cu src/gpu_edge.cu
CXX_SRCS  := src/cpu_edge.cpp src/image_utils.cpp

# ── Compiler flags ────────────────────────────────────────────────────────────
NVCC_FLAGS := \
    -O3 \
    -arch=$(GPU_ARCH) \
    -std=c++17 \
    --generate-line-info \
    --compiler-options "-O3 -march=native -Wall" \
    -Xptxas -v

LDFLAGS := -lm

# ─────────────────────────────────────────────────────────────────────────────

.PHONY: all clean dirs benchmark process generate plot help

all: dirs $(TARGET)

dirs:
	@mkdir -p $(BUILD) input_images output_edges results samples

$(TARGET): $(CUDA_SRCS) $(CXX_SRCS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)
	@echo ""
	@echo "✓  Build successful → $(TARGET)"
	@echo ""

# ── High-level targets ────────────────────────────────────────────────────────

generate:
	@echo "Generating synthetic test images..."
	python3 scripts/generate_images.py

benchmark: $(TARGET)
	@mkdir -p results
	@echo "Running benchmark suite (this may take 1–2 minutes)..."
	./$(TARGET) --benchmark 2>&1 | tee results/benchmark_results.txt

process: $(TARGET)
	@[ -n "$$(ls input_images/*.pgm 2>/dev/null)" ] || \
	    (echo "No images in input_images/ — running: make generate first"; make generate)
	./$(TARGET) --process input_images output_edges

plot:
	python3 scripts/plot_results.py

run_all: generate all benchmark process plot
	@echo ""
	@echo "╔══════════════════════════════════════════╗"
	@echo "║  All done!  Check:                       ║"
	@echo "║    results/  — benchmark data & plots     ║"
	@echo "║    output_edges/ — processed edge images  ║"
	@echo "╚══════════════════════════════════════════╝"

clean:
	rm -rf $(BUILD) $(TARGET) output_edges
	@echo "Cleaned build artifacts."

help:
	@echo "Targets: all  benchmark  process  generate  plot  run_all  clean"

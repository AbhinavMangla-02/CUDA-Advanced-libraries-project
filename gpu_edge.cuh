#pragma once
// =============================================================================
// gpu_edge.cuh — GPU Sobel Edge Detection: Public API
// =============================================================================

#include "image_utils.h"
#include <vector>

// ─── Result struct from GPU benchmark run ────────────────────────────────────

struct GPUBenchmarkResult {
    double time_with_transfer_ms;   // H2D + kernel + D2H
    double time_kernel_only_ms;     // pure kernel execution
    double time_h2d_ms;             // host → device copy
    double time_d2h_ms;             // device → host copy
    double throughput_mpix_per_sec; // megapixels processed per second (kernel only)
    size_t total_pixels;
    int    num_images;
};

// ─── Process a batch of real loaded images ───────────────────────────────────

void sobelBatchGPU(const std::vector<Image>& inputs,
                   std::vector<Image>&       outputs);

// ─── Benchmark helpers ───────────────────────────────────────────────────────

// Benchmark on already-loaded images
GPUBenchmarkResult benchmarkGPU(const std::vector<Image>& images, int runs = 3);

// Benchmark on synthetic (randomly-generated) data — avoids file I/O cost
GPUBenchmarkResult benchmarkGPUSynthetic(int width, int height,
                                         int num_images, int runs = 3);

// CPU synthetic benchmark (for fair comparison in scaling tests)
double benchmarkCPUSynthetic(int width, int height, int num_images);

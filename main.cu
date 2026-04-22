// =============================================================================
// main.cu — CUDA Batch Sobel Edge Detection: Benchmark Driver
// =============================================================================
// Capstone Project: GPU-accelerated edge detection for large image batches
// Demonstrates CPU vs GPU speedup, scaling with batch size and resolution.
// =============================================================================

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

#include "gpu_edge.cuh"
#include "cpu_edge.h"
#include "image_utils.h"

namespace fs = std::filesystem;

// ─── Helpers ─────────────────────────────────────────────────────────────────

void printBanner() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║   CUDA Batch Sobel Edge Detection — Performance Benchmark Suite     ║\n";
    std::cout << "║   Capstone Project · GPU Programming Course                        ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
}

void printSep(char c = '─', int w = 72) {
    std::cout << std::string(w, c) << "\n";
}

void printGPUInfo() {
    int devCount = 0;
    cudaGetDeviceCount(&devCount);
    if (devCount == 0) {
        std::cerr << "[ERROR] No CUDA-capable GPU found!\n";
        std::exit(1);
    }
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    std::cout << "[GPU]  " << p.name << "\n";
    std::cout << "[GPU]  SMs: " << p.multiProcessorCount
              << "  |  Global Mem: " << p.totalGlobalMem / (1024 * 1024) << " MB"
              << "  |  CUDA " << p.major << "." << p.minor << "\n";
    std::cout << "[GPU]  Max threads/block: " << p.maxThreadsPerBlock
              << "  |  Warp size: " << p.warpSize << "\n";
    std::cout << "\n";
}

// ─── Mode: Process real images ─────────────────────────────────────────────

void runImageProcessing(const std::string& input_dir,
                        const std::string& output_dir,
                        int batch_limit)
{
    printSep('═');
    std::cout << "[MODE] Real-Image Processing\n";
    std::cout << "[SRC]  " << input_dir << "\n";
    printSep();

    auto images = loadImageBatch(input_dir, batch_limit);
    if (images.empty()) {
        std::cerr << "[WARN] No .pgm files found in: " << input_dir << "\n";
        std::cerr << "       Run: python3 scripts/generate_images.py  first.\n";
        return;
    }

    int N  = (int)images.size();
    int W  = images[0].width;
    int H  = images[0].height;
    size_t total_px = (size_t)N * W * H;

    std::cout << "[INFO] Loaded " << N << " images  (" << W << "×" << H << ")\n";
    std::cout << "[INFO] Total pixels: " << total_px / 1'000'000.0 << " M\n\n";

    // ── CPU ──────────────────────────────────────────────────────────────
    std::cout << "Running CPU baseline (3 runs avg)...\n";
    double cpu_ms = benchmarkCPU(images, 3);

    // ── GPU ──────────────────────────────────────────────────────────────
    std::cout << "Running GPU benchmark   (3 runs avg)...\n";
    GPUBenchmarkResult gr = benchmarkGPU(images, 3);

    // ── Print results ────────────────────────────────────────────────────
    std::cout << "\n";
    printSep();
    std::cout << std::left;
    std::cout << std::setw(36) << "Metric" << std::setw(16) << "CPU" << "GPU\n";
    printSep();

    auto row = [&](const std::string& label, double cpu_v, double gpu_v, const std::string& unit) {
        std::cout << std::setw(36) << label
                  << std::setw(16) << (std::to_string((int)cpu_v) + " " + unit)
                  << (std::to_string((int)gpu_v) + " " + unit) << "\n";
    };

    std::cout << std::fixed << std::setprecision(2);
    std::cout << std::setw(36) << "Total wall time (ms)"
              << std::setw(16) << cpu_ms
              << gr.time_with_transfer_ms << "\n";
    std::cout << std::setw(36) << "Kernel-only (ms)"
              << std::setw(16) << "—"
              << gr.time_kernel_only_ms << "\n";
    std::cout << std::setw(36) << "H2D transfer (ms)"
              << std::setw(16) << "—"
              << gr.time_h2d_ms << "\n";
    std::cout << std::setw(36) << "D2H transfer (ms)"
              << std::setw(16) << "—"
              << gr.time_d2h_ms << "\n";
    std::cout << std::setw(36) << "Speedup (incl. transfer)"
              << std::setw(16) << "1.00×"
              << (cpu_ms / gr.time_with_transfer_ms) << "×\n";
    std::cout << std::setw(36) << "Speedup (kernel only)"
              << std::setw(16) << "1.00×"
              << (cpu_ms / gr.time_kernel_only_ms) << "×\n";
    std::cout << std::setw(36) << "Throughput (MPix/s)"
              << std::setw(16) << std::setprecision(1) << (total_px / 1e6) / (cpu_ms / 1e3)
              << (total_px / 1e6) / (gr.time_kernel_only_ms / 1e3) << "\n";
    printSep();

    // ── Save output images ────────────────────────────────────────────────
    std::vector<Image> gpu_outputs;
    sobelBatchGPU(images, gpu_outputs);
    fs::create_directories(output_dir);
    saveImageBatch(gpu_outputs, output_dir, "edge_");
    std::cout << "\n[OUT]  Saved " << gpu_outputs.size()
              << " edge images → " << output_dir << "/\n";
}

// ─── Mode: Batch-size scaling ──────────────────────────────────────────────

void runBatchScaling(std::ofstream& csv) {
    printSep('═');
    std::cout << "[BENCH] Batch-Size Scaling  (512×512 images, 3-run avg)\n";
    printSep();

    const int W = 512, H = 512;
    std::vector<int> batches = {1, 5, 10, 25, 50, 100, 200, 500, 1000};

    std::cout << std::left
              << std::setw(8)  << "Batch"
              << std::setw(14) << "CPU (ms)"
              << std::setw(16) << "GPU+Xfer (ms)"
              << std::setw(14) << "Kern (ms)"
              << std::setw(12) << "H2D (ms)"
              << std::setw(12) << "D2H (ms)"
              << std::setw(10) << "Speedup"
              << "Throughput\n";
    printSep('-');

    for (int N : batches) {
        double cpu_ms    = benchmarkCPUSynthetic(W, H, N);
        auto   gr        = benchmarkGPUSynthetic(W, H, N, 3);
        double speedup   = cpu_ms / gr.time_with_transfer_ms;

        std::cout << std::fixed << std::setprecision(2) << std::left
                  << std::setw(8)  << N
                  << std::setw(14) << cpu_ms
                  << std::setw(16) << gr.time_with_transfer_ms
                  << std::setw(14) << gr.time_kernel_only_ms
                  << std::setw(12) << gr.time_h2d_ms
                  << std::setw(12) << gr.time_d2h_ms
                  << std::setw(10) << speedup
                  << std::setprecision(1) << gr.throughput_mpix_per_sec << " MPix/s\n";

        if (csv.is_open())
            csv << "batch_scaling," << N << "," << W << "," << H << ","
                << cpu_ms << "," << gr.time_with_transfer_ms << ","
                << gr.time_kernel_only_ms << "," << speedup << "\n";
    }
}

// ─── Mode: Resolution scaling ──────────────────────────────────────────────

void runResolutionScaling(std::ofstream& csv) {
    printSep('═');
    std::cout << "[BENCH] Resolution Scaling  (50 images per run, 3-run avg)\n";
    printSep();

    struct Res { int w, h; const char* name; };
    std::vector<Res> resolutions = {
        {256,  256,  "256×256"},
        {512,  512,  "512×512"},
        {1024, 1024, "1024×1024"},
        {1920, 1080, "1920×1080 (FHD)"},
        {2560, 1440, "2560×1440 (QHD)"},
        {3840, 2160, "3840×2160 (4K)"}
    };
    const int N = 50;

    std::cout << std::left
              << std::setw(22) << "Resolution"
              << std::setw(14) << "CPU (ms)"
              << std::setw(16) << "GPU+Xfer (ms)"
              << std::setw(14) << "Kern (ms)"
              << std::setw(10) << "Speedup"
              << "Throughput\n";
    printSep('-');

    for (auto& r : resolutions) {
        double cpu_ms  = benchmarkCPUSynthetic(r.w, r.h, N);
        auto   gr      = benchmarkGPUSynthetic(r.w, r.h, N, 3);
        double speedup = cpu_ms / gr.time_with_transfer_ms;

        std::cout << std::fixed << std::setprecision(2) << std::left
                  << std::setw(22) << r.name
                  << std::setw(14) << cpu_ms
                  << std::setw(16) << gr.time_with_transfer_ms
                  << std::setw(14) << gr.time_kernel_only_ms
                  << std::setw(10) << speedup
                  << std::setprecision(1) << gr.throughput_mpix_per_sec << " MPix/s\n";

        if (csv.is_open())
            csv << "res_scaling," << N << "," << r.w << "," << r.h << ","
                << cpu_ms << "," << gr.time_with_transfer_ms << ","
                << gr.time_kernel_only_ms << "," << speedup << "\n";
    }
}

// ─── Mode: Transfer overhead analysis ──────────────────────────────────────

void runTransferAnalysis(std::ofstream& csv) {
    printSep('═');
    std::cout << "[BENCH] Memory Transfer Overhead Analysis  (512×512 images)\n";
    printSep();

    const int W = 512, H = 512;
    std::vector<int> batches = {10, 50, 100, 200, 500, 1000};

    std::cout << std::left
              << std::setw(8)  << "Batch"
              << std::setw(14) << "Data (MB)"
              << std::setw(14) << "H2D (ms)"
              << std::setw(14) << "H2D (GB/s)"
              << std::setw(14) << "D2H (ms)"
              << std::setw(14) << "D2H (GB/s)"
              << "Xfer % total\n";
    printSep('-');

    for (int N : batches) {
        auto gr       = benchmarkGPUSynthetic(W, H, N, 3);
        double mb     = (double)N * W * H / (1024.0 * 1024.0);
        double h2d_bw = (mb / 1024.0) / (gr.time_h2d_ms / 1000.0);
        double d2h_bw = (mb / 1024.0) / (gr.time_d2h_ms / 1000.0);
        double xfer_pct = 100.0 * (gr.time_h2d_ms + gr.time_d2h_ms) / gr.time_with_transfer_ms;

        std::cout << std::fixed << std::setprecision(2) << std::left
                  << std::setw(8)  << N
                  << std::setw(14) << mb
                  << std::setw(14) << gr.time_h2d_ms
                  << std::setw(14) << std::setprecision(1) << h2d_bw
                  << std::setw(14) << std::setprecision(2) << gr.time_d2h_ms
                  << std::setw(14) << std::setprecision(1) << d2h_bw
                  << std::setprecision(1) << xfer_pct << "%\n";

        if (csv.is_open())
            csv << "transfer," << N << "," << W << "," << H << ","
                << gr.time_h2d_ms << "," << gr.time_d2h_ms << ","
                << h2d_bw << "," << d2h_bw << "\n";
    }
}

// ─── Main ─────────────────────────────────────────────────────────────────

void printUsage(const char* prog) {
    std::cout << "Usage:\n";
    std::cout << "  " << prog << "                                  # full benchmark suite\n";
    std::cout << "  " << prog << " --benchmark                       # same\n";
    std::cout << "  " << prog << " --process <in_dir> <out_dir> [N]  # process real images\n";
    std::cout << "  " << prog << " --help\n";
}

int main(int argc, char* argv[]) {
    srand(42);
    printBanner();
    printGPUInfo();

    // Open CSV for results
    fs::create_directories("results");
    std::ofstream csv("results/benchmark_data.csv");
    if (csv.is_open())
        csv << "mode,n_images,width,height,cpu_ms,gpu_total_ms,gpu_kernel_ms,speedup\n";

    std::string mode = (argc > 1) ? argv[1] : "--benchmark";

    if (mode == "--help") {
        printUsage(argv[0]);
        return 0;
    }

    if (mode == "--process") {
        if (argc < 4) {
            std::cerr << "Error: --process needs <input_dir> <output_dir> [batch_limit]\n";
            printUsage(argv[0]);
            return 1;
        }
        int limit = (argc >= 5) ? std::stoi(argv[4]) : -1;
        runImageProcessing(argv[2], argv[3], limit);
        return 0;
    }

    // Default: full benchmark suite
    runBatchScaling(csv);
    std::cout << "\n";
    runResolutionScaling(csv);
    std::cout << "\n";
    runTransferAnalysis(csv);

    printSep('═');
    std::cout << "[DONE] CSV data saved to results/benchmark_data.csv\n";
    std::cout << "       Run:  python3 scripts/plot_results.py  to visualize\n";
    printSep('═');
    return 0;
}

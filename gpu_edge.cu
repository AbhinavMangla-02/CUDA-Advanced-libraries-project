// =============================================================================
// gpu_edge.cu — CUDA Sobel Edge Detection Kernels
// =============================================================================
// Two kernel variants:
//   1. sobelBatchKernel       — naive global-memory access (reference)
//   2. sobelBatchSharedKernel — shared-memory tiled version (optimized)
//
// Batch layout: images are stored contiguously in memory.
//   pixel[img][row][col] → flat index = img*(W*H) + row*W + col
// =============================================================================

#include "gpu_edge.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <numeric>
#include <chrono>

// ─── CUDA error checking ──────────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t _e = (call);                                                 \
        if (_e != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                  \
            throw std::runtime_error(cudaGetErrorString(_e));                    \
        }                                                                        \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// KERNEL 1: Naive global-memory Sobel (simple, illustrative)
// Grid:  1-D over all pixels across the entire batch
// Block: 256 threads
// ─────────────────────────────────────────────────────────────────────────────

__global__ void sobelBatchKernel(
    const unsigned char* __restrict__ input,
          unsigned char* __restrict__ output,
    int width,
    int height,
    int num_images)
{
    int idx          = blockIdx.x * blockDim.x + threadIdx.x;
    int pixels_per_img = width * height;
    int total_pixels = num_images * pixels_per_img;

    if (idx >= total_pixels) return;

    int img_idx   = idx / pixels_per_img;
    int local_idx = idx % pixels_per_img;
    int row       = local_idx / width;
    int col       = local_idx % width;

    // Border → zero gradient
    if (row == 0 || row == height - 1 || col == 0 || col == width - 1) {
        output[idx] = 0;
        return;
    }

    int base = img_idx * pixels_per_img;

    // Sobel Gx:  -1  0  +1
    //            -2  0  +2
    //            -1  0  +1
    int gx = -(int)input[base + (row-1)*width + (col-1)]
             +(int)input[base + (row-1)*width + (col+1)]
          - 2*(int)input[base +  row   *width + (col-1)]
          + 2*(int)input[base +  row   *width + (col+1)]
             -(int)input[base + (row+1)*width + (col-1)]
             +(int)input[base + (row+1)*width + (col+1)];

    // Sobel Gy:  -1  -2  -1
    //             0   0   0
    //            +1  +2  +1
    int gy = -(int)input[base + (row-1)*width + (col-1)]
          - 2*(int)input[base + (row-1)*width +  col   ]
             -(int)input[base + (row-1)*width + (col+1)]
             +(int)input[base + (row+1)*width + (col-1)]
          + 2*(int)input[base + (row+1)*width +  col   ]
             +(int)input[base + (row+1)*width + (col+1)];

    int mag = __float2int_rn(sqrtf((float)(gx*gx + gy*gy)));
    output[idx] = (unsigned char)(mag > 255 ? 255 : mag);
}

// ─────────────────────────────────────────────────────────────────────────────
// KERNEL 2: Shared-memory tiled Sobel (optimized — reduces global mem latency)
//
// Each 2-D block of BLOCK×BLOCK threads loads a (BLOCK+2)×(BLOCK+2) tile
// (including 1-pixel halo on each side) into __shared__ memory, then reads
// from shared memory for the 3×3 Sobel stencil.
//
// Grid dimensions:  (ceil(W/BLOCK), ceil(H/BLOCK), num_images)
// ─────────────────────────────────────────────────────────────────────────────

#define BLOCK  16
#define TILE   (BLOCK + 2)   // 18

__global__ void sobelBatchSharedKernel(
    const unsigned char* __restrict__ input,
          unsigned char* __restrict__ output,
    int width,
    int height,
    int num_images)
{
    __shared__ unsigned char smem[TILE][TILE];

    int img = blockIdx.z;
    int out_col = blockIdx.x * BLOCK + (int)threadIdx.x;
    int out_row = blockIdx.y * BLOCK + (int)threadIdx.y;

    int base = img * width * height;

    // ── Load center element into shared memory ──────────────────────────
    int tc = (int)threadIdx.x + 1;   // tile col (1-indexed due to halo)
    int tr = (int)threadIdx.y + 1;   // tile row

    auto safe_read = [&](int r, int c) -> unsigned char {
        if (r >= 0 && r < height && c >= 0 && c < width)
            return input[base + r * width + c];
        return 0;
    };

    smem[tr][tc] = safe_read(out_row, out_col);

    // ── Load halo pixels ────────────────────────────────────────────────
    // Top row halo
    if (threadIdx.y == 0)
        smem[0][tc] = safe_read(out_row - 1, out_col);
    // Bottom row halo
    if (threadIdx.y == BLOCK - 1)
        smem[TILE-1][tc] = safe_read(out_row + 1, out_col);
    // Left col halo
    if (threadIdx.x == 0)
        smem[tr][0] = safe_read(out_row, out_col - 1);
    // Right col halo
    if (threadIdx.x == BLOCK - 1)
        smem[tr][TILE-1] = safe_read(out_row, out_col + 1);

    // Four corners (only corner threads)
    if (threadIdx.x == 0        && threadIdx.y == 0)
        smem[0][0]           = safe_read(out_row-1, out_col-1);
    if (threadIdx.x == BLOCK-1  && threadIdx.y == 0)
        smem[0][TILE-1]      = safe_read(out_row-1, out_col+1);
    if (threadIdx.x == 0        && threadIdx.y == BLOCK-1)
        smem[TILE-1][0]      = safe_read(out_row+1, out_col-1);
    if (threadIdx.x == BLOCK-1  && threadIdx.y == BLOCK-1)
        smem[TILE-1][TILE-1] = safe_read(out_row+1, out_col+1);

    __syncthreads();

    // ── Compute Sobel using shared memory ────────────────────────────────
    if (out_row >= height || out_col >= width) return;

    if (out_row == 0 || out_row == height-1 || out_col == 0 || out_col == width-1) {
        output[base + out_row * width + out_col] = 0;
        return;
    }

    int gx = -(int)smem[tr-1][tc-1] + (int)smem[tr-1][tc+1]
          - 2*(int)smem[tr  ][tc-1] + 2*(int)smem[tr  ][tc+1]
             -(int)smem[tr+1][tc-1] + (int)smem[tr+1][tc+1];

    int gy = -(int)smem[tr-1][tc-1] - 2*(int)smem[tr-1][tc] - (int)smem[tr-1][tc+1]
             +(int)smem[tr+1][tc-1] + 2*(int)smem[tr+1][tc] + (int)smem[tr+1][tc+1];

    int mag = __float2int_rn(sqrtf((float)(gx*gx + gy*gy)));
    output[base + out_row * width + out_col] = (unsigned char)(mag > 255 ? 255 : mag);
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API implementation
// ─────────────────────────────────────────────────────────────────────────────

void sobelBatchGPU(const std::vector<Image>& inputs,
                   std::vector<Image>&       outputs)
{
    if (inputs.empty()) return;

    int W = inputs[0].width, H = inputs[0].height, N = (int)inputs.size();
    size_t img_bytes   = (size_t)W * H;
    size_t total_bytes = img_bytes * N;

    // Pack images into one contiguous host buffer
    std::vector<uint8_t> h_in(total_bytes);
    for (int i = 0; i < N; ++i)
        std::memcpy(h_in.data() + i * img_bytes,
                    inputs[i].data.data(), img_bytes);

    uint8_t *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  total_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, total_bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), total_bytes, cudaMemcpyHostToDevice));

    // Use shared-memory kernel
    dim3 block(BLOCK, BLOCK);
    dim3 grid((W + BLOCK-1)/BLOCK, (H + BLOCK-1)/BLOCK, N);
    sobelBatchSharedKernel<<<grid, block>>>(d_in, d_out, W, H, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint8_t> h_out(total_bytes);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, total_bytes, cudaMemcpyDeviceToHost));

    outputs.resize(N);
    for (int i = 0; i < N; ++i) {
        outputs[i].width  = W;
        outputs[i].height = H;
        outputs[i].data.resize(img_bytes);
        std::memcpy(outputs[i].data.data(),
                    h_out.data() + i * img_bytes, img_bytes);
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

// ─── Timed benchmark (uses CUDA events for precision) ────────────────────────

static GPUBenchmarkResult runTimedGPU(const uint8_t* h_in, int W, int H, int N,
                                       size_t total_bytes, int runs)
{
    size_t img_bytes = (size_t)W * H;

    uint8_t *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  total_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, total_bytes));

    cudaEvent_t ev[4];
    for (auto& e : ev) CUDA_CHECK(cudaEventCreate(&e));

    std::vector<float> t_h2d(runs), t_kern(runs), t_d2h(runs);

    for (int r = 0; r < runs; ++r) {
        // H2D
        CUDA_CHECK(cudaEventRecord(ev[0]));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, total_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(ev[1]));

        // Kernel (shared-memory version)
        dim3 block(BLOCK, BLOCK);
        dim3 grid((W + BLOCK-1)/BLOCK, (H + BLOCK-1)/BLOCK, N);
        sobelBatchSharedKernel<<<grid, block>>>(d_in, d_out, W, H, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(ev[2]));

        // D2H (discard output — just measure transfer)
        std::vector<uint8_t> tmp(total_bytes);
        CUDA_CHECK(cudaMemcpy(tmp.data(), d_out, total_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(ev[3]));
        CUDA_CHECK(cudaEventSynchronize(ev[3]));

        CUDA_CHECK(cudaEventElapsedTime(&t_h2d[r],  ev[0], ev[1]));
        CUDA_CHECK(cudaEventElapsedTime(&t_kern[r],  ev[1], ev[2]));
        CUDA_CHECK(cudaEventElapsedTime(&t_d2h[r],  ev[2], ev[3]));
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    for (auto& e : ev) CUDA_CHECK(cudaEventDestroy(e));

    auto avg = [&](const std::vector<float>& v) {
        return (double)std::accumulate(v.begin(), v.end(), 0.0f) / runs;
    };

    GPUBenchmarkResult res;
    res.num_images          = N;
    res.total_pixels        = (size_t)N * W * H;
    res.time_h2d_ms         = avg(t_h2d);
    res.time_kernel_only_ms = avg(t_kern);
    res.time_d2h_ms         = avg(t_d2h);
    res.time_with_transfer_ms = res.time_h2d_ms + res.time_kernel_only_ms + res.time_d2h_ms;
    res.throughput_mpix_per_sec =
        (res.total_pixels / 1e6) / (res.time_kernel_only_ms / 1000.0);
    return res;
}

GPUBenchmarkResult benchmarkGPU(const std::vector<Image>& images, int runs) {
    if (images.empty()) return {};
    int W = images[0].width, H = images[0].height, N = (int)images.size();
    size_t img_bytes = (size_t)W * H, total = img_bytes * N;
    std::vector<uint8_t> h_in(total);
    for (int i = 0; i < N; ++i)
        std::memcpy(h_in.data() + i * img_bytes, images[i].data.data(), img_bytes);
    return runTimedGPU(h_in.data(), W, H, N, total, runs);
}

GPUBenchmarkResult benchmarkGPUSynthetic(int W, int H, int N, int runs) {
    size_t img_bytes = (size_t)W * H, total = img_bytes * N;
    std::vector<uint8_t> h_in(total);
    // Fill with deterministic synthetic data
    for (size_t i = 0; i < total; ++i)
        h_in[i] = (uint8_t)((i * 2654435761ULL) >> 24);
    return runTimedGPU(h_in.data(), W, H, N, total, runs);
}

double benchmarkCPUSynthetic(int W, int H, int N) {
    size_t img_bytes = (size_t)W * H, total = img_bytes * N;
    std::vector<uint8_t> in(total), out(total, 0);
    for (size_t i = 0; i < total; ++i)
        in[i] = (uint8_t)((i * 2654435761ULL) >> 24);

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int n = 0; n < N; ++n) {
        const uint8_t* src = in.data()  + n * img_bytes;
              uint8_t* dst = out.data() + n * img_bytes;

        for (int r = 1; r < H - 1; ++r) {
            for (int c = 1; c < W - 1; ++c) {
                auto px = [&](int dr, int dc) -> int {
                    return (int)src[(r+dr)*W + (c+dc)];
                };
                int gx = -px(-1,-1)+px(-1,+1) - 2*px(0,-1)+2*px(0,+1) - px(+1,-1)+px(+1,+1);
                int gy = -px(-1,-1)-2*px(-1,0)-px(-1,+1) + px(+1,-1)+2*px(+1,0)+px(+1,+1);
                int mag = (int)sqrtf((float)(gx*gx + gy*gy));
                dst[r*W + c] = (uint8_t)(mag > 255 ? 255 : mag);
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

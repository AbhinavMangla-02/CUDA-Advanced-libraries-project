// =============================================================================
// cpu_edge.cpp — Single-threaded CPU Sobel Edge Detection
// =============================================================================

#include "cpu_edge.h"
#include <cmath>
#include <chrono>
#include <algorithm>

void sobelCPU(const Image& input, Image& output) {
    int W = input.width, H = input.height;
    output.width  = W;
    output.height = H;
    output.data.assign(W * H, 0);

    for (int r = 1; r < H - 1; ++r) {
        for (int c = 1; c < W - 1; ++c) {
            const uint8_t* src = input.data.data();

            auto px = [&](int dr, int dc) -> int {
                return (int)src[(r+dr)*W + (c+dc)];
            };

            // Gx: horizontal edges
            int gx = -px(-1,-1) + px(-1,+1)
                     -2*px( 0,-1) + 2*px( 0,+1)
                     -px(+1,-1) + px(+1,+1);

            // Gy: vertical edges
            int gy = -px(-1,-1) - 2*px(-1, 0) - px(-1,+1)
                     +px(+1,-1) + 2*px(+1, 0) + px(+1,+1);

            int mag = (int)std::sqrt((float)(gx*gx + gy*gy));
            output.data[r*W + c] = (uint8_t)std::min(mag, 255);
        }
    }
}

void sobelBatchCPU(const std::vector<Image>& inputs,
                   std::vector<Image>&       outputs)
{
    outputs.resize(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i)
        sobelCPU(inputs[i], outputs[i]);
}

double benchmarkCPU(const std::vector<Image>& images, int runs) {
    double total = 0.0;
    std::vector<Image> outputs;

    for (int r = 0; r < runs; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        sobelBatchCPU(images, outputs);
        auto t1 = std::chrono::high_resolution_clock::now();
        total += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    return total / runs;
}

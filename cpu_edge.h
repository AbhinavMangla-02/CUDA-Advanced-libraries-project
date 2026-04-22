#pragma once
// =============================================================================
// cpu_edge.h — Single-threaded CPU Sobel Edge Detection
// =============================================================================
// Used as the performance baseline for GPU comparisons.

#include "image_utils.h"
#include <vector>

// Apply Sobel to one grayscale image (output allocated inside)
void sobelCPU(const Image& input, Image& output);

// Apply Sobel to an entire batch, sequentially
void sobelBatchCPU(const std::vector<Image>& inputs,
                   std::vector<Image>&       outputs);

// Timed batch run, returns average wall-clock ms over `runs` repetitions
double benchmarkCPU(const std::vector<Image>& images, int runs = 3);

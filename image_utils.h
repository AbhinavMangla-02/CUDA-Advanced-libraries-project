#pragma once
// =============================================================================
// image_utils.h — Lightweight grayscale image I/O (PGM binary format)
// =============================================================================
// No external libraries required.  PGM (Portable GrayMap) P5 format.
// All CUDA kernels operate on grayscale (1 channel, uint8) image data.

#include <string>
#include <vector>
#include <cstdint>

// ─── Image container ──────────────────────────────────────────────────────────

struct Image {
    int                   width  = 0;
    int                   height = 0;
    std::vector<uint8_t>  data;      // row-major, 1 byte per pixel

    Image() = default;
    Image(int w, int h) : width(w), height(h), data((size_t)w * h, 0) {}

    size_t numPixels() const { return (size_t)width * height; }
};

// ─── File I/O ────────────────────────────────────────────────────────────────

/// Load a binary PGM (P5) file.  Returns false on error.
bool loadPGM(const std::string& path, Image& img);

/// Save a binary PGM (P5) file.  Returns false on error.
bool savePGM(const std::string& path, const Image& img);

/// Load up to `max_count` .pgm files from `dir` (sorted by name).
/// Pass max_count = -1 to load all.
std::vector<Image> loadImageBatch(const std::string& dir, int max_count = -1);

/// Save a batch of images to `dir` with filenames `prefix0.pgm`, etc.
bool saveImageBatch(const std::vector<Image>& images,
                    const std::string& dir,
                    const std::string& prefix = "edge_");

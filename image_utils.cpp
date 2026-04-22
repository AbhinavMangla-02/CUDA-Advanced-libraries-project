// =============================================================================
// image_utils.cpp — PGM binary image I/O
// =============================================================================

#include "image_utils.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

// ─── PGM reader ──────────────────────────────────────────────────────────────

bool loadPGM(const std::string& path, Image& img) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "[loadPGM] Cannot open: " << path << "\n";
        return false;
    }

    std::string magic;
    f >> magic;

    if (magic != "P5") {
        std::cerr << "[loadPGM] Not a P5 PGM file: " << path << "\n";
        return false;
    }

    // Skip comments
    char c;
    f.get(c);
    while (f.peek() == '#') {
        std::string comment;
        std::getline(f, comment);
    }

    int w, h, maxval;
    f >> w >> h >> maxval;
    f.get(c); // consume single whitespace after maxval

    if (w <= 0 || h <= 0 || maxval <= 0 || maxval > 255) {
        std::cerr << "[loadPGM] Invalid header in: " << path << "\n";
        return false;
    }

    img.width  = w;
    img.height = h;
    img.data.resize((size_t)w * h);
    f.read(reinterpret_cast<char*>(img.data.data()), img.data.size());

    return (bool)f;
}

// ─── PGM writer ──────────────────────────────────────────────────────────────

bool savePGM(const std::string& path, const Image& img) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "[savePGM] Cannot write: " << path << "\n";
        return false;
    }
    f << "P5\n" << img.width << " " << img.height << "\n255\n";
    f.write(reinterpret_cast<const char*>(img.data.data()), (std::streamsize)img.data.size());
    return (bool)f;
}

// ─── Batch helpers ────────────────────────────────────────────────────────────

std::vector<Image> loadImageBatch(const std::string& dir, int max_count) {
    std::vector<std::string> paths;

    for (auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() == ".pgm")
            paths.push_back(entry.path().string());
    }

    std::sort(paths.begin(), paths.end());

    if (max_count > 0 && (int)paths.size() > max_count)
        paths.resize((size_t)max_count);

    std::vector<Image> images;
    images.reserve(paths.size());
    for (auto& p : paths) {
        Image img;
        if (loadPGM(p, img))
            images.push_back(std::move(img));
        else
            std::cerr << "[WARN] Skipped: " << p << "\n";
    }
    return images;
}

bool saveImageBatch(const std::vector<Image>& images,
                    const std::string& dir,
                    const std::string& prefix)
{
    fs::create_directories(dir);
    for (size_t i = 0; i < images.size(); ++i) {
        std::string path = dir + "/" + prefix + std::to_string(i) + ".pgm";
        if (!savePGM(path, images[i])) return false;
    }
    return true;
}

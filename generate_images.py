#!/usr/bin/env python3
"""
generate_images.py — Create synthetic grayscale test images for edge detection.

Outputs N .pgm (binary PGM / P5) files to input_images/.
Five visual patterns are cycled through so the batch is visually diverse:
  0 – geometric shapes (rectangles + circles)
  1 – sinusoidal gradient
  2 – checkerboard
  3 – line drawing (Bresenham)
  4 – noise-modulated wave
"""

import os, struct, math, random, sys

OUTPUT_DIR  = "input_images"
NUM_IMAGES  = 200
IMAGE_SIZE  = 512          # square; override with: python3 generate_images.py 1024

# ─── PGM writer ──────────────────────────────────────────────────────────────

def save_pgm(path: str, pixels: list[int], w: int, h: int):
    with open(path, "wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode())
        f.write(bytes(pixels))

# ─── Pattern generators (each returns a flat list of uint8) ──────────────────

def gen_geometric(w, h, seed):
    rng = random.Random(seed)
    px  = [20] * (w * h)

    # Filled rectangles
    for _ in range(rng.randint(3, 6)):
        x1 = rng.randint(0,  w // 2);  y1 = rng.randint(0, h // 2)
        x2 = rng.randint(x1 + 30, min(x1 + w//2, w-1))
        y2 = rng.randint(y1 + 30, min(y1 + h//2, h-1))
        v  = rng.randint(140, 255)
        for y in range(y1, y2+1):
            for x in range(x1, x2+1):
                px[y*w + x] = v

    # Filled circles
    for _ in range(rng.randint(1, 4)):
        cx = rng.randint(w//4, 3*w//4)
        cy = rng.randint(h//4, 3*h//4)
        r  = rng.randint(20, min(w, h)//6)
        v  = rng.randint(80, 210)
        for y in range(max(0, cy-r), min(h, cy+r+1)):
            for x in range(max(0, cx-r), min(w, cx+r+1)):
                if (x-cx)**2 + (y-cy)**2 <= r*r:
                    px[y*w + x] = v
    return px

def gen_gradient(w, h, seed):
    rng = random.Random(seed)
    fx  = rng.uniform(2, 6)
    fy  = rng.uniform(2, 6)
    ph  = rng.uniform(0, math.pi*2)
    px  = []
    for y in range(h):
        for x in range(w):
            v = 128 + 120 * math.sin(x/w * math.pi*fx + ph) * math.cos(y/h * math.pi*fy)
            px.append(max(0, min(255, int(v))))
    return px

def gen_checkerboard(w, h, seed):
    rng  = random.Random(seed)
    tile = rng.randint(16, 80)
    hi   = rng.randint(180, 255)
    lo   = rng.randint(10,  60)
    return [hi if ((x//tile)+(y//tile))%2==0 else lo
            for y in range(h) for x in range(w)]

def gen_lines(w, h, seed):
    rng = random.Random(seed)
    px  = [25] * (w * h)

    def draw_line(x0, y0, x1, y1, v):
        dx, dy = abs(x1-x0), abs(y1-y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            if 0 <= x0 < w and 0 <= y0 < h:
                px[y0*w+x0] = v
            if x0 == x1 and y0 == y1:
                break
            e2 = 2*err
            if e2 > -dy: err -= dy; x0 += sx
            if e2 <  dx: err += dx; y0 += sy

    for _ in range(rng.randint(8, 25)):
        draw_line(rng.randint(0,w-1), rng.randint(0,h-1),
                  rng.randint(0,w-1), rng.randint(0,h-1),
                  rng.randint(170, 255))
    return px

def gen_noise_wave(w, h, seed):
    rng = random.Random(seed)
    freq = rng.uniform(0.03, 0.10)
    px   = []
    for y in range(h):
        for x in range(w):
            base  = 128 + 100 * math.sin(x * freq) * math.cos(y * freq)
            noise = rng.randint(-18, 18)
            px.append(max(0, min(255, int(base + noise))))
    return px

GENERATORS = [gen_geometric, gen_gradient, gen_checkerboard, gen_lines, gen_noise_wave]

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    size = int(sys.argv[1]) if len(sys.argv) > 1 else IMAGE_SIZE
    n    = int(sys.argv[2]) if len(sys.argv) > 2 else NUM_IMAGES

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Generating {n} images ({size}×{size}) → {OUTPUT_DIR}/")

    for i in range(n):
        gen  = GENERATORS[i % len(GENERATORS)]
        pix  = gen(size, size, seed=i * 6271 + 3571)
        path = os.path.join(OUTPUT_DIR, f"img_{i:04d}.pgm")
        save_pgm(path, pix, size, size)

        if (i+1) % 50 == 0:
            print(f"  {i+1}/{n} images written")

    print(f"Done.  Run:  ./edge_detect --process input_images output_edges")

if __name__ == "__main__":
    main()

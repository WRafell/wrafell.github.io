"""Generate the site's pixel-art images (cyanotype-style prints).

Run from the repo root:
    python _tools/make_pixel_art.py
Requires Pillow and NumPy. Outputs go to assets/images/.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

OUT = Path("assets/images")

# Palette (kept in sync with assets/css/main.css)
PAPER = (243, 235, 221)       # ivory beige
TEAL = (110, 150, 147)        # dusty teal
TEAL_DEEP = (60, 100, 101)    # cyanotype ground
MIST = (196, 210, 202)        # pale teal wash
INK = (75, 37, 51)            # muted burgundy

BAYER4 = np.array(
    [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]
) / 16.0


def dither(gray, levels):
    """Ordered (Bayer 4x4) dither of a [0,1] array into len(levels) tones."""
    h, w = gray.shape
    n = len(levels) - 1
    thresh = np.tile(BAYER4, (h // 4 + 1, w // 4 + 1))[:h, :w]
    idx = np.clip(np.floor(gray * n + thresh), 0, n).astype(int)
    pal = np.array(levels, dtype=np.uint8)
    return pal[idx]


def ragged_edge_mask(h, w, rng, depth=3):
    """Brushed-emulsion border: a rectangle whose edges jitter by a few pixels."""
    mask = np.ones((h, w), dtype=bool)
    for side in range(4):
        length = w if side in (0, 1) else h
        # smooth random walk so the edge looks brushed rather than noisy
        walk = np.cumsum(rng.integers(-1, 2, length))
        walk = np.clip(walk - walk.min(), 0, depth)
        for i, d in enumerate(walk):
            if side == 0:
                mask[:d, i] = False
            elif side == 1:
                mask[h - d:, i] = False
            elif side == 2:
                mask[i, :d] = False
            else:
                mask[i, w - d:] = False
    return mask


def portrait(src, size=96):
    img = Image.open(src).convert("L")
    img = ImageOps.fit(img, (size, size), Image.LANCZOS)
    img = ImageOps.autocontrast(img, cutoff=1)
    gray = (np.asarray(img, dtype=float) / 255.0) ** 0.6  # lift the dark fur
    rgb = dither(gray, [TEAL_DEEP, TEAL, MIST, PAPER])
    mask = ragged_edge_mask(size, size, np.random.default_rng(7), depth=2)
    rgba = np.dstack([rgb, np.where(mask, 255, 0).astype(np.uint8)])
    Image.fromarray(rgba, "RGBA").save(OUT / "portrait-pixel.png", optimize=True)


def barnsley(n=14000, seed=3):
    rng = np.random.default_rng(seed)
    x, y = 0.0, 0.0
    pts = np.empty((n, 2))
    for i in range(n):
        r = rng.random()
        if r < 0.01:
            x, y = 0.0, 0.16 * y
        elif r < 0.86:
            x, y = 0.85 * x + 0.04 * y, -0.04 * x + 0.85 * y + 1.6
        elif r < 0.93:
            x, y = 0.2 * x - 0.26 * y, 0.23 * x + 0.22 * y + 1.6
        else:
            x, y = -0.15 * x + 0.28 * y, 0.26 * x + 0.24 * y + 0.44
        pts[i] = x, y
    return pts


def fern_grid(w, h, pad, angle=0.0):
    pts = barnsley()
    if angle:
        c, s = np.cos(angle), np.sin(angle)
        pts = pts @ np.array([[c, -s], [s, c]]).T
    mn, mx = pts.min(0), pts.max(0)
    scale = min((w - 2 * pad) / (mx[0] - mn[0]), (h - 2 * pad) / (mx[1] - mn[1]))
    gx = ((pts[:, 0] - mn[0]) * scale + (w - (mx[0] - mn[0]) * scale) / 2).astype(int)
    gy = (h - 1 - ((pts[:, 1] - mn[1]) * scale + pad)).astype(int)
    grid = np.zeros((h, w), dtype=int)
    np.add.at(grid, (np.clip(gy, 0, h - 1), np.clip(gx, 0, w - 1)), 1)
    return grid


def fern_print(w=84, h=112):
    """A photogram: ivory fern silhouette exposed onto a teal ground."""
    rng = np.random.default_rng(11)
    grid = fern_grid(w, h, pad=10, angle=-0.12)
    rgb = np.empty((h, w, 3), dtype=np.uint8)
    rgb[:] = TEAL_DEEP
    # uneven exposure: a few lighter teal pixels scattered on the ground
    speckle = rng.random((h, w)) < 0.06
    rgb[speckle] = TEAL
    rgb[grid >= 1] = MIST
    rgb[grid >= 2] = PAPER
    mask = ragged_edge_mask(h, w, rng, depth=3)
    rgba = np.dstack([rgb, np.where(mask, 255, 0).astype(np.uint8)])
    Image.fromarray(rgba, "RGBA").save(OUT / "fern-print.png", optimize=True)


# Hand-drawn 16x16 "윌" (Will): a figure with a stick (ㅇ ㅜ ㅣ) over a rock (ㄹ).
WILL = """
................
...###......#...
..#...#.....#...
..#...#.....#...
...###......#...
............#...
.#########..#...
.....#......#...
.....#......#...
................
.###########....
...........#....
.###########....
.#..............
.###########....
................
"""


def glyph_print(scale, name, border=3):
    """The glyph as a tiny photogram: ivory letter, half-exposed halo, teal ground."""
    rng = np.random.default_rng(5)
    rows = WILL.strip("\n").splitlines()
    glyph = np.array([[ch == "#" for ch in row] for row in rows])
    size = 16 + 2 * border
    mask = np.zeros((size, size), dtype=bool)
    mask[border:border + 16, border:border + 16] = glyph
    # 4-neighbour dilation: light leaking under the edges of the "object"
    halo = mask.copy()
    halo[1:] |= mask[:-1]
    halo[:-1] |= mask[1:]
    halo[:, 1:] |= mask[:, :-1]
    halo[:, :-1] |= mask[:, 1:]
    halo &= ~mask

    rgb = np.empty((size, size, 3), dtype=np.uint8)
    rgb[:] = TEAL_DEEP
    rgb[rng.random((size, size)) < 0.07] = TEAL
    rgb[halo & (rng.random((size, size)) < 0.45)] = TEAL
    rgb[mask] = PAPER
    rgb[mask & (rng.random((size, size)) < 0.08)] = MIST
    edge = ragged_edge_mask(size, size, rng, depth=1)
    rgba = np.dstack([rgb, np.where(edge, 255, 0).astype(np.uint8)])
    img = Image.fromarray(rgba, "RGBA")
    img.resize((size * scale, size * scale), Image.NEAREST).save(OUT / name, optimize=True)


def share_image(name="og-image.png", w=1200, h=630, scale=21):
    """Link-preview card: the 윌 print centred on ivory paper with a hard shadow."""
    glyph_print(1, "_tmp_will.png")
    tile = Image.open(OUT / "_tmp_will.png")
    (OUT / "_tmp_will.png").unlink()
    size = tile.width * scale
    tile = tile.resize((size, size), Image.NEAREST)
    alpha = tile.split()[3]
    card = Image.new("RGB", (w, h), PAPER)
    x, y = (w - size) // 2, (h - size) // 2
    shadow = Image.new("RGB", tile.size, (221, 203, 201))  # --mauve-pale
    off = scale
    card.paste(shadow, (x + off, y + off), alpha)
    card.paste(tile.convert("RGB"), (x, y), alpha)
    card.save(OUT / name, optimize=True)


if __name__ == "__main__":
    portrait(OUT / "profile-photo.jpg")
    fern_print()
    glyph_print(1, "will.png")
    glyph_print(2, "favicon.png")
    share_image()
    print("done")

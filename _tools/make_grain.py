"""Generate the paper-grain background tiles (sparse brown noise).

Run from the repo root:
    python _tools/make_grain.py
Requires Pillow and NumPy. Outputs go to assets/images/.

Brown noise has power ~ 1/f^2: it is built in the frequency domain, so the
tile wraps seamlessly. Only the highest peaks of the field are kept, which
gives sparse soft flecks instead of an even all-over grain.
"""

from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path("assets/images")

INK = (75, 37, 51)        # light theme: burgundy flecks on ivory paper
IVORY = (243, 235, 221)   # dark theme: ivory flecks on near-black paper

SIZE = 384
MAX_WAVELENGTH = 48       # px; drops the largest blobs so the tile does not repeat visibly


def brown_noise(size, seed):
    """Periodic 2D brown noise (amplitude ~ 1/f), normalised to zero mean, unit std."""
    rng = np.random.default_rng(seed)
    f = np.hypot(*np.meshgrid(np.fft.fftfreq(size), np.fft.fftfreq(size)))
    amp = np.zeros_like(f)
    keep = f >= 1 / MAX_WAVELENGTH
    amp[keep] = 1 / f[keep]
    spectrum = amp * np.exp(2j * np.pi * rng.random((size, size)))
    field = np.real(np.fft.ifft2(spectrum))
    return (field - field.mean()) / field.std()


def sparse_alpha(field, coverage, softness, peak):
    """Keep the top `coverage` fraction of the field as soft flecks up to `peak` opacity."""
    t = np.quantile(field, 1 - coverage)
    return np.clip((field - t) / softness, 0, 1) * peak


def grain_tile(name, rgb, peak, seed=7, coverage=0.06, softness=0.6):
    alpha = sparse_alpha(brown_noise(SIZE, seed), coverage, softness, peak)
    rgba = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = np.round(alpha * 255).astype(np.uint8)
    Image.fromarray(rgba, "RGBA").save(OUT / name, optimize=True)


if __name__ == "__main__":
    grain_tile("grain-light.png", INK, peak=0.10)
    grain_tile("grain-dark.png", IVORY, peak=0.06)

"""
generate_weave.py — Tessitura Binary Matrix Visualization

Renders a 25x8 binary matrix as a woven-textile pattern using the
8-Law Weave palette.  Each column = one of the 8 tracked primes (in
LAW_PRIMES order); each row = one card value.  Filled cells glow with
the prime's Law color; empty cells show Nuit (#080620).

Outputs:
  assets/weave_pattern.png       — high-res (2400x800+)
  assets/weave_banner.png        — Twitter-card banner (3200x400)

Uses demonstration data (seed=42, ~20% fill) to protect corpus IP.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
NUIT = (8, 6, 32)  # #080620

LAW_COLORS = [
    (61, 61, 107),   # 11 — Fall of Neutral Events — indaco
    (139, 26, 139),  # 23 — Kaos — viola
    (179, 68, 68),   # 67 — Geometric Essence — mattone
    (255, 140, 66),  # 17 — Arrow of Complexity — arancio
    (220, 201, 100), # 29 — Fall of Events — oro
    (46, 125, 50),   # 19 — Time Matrix — verde
    (74, 144, 226),  # 89 — Synchronicity — azzurro
    (245, 245, 220), # 31 — Geometric Essence 2 — avorio
]

GRID_COLOR = (220, 201, 100)  # gold border between cells

# ---------------------------------------------------------------------------
# Demonstration matrix  (seed=42, ~20% fill rate)
# ---------------------------------------------------------------------------
def make_demo_matrix(rows: int = 25, cols: int = 8, fill: float = 0.20,
                     seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((rows, cols)) < fill).astype(np.uint8)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _lerp_color(c1: tuple, c2: tuple, t: float) -> tuple:
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def _hex_to_rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def render_weave(
    matrix: np.ndarray,
    *,
    cell_w: int = 96,
    cell_h: int = 96,
    corner_r: int = 12,
    border: int = 2,
    glow_radius: int = 6,
    glow_intensity: float = 0.55,
    padding_top: int = 0,
    padding_bottom: int = 0,
    padding_left: int = 0,
    padding_right: int = 0,
) -> Image.Image:
    """Render the binary matrix as a glowing woven-textile image."""
    rows, cols = matrix.shape
    # total canvas size
    img_w = padding_left + cols * (cell_w + border) + border + padding_right
    img_h = padding_top + rows * (cell_h + border) + border + padding_bottom

    # base layer — Nuit background
    base = Image.new("RGB", (img_w, img_h), NUIT)
    draw = ImageDraw.Draw(base)

    # glow accumulation layer (additive, per-cell)
    glow_layer = Image.new("RGB", (img_w, img_h), (0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_layer)

    for r in range(rows):
        for c in range(cols):
            x0 = padding_left + border + c * (cell_w + border)
            y0 = padding_top + border + r * (cell_h + border)
            x1 = x0 + cell_w
            y1 = y0 + cell_h

            if matrix[r, c]:
                color = LAW_COLORS[c]
                # draw rounded rectangle
                draw.rounded_rectangle(
                    [x0, y0, x1, y1], radius=corner_r, fill=color
                )
                # glow: draw a slightly larger rounded rect on glow layer
                expand = glow_radius
                glow_draw.rounded_rectangle(
                    [x0 - expand, y0 - expand, x1 + expand, y1 + expand],
                    radius=corner_r + expand // 2,
                    fill=color,
                )
            else:
                # empty cell — Nuit (already background, but draw subtle
                # inset so the grid lines are visible)
                draw.rounded_rectangle(
                    [x0, y0, x1, y1], radius=corner_r, fill=NUIT
                )

    # Draw grid lines (gold) on the base UNDER the cells — we actually
    # achieve this by drawing thin gold lines in the gaps.
    grid_layer = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(grid_layer)
    # horizontal lines
    for r in range(rows + 1):
        y = padding_top + r * (cell_h + border)
        gdraw.rectangle(
            [padding_left, y, img_w - padding_right, y + border - 1],
            fill=(*GRID_COLOR, 60),
        )
    # vertical lines
    for c in range(cols + 1):
        x = padding_left + c * (cell_w + border)
        gdraw.rectangle(
            [x, padding_top, x + border - 1, img_h - padding_bottom],
            fill=(*GRID_COLOR, 60),
        )

    # blur the glow layer
    glow_blurred = glow_layer.filter(ImageFilter.GaussianBlur(radius=glow_radius * 2))

    # composite: base + grid + glow (screen blend)
    # convert everything to RGBA for compositing
    result = base.convert("RGBA")
    result = Image.alpha_composite(result, grid_layer)

    # screen-blend the glow
    r_arr = np.array(result.convert("RGB"), dtype=np.float32)
    g_arr = np.array(glow_blurred, dtype=np.float32)
    # screen: 1 - (1-a)(1-b)  — scaled by glow_intensity
    g_arr = g_arr * glow_intensity
    screened = 255.0 - (255.0 - r_arr) * (255.0 - g_arr) / 255.0
    screened = np.clip(screened, 0, 255).astype(np.uint8)

    final = Image.fromarray(screened, "RGB")
    return final


def render_banner(matrix: np.ndarray, target_w: int = 3200,
                  target_h: int = 400) -> Image.Image:
    """Render a wide banner version optimized for Twitter card."""
    rows, cols = matrix.shape

    # compute cell sizes to fill target dimensions
    border = 2
    cell_w = (target_w - (cols + 1) * border) // cols
    cell_h = (target_h - (rows + 1) * border) // rows

    # for a wide banner with 25 rows, rows will be tiny — transpose
    # so primes are rows and cards are columns, giving 8 tall rows
    # and 25 narrow columns
    mat_t = matrix.T  # 8 x 25

    rows_t, cols_t = mat_t.shape
    cell_w_b = (target_w - (cols_t + 1) * border) // cols_t
    cell_h_b = (target_h - (rows_t + 1) * border) // rows_t

    # need to reassign colors: in transposed form, row index = prime index
    # we handle this by creating a custom render

    img = Image.new("RGB", (target_w, target_h), NUIT)
    draw = ImageDraw.Draw(img)
    glow_layer = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_layer)

    corner_r = max(4, min(cell_w_b, cell_h_b) // 8)
    glow_radius = 4

    # center the grid
    grid_w = cols_t * (cell_w_b + border) + border
    grid_h = rows_t * (cell_h_b + border) + border
    off_x = (target_w - grid_w) // 2
    off_y = (target_h - grid_h) // 2

    for r in range(rows_t):
        color = LAW_COLORS[r]
        for c in range(cols_t):
            x0 = off_x + border + c * (cell_w_b + border)
            y0 = off_y + border + r * (cell_h_b + border)
            x1 = x0 + cell_w_b
            y1 = y0 + cell_h_b

            if mat_t[r, c]:
                draw.rounded_rectangle(
                    [x0, y0, x1, y1], radius=corner_r, fill=color
                )
                expand = glow_radius
                glow_draw.rounded_rectangle(
                    [x0 - expand, y0 - expand, x1 + expand, y1 + expand],
                    radius=corner_r + expand // 2,
                    fill=color,
                )
            else:
                draw.rounded_rectangle(
                    [x0, y0, x1, y1], radius=corner_r, fill=NUIT
                )

    # grid lines
    grid_overlay = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(grid_overlay)
    for r in range(rows_t + 1):
        y = off_y + r * (cell_h_b + border)
        gdraw.rectangle(
            [off_x, y, off_x + grid_w, y + border - 1],
            fill=(*GRID_COLOR, 50),
        )
    for c in range(cols_t + 1):
        x = off_x + c * (cell_w_b + border)
        gdraw.rectangle(
            [x, off_y, x + border - 1, off_y + grid_h],
            fill=(*GRID_COLOR, 50),
        )

    # glow blur + screen blend
    glow_blurred = glow_layer.filter(
        ImageFilter.GaussianBlur(radius=glow_radius * 2)
    )

    result = img.convert("RGBA")
    result = Image.alpha_composite(result, grid_overlay)
    r_arr = np.array(result.convert("RGB"), dtype=np.float32)
    g_arr = np.array(glow_blurred, dtype=np.float32) * 0.5
    screened = 255.0 - (255.0 - r_arr) * (255.0 - g_arr) / 255.0
    screened = np.clip(screened, 0, 255).astype(np.uint8)

    return Image.fromarray(screened, "RGB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    out_dir = Path(__file__).resolve().parent.parent / "assets"
    out_dir.mkdir(exist_ok=True)

    matrix = make_demo_matrix()
    print(f"Demo matrix: {matrix.shape}, fill rate: {matrix.mean():.1%}")
    print(matrix)

    # --- High-res grid asset ---
    print("\nRendering high-res weave pattern...")
    img = render_weave(
        matrix,
        cell_w=96,
        cell_h=32,
        corner_r=8,
        border=2,
        glow_radius=5,
        glow_intensity=0.55,
        padding_top=20,
        padding_bottom=20,
        padding_left=20,
        padding_right=20,
    )
    path_main = out_dir / "weave_pattern.png"
    img.save(str(path_main), dpi=(300, 300))
    print(f"  Saved: {path_main}  ({img.width}x{img.height})")

    # --- Wide banner ---
    print("Rendering banner...")
    banner = render_banner(matrix, target_w=3200, target_h=400)
    path_banner = out_dir / "weave_banner.png"
    banner.save(str(path_banner), dpi=(144, 144))
    print(f"  Saved: {path_banner}  ({banner.width}x{banner.height})")

    print("\nDone.")


if __name__ == "__main__":
    main()

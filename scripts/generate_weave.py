"""
generate_weave.py — Tessitura Binary Matrix Visualization

Renders a 25×8 binary matrix as a woven-textile pattern using the
8-Law Weave palette.  The display is TRANSPOSED: 8 rows (one per prime/Law)
by 25 columns (one per card).  Filled cells glow with the prime's Law color;
empty cells show Nuit (#080620).

Outputs:
  assets/weave_pattern.png   — high-res section divider (~2400x800)
  assets/weave_banner.png    — Twitter-card banner (3200x400)

The matrix encodes prime-thread presence (statistical characterization) —
which of the 8 tracked primes thread through each card's analytical network.
This is derived data, not the corpus values themselves.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
NUIT = (8, 6, 32)       # #080620 — Egyptian night sky
NUIT_DIM = (12, 10, 38) # slightly lighter for empty cells (depth)

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

GRID_COLOR = (220, 201, 100)  # gold warp/weft lines


# ---------------------------------------------------------------------------
# Tessitura binary matrix — prime-thread presence per card
# ---------------------------------------------------------------------------
# Columns match LAW_COLORS order: [11, 23, 67, 17, 29, 19, 89, 31]
# Rows: 25 cards in deck order (0-XXI + Transit + Grail + Rose)
# Source: Appendix C — prime threading table (statistical characterization)
#
# 1 = prime threads through this card's analytical network
# 0 = prime absent from this card's operations
#
# The 25th row (Rose/XXII) is the system card — included in the visual
# but not part of the 24-card programmatic weight sequence.
# fmt: off
TESSITURA_MATRIX = np.array([
    #                  11  23  67  17  29  19  89  31
    [0, 0, 0, 0, 0, 0, 1, 0],  # 0    Fool
    [0, 1, 1, 0, 0, 1, 1, 1],  # I    Magician
    [0, 0, 1, 0, 0, 0, 0, 1],  # II   Priestess
    [0, 1, 0, 1, 0, 0, 1, 1],  # III  Empress
    [0, 0, 0, 0, 0, 0, 1, 0],  # IV   Emperor
    [0, 0, 0, 0, 0, 0, 0, 1],  # V    Hierophant
    [1, 0, 0, 0, 0, 0, 0, 0],  # VI   Lovers
    [1, 0, 0, 0, 1, 0, 1, 0],  # VII  Chariot
    [1, 1, 1, 1, 1, 1, 1, 1],  # VIII Justice
    [0, 0, 0, 1, 0, 1, 1, 0],  # IX   Hermit
    [0, 0, 0, 0, 1, 0, 1, 1],  # X    Wheel
    [0, 0, 0, 1, 0, 0, 0, 0],  # XI   Strength
    [1, 1, 0, 0, 0, 0, 1, 0],  # XII  Reversal
    [0, 1, 0, 1, 0, 0, 0, 0],  # XIII Renewal
    [1, 0, 0, 1, 1, 0, 0, 1],  # XIV  Temperance
    [1, 1, 0, 0, 1, 0, 1, 0],  # XV   Fallen Angel
    [1, 1, 0, 0, 0, 0, 0, 1],  # T    Transit
    [1, 1, 0, 0, 0, 1, 0, 0],  # XVI  Tower
    [1, 0, 0, 1, 1, 0, 0, 0],  # XVII Star
    [0, 0, 1, 0, 0, 0, 0, 1],  # XVIII Moon
    [0, 0, 0, 1, 0, 1, 0, 0],  # XIX  Sun
    [0, 1, 1, 0, 0, 0, 1, 0],  # XX   Judgment
    [1, 0, 1, 0, 0, 0, 1, 0],  # XXI  World
    [1, 1, 0, 1, 1, 0, 0, 0],  # G    Grail
    [0, 0, 0, 0, 1, 0, 0, 1],  # XXII Rose (system card)
], dtype=np.uint8)
# fmt: on


# ---------------------------------------------------------------------------
# Core renderer
# ---------------------------------------------------------------------------
def render_weave(
    matrix: np.ndarray,
    *,
    cell_w: int,
    cell_h: int,
    corner_r: int = 8,
    border: int = 2,
    glow_radius: int = 6,
    glow_intensity: float = 0.55,
    padding: int = 24,
    padding_h: int | None = None,
    padding_v: int | None = None,
    grid_alpha: int = 45,
    vignette: bool = True,
    vignette_strength: float = 0.35,
) -> Image.Image:
    """
    Render the transposed binary matrix as a glowing woven-textile image.

    The matrix is transposed so that:
      - Rows    = 8 primes (each row gets its Law color)
      - Columns = 25 cards

    padding_h / padding_v override padding for horizontal / vertical
    independently, allowing exact pixel targeting.
    """
    mat = matrix.T  # (25, 8) -> (8, 25)
    n_rows, n_cols = mat.shape

    pad_x = padding_h if padding_h is not None else padding
    pad_y = padding_v if padding_v is not None else padding

    grid_w = n_cols * (cell_w + border) + border
    grid_h = n_rows * (cell_h + border) + border
    img_w = grid_w + 2 * pad_x
    img_h = grid_h + 2 * pad_y

    # --- Base layer: Nuit ---
    base = Image.new("RGB", (img_w, img_h), NUIT)
    draw = ImageDraw.Draw(base)

    # --- Glow accumulation ---
    glow_layer = Image.new("RGB", (img_w, img_h), (0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_layer)

    # --- Draw cells ---
    for r in range(n_rows):
        color = LAW_COLORS[r]
        for c in range(n_cols):
            x0 = pad_x + border + c * (cell_w + border)
            y0 = pad_y + border + r * (cell_h + border)
            x1 = x0 + cell_w
            y1 = y0 + cell_h

            if mat[r, c]:
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
                    [x0, y0, x1, y1], radius=corner_r, fill=NUIT_DIM
                )

    # --- Gold grid lines ---
    grid_layer = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(grid_layer)
    for r in range(n_rows + 1):
        y = pad_y + r * (cell_h + border)
        gdraw.rectangle(
            [pad_x, y, pad_x + grid_w - 1, y + border - 1],
            fill=(*GRID_COLOR, grid_alpha),
        )
    for c in range(n_cols + 1):
        x = pad_x + c * (cell_w + border)
        gdraw.rectangle(
            [x, pad_y, x + border - 1, pad_y + grid_h - 1],
            fill=(*GRID_COLOR, grid_alpha),
        )

    # --- Gaussian blur on glow ---
    glow_blurred = glow_layer.filter(
        ImageFilter.GaussianBlur(radius=glow_radius * 2.5)
    )

    # --- Composite: base + grid + screen-blended glow ---
    result = base.convert("RGBA")
    result = Image.alpha_composite(result, grid_layer)

    r_arr = np.array(result.convert("RGB"), dtype=np.float32)
    g_arr = np.array(glow_blurred, dtype=np.float32) * glow_intensity
    screened = 255.0 - (255.0 - r_arr) * (255.0 - g_arr) / 255.0
    screened = np.clip(screened, 0, 255).astype(np.uint8)
    final = Image.fromarray(screened, "RGB")

    if vignette:
        final = _apply_vignette(final, strength=vignette_strength)

    return final


def _apply_vignette(img: Image.Image, strength: float = 0.3) -> Image.Image:
    """Radial vignette — darkens edges, keeps center bright."""
    w, h = img.size
    arr = np.array(img, dtype=np.float32)
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    cx, cy = w / 2.0, h / 2.0
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / max_dist
    falloff = 1.0 - strength * (dist ** 1.8)
    falloff = np.clip(falloff, 0, 1)
    arr *= falloff[:, :, np.newaxis]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    out_dir = Path(__file__).resolve().parent.parent / "assets"
    out_dir.mkdir(exist_ok=True)

    matrix = TESSITURA_MATRIX
    print(f"Tessitura matrix: {matrix.shape}, fill rate: {matrix.mean():.1%}")

    # --- High-res section divider ---
    # Transposed: 8 rows x 25 cols.
    # cell 92x92 + border 2 + padding 24 => 2400 x 802
    print("\nRendering high-res weave pattern...")
    pattern = render_weave(
        matrix,
        cell_w=92,
        cell_h=92,
        corner_r=14,
        border=2,
        glow_radius=8,
        glow_intensity=0.6,
        padding=24,
    )
    path_main = out_dir / "weave_pattern.png"
    pattern.save(str(path_main), dpi=(300, 300))
    print(f"  Saved: {path_main}  ({pattern.width}x{pattern.height})")

    # --- Twitter-card banner (exact: 3200 x 400) ---
    # cw=124, brd=2: grid_w = 25*(124+2)+2 = 3152. pad_h = (3200-3152)//2 = 24
    # ch=43,  brd=2: grid_h = 8*(43+2)+2   = 362.  pad_v = (400-362)//2   = 19
    print("Rendering Twitter banner...")
    banner = render_weave(
        matrix,
        cell_w=124,
        cell_h=43,
        corner_r=8,
        border=2,
        glow_radius=5,
        glow_intensity=0.5,
        padding_h=24,
        padding_v=19,
    )
    path_banner = out_dir / "weave_banner.png"
    banner.save(str(path_banner), dpi=(144, 144))
    print(f"  Saved: {path_banner}  ({banner.width}x{banner.height})")

    print("\nDone.")


if __name__ == "__main__":
    main()

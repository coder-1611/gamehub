#!/usr/bin/env python3
"""
Extract mini golf course data from a reference screenshot.

Detects: course boundary polygon, ball start, hole position, slope arrows,
white obstacle walls. Outputs JSON + ready-to-paste JavaScript for game.html.

Usage:
    python3 extract_course.py [reference_image_path]
    python3 extract_course.py reference.jpeg
    python3 extract_course.py screenshot.png --size 400x600 -o course.json
"""

import sys
import os
import json
import base64
import io
import argparse

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

# Try to import cv2 for better contour detection; fall back to scipy
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("[INFO] OpenCV not available, using scipy for contour detection.")

# ── Constants ──────────────────────────────────────────────────────────────
W, H = 400, 600  # defaults, overridden at runtime
MARGIN = 25
BG_COLOR = [193, 184, 175]
BASE_GREEN = np.array([77, 142, 67])  # typical course green RGB
SLOPE_FORCE = 0.05


# ── Utility: green mask on original image ──────────────────────────────────
def make_green_mask(arr):
    """Detect green course region."""
    gm = ((arr[:, :, 1] > 80) & (arr[:, :, 0] < 180) & (arr[:, :, 2] < 120) &
           (arr[:, :, 1] > arr[:, :, 0]) & (arr[:, :, 1] > arr[:, :, 2]))
    gm = ndimage.binary_fill_holes(gm)
    gm = ndimage.binary_erosion(gm, iterations=3)
    gm = ndimage.binary_dilation(gm, iterations=3)
    return gm


# ── Step 1: Compute canvas size and scale reference ────────────────────────
def compute_canvas_size(green_mask):
    ys, xs = np.where(green_mask)
    if len(ys) == 0:
        print("[ERROR] No green region detected.")
        sys.exit(1)
    img_w = xs.max() - xs.min()
    img_h = ys.max() - ys.min()
    aspect = img_w / img_h
    TARGET_AREA = 240000
    canvas_h = int(np.sqrt(TARGET_AREA / aspect))
    canvas_w = int(canvas_h * aspect)
    canvas_w = max(200, (canvas_w + 1) // 2 * 2)
    canvas_h = max(200, (canvas_h + 1) // 2 * 2)
    return canvas_w, canvas_h


def scale_image(arr, green_mask, canvas_w, canvas_h):
    ys, xs = np.where(green_mask)
    if len(ys) == 0:
        sys.exit(1)
    img_xmin, img_xmax = xs.min(), xs.max()
    img_ymin, img_ymax = ys.min(), ys.max()
    img_w = img_xmax - img_xmin
    img_h = img_ymax - img_ymin
    scale_x = (canvas_w - 2 * MARGIN) / img_w
    scale_y = (canvas_h - 2 * MARGIN) / img_h
    scale = min(scale_x, scale_y)
    ox = (canvas_w - img_w * scale) / 2 - img_xmin * scale
    oy = (canvas_h - img_h * scale) / 2 - img_ymin * scale
    print(f"  Canvas: {canvas_w}x{canvas_h}, Scale: {scale:.4f}, Offset: ({ox:.1f}, {oy:.1f})")

    canvas = np.full((canvas_h, canvas_w, 3), BG_COLOR, dtype=np.uint8)
    for cy in range(canvas_h):
        for cx in range(canvas_w):
            ix = (cx - ox) / scale
            iy = (cy - oy) / scale
            ixi, iyi = int(ix), int(iy)
            if 0 <= ixi < arr.shape[1] - 1 and 0 <= iyi < arr.shape[0] - 1:
                fx, fy = ix - ixi, iy - iyi
                c00 = arr[iyi, ixi].astype(float)
                c10 = arr[iyi, ixi + 1].astype(float)
                c01 = arr[iyi + 1, ixi].astype(float)
                c11 = arr[iyi + 1, ixi + 1].astype(float)
                val = c00*(1-fx)*(1-fy) + c10*fx*(1-fy) + c01*(1-fx)*fy + c11*fx*fy
                canvas[cy, cx] = np.clip(val, 0, 255).astype(np.uint8)
            elif 0 <= ixi < arr.shape[1] and 0 <= iyi < arr.shape[0]:
                canvas[cy, cx] = arr[iyi, ixi]
    return canvas, scale, ox, oy


def make_canvas_green_mask(canvas):
    gm = ((canvas[:, :, 1] > 80) & (canvas[:, :, 0] < 180) & (canvas[:, :, 2] < 120) &
           (canvas[:, :, 1] > canvas[:, :, 0]) & (canvas[:, :, 1] > canvas[:, :, 2]))
    gm = ndimage.binary_fill_holes(gm)
    return gm


# ── Detect yellow circle ──────────────────────────────────────────────────
def detect_yellow(canvas):
    # Broader detection: R>170, G>130, B<120 (catches gold/amber circles too)
    yellow = ((canvas[:, :, 0] > 170) & (canvas[:, :, 1] > 130) & (canvas[:, :, 2] < 120) &
              (canvas[:, :, 0] > canvas[:, :, 2] + 40))
    count = np.sum(yellow)
    if count > 0:
        ys, xs = np.where(yellow)
        # Dilate to get the full circle area including anti-aliased edges
        yellow = ndimage.binary_dilation(yellow, iterations=5)
        print(f"  Yellow circle: {count} px, center ~({int(np.mean(xs))}, {int(np.mean(ys))})")
    else:
        print("  Yellow circle: not found")
    return yellow


# ── Detect ball (circular outline, not solid white) ───────────────────────
def detect_ball(canvas, green_mask):
    """Find the ball start position. The ball is a faint circular outline (ring)
    on the green, NOT a solid white shape. Look for ring-shaped patterns."""
    # Method 1: Find grayish ring patterns (the ball outline is usually a faint ring)
    r, g, b = canvas[:, :, 0].astype(int), canvas[:, :, 1].astype(int), canvas[:, :, 2].astype(int)

    # Ring pixels deviate from base green — use adaptive base from course median
    green_pixels_r = np.median(r[green_mask])
    green_pixels_g = np.median(g[green_mask])
    green_pixels_b = np.median(b[green_mask])
    dist_from_green = (np.abs(r - green_pixels_r) + np.abs(g - green_pixels_g) + np.abs(b - green_pixels_b))
    # Not white (obstacles), not dark (holes), just off-green
    ring_pixels = (green_mask &
                   (dist_from_green > 15) & (dist_from_green < 450) &
                   (r < 220) & (g < 230) & (b < 220) &
                   (r > 50) & (g > 50) & (b > 50))

    labeled, n = ndimage.label(ring_pixels)
    ball_pos = None
    ball_mask = np.zeros(canvas.shape[:2], dtype=bool)

    ring_candidates = []
    canvas_h = canvas.shape[0]
    for i in range(1, n + 1):
        comp = labeled == i
        size = np.sum(comp)
        if 20 < size < 800:
            ys, xs = np.where(comp)
            cx, cy = int(np.mean(xs)), int(np.mean(ys))
            bw = xs.max() - xs.min() + 1
            bh = ys.max() - ys.min() + 1
            fill_ratio = size / max(1, bw * bh)
            aspect = max(bw, bh) / max(1, min(bw, bh))
            # Hard-reject extreme aspect ratios (lines, not rings)
            if aspect > 5.0:
                continue
            # Rings have low fill ratio (< 0.55) and are roughly circular
            is_ring = fill_ratio < 0.55 and aspect < 2.0 and bw > 15 and bh > 15
            # Larger rings (100-400px, bbox 25x25+) are much more likely to be real
            size_bonus = 0
            if size > 80 and bw > 25 and bh > 25:
                size_bonus = 300  # strong preference for real-sized rings
            elif size < 50:
                size_bonus = -200  # penalize tiny candidates (likely edge artifacts)
            # Penalize candidates very close to course boundary (bottom/top 5%)
            edge_penalty = 0
            if cy > canvas_h * 0.95 or cy < canvas_h * 0.05:
                edge_penalty = -150
            score = (200 if is_ring else 0) + size_bonus + edge_penalty + cy * 0.3
            ring_candidates.append((score, size, cx, cy, comp, bw, bh, is_ring))
            print(f"    Ring candidate: ({cx},{cy}) {bw}x{bh} fill={fill_ratio:.2f} ring={is_ring} score={score:.0f} size={size}")

    if ring_candidates:
        ring_candidates.sort(key=lambda c: -c[0])
        score, size, bx, by, comp, bw, bh, is_ring = ring_candidates[0]
        ball_pos = {"x": bx, "y": by}
        ball_mask = ndimage.binary_dilation(comp, iterations=5)
        print(f"  Ball: {size}px at ({bx},{by}), bbox {bw}x{bh}, ring={is_ring}")
    else:
        # Fallback: look for small solid white dot
        white = (canvas[:, :, 0] > 220) & (canvas[:, :, 1] > 220) & (canvas[:, :, 2] > 220)
        white_in_green = white & green_mask
        labeled2, n2 = ndimage.label(white_in_green)
        for i in range(1, n2 + 1):
            comp = labeled2 == i
            size = np.sum(comp)
            if 5 < size < 50:
                ys, xs = np.where(comp)
                ball_pos = {"x": int(np.mean(xs)), "y": int(np.mean(ys))}
                ball_mask = ndimage.binary_dilation(comp, iterations=4)
                print(f"  Ball (solid dot fallback): {size}px at ({ball_pos['x']},{ball_pos['y']})")
                break

    if ball_pos is None:
        print("  [WARN] Ball not detected.")

    return ball_pos, ball_mask


# ── Detect hole position ─────────────────────────────────────────────────
def detect_hole(canvas, green_mask):
    """Find hole position. Looks for:
    1. Dark circle on green (hole itself)
    2. Flag pole (thin dark vertical line) with red flag on top
    3. Position near flag/red pixels"""

    # Method 1: dark circle on green
    dark = (canvas[:, :, 0] < 50) & (canvas[:, :, 1] < 50) & (canvas[:, :, 2] < 50)
    dark_on_green = dark & green_mask

    labeled, n = ndimage.label(dark_on_green)
    hole_candidates = []
    for i in range(1, n + 1):
        comp = labeled == i
        size = np.sum(comp)
        if 3 < size < 2000:
            ys, xs = np.where(comp)
            hole_candidates.append((size, int(np.mean(xs)), int(np.mean(ys))))

    # Method 2: find flag (red pixels) and infer hole position near pole base
    red = (canvas[:, :, 0] > 150) & (canvas[:, :, 1] < 100) & (canvas[:, :, 2] < 100)
    red_on_green = red & ndimage.binary_dilation(green_mask, iterations=10)
    flag_x, flag_y = None, None
    if np.sum(red_on_green) > 5:
        rys, rxs = np.where(red_on_green)
        flag_x, flag_y = int(np.mean(rxs)), int(np.mean(rys))
        # The hole is typically at the base of the flag pole
        pole_bottom_y = flag_y
        for check_y in range(flag_y, min(flag_y + 60, canvas.shape[0])):
            col_slice = canvas[check_y, max(0, flag_x-3):min(canvas.shape[1], flag_x+4), :]
            if np.any((col_slice[:, 0] < 60) & (col_slice[:, 1] < 60) & (col_slice[:, 2] < 60)):
                pole_bottom_y = check_y
        hole_from_flag = (100, flag_x, pole_bottom_y)  # fallback priority
        hole_candidates.append(hole_from_flag)
        print(f"  Flag found at ({flag_x}, {flag_y}), pole bottom ~{pole_bottom_y}")

    if hole_candidates:
        # If we have a flag, find the hole at the bottom of the dark region
        # near the pole base (the hole circle is at the base of the pole)
        if flag_x is not None:
            # Find the hole circle at the base of the pole.
            # Look for dark pixels in a small horizontal band near pole bottom.
            # The hole circle extends wider than the thin pole.
            pole_bottom_y = max(c[2] for c in hole_candidates)
            search_y_min = max(0, pole_bottom_y - 8)
            search_y_max = min(canvas.shape[0], pole_bottom_y + 4)
            search_x_min = max(0, flag_x - 15)
            search_x_max = min(canvas.shape[1], flag_x + 15)
            # Use raw dark pixels (not masked by green since dark != green)
            local_dark = dark[search_y_min:search_y_max, search_x_min:search_x_max]
            if np.sum(local_dark) > 2:
                lys, lxs = np.where(local_dark)
                hx = int(np.mean(lxs)) + search_x_min
                hy = int(np.mean(lys)) + search_y_min
                print(f"  Hole: dark cluster at ({hx}, {hy}) near pole base")
                return {"x": hx, "y": hy}

        # Fallback: largest candidate
        hole_candidates.sort(key=lambda c: -c[0])
        size, hx, hy = hole_candidates[0]
        print(f"  Hole: at ({hx}, {hy}) (confidence={size})")
        return {"x": hx, "y": hy}
    else:
        print("  [WARN] Hole not detected.")
        return None


# ── Detect flag/pole for removal ─────────────────────────────────────────
def detect_flag(canvas, green_mask):
    """Find flag/pole pixels for background removal."""
    # Red flag pixels
    red = (canvas[:, :, 0] > 140) & (canvas[:, :, 1] < 100) & (canvas[:, :, 2] < 100)
    # Dark pole pixels (thin vertical dark line near red)
    dark = (canvas[:, :, 0] < 60) & (canvas[:, :, 1] < 60) & (canvas[:, :, 2] < 60)

    red_on_green = red & ndimage.binary_dilation(green_mask, iterations=10)

    # Find the flag area and expand to include pole
    flag_mask = np.zeros(canvas.shape[:2], dtype=bool)

    if np.sum(red_on_green) > 3:
        rys, rxs = np.where(red_on_green)
        flag_x = int(np.mean(rxs))
        flag_y = int(np.mean(rys))

        # Mark a generous rectangle around the flag + pole
        # Flag is above, pole extends below
        y_min = max(0, min(rys) - 5)
        y_max = min(canvas.shape[0], max(rys) + 60)  # pole extends below flag
        x_min = max(0, flag_x - 12)
        x_max = min(canvas.shape[1], flag_x + 15)
        flag_mask[y_min:y_max, x_min:x_max] = True

        # Only remove pixels that are red, dark, or near them (not white obstacles)
        near_flag = flag_mask & (red | dark | ndimage.binary_dilation(red, iterations=3))
        # Also get the pole: dark pixels in the flag column area
        pole_col_mask = np.zeros_like(flag_mask)
        pole_col_mask[y_min:y_max, max(0,flag_x-2):min(canvas.shape[1],flag_x+3)] = True
        near_flag |= (pole_col_mask & dark)

        flag_mask = ndimage.binary_dilation(near_flag, iterations=3)
        count = np.sum(flag_mask)
        print(f"  Flag/pole: {count} px, area y=[{y_min},{y_max}] x=[{x_min},{x_max}]")
    else:
        print("  Flag/pole: not found")

    return flag_mask


def _merge_gradient_bands(bands, gap=20):
    """Merge gradient bands that overlap or are within `gap` pixels of each other.
    Light and dark halves of the same slope should merge into one region."""
    if not bands:
        return []
    merged = list(bands)
    changed = True
    while changed:
        changed = False
        new_merged = []
        used = [False] * len(merged)
        for i in range(len(merged)):
            if used[i]:
                continue
            a = merged[i]
            ax1, ay1 = a['x'], a['y']
            ax2, ay2 = ax1 + a['w'], ay1 + a['h']
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                b = merged[j]
                bx1, by1 = b['x'], b['y']
                bx2, by2 = bx1 + b['w'], by1 + b['h']
                # Check overlap with gap tolerance
                if (ax1 - gap <= bx2 and ax2 + gap >= bx1 and
                    ay1 - gap <= by2 and ay2 + gap >= by1):
                    # Merge bounding boxes
                    nx1 = min(ax1, bx1)
                    ny1 = min(ay1, by1)
                    nx2 = max(ax2, bx2)
                    ny2 = max(ay2, by2)
                    a = {'x': nx1, 'y': ny1, 'w': nx2 - nx1, 'h': ny2 - ny1,
                         'dir': a['dir'], 'type': 'merged', 'size': a['size'] + b['size']}
                    ax1, ay1, ax2, ay2 = nx1, ny1, nx2, ny2
                    used[j] = True
                    changed = True
            new_merged.append(a)
        merged = new_merged
    return merged


# ── Detect slope indicators ──────────────────────────────────────────────
def detect_slopes(canvas, green_mask, wall_positions=None):
    """
    Detect slopes by finding:
    1. Faint white triangles on the green (slope direction indicators)
    2. Lighter/darker green gradient bands (slope regions)

    wall_positions: list of (cx, cy) of detected wall obstacles, used to filter
    out triangles that are actually wall obstacles (not slope indicators).
    """
    slopes = []

    # Method 1: Find faint white triangles (slope direction arrows)
    triangles = _find_triangles(canvas, green_mask)

    # Method 2: Find gradient bands (lighter or darker green rectangles)
    # These are the actual slope regions
    gradient_bands = _find_gradient_bands(canvas, green_mask)

    # Merge gradient bands that overlap or are adjacent (light + dark halves of same slope)
    merged_bands = _merge_gradient_bands(gradient_bands)

    # For each merged band, use a nearby triangle indicator for direction if available
    for band in merged_bands:
        bx, by, bw, bh = band['x'], band['y'], band['w'], band['h']
        bcx, bcy = bx + bw / 2, by + bh / 2
        # Find triangle indicator inside or near this band (use generous 30px tolerance
        # because bands may not have expanded fully to cover the triangle location)
        best_tri = None
        best_dist = 999
        for tri in triangles:
            if (bx - 30 <= tri['x'] <= bx + bw + 30 and
                by - 30 <= tri['y'] <= by + bh + 30):
                dist = abs(tri['x'] - bcx) + abs(tri['y'] - bcy)
                if dist < best_dist:
                    best_dist = dist
                    best_tri = tri
        # Use triangle direction if found, otherwise fall back to gradient direction
        direction = best_tri['dir'] if best_tri else band['dir']
        slopes.append({
            "x": bx, "y": by, "w": bw, "h": bh, "dir": direction
        })
        src = f"tri@({best_tri['x']},{best_tri['y']})" if best_tri else "gradient"
        print(f"  Slope: ({bx},{by}) {bw}x{bh} dir={direction} (from {src})")

    if not slopes:
        print("  No slopes detected.")

    return slopes


def _find_triangles(canvas, green_mask):
    """Find small white/light triangular shapes on the green that indicate slope direction."""
    r, g, b = canvas[:, :, 0].astype(int), canvas[:, :, 1].astype(int), canvas[:, :, 2].astype(int)

    # White/light pixels on green
    white = (canvas[:, :, 0] > 200) & (canvas[:, :, 1] > 200) & (canvas[:, :, 2] > 200)
    white_on_green = white & green_mask

    labeled, n = ndimage.label(white_on_green)
    triangles = []

    for i in range(1, n + 1):
        comp = labeled == i
        size = np.sum(comp)
        # Triangles are small (20-500px), not large squares/rectangles
        if size < 15 or size > 600:
            continue

        ys, xs = np.where(comp)
        cx, cy = int(np.mean(xs)), int(np.mean(ys))
        bw = xs.max() - xs.min() + 1
        bh = ys.max() - ys.min() + 1

        # Check if it's roughly triangular (not a square/rectangle)
        # Triangles have significantly less fill than their bounding box
        fill_ratio = size / max(1, bw * bh)
        if fill_ratio > 0.75:
            continue  # too square, probably a rectangle obstacle

        # Skip if very elongated (probably a line, not triangle)
        if bw > bh * 4 or bh > bw * 4:
            continue

        # Determine direction using taper analysis
        direction = _infer_direction_from_shape(comp)

        triangles.append({"x": cx, "y": cy, "dir": direction, "size": size})
        print(f"    Triangle: ({cx},{cy}) {bw}x{bh} fill={fill_ratio:.2f} dir={direction} size={size}")

    return triangles


def _find_gradient_bands(canvas, green_mask):
    """Find rectangular areas where green is noticeably lighter or darker than base.
    Uses adaptive thresholds based on the actual green channel distribution."""
    g = canvas[:, :, 1].astype(float)
    r = canvas[:, :, 0].astype(float)

    # Compute adaptive thresholds from the green channel median.
    # Use proportional offsets: ~+17% for light, ~-31% for dark.
    # This generalizes across courses with different base green colors.
    green_g_values = g[green_mask]
    if len(green_g_values) == 0:
        return []
    median_g = float(np.median(green_g_values))
    median_r = float(np.median(r[green_mask]))

    light_thresh = median_g * 1.10   # ~+10% brighter than base
    dark_thresh = median_g * 0.78    # ~-22% darker than base

    print(f"    Gradient thresholds: median_G={median_g:.0f}, light>{light_thresh:.0f}, dark<{dark_thresh:.0f}")

    lighter = green_mask & (g > light_thresh) & (r > median_r - 20) & (r < median_r + 50)
    darker = green_mask & (g < dark_thresh) & (g > 60) & (r < median_r)

    bands = []
    for band_type, mask in [("light", lighter), ("dark", darker)]:
        if np.sum(mask) < 80:
            continue

        labeled, n = ndimage.label(mask)
        for i in range(1, n + 1):
            comp = labeled == i
            size = np.sum(comp)
            if size < 80:
                continue

            ys, xs = np.where(comp)
            bx, by = int(xs.min()), int(ys.min())
            bw = int(xs.max() - xs.min())
            bh = int(ys.max() - ys.min())

            # Skip bands that are too thin (< 3px in either dimension)
            if bw < 3 or bh < 3:
                continue

            direction = _infer_direction_from_gradient(canvas, comp)

            # Expand band: the detected region is only the extreme edge of the gradient.
            # The slope gradient transitions smoothly from the bright/dark extreme to
            # normal green, so we use a very soft threshold — expand as long as pixels
            # are even slightly off-median (0.5% deviation). Cap at 4x original size.
            soft_thresh = median_g * 1.005
            soft_dark = median_g * 0.995
            max_expand = max(bw, bh) * 4  # don't expand more than 4x original
            ex, ey, ew, eh = bx, by, bw, bh
            total_expanded = 0
            expanded = True
            while expanded and total_expanded < max_expand:
                expanded = False
                for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                    if dx == -1:
                        if ex <= 0: continue
                        check_col = g[ey:ey+eh+1, ex-1] if ey+eh+1 <= g.shape[0] and ex-1 >= 0 else np.array([])
                        check_green = green_mask[ey:ey+eh+1, ex-1] if ey+eh+1 <= green_mask.shape[0] and ex-1 >= 0 else np.array([])
                    elif dx == 1:
                        nx = ex + ew + 1
                        if nx >= g.shape[1]: continue
                        check_col = g[ey:ey+eh+1, nx] if ey+eh+1 <= g.shape[0] else np.array([])
                        check_green = green_mask[ey:ey+eh+1, nx] if ey+eh+1 <= green_mask.shape[0] else np.array([])
                    elif dy == -1:
                        if ey <= 0: continue
                        check_col = g[ey-1, ex:ex+ew+1] if ex+ew+1 <= g.shape[1] and ey-1 >= 0 else np.array([])
                        check_green = green_mask[ey-1, ex:ex+ew+1] if ex+ew+1 <= green_mask.shape[1] and ey-1 >= 0 else np.array([])
                    else:
                        ny = ey + eh + 1
                        if ny >= g.shape[0]: continue
                        check_col = g[ny, ex:ex+ew+1] if ex+ew+1 <= g.shape[1] else np.array([])
                        check_green = green_mask[ny, ex:ex+ew+1] if ex+ew+1 <= green_mask.shape[1] else np.array([])

                    if len(check_col) == 0 or len(check_green) == 0:
                        continue
                    green_frac = np.mean(check_green)
                    mean_g = np.mean(check_col)
                    if green_frac > 0.7 and (mean_g > soft_thresh or mean_g < soft_dark):
                        if dx == -1:
                            ex -= 1; ew += 1
                        elif dx == 1:
                            ew += 1
                        elif dy == -1:
                            ey -= 1; eh += 1
                        else:
                            eh += 1
                        expanded = True
                        total_expanded += 1

            bands.append({
                "x": ex, "y": ey, "w": ew, "h": eh,
                "dir": direction, "type": band_type, "size": size
            })
            print(f"    Gradient band: ({ex},{ey}) {ew}x{eh} {band_type} dir={direction} size={size} (expanded from {bw}x{bh})")

    return bands


def _infer_direction_from_shape(mask):
    """Determine arrow direction from triangle shape taper."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return 'S'

    ymin, ymax = ys.min(), ys.max()
    xmin, xmax = xs.min(), xs.max()
    height = ymax - ymin + 1
    width = xmax - xmin + 1

    if height < 2 and width < 2:
        return 'S'

    row_widths = np.array([np.sum(mask[y, :]) for y in range(ymin, ymax + 1)])
    col_heights = np.array([np.sum(mask[:, x]) for x in range(xmin, xmax + 1)])

    # Compute taper correlation
    if len(row_widths) >= 2:
        row_positions = np.arange(len(row_widths), dtype=float)
        row_corr = np.corrcoef(row_positions, row_widths.astype(float))[0, 1]
    else:
        row_corr = 0.0

    if len(col_heights) >= 2:
        col_positions = np.arange(len(col_heights), dtype=float)
        col_corr = np.corrcoef(col_positions, col_heights.astype(float))[0, 1]
    else:
        col_corr = 0.0

    if abs(row_corr) > abs(col_corr):
        return 'S' if row_corr < 0 else 'N'
    else:
        return 'E' if col_corr < 0 else 'W'


def _infer_direction_from_gradient(canvas, mask):
    """Infer slope direction from green channel brightness gradient."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return 'S'

    g_values = canvas[ys, xs, 1].astype(float)
    ymid = np.mean(ys)
    xmid = np.mean(xs)

    top = g_values[ys < ymid].mean() if np.any(ys < ymid) else 0
    bot = g_values[ys >= ymid].mean() if np.any(ys >= ymid) else 0
    left = g_values[xs < xmid].mean() if np.any(xs < xmid) else 0
    right = g_values[xs >= xmid].mean() if np.any(xs >= xmid) else 0

    h_diff = abs(top - bot)
    v_diff = abs(left - right)

    if h_diff > v_diff:
        return 'S' if top > bot else 'N'
    else:
        return 'E' if left > right else 'W'


# ── Detect white obstacles and convert to wall segments ──────────────────
def detect_walls(canvas, green_mask, ball_pos=None, slopes=None):
    """Find white obstacle shapes on the green and convert to wall segments.
    Filters out the ball position and slope indicator triangles.
    Returns (walls, wall_centers) where wall_centers is a list of (cx, cy)."""
    white = (canvas[:, :, 0] > 230) & (canvas[:, :, 1] > 230) & (canvas[:, :, 2] > 230)
    white_on_green = white & green_mask

    labeled, n = ndimage.label(white_on_green)

    walls = []
    wall_centers = []
    for i in range(1, n + 1):
        comp = labeled == i
        size = np.sum(comp)
        if size < 8 or size > 5000:
            continue

        ys, xs = np.where(comp)
        cx, cy = int(np.mean(xs)), int(np.mean(ys))

        # Skip clusters near the ball position (within 15px)
        if ball_pos and abs(cx - ball_pos['x']) < 15 and abs(cy - ball_pos['y']) < 15:
            print(f"  Skipping ball-like cluster: {size}px at ~({cx},{cy})")
            continue

        # Skip clusters that are slope direction indicators (small triangles inside slope bands)
        # Filter clusters that are inside a slope bbox AND have compact bounding box (both dims < 20px)
        bw = int(xs.max() - xs.min())
        bh = int(ys.max() - ys.min())
        if slopes and bw < 20 and bh < 20:
            is_slope_indicator = False
            for s in slopes:
                sx, sy, sw, sh = s['x'], s['y'], s['w'], s['h']
                if (sx - 10 <= cx <= sx + sw + 10) and (sy - 10 <= cy <= sy + sh + 10):
                    is_slope_indicator = True
                    break
            if is_slope_indicator:
                print(f"  Skipping slope indicator: {size}px ({bw}x{bh}) at ~({cx},{cy})")
                continue

        # Skip very small clusters that are likely noise (< 20px)
        if size < 20:
            print(f"  Skipping tiny cluster: {size}px at ~({cx},{cy})")
            continue

        segments = _contour_to_segments(comp)
        if segments:
            walls.extend(segments)
            wall_centers.append((cx, cy))
            print(f"  Wall obstacle: {size}px at ~({cx},{cy}), {len(segments)} segments")

    # Second pass: detect obstacles hidden under slope gradients.
    # Slope gradients are semi-transparent and rendered ON TOP of white obstacles,
    # blending white down below the >230 threshold. Within slope bboxes, look for
    # pixels that are significantly brighter than green-under-gradient (i.e. they
    # must be white objects underneath). Green under gradient has low R and B;
    # white under gradient retains high R and B (typically R>140, B>120).
    if slopes:
        r = canvas[:, :, 0].astype(int)
        g_ch = canvas[:, :, 1].astype(int)
        b = canvas[:, :, 2].astype(int)
        # Compute median green color for reference
        med_r = int(np.median(r[green_mask]))
        med_b = int(np.median(b[green_mask]))
        # Thresholds: pixels with R and B well above what green-under-gradient would have
        r_thresh = max(med_r + 80, 120)
        b_thresh = max(med_b + 80, 100)
        for s in slopes:
            sx, sy, sw, sh = s['x'], s['y'], s['w'], s['h']
            # Expand search area slightly beyond slope bbox
            pad = 5
            y0 = max(0, sy - pad)
            y1 = min(canvas.shape[0], sy + sh + pad)
            x0 = max(0, sx - pad)
            x1 = min(canvas.shape[1], sx + sw + pad)
            # Find bright anomalies in this region (white under gradient)
            region_mask = np.zeros(canvas.shape[:2], dtype=bool)
            region_r = r[y0:y1, x0:x1]
            region_b = b[y0:y1, x0:x1]
            region_g = g_ch[y0:y1, x0:x1]
            region_green = green_mask[y0:y1, x0:x1]
            # White-under-gradient: R well above green, B well above green, G also elevated
            bright = (region_r > r_thresh) & (region_b > b_thresh) & (region_g > 150) & region_green
            # Exclude pixels already detected as white (>230 threshold)
            already_white = (region_r > 230) & (region_g > 230) & (region_b > 230)
            bright = bright & ~already_white
            if np.sum(bright) < 15:
                continue
            # Place back into full-size mask
            region_mask[y0:y1, x0:x1] = bright
            # Also include any pure-white fragments in this region that were
            # previously filtered as slope indicators
            pure_white_region = (r > 230) & (g_ch > 230) & (b > 230) & green_mask
            pure_white_in_slope = np.zeros_like(region_mask)
            pure_white_in_slope[y0:y1, x0:x1] = pure_white_region[y0:y1, x0:x1]
            region_mask |= pure_white_in_slope
            # Cluster and filter
            labeled_s, n_s = ndimage.label(region_mask)
            # Merge all clusters in this slope region into one obstacle
            all_obstacle_px = np.zeros_like(region_mask)
            for i in range(1, n_s + 1):
                comp = labeled_s == i
                size = np.sum(comp)
                if size < 8:
                    continue
                ys_c, xs_c = np.where(comp)
                cx_c, cy_c = int(np.mean(xs_c)), int(np.mean(ys_c))
                bw_c = int(xs_c.max() - xs_c.min())
                bh_c = int(ys_c.max() - ys_c.min())
                # Skip if it looks like a slope indicator (small, compact)
                if bw_c < 16 and bh_c < 16 and size < 100:
                    continue
                # Skip if near ball
                if ball_pos and abs(cx_c - ball_pos['x']) < 15 and abs(cy_c - ball_pos['y']) < 15:
                    continue
                # Skip if already detected as a wall (center within 10px of existing)
                already_found = False
                for wc in wall_centers:
                    if abs(cx_c - wc[0]) < 10 and abs(cy_c - wc[1]) < 10:
                        already_found = True
                        break
                if already_found:
                    continue
                all_obstacle_px |= comp
            if np.sum(all_obstacle_px) >= 15:
                segments = _contour_to_segments(all_obstacle_px)
                if segments:
                    ys_a, xs_a = np.where(all_obstacle_px)
                    cx_a, cy_a = int(np.mean(xs_a)), int(np.mean(ys_a))
                    walls.extend(segments)
                    wall_centers.append((cx_a, cy_a))
                    print(f"  Wall obstacle (under slope gradient): {np.sum(all_obstacle_px)}px at ~({cx_a},{cy_a}), {len(segments)} segments")

    return walls, wall_centers


def _contour_to_segments(binary_mask):
    if HAS_CV2:
        return _contour_to_segments_cv2(binary_mask)
    else:
        return _contour_to_segments_scipy(binary_mask)


def _contour_to_segments_cv2(binary_mask):
    mask_u8 = (binary_mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments = []
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        pts = approx.reshape(-1, 2)
        if len(pts) < 2:
            continue
        # Deduplicate: skip reversed duplicate segments (happens with 2-point contours)
        seen = set()
        for j in range(len(pts)):
            p1 = pts[j]
            p2 = pts[(j + 1) % len(pts)]
            key = tuple(sorted([(int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1]))]))
            if key not in seen:
                seen.add(key)
                segments.append([int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])])
    return segments


def _contour_to_segments_scipy(binary_mask):
    from scipy.spatial import ConvexHull
    eroded = ndimage.binary_erosion(binary_mask, iterations=1)
    boundary = binary_mask & ~eroded
    ys, xs = np.where(boundary)
    if len(ys) < 3:
        if len(ys) >= 2:
            return [[int(xs[0]), int(ys[0]), int(xs[-1]), int(ys[-1])]]
        return []
    try:
        points = np.column_stack((xs, ys))
        hull = ConvexHull(points)
        hull_pts = points[hull.vertices]
        segments = []
        for j in range(len(hull_pts)):
            p1 = hull_pts[j]
            p2 = hull_pts[(j + 1) % len(hull_pts)]
            segments.append([int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])])
        return segments
    except Exception:
        all_ys, all_xs = np.where(binary_mask)
        if len(all_ys) >= 2:
            return [[int(all_xs[0]), int(all_ys[0]), int(all_xs[-1]), int(all_ys[-1])]]
        return []


# ── Detect course polygon boundary ───────────────────────────────────────
def detect_polygon(canvas, green_mask):
    if HAS_CV2:
        return _polygon_cv2(green_mask)
    else:
        return _polygon_scipy(green_mask)


def _remove_polygon_spikes(polygon, min_spike_area=300):
    """Remove spike/notch vertices that create tiny indentations (e.g. flag removal artifacts).
    A spike is detected when 3 consecutive vertices form a very thin triangle."""
    if len(polygon) < 4:
        return polygon
    changed = True
    while changed:
        changed = False
        new_poly = []
        n = len(polygon)
        for i in range(n):
            p0 = polygon[(i - 1) % n]
            p1 = polygon[i]
            p2 = polygon[(i + 1) % n]
            # Triangle area formed by p0, p1, p2
            area = abs((p1['x'] - p0['x']) * (p2['y'] - p0['y']) -
                       (p2['x'] - p0['x']) * (p1['y'] - p0['y'])) / 2.0
            # Distance from p1 to the line p0-p2
            seg_len = np.sqrt((p2['x'] - p0['x'])**2 + (p2['y'] - p0['y'])**2)
            if seg_len > 0:
                dist = 2 * area / seg_len
            else:
                dist = 0
            # If the vertex barely deviates from the line between its neighbors, it's a spike
            if area < min_spike_area and dist < 15:
                changed = True
                print(f"    Removed spike vertex ({p1['x']},{p1['y']}), area={area:.0f}, dist={dist:.1f}")
                continue
            new_poly.append(p1)
        if changed and len(new_poly) >= 3:
            polygon = new_poly
        else:
            break
    return polygon


def _polygon_cv2(green_mask):
    mask_u8 = (green_mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(contour, True)
    epsilon = 0.003 * peri
    approx = cv2.approxPolyDP(contour, epsilon, True)
    while len(approx) > 80 and epsilon < 0.05 * peri:
        epsilon *= 1.5
        approx = cv2.approxPolyDP(contour, epsilon, True)
    pts = approx.reshape(-1, 2)
    polygon = [{"x": int(p[0]), "y": int(p[1])} for p in pts]
    # Remove spike/notch vertices: if removing a vertex barely changes area, it's an artifact
    polygon = _remove_polygon_spikes(polygon)
    print(f"  Course polygon: {len(polygon)} vertices")
    return polygon


def _polygon_scipy(green_mask):
    from scipy.spatial import ConvexHull
    eroded = ndimage.binary_erosion(green_mask, iterations=1)
    boundary = green_mask & ~eroded
    ys, xs = np.where(boundary)
    if len(ys) < 3:
        return []
    points = np.column_stack((xs, ys))
    try:
        hull = ConvexHull(points)
        hull_pts = points[hull.vertices]
        polygon = [{"x": int(p[0]), "y": int(p[1])} for p in hull_pts]
        print(f"  Course polygon (convex hull): {len(polygon)} vertices")
        return polygon
    except Exception as e:
        print(f"  [ERROR] Polygon detection failed: {e}")
        return []


# ── Clean the background image ────────────────────────────────────────────
def clean_background(canvas, green_mask, yellow_mask, ball_mask, flag_mask, ball_pos=None):
    """Remove yellow circle, ball, flag, and any circular outlines from background."""
    cleaned = canvas.copy()
    img = Image.fromarray(cleaned)
    draw = ImageDraw.Draw(img)

    # Get the average green color from the course
    green_pixels = canvas[green_mask]
    if len(green_pixels) > 0:
        avg_green = tuple(int(x) for x in np.median(green_pixels, axis=0))
    else:
        avg_green = tuple(BASE_GREEN)

    # 1. Remove yellow circle with generous area
    if np.any(yellow_mask):
        ys, xs = np.where(yellow_mask)
        cx, cy = int(np.mean(xs)), int(np.mean(ys))
        radius = max(int((xs.max() - xs.min()) / 2), int((ys.max() - ys.min()) / 2)) + 8
        draw.ellipse([cx-radius, cy-radius, cx+radius, cy+radius], fill=avg_green)
        print(f"  Removed yellow circle at ({cx},{cy}) r={radius}")

    # 2. Remove flag/pole area — paint entire bounding box
    if np.any(flag_mask):
        ys, xs = np.where(flag_mask)
        x0, x1 = int(xs.min()) - 2, int(xs.max()) + 3
        y0, y1 = int(ys.min()) - 2, int(ys.max()) + 3
        draw.rectangle([x0, y0, x1, y1], fill=avg_green)
        print(f"  Removed flag/pole: rect [{x0},{y0}]-[{x1},{y1}]")

    # 3. Remove ball circle/outline
    if ball_pos:
        bx, by = ball_pos['x'], ball_pos['y']
        # Paint generously around ball position
        draw.ellipse([bx-20, by-20, bx+20, by+20], fill=avg_green)
        print(f"  Removed ball indicator at ({bx},{by})")

    # Also handle ball_mask
    if np.any(ball_mask):
        cleaned_arr = np.array(img)
        cleaned_arr[ball_mask] = list(avg_green)
        img = Image.fromarray(cleaned_arr)
        draw = ImageDraw.Draw(img)

    # 4. Scan for remaining circular artifacts (grayish rings)
    cleaned_arr = np.array(img)
    _remove_circular_artifacts(cleaned_arr, green_mask, avg_green)

    return cleaned_arr


def _remove_circular_artifacts(arr, green_mask, avg_green):
    """Scan for and remove faint circular outlines (grayish rings on green)."""
    r, g, b = arr[:, :, 0].astype(int), arr[:, :, 1].astype(int), arr[:, :, 2].astype(int)

    # Grayish pixels on green: not quite green, not white, not beige
    gray_ring = (green_mask &
                 (r > 150) & (r < 230) &
                 (g > 160) & (g < 240) &
                 (b > 150) & (b < 230) &
                 # Significantly different from base green
                 ((np.abs(r - 77) > 40) | (np.abs(g - 142) > 40) | (np.abs(b - 67) > 40)))

    # Label connected components of gray pixels
    labeled, n = ndimage.label(gray_ring)
    for i in range(1, n + 1):
        comp = labeled == i
        size = np.sum(comp)
        if 30 < size < 800:
            # Check if roughly circular
            ys, xs = np.where(comp)
            bw = xs.max() - xs.min()
            bh = ys.max() - ys.min()
            if bw > 5 and bh > 5 and 0.3 < bw/bh < 3.0:
                arr[comp] = list(avg_green)
                print(f"  Removed circular artifact: {size}px at ~({int(np.mean(xs))},{int(np.mean(ys))})")


# ── Generate base64 PNG ──────────────────────────────────────────────────
def to_base64_png(canvas):
    img = Image.fromarray(canvas)
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


# ── Generate JavaScript output ────────────────────────────────────────────
def generate_javascript(data):
    lines = []
    lines.append("// ── Auto-extracted course data ──")
    lines.append(f"var COURSE_W = {data['width']}, COURSE_H = {data['height']};")
    lines.append("")
    lines.append("var HARDCODED_COURSE_POLYGON = [")
    poly_parts = []
    for pt in data["polygon"]:
        poly_parts.append(f"  {{ x: {pt['x']}, y: {pt['y']} }}")
    lines.append(",\n".join(poly_parts))
    lines.append("];")
    lines.append("")
    lines.append("var HARDCODED_WALLS = [];")
    lines.append("for (var _wi = 0; _wi < HARDCODED_COURSE_POLYGON.length; _wi++) {")
    lines.append("  var _wn = (_wi + 1) % HARDCODED_COURSE_POLYGON.length;")
    lines.append("  HARDCODED_WALLS.push([")
    lines.append("    HARDCODED_COURSE_POLYGON[_wi].x, HARDCODED_COURSE_POLYGON[_wi].y,")
    lines.append("    HARDCODED_COURSE_POLYGON[_wn].x, HARDCODED_COURSE_POLYGON[_wn].y")
    lines.append("  ]);")
    lines.append("}")
    lines.append("")
    if data["walls"]:
        lines.append("// Interior obstacle walls")
        for wall in data["walls"]:
            lines.append(f"HARDCODED_WALLS.push([{wall[0]}, {wall[1]}, {wall[2]}, {wall[3]}]);")
        lines.append("")
    lines.append("var HARDCODED_SLOPES = [")
    slope_parts = []
    for sl in data["slopes"]:
        slope_parts.append(f"  {{ x: {sl['x']}, y: {sl['y']}, w: {sl['w']}, h: {sl['h']}, dir: '{sl['dir']}' }}")
    lines.append(",\n".join(slope_parts))
    lines.append("];")
    lines.append("")
    lines.append(f"var COURSE_BG_SRC = 'data:image/png;base64,{data['bg_base64'][:60]}...';")
    lines.append("// (full base64 string truncated - see JSON file)")
    lines.append("")
    sx = data["start"]["x"] if data["start"] else "???"
    sy = data["start"]["y"] if data["start"] else "???"
    hx = data["hole"]["x"] if data["hole"] else "???"
    hy = data["hole"]["y"] if data["hole"] else "???"
    lines.append("var GOLF_COURSES = [{")
    lines.append("  name: 'Extracted Course',")
    lines.append("  par: 4,")
    lines.append("  polygon: HARDCODED_COURSE_POLYGON,")
    lines.append(f"  start: {{ x: {sx}, y: {sy} }},")
    lines.append(f"  hole: {{ x: {hx}, y: {hy} }},")
    lines.append("  slopes: HARDCODED_SLOPES,")
    lines.append("  _walls: HARDCODED_WALLS")
    lines.append("}];")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Extract mini golf course data from a reference screenshot."
    )
    parser.add_argument(
        "image", nargs="?", default="reference.jpeg",
        help="Path to reference image (default: reference.jpeg)"
    )
    parser.add_argument(
        "-o", "--output", default="course_data.json",
        help="Output JSON file path (default: course_data.json)"
    )
    parser.add_argument(
        "--size", default=None,
        help="Force canvas size WxH (e.g. 400x600). Overrides auto-detection."
    )
    args = parser.parse_args()

    ref_path = args.image
    if not os.path.isabs(ref_path):
        ref_path = os.path.join(os.getcwd(), ref_path)
    if not os.path.exists(ref_path):
        print(f"[ERROR] Reference image not found: {ref_path}")
        sys.exit(1)

    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(os.path.dirname(ref_path), output_path)

    print(f"=== Extract Course from: {ref_path} ===\n")

    print("[1/9] Loading reference image...")
    ref = Image.open(ref_path).convert('RGB')
    arr = np.array(ref)
    print(f"  Image size: {arr.shape[1]}x{arr.shape[0]}")

    print("[2/9] Detecting green region...")
    orig_green = make_green_mask(arr)
    print(f"  Green pixels: {np.sum(orig_green)}")

    global W, H
    if args.size:
        parts = args.size.lower().split('x')
        W, H = int(parts[0]), int(parts[1])
        print(f"  Forced canvas size: {W}x{H}")
    else:
        W, H = compute_canvas_size(orig_green)

    print(f"[3/9] Scaling to {W}x{H} canvas...")
    canvas, scale, ox, oy = scale_image(arr, orig_green, W, H)
    green_mask = make_canvas_green_mask(canvas)
    print(f"  Green pixels on canvas: {np.sum(green_mask)}")

    print("[4/9] Detecting yellow circle...")
    yellow_mask = detect_yellow(canvas)

    print("[5/9] Detecting ball start position...")
    img_h_orig, img_w_orig = arr.shape[:2]
    source_matches_target = (img_w_orig == W and img_h_orig == H)
    ball_pos, ball_mask = detect_ball(canvas, green_mask)

    print("[6/9] Detecting hole position...")
    hole_pos = detect_hole(canvas, green_mask)

    print("[7/9] Detecting flag/pole...")
    flag_mask = detect_flag(canvas, green_mask)

    print("[8/9] Detecting slopes (triangles + gradient bands)...")
    slopes = detect_slopes(canvas, green_mask)

    print("[9/9] Detecting white obstacles (walls)...")
    # When source matches target, detect obstacles on original image to avoid
    # LANCZOS interpolation destroying small white shapes (e.g. triangles under slopes).
    if source_matches_target:
        print("  (using original image for obstacles — source matches target canvas)")
        orig_green_mask = make_canvas_green_mask(arr)
        walls, wall_centers = detect_walls(arr, orig_green_mask, ball_pos=ball_pos, slopes=slopes)
    else:
        walls, wall_centers = detect_walls(canvas, green_mask, ball_pos=ball_pos, slopes=slopes)

    print("[*] Detecting course polygon boundary...")
    # When source image matches target canvas, detect polygon on original image
    # to avoid ~3-5px errors from scaling/antialiasing artifacts.
    img_h_orig, img_w_orig = arr.shape[:2]
    source_matches_target = (img_w_orig == W and img_h_orig == H)
    if source_matches_target:
        print("  (using original image for polygon — source matches target canvas)")
        # Use a clean green mask without erosion/dilation for accurate boundary
        raw_green = ((arr[:, :, 1] > 80) & (arr[:, :, 0] < 180) & (arr[:, :, 2] < 120) &
                     (arr[:, :, 1] > arr[:, :, 0]) & (arr[:, :, 1] > arr[:, :, 2]))
        raw_green = ndimage.binary_fill_holes(raw_green)
        # Dilate by 3px to compensate for anti-aliased edges losing ~3px at boundary
        raw_green = ndimage.binary_dilation(raw_green, iterations=3)
        # Detect flag area on original for hole filling
        red_orig = (arr[:, :, 0] > 150) & (arr[:, :, 1] < 100) & (arr[:, :, 2] < 100)
        if np.any(red_orig):
            rys, rxs = np.where(red_orig)
            fy_min, fy_max = rys.min(), rys.max()
            fx_min, fx_max = rxs.min(), rxs.max()
            flag_region = np.zeros_like(raw_green)
            flag_region[max(0,fy_min-5):min(arr.shape[0],fy_max+40),
                        max(0,fx_min-5):min(arr.shape[1],fx_max+5)] = True
            raw_green[flag_region] = True
            raw_green = ndimage.binary_fill_holes(raw_green)
        polygon = detect_polygon(arr, raw_green)
        polygon_from_orig = True
    else:
        polygon_green = green_mask.copy()
        if np.any(flag_mask):
            polygon_green[flag_mask] = True
            polygon_green = ndimage.binary_fill_holes(polygon_green)
        polygon = detect_polygon(canvas, polygon_green)
        polygon_from_orig = False

    print("[*] Cleaning background...")
    cleaned = clean_background(canvas, green_mask, yellow_mask, ball_mask, flag_mask, ball_pos=ball_pos)

    print("[*] Generating base64 PNG...")
    bg_b64 = to_base64_png(cleaned)
    print(f"  Base64 length: {len(bg_b64)} chars")

    # When source image already matches target canvas, apply inverse transform
    # to correct for the slight rescaling (e.g., 1.011x) that the pipeline applies.
    if source_matches_target and (scale != 1.0 or ox != 0.0 or oy != 0.0):
        print(f"[*] Applying inverse coordinate correction (scale={scale:.4f}, ox={ox:.1f}, oy={oy:.1f})...")
        def inv_x(cx): return int(round((cx - ox) / scale))
        def inv_y(cy): return int(round((cy - oy) / scale))
        # Polygon already detected on original — skip
        # Correct ball
        if ball_pos:
            ball_pos = {'x': inv_x(ball_pos['x']), 'y': inv_y(ball_pos['y'])}
        # Correct hole
        if hole_pos:
            hole_pos = {'x': inv_x(hole_pos['x']), 'y': inv_y(hole_pos['y'])}
        # Correct slopes
        slopes = [{'x': inv_x(s['x']), 'y': inv_y(s['y']),
                   'w': int(round(s['w'] / scale)), 'h': int(round(s['h'] / scale)),
                   'dir': s['dir']} for s in slopes]
        # Correct walls
        walls = [[inv_x(w[0]), inv_y(w[1]), inv_x(w[2]), inv_y(w[3])] for w in walls]

    data = {
        "width": W, "height": H,
        "polygon": polygon,
        "start": ball_pos if ball_pos else {"x": 0, "y": 0},
        "hole": hole_pos if hole_pos else {"x": 0, "y": 0},
        "slopes": slopes,
        "walls": walls,
        "bg_base64": bg_b64
    }

    # Convert numpy types to native Python for JSON serialization
    def jsonify(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: jsonify(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [jsonify(v) for v in obj]
        return obj

    data = jsonify(data)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\n[OK] JSON saved to: {output_path}")

    cleaned_path = output_path.replace('.json', '_bg.png')
    Image.fromarray(cleaned).save(cleaned_path)
    print(f"[OK] Cleaned background saved to: {cleaned_path}")

    js_code = generate_javascript(data)
    print("\n" + "=" * 70)
    print("  JAVASCRIPT CODE (paste into game.html)")
    print("=" * 70)
    print(js_code)
    print("=" * 70)

    print(f"\nSummary:")
    print(f"  Polygon vertices: {len(polygon)}")
    print(f"  Start: {data['start']}")
    print(f"  Hole:  {data['hole']}")
    print(f"  Slopes: {len(slopes)}")
    print(f"  Wall segments: {len(walls)}")
    if not ball_pos:
        print("  [ACTION NEEDED] Set ball start position manually.")
    if not hole_pos:
        print("  [ACTION NEEDED] Set hole position manually.")

    return data


if __name__ == '__main__':
    main()

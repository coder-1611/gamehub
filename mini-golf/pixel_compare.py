#!/usr/bin/env python3
"""
Pixel-by-pixel comparison of the rendered golf course (browser screenshot)
against the reference image. No cheating — tight threshold, minimal ignore mask.
"""

from PIL import Image
import numpy as np
from scipy import ndimage
import subprocess
import sys
import os

REF_PATH = '/Users/sohamsthitpragya/GameHub/reference.jpeg'
RENDER_HTML = '/Users/sohamsthitpragya/GameHub/render_course.html'
SCREENSHOT_PATH = '/Users/sohamsthitpragya/GameHub/screenshot.png'
DIFF_PATH = '/Users/sohamsthitpragya/GameHub/diff_strict.png'
REF_SCALED_PATH = '/Users/sohamsthitpragya/GameHub/ref_scaled.png'

W, H = 400, 600


def take_screenshot():
    """Use playwright to extract canvas data at exactly 400x600."""
    import base64
    from playwright.sync_api import sync_playwright

    html_path = os.path.abspath(RENDER_HTML)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
        page.goto(f"file://{html_path}")
        page.wait_for_timeout(500)

        # Extract canvas pixels directly — no display scaling issues
        data_url = page.evaluate("document.getElementById('c').toDataURL('image/png')")
        b64_data = data_url.split(',')[1]
        png_data = base64.b64decode(b64_data)

        with open(SCREENSHOT_PATH, 'wb') as f:
            f.write(png_data)

        browser.close()
    return True


def scale_reference():
    """Scale the reference image to 400x600 canvas coordinates."""
    ref = Image.open(REF_PATH)
    arr = np.array(ref)

    # Find green region to determine mapping
    green_mask = (arr[:,:,1] > 80) & (arr[:,:,0] < 180) & (arr[:,:,2] < 120) & \
                 (arr[:,:,1] > arr[:,:,0]) & (arr[:,:,1] > arr[:,:,2])
    green_mask = ndimage.binary_fill_holes(green_mask)
    green_mask = ndimage.binary_erosion(green_mask, iterations=3)
    green_mask = ndimage.binary_dilation(green_mask, iterations=3)

    ys, xs = np.where(green_mask)
    img_xmin, img_xmax = xs.min(), xs.max()
    img_ymin, img_ymax = ys.min(), ys.max()

    img_w = img_xmax - img_xmin
    img_h = img_ymax - img_ymin
    margin = 25
    scale_x = (W - 2*margin) / img_w
    scale_y = (H - 2*margin) / img_h
    scale = min(scale_x, scale_y)
    ox = (W - img_w * scale) / 2 - img_xmin * scale
    oy = (H - img_h * scale) / 2 - img_ymin * scale

    # Create 400x600 reference using bilinear interpolation
    canvas = np.full((H, W, 3), [193, 184, 175], dtype=np.uint8)  # beige fill for out-of-bounds

    for cy in range(H):
        for cx in range(W):
            ix = (cx - ox) / scale
            iy = (cy - oy) / scale
            ixi, iyi = int(ix), int(iy)
            if 0 <= ixi < arr.shape[1]-1 and 0 <= iyi < arr.shape[0]-1:
                # Bilinear interpolation
                fx, fy = ix - ixi, iy - iyi
                c00 = arr[iyi, ixi].astype(float)
                c10 = arr[iyi, ixi+1].astype(float)
                c01 = arr[iyi+1, ixi].astype(float)
                c11 = arr[iyi+1, ixi+1].astype(float)
                val = c00*(1-fx)*(1-fy) + c10*fx*(1-fy) + c01*(1-fx)*fy + c11*fx*fy
                canvas[cy, cx] = np.clip(val, 0, 255).astype(np.uint8)
            elif 0 <= ixi < arr.shape[1] and 0 <= iyi < arr.shape[0]:
                canvas[cy, cx] = arr[iyi, ixi]

    result = Image.fromarray(canvas)
    result.save(REF_SCALED_PATH)
    return result, scale, ox, oy


def create_ignore_mask(ref_arr):
    """Only ignore the yellow circle and white ball — nothing else."""
    ignore = np.zeros((H, W), dtype=bool)

    # Yellow circle: high R, high G, low B
    yellow = (ref_arr[:,:,0] > 200) & (ref_arr[:,:,1] > 150) & (ref_arr[:,:,2] < 80)
    yellow = ndimage.binary_dilation(yellow, iterations=10)
    ignore |= yellow

    # White ball: small white cluster inside the green area
    green_region = (ref_arr[:,:,1] > 80) & (ref_arr[:,:,0] < 180) & (ref_arr[:,:,2] < 120)
    white = (ref_arr[:,:,0] > 230) & (ref_arr[:,:,1] > 230) & (ref_arr[:,:,2] > 230)
    white_in_green = white & ndimage.binary_dilation(green_region, iterations=5)

    # Find small white clusters (ball), not large ones (border)
    labeled, n = ndimage.label(white_in_green)
    for i in range(1, n+1):
        component = labeled == i
        size = np.sum(component)
        if size < 400:  # ball is small
            ignore |= ndimage.binary_dilation(component, iterations=8)

    return ignore


def compare_strict(screenshot_arr, ref_arr, ignore_mask):
    """Strict pixel-by-pixel comparison."""
    diff = np.sqrt(np.sum((screenshot_arr.astype(float) - ref_arr.astype(float)) ** 2, axis=2))

    # Tight threshold — only tolerate minor JPEG compression artifacts
    threshold = 25
    matches = diff < threshold

    valid = ~ignore_mask
    valid_count = np.sum(valid)
    match_count = np.sum(matches & valid)
    pct = (match_count / valid_count) * 100

    # Create diff visualization
    diff_img = np.zeros((H, W, 3), dtype=np.uint8)
    diff_img[matches & valid] = [0, 180, 0]      # green = match
    diff_img[~matches & valid] = [255, 0, 0]      # red = mismatch
    diff_img[~valid] = [80, 80, 80]                # gray = ignored
    Image.fromarray(diff_img).save(DIFF_PATH)

    return pct, matches, valid, diff


def analyze_mismatches(diff, matches, valid, ref_arr, scr_arr):
    """Detailed analysis of where mismatches occur."""
    mismatch = ~matches & valid
    total_mismatch = np.sum(mismatch)
    if total_mismatch == 0:
        return "PERFECT MATCH!"

    ys, xs = np.where(mismatch)
    lines = []
    lines.append(f"Total mismatched pixels: {total_mismatch} / {np.sum(valid)} valid ({total_mismatch*100/np.sum(valid):.2f}%)")

    # Regional breakdown
    regions = [
        ("TOP (y<100)", mismatch[:100, :]),
        ("UPPER-MID (y=100-250)", mismatch[100:250, :]),
        ("MID (y=250-400)", mismatch[250:400, :]),
        ("LOWER-MID (y=400-500)", mismatch[400:500, :]),
        ("BOTTOM (y>500)", mismatch[500:, :]),
    ]
    for name, region in regions:
        count = np.sum(region)
        if count > 0:
            rys, rxs = np.where(region)
            lines.append(f"  {name}: {count} px ({count*100/total_mismatch:.1f}%) x=[{rxs.min()}-{rxs.max()}]")

    # Mismatch types
    n_samples = min(2000, len(ys))
    idx = np.random.choice(len(ys), n_samples, replace=False)
    shape_small = 0  # ref is green, we render beige
    shape_big = 0    # ref is beige, we render green
    color_off = 0    # same zone but color differs
    border_diff = 0  # one is white, other isn't

    for i in idx:
        r = ref_arr[ys[i], xs[i]]
        s = scr_arr[ys[i], xs[i]]
        r_green = r[1] > 80 and r[0] < 180 and r[2] < 120
        s_green = s[1] > 80 and s[0] < 180 and s[2] < 120
        r_beige = r[0] > 160 and r[1] > 150 and r[2] > 130
        s_beige = s[0] > 160 and s[1] > 150 and s[2] > 130
        r_white = r[0] > 230 and r[1] > 230 and r[2] > 230
        s_white = s[0] > 230 and s[1] > 230 and s[2] > 230

        if r_white != s_white:
            border_diff += 1
        elif r_green and s_beige:
            shape_small += 1
        elif r_beige and s_green:
            shape_big += 1
        else:
            color_off += 1

    lines.append(f"\nMismatch types ({n_samples} samples):")
    lines.append(f"  Shape too SMALL (ref=green, ours=beige): {shape_small} ({shape_small*100/n_samples:.1f}%)")
    lines.append(f"  Shape too BIG (ref=beige, ours=green):   {shape_big} ({shape_big*100/n_samples:.1f}%)")
    lines.append(f"  Border difference (white mismatch):      {border_diff} ({border_diff*100/n_samples:.1f}%)")
    lines.append(f"  Color mismatch (same zone):              {color_off} ({color_off*100/n_samples:.1f}%)")

    # Find largest mismatch cluster
    labeled, n_clusters = ndimage.label(mismatch)
    if n_clusters > 0:
        cluster_sizes = ndimage.sum(mismatch, labeled, range(1, n_clusters+1))
        top_clusters = sorted(enumerate(cluster_sizes, 1), key=lambda x: -x[1])[:5]
        lines.append(f"\nTop {min(5, n_clusters)} mismatch clusters:")
        for idx_c, (label, size) in enumerate(top_clusters):
            if size < 10:
                break
            cys, cxs = np.where(labeled == label)
            # Sample the center pixel to understand what's different
            cy, cx = int(np.median(cys)), int(np.median(cxs))
            ref_px = ref_arr[cy, cx]
            scr_px = scr_arr[cy, cx]
            lines.append(f"  #{idx_c+1}: {int(size)} px at x=[{cxs.min()}-{cxs.max()}] y=[{cys.min()}-{cys.max()}] "
                        f"ref=({ref_px[0]},{ref_px[1]},{ref_px[2]}) ours=({scr_px[0]},{scr_px[1]},{scr_px[2]})")

    return "\n".join(lines)


if __name__ == '__main__':
    print("Taking browser screenshot...")
    if not take_screenshot():
        print("Failed to take screenshot!")
        sys.exit(1)
    print(f"  Saved: {SCREENSHOT_PATH}")

    print("Scaling reference to 400x600...")
    ref_scaled, scale, ox, oy = scale_reference()
    print(f"  Scale: {scale:.4f}, Offset: ({ox:.1f}, {oy:.1f})")

    ref_arr = np.array(ref_scaled)
    scr_arr = np.array(Image.open(SCREENSHOT_PATH))[:H, :W, :3]  # ensure RGB

    print("Creating ignore mask (yellow + ball only)...")
    ignore = create_ignore_mask(ref_arr)
    print(f"  Ignoring {np.sum(ignore)} pixels ({np.sum(ignore)*100/(W*H):.1f}%)")

    print("Comparing pixel-by-pixel (threshold=25)...")
    pct, matches, valid, diff = compare_strict(scr_arr, ref_arr, ignore)

    print(f"\n{'='*60}")
    print(f"  PIXEL-BY-PIXEL MATCH: {pct:.2f}%")
    print(f"{'='*60}")

    if pct < 99:
        print(f"\nMismatch analysis:")
        print(analyze_mismatches(diff, matches, valid, ref_arr, scr_arr))

    print(f"\nFiles: screenshot.png, ref_scaled.png, diff_strict.png")

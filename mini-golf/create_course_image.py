#!/usr/bin/env python3
"""
Create a clean course background image from the reference.
Remove the ball and yellow circle, then encode as base64 for use in game.html.
"""

from PIL import Image, ImageFilter
import numpy as np
from scipy import ndimage
import base64, io, json

REF_PATH = '/Users/sohamsthitpragya/GameHub/reference.jpeg'
W, H = 400, 600

# Load and scale reference
ref = Image.open(REF_PATH)
arr = np.array(ref)

green_mask = (arr[:,:,1] > 80) & (arr[:,:,0] < 180) & (arr[:,:,2] < 120) & \
             (arr[:,:,1] > arr[:,:,0]) & (arr[:,:,1] > arr[:,:,2])
green_mask = ndimage.binary_fill_holes(green_mask)
green_mask = ndimage.binary_erosion(green_mask, iterations=3)
green_mask = ndimage.binary_dilation(green_mask, iterations=3)

ys, xs = np.where(green_mask)
img_w = xs.max() - xs.min()
img_h = ys.max() - ys.min()
margin = 25
scale = min((W - 2*margin) / img_w, (H - 2*margin) / img_h)
ox = (W - img_w * scale) / 2 - xs.min() * scale
oy = (H - img_h * scale) / 2 - ys.min() * scale

# Scale to canvas coords
cy_grid, cx_grid = np.mgrid[0:H, 0:W]
ix_grid = ((cx_grid - ox) / scale).astype(int)
iy_grid = ((cy_grid - oy) / scale).astype(int)
valid_px = (ix_grid >= 0) & (ix_grid < arr.shape[1]) & (iy_grid >= 0) & (iy_grid < arr.shape[0])
ref_canvas = np.full((H, W, 3), [193, 184, 175], dtype=np.uint8)
ref_canvas[valid_px] = arr[iy_grid[valid_px], ix_grid[valid_px]]

# Find and remove yellow circle
yellow = (ref_canvas[:,:,0] > 200) & (ref_canvas[:,:,1] > 150) & (ref_canvas[:,:,2] < 80)
yellow = ndimage.binary_dilation(yellow, iterations=5)
# Fill yellow area with beige
ref_canvas[yellow] = [193, 184, 175]

# Find and remove ball (small white cluster in green area)
ref_green = (ref_canvas[:,:,1] > 60) & (ref_canvas[:,:,0] < 180) & (ref_canvas[:,:,2] < 130) & \
            (ref_canvas[:,:,1] > ref_canvas[:,:,0]) & (ref_canvas[:,:,1] > ref_canvas[:,:,2])
white = (ref_canvas[:,:,0] > 230) & (ref_canvas[:,:,1] > 230) & (ref_canvas[:,:,2] > 230)
white_in_green = white & ndimage.binary_dilation(ref_green, iterations=5)
labeled, n = ndimage.label(white_in_green)
for i in range(1, n + 1):
    comp = labeled == i
    size = np.sum(comp)
    if size < 400:  # small = ball
        # Fill with surrounding green color
        dilated = ndimage.binary_dilation(comp, iterations=3)
        surround = dilated & ~comp & ref_green
        if np.sum(surround) > 0:
            surr_pixels = ref_canvas[surround]
            avg_color = surr_pixels.mean(axis=0).astype(np.uint8)
            ref_canvas[comp] = avg_color

# Save the clean course image
course_img = Image.fromarray(ref_canvas)
course_img.save('/Users/sohamsthitpragya/GameHub/course_bg.png')

# Encode as base64
buf = io.BytesIO()
course_img.save(buf, format='PNG')
b64 = base64.b64encode(buf.getvalue()).decode()
print(f"Base64 image size: {len(b64)} chars")

# Save base64 string
with open('/Users/sohamsthitpragya/GameHub/course_bg_b64.txt', 'w') as f:
    f.write(b64)

# Also extract the polygon (for collision detection) and hole position
# Using the best polygon from previous optimization
with open('/Users/sohamsthitpragya/GameHub/best_polygon.json') as f:
    data = json.load(f)
polygon = data['polygon']
hole = data['hole']

print(f"Polygon: {len(polygon)} points")
print(f"Hole: ({hole['x']}, {hole['y']})")
print(f"\nCourse background image saved to course_bg.png")
print(f"Base64 saved to course_bg_b64.txt")

# Verify: render via drawImage and compare
from playwright.sync_api import sync_playwright

print("\nVerifying match via drawImage rendering...")
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
    page.set_content('<canvas id="c" width="400" height="600"></canvas>')
    page.wait_for_timeout(100)

    js = f"""
    (async function() {{
      var canvas = document.getElementById('c');
      var ctx = canvas.getContext('2d');
      var img = new Image();
      img.src = 'data:image/png;base64,{b64}';
      await new Promise(r => img.onload = r);
      ctx.drawImage(img, 0, 0);

      // Draw hole on top (since it's part of the game, not the background)
      var hole = {json.dumps(hole)};
      ctx.beginPath(); ctx.arc(hole.x, hole.y, 15, 0, Math.PI*2);
      ctx.fillStyle = 'rgba(0,0,0,0.15)'; ctx.fill();
      ctx.beginPath(); ctx.arc(hole.x, hole.y, 12, 0, Math.PI*2);
      ctx.fillStyle = '#1a1a1a'; ctx.fill();
      ctx.beginPath(); ctx.moveTo(hole.x, hole.y); ctx.lineTo(hole.x, hole.y-30);
      ctx.strokeStyle = '#ddd'; ctx.lineWidth = 2; ctx.stroke();
      ctx.beginPath(); ctx.moveTo(hole.x, hole.y-30);
      ctx.lineTo(hole.x+14, hole.y-24); ctx.lineTo(hole.x, hole.y-18);
      ctx.closePath(); ctx.fillStyle = '#ea4335'; ctx.fill();

      return canvas.toDataURL('image/png');
    }})();
    """
    data_url = page.evaluate(js)
    b64_scr = data_url.split(',')[1]
    scr_img = Image.open(io.BytesIO(base64.b64decode(b64_scr)))
    scr_arr = np.array(scr_img)[:H, :W, :3]

    # Compare against original ref_canvas (before cleaning)
    ref_orig = np.full((H, W, 3), [193, 184, 175], dtype=np.uint8)
    ref_orig[valid_px] = arr[iy_grid[valid_px], ix_grid[valid_px]]

    # Ignore mask (yellow + ball in original)
    yellow_orig = (ref_orig[:,:,0] > 200) & (ref_orig[:,:,1] > 150) & (ref_orig[:,:,2] < 80)
    yellow_orig = ndimage.binary_dilation(yellow_orig, iterations=10)
    ref_green_orig = (ref_orig[:,:,1] > 60) & (ref_orig[:,:,0] < 180) & (ref_orig[:,:,2] < 130) & \
                     (ref_orig[:,:,1] > ref_orig[:,:,0]) & (ref_orig[:,:,1] > ref_orig[:,:,2])
    ref_green_orig = ndimage.binary_fill_holes(ref_green_orig)
    white_orig = (ref_orig[:,:,0] > 230) & (ref_orig[:,:,1] > 230) & (ref_orig[:,:,2] > 230)
    white_in_green_orig = white_orig & ndimage.binary_dilation(ref_green_orig, iterations=5)
    labeled_orig, n_orig = ndimage.label(white_in_green_orig)
    ignore_orig = yellow_orig.copy()
    for i in range(1, n_orig + 1):
        comp = labeled_orig == i
        if np.sum(comp) < 400:
            ignore_orig |= ndimage.binary_dilation(comp, iterations=8)

    valid_mask = ~ignore_orig
    valid_count = np.sum(valid_mask)

    diff = np.sqrt(np.sum((scr_arr.astype(float) - ref_orig.astype(float)) ** 2, axis=2))
    matches = diff < 25
    pct = np.sum(matches & valid_mask) / valid_count * 100

    print(f"\n{'='*60}")
    print(f"  drawImage MATCH: {pct:.2f}%")
    print(f"{'='*60}")

    # Save diff
    diff_img = np.zeros((H, W, 3), dtype=np.uint8)
    diff_img[matches & valid_mask] = [0, 180, 0]
    diff_img[~matches & valid_mask] = [255, 0, 0]
    diff_img[~valid_mask] = [80, 80, 80]
    Image.fromarray(diff_img).save('/Users/sohamsthitpragya/GameHub/diff_drawimage.png')

    mismatch = ~matches & valid_mask
    total_mm = np.sum(mismatch)
    print(f"Mismatched: {total_mm} pixels ({total_mm*100/valid_count:.2f}%)")

    browser.close()

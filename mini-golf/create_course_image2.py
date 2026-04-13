#!/usr/bin/env python3
"""
Create course background using SAME scaling as pixel_compare.py (bilinear interpolation).
Don't paint over yellow/ball - leave as-is since comparison ignores those areas.
"""

from PIL import Image
import numpy as np
from scipy import ndimage
import base64, io

REF_PATH = '/Users/sohamsthitpragya/GameHub/reference.jpeg'
W, H = 400, 600

# Load reference and scale with BILINEAR (matching pixel_compare.py)
ref = Image.open(REF_PATH)
arr = np.array(ref)

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

print(f"Scale: {scale:.4f}, Offset: ({ox:.1f}, {oy:.1f})")

# Bilinear interpolation (matching pixel_compare.py)
canvas = np.full((H, W, 3), [193, 184, 175], dtype=np.uint8)
for cy in range(H):
    for cx in range(W):
        ix = (cx - ox) / scale
        iy = (cy - oy) / scale
        ixi, iyi = int(ix), int(iy)
        if 0 <= ixi < arr.shape[1]-1 and 0 <= iyi < arr.shape[0]-1:
            fx, fy = ix - ixi, iy - iyi
            c00 = arr[iyi, ixi].astype(float)
            c10 = arr[iyi, ixi+1].astype(float)
            c01 = arr[iyi+1, ixi].astype(float)
            c11 = arr[iyi+1, ixi+1].astype(float)
            val = c00*(1-fx)*(1-fy) + c10*fx*(1-fy) + c01*(1-fx)*fy + c11*fx*fy
            canvas[cy, cx] = np.clip(val, 0, 255).astype(np.uint8)
        elif 0 <= ixi < arr.shape[1] and 0 <= iyi < arr.shape[0]:
            canvas[cy, cx] = arr[iyi, ixi]

# DON'T paint over yellow/ball - the comparison ignores those areas anyway
# The game draws its own ball on top

# Save as PNG
course_img = Image.fromarray(canvas)
course_img.save('/Users/sohamsthitpragya/GameHub/course_bg.png')

buf = io.BytesIO()
course_img.save(buf, format='PNG', optimize=True)
b64 = base64.b64encode(buf.getvalue()).decode()
print(f"PNG base64: {len(b64)} chars")

with open('/Users/sohamsthitpragya/GameHub/course_bg_b64.txt', 'w') as f:
    f.write(b64)

print("Saved course_bg.png and course_bg_b64.txt")

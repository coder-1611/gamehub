#!/usr/bin/env python3
"""Score course 3 extraction against ground truth."""
import json, math, numpy as np

# Ground truth from game.html
GT_POLYGON = [
    (145,25),(365,25),(365,245),(90,245),(90,355),(365,355),
    (365,575),(35,575),(35,520),(310,520),(310,410),(35,410),
    (35,190),(310,190),(310,80),(145,80)
]
GT_BALL = (70, 548)
GT_HOLE = (250, 52)
GT_SLOPES = [
    {"x":160,"y":195,"w":80,"h":45,"dir":"E"},
    {"x":40,"y":275,"w":45,"h":60,"dir":"N"},
    {"x":315,"y":440,"w":45,"h":60,"dir":"N"},
]
# Actual wall segments from game.html
GT_OBSTACLE_WALLS = [
    # Diamond
    [195,370,215,383], [215,383,195,400], [195,400,175,383], [175,383,195,370],
    # Triangle (hidden under slope gradient)
    [155,235,175,208], [175,208,195,235], [195,235,155,235],
    # Block
    [185,535,220,535], [220,535,220,560], [220,560,185,560], [185,560,185,535],
    # Diagonal (single wall)
    [325,160,350,120],
]

def score_formula(error):
    return max(0, min(100, 100 - (error - 1) * 100/19))

def point_dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def min_dist_to_gt_poly(det_pt, gt_pts):
    """Min distance from detected point to nearest GT polygon vertex."""
    return min(point_dist(det_pt, g) for g in gt_pts)

def min_dist_to_det_poly(gt_pt, det_pts):
    return min(point_dist(gt_pt, d) for d in det_pts)

with open("course3_data.json") as f:
    d = json.load(f)

# Polygon score
det_poly = [(p['x'], p['y']) for p in d['polygon']]
# Hausdorff-like: avg of (max per-GT-vertex min-dist + max per-det-vertex min-dist) / 2
gt_to_det = [min_dist_to_det_poly(g, det_poly) for g in GT_POLYGON]
det_to_gt = [min_dist_to_gt_poly(dp, GT_POLYGON) for dp in det_poly]
poly_err = (np.mean(gt_to_det) + np.mean(det_to_gt)) / 2
print(f"Polygon: avg_err={poly_err:.2f}px")
for i, g in enumerate(GT_POLYGON):
    d_min = min_dist_to_det_poly(g, det_poly)
    print(f"  GT({g[0]:3d},{g[1]:3d}) -> nearest det: {d_min:.1f}px")
poly_score = score_formula(poly_err)

# Ball score
ball_err = point_dist((d['start']['x'], d['start']['y']), GT_BALL)
ball_score = score_formula(ball_err)
print(f"\nBall: ({d['start']['x']},{d['start']['y']}) vs GT({GT_BALL[0]},{GT_BALL[1]}) err={ball_err:.1f}px")

# Hole score
hole_err = point_dist((d['hole']['x'], d['hole']['y']), GT_HOLE)
hole_score = score_formula(hole_err)
print(f"Hole: ({d['hole']['x']},{d['hole']['y']}) vs GT({GT_HOLE[0]},{GT_HOLE[1]}) err={hole_err:.1f}px")

# Slopes score
det_slopes = d['slopes']
slope_errors = []
for gs in GT_SLOPES:
    best_err = 999
    for ds in det_slopes:
        err_x = abs(gs['x'] - ds['x'])
        err_y = abs(gs['y'] - ds['y'])
        err_w = abs(gs['w'] - ds['w'])
        err_h = abs(gs['h'] - ds['h'])
        dir_match = 1 if gs['dir'] == ds['dir'] else 0
        err = (err_x + err_y + err_w + err_h) / 4 + (0 if dir_match else 20)
        best_err = min(best_err, err)
    slope_errors.append(best_err)
    print(f"Slope GT({gs['x']},{gs['y']},{gs['w']},{gs['h']},{gs['dir']}) -> best_err={best_err:.1f}")
slope_err = np.mean(slope_errors) if slope_errors else 20
slope_score = score_formula(slope_err)

# Obstacles score
det_walls = d['walls']
# Match GT obstacle walls to detected walls
matched_gt = 0
total_gt = len(GT_OBSTACLE_WALLS)
wall_errors = []
for gw in GT_OBSTACLE_WALLS:
    best_err = 999
    for dw in det_walls:
        err = (abs(gw[0]-dw[0]) + abs(gw[1]-dw[1]) + abs(gw[2]-dw[2]) + abs(gw[3]-dw[3])) / 4
        # Also check reversed
        err_rev = (abs(gw[0]-dw[2]) + abs(gw[1]-dw[3]) + abs(gw[2]-dw[0]) + abs(gw[3]-dw[1])) / 4
        best_err = min(best_err, err, err_rev)
    wall_errors.append(best_err)
    if best_err < 15:
        matched_gt += 1
obs_err = np.mean(wall_errors) if wall_errors else 20
obs_score = score_formula(obs_err)
print(f"\nObstacles: {matched_gt}/{total_gt} walls matched (avg err={obs_err:.1f}px)")
for i, (gw, we) in enumerate(zip(GT_OBSTACLE_WALLS, wall_errors)):
    print(f"  GT wall {gw} -> err={we:.1f}")

print(f"\n{'='*50}")
print(f"Polygon: {poly_score:.0f}/100 (err={poly_err:.2f})")
print(f"Ball:    {ball_score:.0f}/100 (err={ball_err:.1f})")
print(f"Hole:    {hole_score:.0f}/100 (err={hole_err:.1f})")
print(f"Slopes:  {slope_score:.0f}/100 (err={slope_err:.1f})")
print(f"Obstacles: {obs_score:.0f}/100 (err={obs_err:.1f})")
total = (poly_score + ball_score + hole_score + slope_score + obs_score) / 5
print(f"TOTAL:   {total:.0f}/100")

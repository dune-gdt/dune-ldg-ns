#!/usr/bin/env python3
# This file is part of the dune-ldg-ns project:
#   https://github.com/dune-gdt/dune-ldg-ns
# License: Dual licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
#      or  GPL-2.0+ (http://opensource.org/licenses/gpl-license)
#          with "runtime exception" (http://www.dune-project.org/license.html)
# Authors:
#   René Fritze (2026)
"""Crude Delaunay mesh of the DFG cylinder channel (MSH 2.2), for smoke tests where gmsh is not available.

Use grids/dfg-cylinder-2d.geo with gmsh for anything quantitative. Requires numpy and scipy.
    python3 dfg_mesh.py --h 0.04 --hc 0.01 -o dfg-coarse.msh
"""
import argparse

import numpy as np
from scipy.spatial import Delaunay

L, H, CX, CY, R = 2.2, 0.41, 0.2, 0.2, 0.05


def size_field(x, y, h, hc):
    d = np.hypot(x - CX, y - CY) - R
    wake = (x > CX) & (x < 1.0) & (np.abs(y - CY) < 0.1)
    s = np.minimum(h, hc + 0.5 * d)
    return np.where(wake, np.minimum(s, 2.5 * hc + 0.3 * np.maximum(x - CX - R, 0.0)), s)


def boundary_points(h, hc):
    pts = []
    for (x0, y0), (x1, y1) in [((0, 0), (L, 0)), ((L, 0), (L, H)), ((L, H), (0, H)), ((0, H), (0, 0))]:
        length = np.hypot(x1 - x0, y1 - y0)
        n = max(int(np.ceil(length / h)), 1)
        for t in np.linspace(0, 1, n, endpoint=False):
            pts.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    nc = int(np.ceil(2 * np.pi * R / hc))
    for phi in np.linspace(0, 2 * np.pi, nc, endpoint=False):
        pts.append((CX + R * np.cos(phi), CY + R * np.sin(phi)))
    return np.array(pts)


def interior_points(h, hc, rng):
    pts = []
    y_levels = np.arange(hc, H, hc)
    for y in y_levels:
        x = hc
        while x < L - hc / 2:
            s = size_field(np.array([x]), np.array([y]), h, hc)[0]
            if rng.random() < (hc / s) ** 2:
                px = x + 0.2 * s * (rng.random() - 0.5)
                py = y + 0.2 * s * (rng.random() - 0.5)
                if 0 < px < L and 0 < py < H and np.hypot(px - CX, py - CY) > R + 0.5 * s:
                    pts.append((px, py))
            x += hc
    return np.array(pts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h", type=float, default=0.04)
    parser.add_argument("--hc", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-o", "--output", default="dfg-coarse.msh")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    pts = np.vstack([boundary_points(args.h, args.hc), interior_points(args.h, args.hc, rng)])
    tri = Delaunay(pts)
    cells = []
    for simplex in tri.simplices:
        c = pts[simplex].mean(axis=0)
        if np.hypot(c[0] - CX, c[1] - CY) < R:
            continue
        a, b, d = pts[simplex]
        area = 0.5 * ((b[0] - a[0]) * (d[1] - a[1]) - (b[1] - a[1]) * (d[0] - a[0]))
        if abs(area) < 1e-14:
            continue
        cells.append(simplex if area > 0 else simplex[[0, 2, 1]])
    used = np.unique(np.array(cells).ravel())
    index = {old: new + 1 for new, old in enumerate(used)}
    with open(args.output, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n%d\n" % len(used))
        for old in used:
            f.write("%d %.16g %.16g 0\n" % (index[old], pts[old][0], pts[old][1]))
        f.write("$EndNodes\n$Elements\n%d\n" % len(cells))
        for k, c in enumerate(cells):
            f.write("%d 2 2 1 1 %d %d %d\n" % (k + 1, index[c[0]], index[c[1]], index[c[2]]))
        f.write("$EndElements\n")
    print("%s: %d vertices, %d triangles" % (args.output, len(used), len(cells)))


if __name__ == "__main__":
    main()

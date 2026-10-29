"""Export a replayed molecule and its lattice as VTK models for 3D Slicer (L105).

    python scripts/replay_molecule.py <run>/runs_a2nm --chunk 0 --molecule 15
    python scripts/export_slicer.py results/trajectories/<name>

Writes legacy-ASCII VTK PolyData, which Slicer loads directly from File > Add Data. Written
by hand rather than through `vtk`/`pyvista` so the project gains no dependency it would have
to install on a cluster.

Files, each a separate Slicer model node so colour and opacity can be set independently:

    <name>_trajectory.vtk    the path as one polyline, with per-point scalars
                             time_s, state (0 free / 1 bound) and cell_id
    <name>_cells.vtk         neighbouring cells, scalar `ring`
    <name>_cells_target.vtk  cells of the target ring
    <name>_cell_capture.vtk  the single cell the molecule was captured on
    <name>_markers.mrk.json  start and capture points as labelled fiducials

Units: the model is in micrometres, and Slicer will read those numbers as millimetres. That
is only a scale — a 10 um cell shows up as a 10 mm sphere — but it means the ruler lies by
1000x unless you scale the transform.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from fpsim.config import Geometry
from fpsim.lattice import make_cluster


def write_polydata(path, points, *, lines=None, polys=None, verts=None, scalars=None, title="fpsim"):
    """Legacy ASCII VTK PolyData. `scalars` maps name -> one value per point."""
    points = np.asarray(points, dtype=float)
    out = ["# vtk DataFile Version 3.0", title, "ASCII", "DATASET POLYDATA",
           f"POINTS {len(points)} float"]
    out += [" ".join(f"{v:.6g}" for v in p) for p in points]
    for name, cells in (("VERTICES", verts), ("LINES", lines), ("POLYGONS", polys)):
        if not cells:
            continue
        out.append(f"{name} {len(cells)} {sum(len(c) + 1 for c in cells)}")
        out += [f"{len(c)} " + " ".join(str(int(i)) for i in c) for c in cells]
    if scalars:
        out.append(f"POINT_DATA {len(points)}")
        for name, values in scalars.items():
            out += [f"SCALARS {name} float 1", "LOOKUP_TABLE default"]
            out += [" ".join(f"{v:.6g}" for v in chunk)
                    for chunk in np.array_split(np.asarray(values, dtype=float),
                                                max(1, len(values) // 9))]
    path.write_text("\n".join(out) + "\n")
    return path


def sphere_mesh(centre, r, n_theta=28, n_phi=14):
    """UV sphere as (points, triangles), poles included."""
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    ph = np.linspace(0, np.pi, n_phi + 2)[1:-1]            # exclude the poles; added separately
    g_th, g_ph = np.meshgrid(th, ph, indexing="ij")
    pts = np.column_stack([(r * np.sin(g_ph) * np.cos(g_th)).ravel(),
                           (r * np.sin(g_ph) * np.sin(g_th)).ravel(),
                           (r * np.cos(g_ph)).ravel()])
    north, south = len(pts), len(pts) + 1
    pts = np.vstack([pts, [0, 0, r], [0, 0, -r]]) + centre

    def idx(i, j):
        return (i % n_theta) * n_phi + j

    tris = []
    for i in range(n_theta):
        for j in range(n_phi - 1):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            tris += [(a, b, c), (a, c, d)]
        tris.append((north, idx(i + 1, 0), idx(i, 0)))
        tris.append((south, idx(i, n_phi - 1), idx(i + 1, n_phi - 1)))
    return pts, tris


def merge_spheres(centres, r, values, n_theta=28, n_phi=14):
    """One polydata for many spheres, carrying a per-point scalar from `values`."""
    P, T, S, off = [], [], [], 0
    for c, v in zip(centres, values):
        pts, tris = sphere_mesh(c, r, n_theta, n_phi)
        P.append(pts)
        T += [(a + off, b + off, d + off) for a, b, d in tris]
        S.append(np.full(len(pts), v))
        off += len(pts)
    return (np.vstack(P), T, np.concatenate(S)) if P else (np.empty((0, 3)), [], np.empty(0))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trajectory", type=Path, help="path without extension, from replay_molecule.py")
    p.add_argument("--out-dir", type=Path, default=None, help="default: <trajectory>_slicer/")
    p.add_argument("--context", type=float, default=3.0,
                   help="include cells within this many lattice spacings of the path (99 = all)")
    p.add_argument("--facets", type=int, default=28, help="sphere resolution around the equator")
    args = p.parse_args()

    base = args.trajectory.parent / args.trajectory.name.removesuffix(".npy").removesuffix(".json")
    traj = np.load(base.parent / f"{base.name}.npy")
    scene = json.loads((base.parent / f"{base.name}.json").read_text())
    row, geo = scene["molecule_row"], dict(scene["geometry"])
    keep = {f.name for f in dataclasses.fields(Geometry)}
    geo["dim"] = traj.shape[1] - 2
    g = Geometry(**{k: v for k, v in geo.items() if k in keep})
    if g.dim != 3:
        raise SystemExit(f"Slicer export is 3D only; trajectory is {g.dim}D")
    cl = make_cluster(g)
    out = args.out_dir or base.parent / f"{base.name}_slicer"
    out.mkdir(parents=True, exist_ok=True)

    xyz = traj[:, :3]
    t = np.arange(len(traj)) * scene["interval"]
    write_polydata(out / f"{base.name}_trajectory.vtk", xyz,
                   lines=[list(range(len(xyz)))],
                   scalars={"time_s": t, "state_bound": traj[:, 3], "cell_id": traj[:, 4]},
                   title=f"molecule {scene['molecule']} of {scene['run']}")

    near = np.flatnonzero(np.linalg.norm(cl.centers - xyz.mean(0), axis=1) <= args.context * g.spacing)
    capture = int(row["final_cell"])
    ring = cl.ring
    groups = {
        f"{base.name}_cells.vtk": [c for c in near if ring[c] != g.target_ring],
        f"{base.name}_cells_target.vtk": [c for c in near if ring[c] == g.target_ring and c != capture],
        f"{base.name}_cell_capture.vtk": [capture],
    }
    for name, cells in groups.items():
        if not cells:
            continue
        pts, tris, sc = merge_spheres(cl.centers[cells], g.cell_radius, ring[cells],
                                      args.facets, max(6, args.facets // 2))
        write_polydata(out / name, pts, polys=tris, scalars={"ring": sc}, title=name)
        print(f"  {name:48} {len(cells):4d} cells, {len(tris):6d} triangles")

    markers = {
        "@schema": "https://raw.githubusercontent.com/Slicer/Slicer/main/Modules/Loadable/"
                   "Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": [{
            "type": "Fiducial", "coordinateSystem": "RAS", "controlPoints": [
                {"id": "1", "label": "start", "position": [float(v) for v in xyz[0]]},
                {"id": "2", "label": f"receptor found ({row['t_end']:.1f} s)",
                 "position": [float(v) for v in xyz[-1]]}]}]}
    (out / f"{base.name}_markers.mrk.json").write_text(json.dumps(markers, indent=2))

    (out / "README.txt").write_text(
        f"3D Slicer scene for molecule {scene['molecule']} of {scene['run']}\n"
        f"{'=' * 72}\n\n"
        "Load: File > Add Data > choose this folder's .vtk files (and the .mrk.json).\n"
        "Each .vtk becomes a Models node; set colour and opacity per node in the Models\n"
        "module. A good starting point:\n\n"
        "  _cells.vtk          grey,  opacity 0.15   (the crowd)\n"
        "  _cells_target.vtk   blue,  opacity 0.25   (ring-2 cells: the goal)\n"
        "  _cell_capture.vtk   green, opacity 0.35   (the cell it was caught on)\n"
        "  _trajectory.vtk     scalar colouring, see below\n\n"
        "Trajectory scalars (Models > Scalars > pick the array, then Visible):\n"
        "  time_s        seconds, 0 .. %.1f\n"
        "  state_bound   0 = free in the gaps, 1 = bound to a membrane\n"
        "  cell_id       which cell it was on or nearest\n\n"
        "Colour by state_bound to see the 2D/3D alternation; colour by time_s to read the\n"
        "path's direction. Line width is under Models > Display > Line Width.\n\n"
        "Units: coordinates are micrometres and Slicer reads them as millimetres, so the\n"
        "cells are 10 mm across on the ruler rather than 10 um. Scale by 0.001 with a\n"
        "linear transform if the real units matter.\n"
        % float(t[-1]))

    print(f"\n{len(xyz)} trajectory points -> {out}")


if __name__ == "__main__":
    main()

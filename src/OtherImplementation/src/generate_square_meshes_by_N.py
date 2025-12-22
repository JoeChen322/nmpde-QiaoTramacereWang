#!/usr/bin/env python3
"""
Generate a family of unit-square Gmsh meshes with filenames like:
  mesh-square-40.msh

Convention:
  N  := "nominal divisions per side"
  lc := 1.0 / N  (target element size)

This matches current naming style and makes the meaning of "40" explicit.
Usage example:
    python3 generate_square_meshes_by_N.py --Ns 20 40 80
"""

from __future__ import annotations
import argparse
from pathlib import Path
import gmsh


def build_unit_square_geo(lc: float) -> tuple[int, list[int]]:
    """
    Build [0,1]x[0,1] geometry in the GEO kernel (fast) with characteristic length lc.
    Returns (surface_tag, [line_tags]).
    """
    gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc, 1)
    gmsh.model.geo.addPoint(1.0, 0.0, 0.0, lc, 2)
    gmsh.model.geo.addPoint(1.0, 1.0, 0.0, lc, 3)
    gmsh.model.geo.addPoint(0.0, 1.0, 0.0, lc, 4)

    l1 = gmsh.model.geo.addLine(1, 2, 1)
    l2 = gmsh.model.geo.addLine(2, 3, 2)
    l3 = gmsh.model.geo.addLine(3, 4, 3)
    l4 = gmsh.model.geo.addLine(4, 1, 4)

    loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4], 1)
    surf = gmsh.model.geo.addPlaneSurface([loop], 1)

    gmsh.model.geo.synchronize()

    return surf, [l1, l2, l3, l4]


def add_physical_groups(surface_tag: int, line_tags: list[int]) -> None:
    """
    Add:
      Physical Curve(1)  = boundary
      Physical Surface(10) = domain
    """
    gmsh.model.addPhysicalGroup(1, line_tags, 1)
    gmsh.model.setPhysicalName(1, 1, "boundary")

    gmsh.model.addPhysicalGroup(2, [surface_tag], 10)
    gmsh.model.setPhysicalName(2, 10, "domain")


def generate_mesh_for_N(N: int, out_file: Path, algo: int = 6) -> tuple[int, int, float]:
    """
    Generate mesh with target size lc=1/N and write to out_file.
    Returns (num_elements, num_nodes, lc).
    """
    if N <= 0:
        raise ValueError("N must be positive")

    lc = 1.0 / float(N)

    gmsh.model.add(f"unit_square_N{N}")
    surf, lines = build_unit_square_geo(lc)
    add_physical_groups(surf, lines)

    # Output format + reproducibility-friendly options
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.option.setNumber("Mesh.Binary", 0)  # ASCII for inspectability

    # Try to keep sizes close to lc
    gmsh.option.setNumber("Mesh.MeshSizeMin", lc)
    gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)

    # Triangular meshing algorithm (robust default: 6=Frontal-Delaunay)
    gmsh.option.setNumber("Mesh.Algorithm", algo)

    gmsh.model.mesh.generate(2)

    # Stats
    elem_types, elem_tags, _ = gmsh.model.mesh.getElements(2)
    num_elements = len(elem_tags[0]) if elem_tags else 0
    _, node_coords, _ = gmsh.model.mesh.getNodes()
    num_nodes = len(node_coords) // 3

    gmsh.write(str(out_file))

    # Clear model for next iteration
    gmsh.model.remove()

    return num_elements, num_nodes, lc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate unit-square mesh family named mesh-square-N.msh where lc=1/N."
    )
    parser.add_argument("--out-dir", type=str, default="mesh", help="Output directory.")
    parser.add_argument(
        "--Ns",
        type=int,
        nargs="+",
        required=True,
        help="List of N values (nominal divisions per side), e.g. --Ns 20 40 80 160",
    )
    parser.add_argument("--prefix", type=str, default="mesh-square", help="Filename prefix.")
    parser.add_argument(
        "--algo",
        type=int,
        default=6,
        help="Gmsh 2D algorithm (default 6=Frontal-Delaunay).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    try:
        print(f"Writing meshes to: {out_dir.resolve()}")
        print("N, lc=1/N, elements, nodes, filename")

        for N in args.Ns:
            out_file = out_dir / f"{args.prefix}-{N}.msh"
            ne, nn, lc = generate_mesh_for_N(N, out_file, algo=args.algo)
            print(f"{N}, {lc:.12g}, {ne}, {nn}, {out_file}")

    finally:
        gmsh.finalize()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

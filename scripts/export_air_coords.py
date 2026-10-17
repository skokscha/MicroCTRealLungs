"""Export voxel coordinates of a phase as a plain text list: one line "x y z".

Same 510^3 grid as air_phase_voxels_510.npz/.raw (artificial boundary frame
removed), for tools that want coordinates instead of a 3D matrix.

Usage: python export_air_coords.py [air|tissue]   (default: air)

Output: data/preview/{air|tissue}_voxel_coordinates_510.txt
  header lines start with '#', then one voxel per line:
  x y z        (0-based)
  voxel centre in STL micrometres: (4.9*(x+0.5), 4.9*(y+0.5), 4.9*(z+0.5))
  for 1-based coordinates add 1 to every number
Lines are sorted by x (slice axis), then y, then z.
Air:      air[x, y, z] == 1    in air_phase_voxels_510.npz
Tissue:  ~air[x, y, z] == 1    (tissue is the exact complement of air)
"""
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
PREVIEW = PROJECT / "data" / "preview"
VOXEL_UM = 4.9


def main() -> None:
    phase = sys.argv[1] if len(sys.argv) > 1 else "air"
    assert phase in ("air", "tissue"), "phase must be 'air' or 'tissue'"

    air = np.load(PREVIEW / "air_phase_voxels_510.npz")["air"]
    mask = air if phase == "air" else ~air
    n = int(mask.sum())
    shape = mask.shape
    print(f"{phase} voxels: {n:,} on grid {shape}", flush=True)

    k, j, i = np.nonzero(mask)
    del mask, air
    print("coordinates computed, writing...", flush=True)

    header = (
        f"# {phase.capitalize()} phase voxels of the 2.5 mm human lung cube"
        " (HiP-CT, 4.9 um data)\n"
        f"# grid {shape[0]}x{shape[1]}x{shape[2]}, voxel {VOXEL_UM} um, origin at cube corner\n"
        f"# {n} {phase} voxels, one per line: x y z (0-based)\n"
        + (
            "# matches air_phase_voxels_510.npz: air[x, y, z] == 1\n"
            if phase == "air"
            else "# matches air_phase_voxels_510.npz: air[x, y, z] == 0 (tissue = complement of air)\n"
        )
        + "# STL: voxel centre at (4.9*(x+0.5), 4.9*(y+0.5), 4.9*(z+0.5)) um; 1-based = add 1\n"
    )

    out = PREVIEW / f"{phase}_voxel_coordinates_510.txt"
    with open(out, "w", buffering=1 << 24) as f:
        f.write(header)
        step = 4_000_000
        for s in range(0, n, step):
            e = min(s + step, n)
            f.write(
                "\n".join(
                    f"{a} {b} {c}"
                    for a, b, c in zip(k[s:e].tolist(), j[s:e].tolist(), i[s:e].tolist())
                )
            )
            f.write("\n")
            print(f"  {e:,}/{n:,}", flush=True)

    print(f"saved {out.name}: {out.stat().st_size / 1e6:.0f} MB", flush=True)


if __name__ == "__main__":
    main()

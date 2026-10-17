"""Tissue + air surface meshes of the 2.5 mm cube for OpenFOAM.

Both phases are built by the SAME function from the SAME cropped voxel grid, so
  - they share one marching-cubes sampling grid -> interface triangles mirror 1:1
  - both are closed with flat caps on all 6 cube faces (watertight)
  - there is no artificial 1-voxel air frame: the zeroed outer layer of the
    512^3 mask (index 0 and 511) is cropped away first.

Output (data/preview/):
  lung_cube_tissue_2p5mm.stl   - tissue solid (cavities are voids)
  lung_cube_air_2p5mm.stl      - air solid (tissue is void)
Coordinates are in micrometres, origin at the cube corner.
For OpenFOAM scale to metres: surfaceTransformPoints -scale '(1e-6 1e-6 1e-6)' in.stl out.stl
"""
from pathlib import Path

import numpy as np
from skimage.measure import marching_cubes
from trimesh import Trimesh
from trimesh.exchange.export import export_mesh

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "data" / "preview"
VOXEL_UM = 4.9
STEP = 2  # 1 = full resolution (huge), 2 = ~8M faces for air


def phase_mesh(phase: np.ndarray, step: int) -> Trimesh:
    """Closed, capped surface of a boolean phase on a cubic grid."""
    n = phase.shape[0]
    # padded size must be 1 + step*k, otherwise the last (high-side) pad layer
    # is never sampled and the +x/+y/+z faces stay open
    pad_hi = (-n) % step or step
    padded = np.pad(phase, ((1, pad_hi),) * 3, constant_values=False)
    verts, faces, _, _ = marching_cubes(
        padded.astype(np.float32),
        level=0.5,
        spacing=(VOXEL_UM,) * 3,
        step_size=step,
    )
    verts -= VOXEL_UM  # low cap sits at padded index 1 -> move cube corner to 0
    m = Trimesh(vertices=verts, faces=faces)
    m.fix_normals()
    if m.volume < 0:
        m.invert()
    return m


def report(name: str, m: Trimesh, v_true_mm3: float) -> None:
    print(
        f"{name}: {len(m.faces):,} faces, watertight={m.is_watertight}, "
        f"volume {m.volume / 1e9:.3f} mm3 vs voxel truth {v_true_mm3:.3f} "
        f"({100 * m.volume / 1e9 / v_true_mm3:.1f}%), "
        f"bounds {m.bounds.min(0).round(1)} .. {m.bounds.max(0).round(1)}",
        flush=True,
    )


def main() -> None:
    mask = np.load(PROJECT / "data" / "cube_4.9um_512_mask.npz")["mask"]
    tissue = mask[1:-1, 1:-1, 1:-1]  # drop zeroed boundary layers -> 510^3
    air = ~tissue                    # exact voxel complement
    print(f"cube {tissue.shape}, tissue fraction {tissue.mean():.3f}", flush=True)

    for name, phase in [("tissue", tissue), ("air", air)]:
        m = phase_mesh(phase, STEP)
        report(name, m, phase.sum() * VOXEL_UM**3 / 1e9)
        path = OUT / f"lung_cube_{name}_2p5mm.stl"
        export_mesh(m, str(path))
        print(f"saved {path.name}, {path.stat().st_size / 1e6:.0f} MB", flush=True)


if __name__ == "__main__":
    main()

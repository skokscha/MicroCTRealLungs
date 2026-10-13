"""Air-phase surface mesh of the full 2.5 mm cube (512^3).

Same recipe as the RVE500 air mesh: air = ~tissue_mask, pad 1 layer, marching
cubes at full resolution (step 1, spacing 4.9 um), Taubin smoothing (25 iters,
no shrinkage), no decimation. Face count ~9M expected (surface grows ~L^2).

Output: data/air_2.5mm/lung_air_2p5mm_taubin.stl
"""
from pathlib import Path

import numpy as np
from skimage.measure import marching_cubes
from trimesh import Trimesh
from trimesh.exchange.export import export_mesh
from trimesh.smoothing import filter_taubin

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "data" / "air_2.5mm"
OUT.mkdir(exist_ok=True)
VOXEL_UM = 4.9


def main() -> None:
    mask = np.load(PROJECT / "data" / "cube_4.9um_512_mask.npz")["mask"]
    air = ~mask  # faces of the cube are already zeroed (air) -> open boundary
    v_air = air.sum() * VOXEL_UM**3 / 1e9
    print(f"cube 512^3, air fraction {air.mean():.3f}, air volume {v_air:.4f} mm3", flush=True)

    padded = np.pad(air, 1, constant_values=False)
    verts, faces, _, _ = marching_cubes(
        padded.astype(np.float32), level=0.5, spacing=(VOXEL_UM,) * 3
    )
    print(f"marching cubes step1: {len(faces):,} faces", flush=True)

    m = Trimesh(vertices=verts, faces=faces)
    del padded
    m.fix_normals()
    filter_taubin(m, lamb=0.5, nu=0.53, iterations=25)
    m.fix_normals()

    path = OUT / "lung_air_2p5mm_taubin.stl"
    export_mesh(m, str(path))
    print(
        f"saved {path.name}: {len(m.faces):,} faces, "
        f"{path.stat().st_size / 1e6:.0f} MB, watertight={m.is_watertight}",
        flush=True,
    )


if __name__ == "__main__":
    main()

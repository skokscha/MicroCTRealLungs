"""Air-phase surface mesh of the full 2.5 mm cube (512^3) - CORRECT recipe.

The air mask is the exact voxel complement of the tissue mask (voxel not in
tissue -> air), so interface triangles mirror the tissue surface 1:1.
Marching cubes at step_size=2 (same sampling as the tissue preview mesh),
NO cluster decimation (that is what wrecked the first attempt), normals fixed
outward and validated against the voxel-counted air volume (13.12 mm3).

Output: data/preview/lung_air_2p5um_mc2.stl  (~8M faces, ~400 MB)
"""
from pathlib import Path

import numpy as np
from skimage.measure import marching_cubes
from trimesh import Trimesh
from trimesh.exchange.export import export_mesh

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "data" / "preview"
VOXEL_UM = 4.9


def main() -> None:
    mask = np.load(PROJECT / "data" / "cube_4.9um_512_mask.npz")["mask"]
    air = ~mask  # exact complement: not-tissue -> air
    v_true = air.sum() * VOXEL_UM**3 / 1e9
    print(f"air: {air.mean():.3f} fraction, {v_true:.3f} mm3 (voxel truth)", flush=True)

    # pad 2 layers: step_size=2 samples every 2nd voxel ([::2]), so a 1-layer
    # pad lands on an odd index at the max sides and gets dropped -> 3 missing
    # box faces (user caught this by overlaying tissue vs air in Blender)
    padded = np.pad(air, 2, constant_values=False)
    verts, faces, _, _ = marching_cubes(
        padded.astype(np.float32), level=0.5, spacing=(VOXEL_UM,) * 3, step_size=2
    )
    del padded
    print(f"marching cubes step2: {len(faces):,} faces", flush=True)

    m = Trimesh(vertices=verts, faces=faces)
    m.fix_normals()
    if m.volume < 0:  # normals ended up pointing into the air solid -> flip
        m.invert()
    print(
        f"mesh volume {m.volume / 1e9:.3f} mm3 vs truth {v_true:.3f} "
        f"({100 * m.volume / 1e9 / v_true:.1f}%)",
        flush=True,
    )

    path = OUT / "lung_air_2p5um_mc2.stl"
    export_mesh(m, str(path))
    print(
        f"saved {path.name}: {len(m.faces):,} faces, {path.stat().st_size / 1e6:.0f} MB",
        flush=True,
    )


if __name__ == "__main__":
    main()

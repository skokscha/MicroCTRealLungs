"""Voxel-true section of the air phase for visualization (no mesh booleans).

Fill the upper half of the cube with 'tissue' (False), run marching cubes on
the lower half -> closed half-box whose flat cut face shows the alveolar
cavity cross-sections. This is the honest cutaway: built from voxels, not from
a boolean on a leaky mesh.

The section STL is a render prop: it is exported, rendered by blender, then the
STL itself is deleted (only the PNG is kept).
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
    air = ~mask
    air[256:, :, :] = False  # upper half -> solid: cut plane at z = 1.25 mm

    # pad 2 layers: with step_size=2 a 1-layer pad is dropped at the max sides
    padded = np.pad(air, 2, constant_values=False)
    verts, faces, _, _ = marching_cubes(
        padded.astype(np.float32), level=0.5, spacing=(VOXEL_UM,) * 3, step_size=2
    )
    m = Trimesh(vertices=verts, faces=faces)
    m.fix_normals()
    if m.volume < 0:
        m.invert()
    path = OUT / "_air_section_prop.stl"
    export_mesh(m, str(path))
    print(f"section prop: {len(m.faces):,} faces, {path.stat().st_size / 1e6:.0f} MB", flush=True)


if __name__ == "__main__":
    main()

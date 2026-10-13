"""Air-phase meshes of the 2.5 mm cube at the same class as data/preview tissue STLs.

Fast version (user-impatient): one marching-cubes pass at step_size=2 (like
lung_alveolar_rve_4.9um_preview.stl), then vertex-cluster decimation to two
levels mirroring the existing pair:
  lung_air_2p5um_preview.stl  - cluster 14.7 um  (counterpart of tissue _preview, 2.87M)
  lung_air_2p5um_view.stl     - cluster 19.6 um  (light, counterpart of tissue _view, 1.22M)

No Taubin (tissue counterparts in data/preview are raw too).
The full-resolution Taubin air mesh remains available via make_air_2p5mm.py.
"""
from pathlib import Path

import numpy as np
from skimage.measure import marching_cubes
from trimesh import Trimesh
from trimesh.exchange.export import export_mesh

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "data" / "preview"
VOXEL_UM = 4.9


def decimate(vertices, faces, step_um):
    key = np.round(vertices / step_um).astype(np.int64)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    new_verts = np.zeros((len(uniq), 3))
    cnt = np.zeros(len(uniq))
    np.add.at(new_verts, inv, vertices)
    np.add.at(cnt, inv, 1)
    new_verts /= cnt[:, None]
    f = inv[faces]
    keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    return new_verts, f[keep]


def main() -> None:
    mask = np.load(PROJECT / "data" / "cube_4.9um_512_mask.npz")["mask"]
    air = ~mask
    print(f"air fraction {air.mean():.3f}, volume {air.sum() * VOXEL_UM**3 / 1e9:.4f} mm3", flush=True)

    padded = np.pad(air, 1, constant_values=False)
    verts, faces, _, _ = marching_cubes(
        padded.astype(np.float32), level=0.5, spacing=(VOXEL_UM,) * 3, step_size=2
    )
    del padded
    print(f"mc step2: {len(faces):,} faces", flush=True)

    for step_um, name in [
        (VOXEL_UM * 3, "lung_air_2p5um_preview.stl"),
        (VOXEL_UM * 4, "lung_air_2p5um_view.stl"),
    ]:
        v2, f2 = decimate(verts, faces, step_um)
        m = Trimesh(vertices=v2, faces=f2)
        m.fix_normals()
        export_mesh(m, str(OUT / name))
        print(f"{name}: {len(f2):,} faces, {(OUT / name).stat().st_size / 1e6:.0f} MB", flush=True)


if __name__ == "__main__":
    main()

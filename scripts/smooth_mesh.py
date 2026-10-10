"""Nicer, smoother RVE mesh:
mask cleanup (drop small debris) -> marching cubes at full step ->
cluster decimation -> Taubin smoothing (no shrinkage) -> STLs.
"""
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import marching_cubes
from trimesh.exchange.export import export_mesh
import trimesh
from trimesh.smoothing import filter_taubin

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "data" / "preview"
VOXEL_UM = 4.9


def cluster_decimate(v, f, step_um):
    key = np.round(v / step_um).astype(np.int64)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    nv = np.zeros((len(uniq), 3))
    cnt = np.zeros(len(uniq))
    np.add.at(nv, inv, v)
    np.add.at(cnt, inv, 1)
    nv /= cnt[:, None]
    ff = inv[f]
    keep = (ff[:, 0] != ff[:, 1]) & (ff[:, 1] != ff[:, 2]) & (ff[:, 0] != ff[:, 2])
    return nv, ff[keep]


def main() -> None:
    mask = np.load(PROJECT / "data" / "cube_4.9um_512_mask.npz")["mask"]

    # drop tiny floating debris (< 500 voxels) that reads as visual noise
    lab, n = ndi.label(mask)
    if n > 1:
        sizes = np.bincount(lab.ravel())
        keep_ids = np.where(sizes >= 500)[0]
        keep_ids = keep_ids[keep_ids != 0]
        mask = np.isin(lab, keep_ids)
        print(f"components: {n} -> kept {len(keep_ids)}, tissue {mask.mean():.3f}", flush=True)
    del lab

    verts, faces, _, _ = marching_cubes(
        mask.astype(np.float32), level=0.5, spacing=(VOXEL_UM,) * 3, step_size=1
    )
    print(f"full-res surface: {len(faces) / 1e6:.1f}M faces", flush=True)
    del mask

    for step, iters, name in [
        (VOXEL_UM * 2, 45, "lung_alveolar_rve_4.9um_smooth.stl"),
        (VOXEL_UM * 4, 30, "lung_alveolar_rve_4.9um_smooth_light.stl"),
    ]:
        v, f = cluster_decimate(verts, faces, step)
        m = trimesh.Trimesh(vertices=v, faces=f)
        m.fix_normals()
        filter_taubin(m, lamb=0.5, nu=0.53, iterations=iters)
        m.fix_normals()
        export_mesh(m, str(OUT / name))
        print(
            f"{name}: {len(m.faces) / 1e6:.2f}M faces, "
            f"{(OUT / name).stat().st_size / 1e6:.0f} MB",
            flush=True,
        )


if __name__ == "__main__":
    main()

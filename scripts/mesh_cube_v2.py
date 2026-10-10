"""Rebuild the alveolar RVE mesh with manageable sizes:
- recompute + archive the binary mask (npz)
- marching cubes step 2 -> decimated preview STLs
- shaded matplotlib render for visual check
"""
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.measure import marching_cubes

PROJECT = Path(__file__).resolve().parent.parent
CUBE = PROJECT / "data" / "cube_4.9um_512.npy"
OUT = PROJECT / "data" / "preview"
VOXEL_UM = 4.9


def otsu_on_sample(data: np.ndarray) -> float:
    return float(threshold_otsu(data[::4, ::4, ::4].ravel()))


def decimate(vertices: np.ndarray, faces: np.ndarray, step_um: float):
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


def render(verts: np.ndarray, faces: np.ndarray, path: Path, max_faces: int = 150_000):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if len(faces) > max_faces:
        sel = np.random.default_rng(0).choice(len(faces), max_faces, replace=False)
        f = faces[sel]
    else:
        f = faces
    tri = verts[f]

    fig = plt.figure(figsize=(10, 10), dpi=110)
    ax = fig.add_subplot(projection="3d")
    # light shading by normal
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    light = np.array([0.4, -0.5, 0.77])
    shade = np.clip(np.abs(n @ light), 0.25, 1.0)
    colors = np.dstack([shade * 0.55, shade * 0.75, shade * 0.45, np.ones_like(shade)])
    pc = Poly3DCollection(tri, facecolors=colors[0], edgecolor="none")
    ax.add_collection3d(pc)
    r = verts.max(0) - verts.min(0)
    c = (verts.max(0) + verts.min(0)) / 2
    ax.set_xlim(c[0] - r.max() / 2, c[0] + r.max() / 2)
    ax.set_ylim(c[1] - r.max() / 2, c[1] + r.max() / 2)
    ax.set_zlim(c[2] - r.max() / 2, c[2] + r.max() / 2)
    ax.set_axis_off()
    ax.view_init(elev=18, azim=-60)
    fig.tight_layout(pad=0)
    fig.savefig(path)
    plt.close(fig)
    print("render saved:", path.name, flush=True)


def main() -> None:
    data = np.load(CUBE, mmap_mode="r")
    thr = otsu_on_sample(np.asarray(data))
    print(f"otsu threshold: {thr:.0f}", flush=True)

    smooth = np.empty(data.shape, dtype=np.float32)
    for k in range(data.shape[0]):
        smooth[k] = ndi.gaussian_filter(data[k].astype(np.float32), sigma=0.8)
    mask = smooth > thr
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3, 3)), iterations=1)
    mask[0, :, :] = mask[-1, :, :] = False
    mask[:, 0, :] = mask[:, -1, :] = False
    mask[:, :, 0] = mask[:, :, -1] = False
    print(f"tissue fraction: {mask.mean():.3f}", flush=True)

    np.savez_compressed(PROJECT / "data" / "cube_4.9um_512_mask.npz", mask=mask)
    print("mask archived (npz)", flush=True)
    del smooth

    verts, faces, _, _ = marching_cubes(
        mask.astype(np.float32), level=0.5, spacing=(VOXEL_UM,) * 3, step_size=2
    )
    print(f"mc step2: {len(faces)} faces", flush=True)

    import trimesh
    from trimesh.exchange.export import export_mesh

    for step, name in [(VOXEL_UM * 3, "lung_alveolar_rve_4.9um_preview.stl")]:
        v2, f2 = decimate(verts, faces, step_um=step)
        m = trimesh.Trimesh(vertices=v2, faces=f2)
        m.fix_normals()
        export_mesh(m, str(OUT / name))
        print(
            f"{name}: {len(f2)} faces, {(OUT / name).stat().st_size / 1e6:.1f} MB, "
            f"watertight={m.is_watertight}",
            flush=True,
        )

    render(v2, f2, OUT / "rve_render.png")


if __name__ == "__main__":
    main()

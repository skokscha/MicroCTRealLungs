"""Segment the 512^3 cube (tissue vs air) and build an alveolar-geometry STL.

Pipeline: histogram -> Otsu threshold -> gaussian smoothing -> closing ->
marching cubes (spacing = 4.9 um) -> STL (full + decimated) + mask previews.
"""
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.measure import marching_cubes
from trimesh.exchange.export import export_mesh

PROJECT = Path(__file__).resolve().parent.parent
CUBE = PROJECT / "data" / "cube_4.9um_512.npy"
OUT = PROJECT / "data" / "preview"
VOXEL_UM = 4.9


def otsu_on_sample(data: np.ndarray) -> float:
    sample = data[::4, ::4, ::4].ravel()
    return float(threshold_otsu(sample))


def save_overlay(gray: np.ndarray, mask: np.ndarray, path: Path) -> None:
    lo, hi = np.percentile(gray, [1, 99])
    g = np.clip((gray - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = np.dstack([g, g, g])
    edge = mask ^ ndi.binary_erosion(mask)
    rgb[edge] = [1.0, 0.2, 0.2]
    Image.fromarray((rgb * 255).astype(np.uint8)).save(path)


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


def main() -> None:
    data = np.load(CUBE, mmap_mode="r")
    thr = otsu_on_sample(np.asarray(data))
    print(f"otsu threshold: {thr:.0f}", flush=True)

    # smoothing in chunks to keep RAM low
    smooth = np.empty(data.shape, dtype=np.float32)
    for k in range(data.shape[0]):
        smooth[k] = ndi.gaussian_filter(data[k].astype(np.float32), sigma=0.8)
    mask = smooth > thr
    print(f"tissue fraction: {mask.mean():.3f}", flush=True)

    mask = ndi.binary_closing(mask, structure=np.ones((3, 3, 3)), iterations=1)
    mask[:, 0, :] = mask[:, -1, :] = False
    mask[0, :, :] = mask[-1, :, :] = False
    mask[:, :, 0] = mask[:, :, -1] = False
    print(f"tissue fraction after closing: {mask.mean():.3f}", flush=True)

    # previews: mask edge overlay on raw slices
    for k, tag in [(128, "z128"), (256, "z256"), (384, "z384")]:
        save_overlay(np.asarray(data[k]), mask[k], OUT / f"mask_overlay_{tag}.png")
    print("overlays saved", flush=True)

    verts, faces, _, _ = marching_cubes(
        mask.astype(np.float32), level=0.5, spacing=(VOXEL_UM,) * 3, step_size=1
    )
    print(f"full mesh: {len(verts)} verts, {len(faces)} faces", flush=True)

    import trimesh

    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    path = OUT / "lung_alveolar_rve_4.9um.stl"
    export_mesh(mesh, str(path))
    print(f"saved {path.name} ({path.stat().st_size / 1e6:.0f} MB)", flush=True)

    v2, f2 = decimate(verts, faces, step_um=VOXEL_UM * 2)
    small = trimesh.Trimesh(vertices=v2, faces=f2)
    small.fix_normals()
    path2 = OUT / "lung_alveolar_rve_4.9um_small.stl"
    export_mesh(small, str(path2))
    print(
        f"saved {path2.name}: {len(f2)} faces, "
        f"{path2.stat().st_size / 1e6:.1f} MB, watertight={small.is_watertight}",
        flush=True,
    )


if __name__ == "__main__":
    main()

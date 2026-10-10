"""Build a ~500 um RVE sub-cube (102 voxels x 4.9 um) inside the 512^3 cube.

- scans candidate offsets, picks the one with tissue fraction closest to the
  global mean (clean parenchyma, no vessel bias)
- exports gray cube, tissue/air masks, and full-resolution STLs of BOTH phases
  (marching cubes at step 1 + Taubin smoothing, no decimation)
- saves overlay previews for visual control
"""
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.measure import marching_cubes
from trimesh import Trimesh
from trimesh.exchange.export import export_mesh
from trimesh.smoothing import filter_taubin

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
OUT = DATA / "rve500"
OUT.mkdir(exist_ok=True)
VOXEL_UM = 4.9
N = 102  # 102 * 4.9 um = 499.8 um


def mesh_from_mask(mask: np.ndarray, name: str, taubin_iters: int = 25) -> Trimesh:
    padded = np.pad(mask, 1, constant_values=False)
    verts, faces, _, _ = marching_cubes(
        padded.astype(np.float32), level=0.5, spacing=(VOXEL_UM,) * 3
    )
    m = Trimesh(vertices=verts, faces=faces)
    m.fix_normals()
    filter_taubin(m, lamb=0.5, nu=0.53, iterations=taubin_iters)
    m.fix_normals()
    export_mesh(m, str(OUT / name))
    print(f"{name}: {len(m.faces)} faces, {(OUT / name).stat().st_size / 1e6:.1f} MB", flush=True)
    return m


def save_overlay(gray, mask, path):
    lo, hi = np.percentile(gray, [1, 99])
    g = np.clip((gray - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = np.dstack([g, g, g])
    edge = mask ^ ndi.binary_erosion(mask)
    rgb[edge] = [1.0, 0.25, 0.25]
    Image.fromarray((rgb * 255).astype(np.uint8)).save(path)


def main() -> None:
    mask_full = np.load(DATA / "cube_4.9um_512_mask.npz")["mask"]
    global_frac = mask_full.mean()

    # scan candidate offsets on a grid, keep the most "average" one
    best, best_score = None, 1e9
    for ox in range(0, 512 - N + 1, 51):
        for oy in range(0, 512 - N + 1, 51):
            for oz in range(0, 512 - N + 1, 51):
                frac = mask_full[oz:oz + N, oy:oy + N, ox:ox + N].mean()
                score = abs(frac - global_frac)
                if score < best_score:
                    best, best_score = (ox, oy, oz), score
    ox, oy, oz = best
    print(f"chosen offset x={ox} y={oy} z={oz}, tissue frac={mask_full[oz:oz+N, oy:oy+N, ox:ox+N].mean():.3f}", flush=True)

    gray = np.load(DATA / "cube_4.9um_512.npy", mmap_mode="r")
    sub_gray = np.asarray(gray[oz:oz + N, oy:oy + N, ox:ox + N])
    sub_mask = mask_full[oz:oz + N, oy:oy + N, ox:ox + N].copy()

    np.save(OUT / "cube_gray.npy", sub_gray)
    np.savez_compressed(OUT / "mask_tissue.npz", mask=sub_mask)

    v_tissue = sub_mask.sum() * VOXEL_UM**3
    v_air = (1 - sub_mask.mean()) * N**3 * VOXEL_UM**3
    print(f"tissue volume: {v_tissue / 1e9:.4f} mm3, air volume: {v_air / 1e9:.4f} mm3", flush=True)

    mesh_from_mask(sub_mask, "lung_rve500_tissue.stl")
    mesh_from_mask(~sub_mask, "lung_rve500_air.stl")

    k = N // 2
    save_overlay(sub_gray[k], sub_mask[k], OUT / "overlay_mid.png")
    print("overlay saved", flush=True)


if __name__ == "__main__":
    main()

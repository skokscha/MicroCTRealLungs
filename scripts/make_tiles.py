"""Generate 10 tileable 102^3 RVE tiles from the 512^3 mask.

All tiles live on ONE lattice: base offset 1 voxel, stride 102 voxels, 5 slots
per axis. Any subset of the 5x5x5 lattice glues seamlessly into a bigger volume
because the tiles are disjoint and share face planes exactly.

This script produces a 5x2x1 slab (10 tiles): all 5 x-slots, the 2 best y-slots
and the best z-slot (tissue fraction closest to the global mean).

Per tile (data/tiles/):
  tile_x{ix}_y{a|b}_z{iz}_tissue_raw.stl  - marching cubes, no smoothing
  tile_x{ix}_y{a|b}_z{iz}_air_raw.stl     - air phase, same
  tile_x{ix}_y{a|b}_z{iz}_mask.npz        - tile mask + voxel offset (z,y,x)
  tile_x{ix}_y{a|b}_z{iz}_overlay.png     - segmentation control (mid slice)

Smoothing is done separately in Blender: scripts/blender_smooth_tiles.py.
"""
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.measure import marching_cubes
from trimesh import Trimesh
from trimesh.exchange.export import export_mesh

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
OUT = DATA / "tiles"
OUT.mkdir(exist_ok=True)
VOXEL_UM = 4.9
N = 102
BASE = 1  # keeps tiles off the zeroed face planes (index 0 and 511) of the 512 cube
SLOTS = [BASE + i * N for i in range(5)]  # 1, 103, 205, 307, 409


def mesh_from_mask(mask: np.ndarray, name: str) -> int:
    padded = np.pad(mask, 1, constant_values=False)
    verts, faces, _, _ = marching_cubes(
        padded.astype(np.float32), level=0.5, spacing=(VOXEL_UM,) * 3
    )
    m = Trimesh(vertices=verts, faces=faces)
    m.fix_normals()
    export_mesh(m, str(OUT / name))
    return len(faces)


def save_overlay(gray: np.ndarray, mask2d: np.ndarray, path: Path) -> None:
    lo, hi = np.percentile(gray, [1, 99])
    g = np.clip((gray - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = np.dstack([g, g, g])
    edge = mask2d ^ ndi.binary_erosion(mask2d)
    rgb[edge] = [1.0, 0.25, 0.25]
    Image.fromarray((rgb * 255).astype(np.uint8)).save(path)


def main() -> None:
    mask = np.load(DATA / "cube_4.9um_512_mask.npz")["mask"]
    g = float(mask.mean())
    print(f"global tissue frac: {g:.4f}", flush=True)

    # pick the z-slot and two y-slots whose slab fraction is closest to global
    z_scores = [abs(mask[z : z + N, :, :].mean() - g) for z in SLOTS]
    iz = int(np.argmin(z_scores))
    y_scores = [abs(mask[:, y : y + N, :].mean() - g) for y in SLOTS]
    iys = [int(i) for i in np.argsort(y_scores)[:2]]
    print(
        f"chosen z slot {SLOTS[iz]} (dev {z_scores[iz]:.4f}), "
        f"y slots {[SLOTS[i] for i in iys]} (dev {[round(y_scores[i], 4) for i in iys]})",
        flush=True,
    )

    gray = np.load(DATA / "cube_4.9um_512.npy", mmap_mode="r")
    y_tag = {iys[0]: "a", iys[1]: "b"}

    for iy in iys:
        for ix in range(5):
            ox, oy, oz = SLOTS[ix], SLOTS[iy], SLOTS[iz]
            tag = f"tile_x{ix}_y{y_tag[iy]}_z{iz}"
            sub = mask[oz : oz + N, oy : oy + N, ox : ox + N]
            ft = mesh_from_mask(sub, f"{tag}_tissue_raw.stl")
            fa = mesh_from_mask(~sub, f"{tag}_air_raw.stl")
            np.savez_compressed(OUT / f"{tag}_mask.npz", mask=sub, offset=(oz, oy, ox))
            save_overlay(
                np.asarray(gray[oz + N // 2, oy : oy + N, ox : ox + N]),
                sub[N // 2],
                OUT / f"{tag}_overlay.png",
            )
            print(
                f"{tag}: offset(z,y,x)={oz},{oy},{ox} frac={sub.mean():.3f} "
                f"faces tissue/air={ft}/{fa}",
                flush=True,
            )

    total = sum(p.stat().st_size for p in OUT.glob("*.stl")) / 1e6
    print(f"raw STL total: {total:.0f} MB in {OUT}", flush=True)


if __name__ == "__main__":
    main()

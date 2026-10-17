"""Export the AIR (gas) phase as voxel matrices for connectivity analysis.

Source: data/cube_4.9um_512_mask.npz (tissue mask). The artificially zeroed
1-voxel boundary frame is cropped away -> 510^3 grid, exactly matching the
lung_cube_tissue/air_2p5mm STLs (no fake air frame: honest connectivity).

Outputs (data/preview/):
  air_phase_voxels_510.npz - compressed bool array 'air' (True = air voxel)
  air_phase_voxels_510.raw - same volume as plain uint8 0/1, no metadata
Axis order is numpy [z, y, x]; voxel 4.9 um isotropic; origin at cube corner
(STL coordinate of voxel centre i,j,k = 4.9*(i+0.5), 4.9*(j+0.5), 4.9*(k+0.5)).
Also prints air-phase connectivity stats (6-connectivity).
"""
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
OUT = DATA / "preview"
VOXEL_UM = 4.9


def main() -> None:
    mask = np.load(DATA / "cube_4.9um_512_mask.npz")["mask"]
    air = ~mask[1:-1, 1:-1, 1:-1]  # drop the forced-zero frame -> 510^3
    print(f"grid {air.shape}, air fraction {air.mean():.3f}", flush=True)

    np.savez_compressed(
        OUT / "air_phase_voxels_510.npz", air=air, voxel_um=np.float32(VOXEL_UM)
    )
    air.astype(np.uint8).tofile(OUT / "air_phase_voxels_510.raw")
    print("saved air_phase_voxels_510.npz and .raw", flush=True)

    lab, n = ndi.label(air)  # default structure = 6-connectivity (faces)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0  # label 0 is tissue
    order = np.argsort(sizes)[::-1]
    big = order[0]
    comp = lab == big
    touches = [
        bool(comp[0].any()), bool(comp[-1].any()),
        bool(comp[:, 0].any()), bool(comp[:, -1].any()),
        bool(comp[:, :, 0].any()), bool(comp[:, :, -1].any()),
    ]
    frac = sizes[order[:5]] / air.sum()
    print(f"air components (6-conn): {n}", flush=True)
    print(
        f"largest: {sizes[big]:,} voxels = {frac[0] * 100:.2f}% of air; "
        f"touches all 6 cube faces: {all(touches)} {touches}",
        flush=True,
    )
    print(f"top-5 component shares of air: {np.round(frac * 100, 3)}%", flush=True)
    print(f"air outside largest component: {(1 - frac[0]) * 100:.3f}%", flush=True)
    print(f"isolated pockets < 1000 voxels: {(sizes[1:] < 1000).sum()}", flush=True)


if __name__ == "__main__":
    main()

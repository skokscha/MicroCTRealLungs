"""Smoke test of the full pipeline at the coarse scale (s4, 39.2 um voxels):
stream the volume from GCS, threshold tissue vs air, marching cubes, STL.
The real mesh will later be built from the full-resolution s0 data.
"""
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import marching_cubes
from trimesh.exchange.export import export_mesh

import hoa_tools.dataset

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "data" / "preview"
OUT.mkdir(parents=True, exist_ok=True)

DATASET = "LADAF-2020-27_lung_left_VOI-06-lower-lobe-basal_2.45um_bm05"
VOXEL_UM = 2.45 * 16  # s4 downsampling factor is 16


def main() -> None:
    print("streaming s4 ...", flush=True)
    ds = hoa_tools.dataset.get_dataset(DATASET)
    data = np.asarray(ds.data_array(downsample_level=4).data)

    hist_lo, hist_hi = np.percentile(data, [0.5, 99.5])
    print(f"percentile range: {hist_lo:.0f}-{hist_hi:.0f}")

    # smooth a bit, then Otsu-style midpoint threshold on the histogram
    smooth = ndi.gaussian_filter(data.astype(np.float32), sigma=1.0)
    hist, edges = np.histogram(smooth, bins=256, range=(hist_lo, hist_hi))
    mids = 0.5 * (edges[:-1] + edges[1:])
    p = hist / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * mids)
    mu_t = mu[-1]
    sigma_b = (mu_t * omega - mu) ** 2 / np.maximum(omega * (1 - omega), 1e-12)
    thr = mids[int(np.argmax(sigma_b))]
    print(f"otsu threshold: {thr:.0f}")

    mask = smooth > thr
    print(f"tissue fraction: {mask.mean():.3f}")

    # pad with air so the volume boundary caps the sponge like a cut RVE cube
    mask = np.pad(mask, 1, constant_values=False)
    # close tiny holes so walls are watertight at this coarse scale
    mask = ndi.binary_closing(mask, iterations=1)

    spacing = (VOXEL_UM,) * 3
    verts, faces, _, _ = marching_cubes(mask.astype(np.float32), level=0.5, spacing=spacing)
    print(f"mesh: {len(verts)} verts, {len(faces)} faces")

    import trimesh

    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.fix_normals()
    path = OUT / "lung_s4_smoke_test.stl"
    export_mesh(mesh, str(path))
    print(f"saved {path} ({path.stat().st_size / 1e6:.1f} MB), watertight={mesh.is_watertight}")


if __name__ == "__main__":
    main()

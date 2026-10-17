"""Binary (.npy) versions of the phase coordinate lists.

Row r of the file == line r of {air,tissue}_voxel_coordinates_510.txt.
dtype int16 (little-endian), shape (N, 3), columns x, y, z (0-based);
coordinates fit int16 exactly (max index 509 < 32767).
The 128-byte npy header can be skipped by non-numpy tools:
np.fromfile(path, np.int16, offset=128).reshape(-1, 3).

Output: data/preview/{air,tissue}_voxel_coordinates_510.npy
"""
from pathlib import Path

import numpy as np

PREVIEW = Path(__file__).resolve().parent.parent / "data" / "preview"


def dump(name: str, mask: np.ndarray) -> None:
    k, j, i = np.nonzero(mask)
    coords = np.empty((len(k), 3), dtype=np.int16)
    coords[:, 0] = k
    del k
    coords[:, 1] = j
    del j
    coords[:, 2] = i
    del i
    np.save(PREVIEW / f"{name}_voxel_coordinates_510.npy", coords)
    print(f"{name}: {coords.shape[0]:,} triplets saved", flush=True)


def main() -> None:
    air = np.load(PREVIEW / "air_phase_voxels_510.npz")["air"]
    dump("air", air)
    dump("tissue", ~air)


if __name__ == "__main__":
    main()

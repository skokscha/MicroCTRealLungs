"""Compress a voxel phase with our own RLE (nested loops).

For x in slices, for y in rows, walk along z and record runs of equal values
as (value, length) pairs.

Usage: python compress_rle.py [air|tissue]   (default: air)

Note: run boundaries are identical for both phases (tissue = ~air), only the
0/1 values swap — the two RLE files carry the same information, and either
one alone regenerates the whole volume (invert the values for the other phase).

Outputs (data/preview/):
  {phase}_phase_510_rle.bin        - packed runs: uint8 value + uint16 length, 3 B/run
  {phase}_phase_510_rle_zlib.bin   - the same stream, zlib level 9

The original mask (and therefore the coordinate lists) is regenerated from
the RLE file alone: np.repeat(values, lengths).reshape(Nx, Ny, Nz).
Verified bit-for-bit against the source mask at the end.
"""
import sys
import zlib
from pathlib import Path

import numpy as np

PREVIEW = Path(__file__).resolve().parent.parent / "data" / "preview"
Nx = Ny = Nz = 510


def main() -> None:
    phase = sys.argv[1] if len(sys.argv) > 1 else "air"
    assert phase in ("air", "tissue"), "phase must be 'air' or 'tissue'"

    air = np.load(PREVIEW / "air_phase_voxels_510.npz")["air"]
    mask = air if phase == "air" else ~air

    vals: list = []
    lens: list = []
    for x in range(Nx):                      # цикл 1: срезы
        for y in range(Ny):                  # цикл 2: строки
            row = mask[x, y]                 # вдоль z (510 значений)
            change = np.flatnonzero(row[1:] != row[:-1]) + 1
            starts = np.concatenate(([0], change))
            ends = np.concatenate((change, [Nz]))
            vals.append(row[starts].astype(np.uint8))
            lens.append((ends - starts).astype(np.uint16))

    v = np.concatenate(vals)
    l = np.concatenate(lens)
    runs = np.empty(len(v), dtype=[("v", "u1"), ("l", "<u2")])  # 3 байта на отрезок
    runs["v"] = v
    runs["l"] = l
    rle_path = PREVIEW / f"{phase}_phase_510_rle.bin"
    rle_path.write_bytes(runs.tobytes())
    print(f"runs: {len(runs):,}  ->  rle {rle_path.stat().st_size / 1e6:.1f} MB", flush=True)

    zlib_path = PREVIEW / f"{phase}_phase_510_rle_zlib.bin"
    zlib_path.write_bytes(zlib.compress(runs.tobytes(), 9))
    print(f"rle+zlib: {zlib_path.stat().st_size / 1e6:.1f} MB", flush=True)

    # проверка: декодируем и сверяем бит-в-бит
    back = np.repeat(runs["v"].astype(bool), runs["l"].astype(np.int64)).reshape(Nx, Ny, Nz)
    assert np.array_equal(back, mask), "RLE decode mismatch!"
    print(f"decode check: {phase} mask restored bit-for-bit  OK", flush=True)


if __name__ == "__main__":
    main()

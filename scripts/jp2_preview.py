"""Decode sample slices from the 4.9um JP2 zip and build a preview montage
to choose a clean parenchyma region for the RVE cube.
"""
import io
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parent.parent
ZIP = PROJECT / "4.9um_LADAF-2020-27_lung-left_06_pag-0.02_0.04_jp2_.zip"
OUT = PROJECT / "data" / "preview"
OUT.mkdir(parents=True, exist_ok=True)

STEP = 66  # decode every 66th slice -> 16 slices over the stack


def main() -> None:
    z = zipfile.ZipFile(ZIP)
    names = sorted(n for n in z.namelist() if n.endswith(".jp2"))
    idxs = list(range(0, len(names), STEP))
    tiles = []
    all_vals = []
    for i in idxs:
        a = np.array(Image.open(io.BytesIO(z.read(names[i]))))
        all_vals.append(a[::4, ::4])  # subsample for stats
        lo, hi = np.percentile(a, [1, 99])
        img = np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
        tiles.append(Image.fromarray((img * 255).astype(np.uint8)).resize((476, 476)))

    vals = np.concatenate([v.ravel() for v in all_vals])
    print("sampled intensity percentiles:")
    for p in [0.1, 1, 10, 50, 90, 99, 99.9]:
        print(f"  p{p}: {np.percentile(vals, p):.0f}")
    print("min/max:", vals.min(), vals.max())

    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    montage = Image.new("L", (cols * 476, rows * 476))
    for k, t in enumerate(tiles):
        montage.paste(t, ((k % cols) * 476, (k // cols) * 476))
    path = OUT / "jp2_4.9um_slices_montage.png"
    montage.save(path)
    print("saved", path, f"({len(tiles)} slices, indices {idxs})")


if __name__ == "__main__":
    main()

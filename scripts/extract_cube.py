"""Extract a 512^3 uint16 cube from the 4.9um JP2 zip into a single .npy."""
import io
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parent.parent
ZIP = PROJECT / "4.9um_LADAF-2020-27_lung-left_06_pag-0.02_0.04_jp2_.zip"
OUT = PROJECT / "data" / "cube_4.9um_512.npy"

X0, Y0 = 295, 1044  # cube top-left corner in slice coords
N = 512
Z0 = 400


def main() -> None:
    z = zipfile.ZipFile(ZIP)
    names = sorted(n for n in z.namelist() if n.endswith(".jp2"))
    assert Z0 + N <= len(names), "not enough slices"
    out = np.lib.format.open_memmap(
        OUT, mode="w+", dtype=np.uint16, shape=(N, N, N)
    )
    for k, zi in enumerate(range(Z0, Z0 + N)):
        a = np.array(Image.open(io.BytesIO(z.read(names[zi]))))
        out[k] = a[Y0 : Y0 + N, X0 : X0 + N]
        if k % 100 == 0:
            print(f"{k}/{N}", flush=True)
    out.flush()
    print("saved", OUT, f"{OUT.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()

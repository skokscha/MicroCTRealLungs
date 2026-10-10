"""Quick connectivity test + preview: load the coarsest scale (s4) of the HOA
zoom dataset and save a couple of PNG slices to data/preview/.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from hoa_utils import get_dataset_array  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
PREVIEW = PROJECT / "data" / "preview"
PREVIEW.mkdir(parents=True, exist_ok=True)

DATASET = "LADAF-2020-27_lung_left_VOI-06-lower-lobe-basal_2.45um_bm05"


def save_png(arr: np.ndarray, path: Path) -> None:
    lo, hi = np.percentile(arr, [1, 99])
    img = np.clip((arr.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
    Image.fromarray((img * 255).astype(np.uint8)).save(path)


def main() -> None:
    print("Opening dataset (downsample level 4)...")
    arr = get_dataset_array(DATASET, downsample_level=4)
    print("shape:", arr.shape, "dtype:", arr.dtype)

    data = np.asarray(arr)
    print("loaded into memory:", data.nbytes / 1e6, "MB")
    print("intensity range:", data.min(), "-", data.max())

    nz, ny, nx = data.shape
    save_png(data[nz // 2, :, :], PREVIEW / "s4_axial_mid.png")
    save_png(data[:, ny // 2, :], PREVIEW / "s4_coronal_mid.png")
    save_png(data[:, :, nx // 2], PREVIEW / "s4_sagittal_mid.png")
    print("saved previews to", PREVIEW)


if __name__ == "__main__":
    main()

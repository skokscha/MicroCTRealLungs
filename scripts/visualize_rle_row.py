"""Visual explainer: how RLE packs one real row of the lung cube.

Reads the first runs of air_phase_510_rle.bin (they belong to row x=0, y=0),
draws the 510-voxel strip and its (value, length) pairs ->
data/preview/rle_row_explained.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

PREVIEW = Path(__file__).resolve().parent.parent / "data" / "preview"
AIR_C, TIS_C = "#9ecae1", "#8c6d31"


def main() -> None:
    runs = np.fromfile(PREVIEW / "air_phase_510_rle.bin", dtype=[("v", "u1"), ("l", "<u2")])
    ends = np.cumsum(runs["l"])
    k = int(np.searchsorted(ends, 510)) + 1  # runs of row (x=0, y=0)
    vals, lens = runs["v"][:k].astype(int), runs["l"][:k].astype(int)
    assert ends[k - 1] == 510, "first row must end exactly at 510"
    row = np.repeat(vals, lens)  # the real 510 voxels of row (0,0)

    fig, ax = plt.subplots(figsize=(14, 3.6), dpi=130)
    ax.imshow(row[None, :], aspect="auto", cmap=ListedColormap([TIS_C, AIR_C]),
              extent=(-0.5, 509.5, 0, 1))
    for b in (np.cumsum(lens)[:-1] - 0.5):
        ax.axvline(b, color="black", lw=1.0, alpha=0.8)
    z0 = 0
    for j, (v, l) in enumerate(zip(vals, lens)):
        if j < 8:
            name = "воздух" if v else "стенка"
            ax.text(z0 + (l - 1) / 2, 1.15 + 0.55 * (j % 2), f"({v}, {l})\n{name}",
                    ha="center", va="bottom", fontsize=9)
        z0 += l
    ax.text(508, 1.7, "…", ha="right", fontsize=16)
    ax.set_ylim(0, 2.7)
    ax.set_yticks([])
    ax.set_xlabel("вдоль оси z — один ряд куба из 510 вокселей (x=0, y=0)")
    ax.set_title(f"Ряд куба: 510 вокселей → {k} пар RLE ({k * 3} байт вместо 510); "
                 f"таких рядов в кубе 260 100, всего пар 5 741 783")
    fig.tight_layout()
    out = PREVIEW / "rle_row_explained.png"
    fig.savefig(out)
    print("saved", out)
    print("row (0,0) runs:", list(zip(vals.tolist(), lens.tolist())))


if __name__ == "__main__":
    main()

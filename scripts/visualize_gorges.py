"""Visualize the bottleneck map from connectivity_paths.py.

Per flow axis: (left) total path visits per cross-section along the flow axis
- peaks mark gorge planes; (right) the most-loaded cross-section: gray = air,
brown = tissue, yellow-to-red = how many of the traced paths cross the voxel.

Output: data/preview/connectivity_gorges.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm
import numpy as np

PREVIEW = Path(__file__).resolve().parent.parent / "data" / "preview"


def main() -> None:
    air = np.load(PREVIEW / "air_phase_voxels_510.npz")["air"]
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), dpi=120,
                             gridspec_kw={"width_ratios": [1, 1.6], "wspace": 0.18})

    for row, name in enumerate(["x", "y", "z"]):
        d = np.load(PREVIEW / f"connectivity_visits_{name}.npz")
        v = d["visits"]
        prof = v.sum(axis=tuple(i for i in range(3) if i != row))  # per cross-section
        ax = axes[row, 0]
        ax.plot(prof, color="#d95f02", lw=1.2)
        peak = int(np.argmax(prof))
        ax.axvline(peak, color="k", ls="--", lw=1)
        ax.set_title(f"ось {name}: сумма проходов путей по сечениям\n"
                     f"пик на сечении {peak} ({prof.max():,} проходов)", fontsize=10)
        ax.set_xlabel(f"координата вдоль оси {name}")
        ax.set_ylabel("сумма проходов")
        ax.grid(alpha=0.3)

        sl = np.take(v, peak, axis=row)  # the hottest cross-section
        air_sl = np.take(air, peak, axis=row)
        rgb = np.full(air_sl.shape + (3,), 0.62)  # neutral: air gray
        rgb[~air_sl] = (0.55, 0.43, 0.19)         # tissue brown
        m = sl > 0
        frac = np.clip(sl[m] / max(1, sl.max()), 0, 1)
        cmap = plt.get_cmap("inferno")
        rgb[m] = cmap(0.25 + 0.75 * frac)[:, :3]
        ax2 = axes[row, 1]
        ax2.imshow(rgb, origin="lower")
        ax2.set_title(f"самое нагруженное сечение {name}={peak}\n"
                      f"жёлтое-красное = «ущелья» (макс. {sl.max()} путей через воксель)",
                      fontsize=10)
        ax2.set_xlabel("координата поперёк потока 1")
        ax2.set_ylabel("координата поперёк потока 2")
        del v, sl

    fig.suptitle("Карта узких мест воздушной фазы: 128 кратчайших путей на ось "
                 "(6-связность), счётчик проходов через воксели", y=0.995)
    out = PREVIEW / "connectivity_gorges.png"
    fig.savefig(out, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()

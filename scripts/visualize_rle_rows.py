"""Visual explainer #2: how (x, y, z) coordinates map onto the 510^3 cube.

Top: schematic - the cube is 510 slices (x) of 510 rows (y), each row is 510
voxels along z. Bottom: six REAL rows of the cube (from the air mask), each
labeled with its (x, y) and its actual RLE pairs.

Output: data/preview/rle_rows_xyz.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
PREVIEW = PROJECT / "data" / "preview"
AIR_C, TIS_C = "#9ecae1", "#8c6d31"
HL = "#d95f02"


def load_air() -> np.ndarray:
    p = PREVIEW / "air_phase_voxels_510.npz"
    if p.exists():
        return np.load(p)["air"]
    m = np.load(PROJECT / "data" / "cube_4.9um_512_mask.npz")["mask"]
    return ~m[1:-1, 1:-1, 1:-1]


def row_runs(row: np.ndarray):
    ch = np.flatnonzero(row[1:] != row[:-1]) + 1
    starts = np.concatenate(([0], ch))
    ends = np.concatenate((ch, [len(row)]))
    return row[starts].astype(int), (ends - starts)


def draw_schematic(ax) -> None:
    ax.set_xlim(-15, 175)
    ax.set_ylim(-52, 150)
    ax.set_aspect("equal")
    ax.axis("off")
    s, off = 105, 16
    for i in (2, 1, 0):  # stack along x, far slices first
        x0, y0 = i * off, i * off * 0.55
        ax.add_patch(Rectangle((x0, y0), s, s, fill=False, ec="#999999", lw=1.2,
                               alpha=1.0 if i == 0 else 0.55))
    f = 0  # front square
    for t in np.linspace(0, s, 9):
        ax.plot([f + t, f + t], [f, f + s], color="#cccccc", lw=0.6, zorder=0)
        ax.plot([f, f + s], [f + t, f + t], color="#cccccc", lw=0.6, zorder=0)
    y_row = f + s * 0.55
    ax.plot([f, f + s], [y_row, y_row], color=HL, lw=3, zorder=5)
    ax.annotate("", xy=(f + s, f - 14), xytext=(f, f - 14),
                arrowprops=dict(arrowstyle="->", color="k"))
    ax.text(f + s / 2, f - 24, "z: 510 вокселей вдоль ряда", ha="center", fontsize=9)
    ax.annotate("", xy=(f - 12, f + s), xytext=(f - 12, f),
                arrowprops=dict(arrowstyle="->", color="k"))
    ax.text(f - 16, f + s / 2, "y: 510 строк", rotation=90, va="center",
            ha="right", fontsize=9)
    ax.annotate("x: 510 срезов", xy=(s + 30, s + 22), xytext=(s - 6, s + 34),
                arrowprops=dict(arrowstyle="->", color="k"), fontsize=9)
    ax.annotate("ряд (x, y)", xy=(s * 0.62, y_row), xytext=(s * 0.45, y_row - 26),
                arrowprops=dict(arrowstyle="->", color=HL), color=HL, fontsize=10)
    ax.text(52, -46, "куб = 510 срезов (x) × 510 строк (y) × 510 вокселей (z)",
            fontsize=10, ha="center")


def main() -> None:
    air = load_air()
    rows = [(0, 0), (0, 1), (0, 2), (127, 300), (255, 255), (509, 509)]

    fig = plt.figure(figsize=(13, 9.5), dpi=120)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 2.0], hspace=0.30)
    top = gs[0].subgridspec(1, 2, width_ratios=[1, 1.15], wspace=0.02)
    draw_schematic(fig.add_subplot(top[0]))
    txt = fig.add_subplot(top[1])
    txt.axis("off")
    txt.text(0.0, 0.72,
             "x — номер среза (0…509): какой «ломтик» куба по глубине;\n"
             "y — номер строки внутри среза (0…509);\n"
             "z — номер вокселя вдоль строки (0…509).\n\n"
             "Воздух = 1 (голубой), стенка = 0 (коричневый).\n"
             "Ниже — шесть НАСТОЯЩИХ рядов куба и их RLE-пары:\n"
             "пары читаются слева направо, значение чередуется 1→0→1→0…",
             fontsize=11, va="top")

    bot = gs[1].subgridspec(len(rows), 1, hspace=1.15)
    for i, (x, y) in enumerate(rows):
        ax = fig.add_subplot(bot[i])
        row = air[x, y].astype(int)
        v, l = row_runs(row)
        ax.imshow(row[None, :], aspect="auto", cmap=ListedColormap([TIS_C, AIR_C]),
                  extent=(-0.5, 509.5, 0, 1))
        for b in np.cumsum(l)[:-1] - 0.5:
            ax.axvline(b, color="black", lw=0.7, alpha=0.75)
        first = "".join(f"({a},{b})" for a, b in zip(v[:6], l[:6]))
        ax.text(516, 0.5, f"{len(l)} пар: {first}…", fontsize=8.5, va="center")
        ax.set_xlim(-0.5, 780)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(f"x={x}\ny={y}", rotation=0, ha="right", va="center", fontsize=10)
        ax.set_xticks([] if i < len(rows) - 1 else [0, 100, 200, 300, 400, 500])
        if i == len(rows) - 1:
            ax.set_xlabel("z — номер вокселя в ряду", fontsize=10)
    fig.suptitle("Координаты (x, y, z) в кубе 510³ и RLE-упаковка реальных рядов",
                 fontsize=13, y=0.985)
    out = PREVIEW / "rle_rows_xyz.png"
    fig.savefig(out, bbox_inches="tight")
    print("saved", out)
    for x, y in rows:
        v, l = row_runs(air[x, y].astype(int))
        print(f"row (x={x}, y={y}): {len(l)} пар, сумма длин = {l.sum()}")


if __name__ == "__main__":
    main()

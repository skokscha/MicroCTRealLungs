"""Export the 2.5 mm tissue cube as a finite-element mesh (Abaqus/CalculiX .inp).

Each tissue voxel -> one C3D8 hex element; corner nodes are shared between
neighbouring voxels (deduplicated). Geometry units: mm (voxel = 0.0049).
Grid 510^3, same as lung_cube_tissue_2p5mm.stl (voxel mask IS the mesh).

Output: data/lung_cube_tissue_2p5mm_hex.inp
  ~23M C3D8 elements, ~28M nodes (exact counts printed).
  NSETs: XMIN/XMAX/YMIN/YMAX/ZMIN/ZMAX - boundary nodes for BCs.
  Material block is a PLACEHOLDER (E=1.0 MPa, nu=0.45) - replace with real
  lung tissue values; units mm/N/MPa.

Run: python scripts/export_hex_inp.py   (~5-10 min, few GB RAM)
"""
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "data" / "lung_cube_tissue_2p5mm_hex.inp"
N = 510
VOXEL_MM = 0.0049

# 8 corners of a voxel (Abaqus C3D8 order: bottom CCW, top CCW)
OFF = np.array(
    [
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ],
    dtype=np.int32,
)


def main() -> None:
    air = np.load(PROJECT / "data" / "preview" / "air_phase_voxels_510.npz")["air"]
    tissue = ~air
    del air
    vox = np.argwhere(tissue).astype(np.int32)  # (Nv, 3)
    del tissue
    nv = len(vox)
    print(f"tissue voxels: {nv:,}", flush=True)

    def chunk_keys(s, e):
        c = vox[s:e][:, None, :] + OFF[None, :, :]   # (m, 8, 3)
        return (c[..., 0] * (N + 1) ** 2 + c[..., 1] * (N + 1) + c[..., 2]).astype(np.int64)

    # pass 1: collect unique corner keys (sorted) = node table
    step = 2_000_000
    parts = [chunk_keys(s, min(s + step, nv)).ravel() for s in range(0, nv, step)]
    uniq = np.unique(np.concatenate(parts))
    del parts
    n_nodes = len(uniq)
    print(f"unique nodes: {n_nodes:,}", flush=True)

    ni = uniq // (N + 1) ** 2
    nj = (uniq // (N + 1)) % (N + 1)
    nk = uniq % (N + 1)

    with open(OUT, "w", buffering=1 << 24) as f:
        f.write("*HEADING\n")
        f.write("Human lung tissue 2.5mm cube, voxel mesh 510^3, voxel=0.0049 mm\n")
        f.write("HiP-CT LADAF-2020-27, DOI 10.15151/ESRF-DC-572194514\n")
        f.write("*NODE\n")
        for s in range(0, n_nodes, step):
            e = min(s + step, n_nodes)
            f.write(
                "\n".join(
                    f"{i+1}, {a*VOXEL_MM:.6f}, {b*VOXEL_MM:.6f}, {c*VOXEL_MM:.6f}"
                    for i, a, b, c in zip(range(s, e), ni[s:e], nj[s:e], nk[s:e])
                )
            )
            f.write("\n")

        f.write("*ELEMENT, TYPE=C3D8, ELSET=TISSUE\n")
        eid = 0
        for s in range(0, nv, step):
            e = min(s + step, nv)
            k = chunk_keys(s, e)                       # (m, 8)
            ids = np.searchsorted(uniq, k) + 1         # 1-based node ids
            f.write(
                "\n".join(
                    f"{i+1}, " + ", ".join(map(str, row))
                    for i, row in zip(range(eid, eid + (e - s)), ids)
                )
            )
            f.write("\n")
            eid += e - s
            print(f"  elements {eid:,}/{nv:,}", flush=True)

        # boundary node sets from the node table (plane indices 0 / N)
        for name, cond in [
            ("XMIN", ni == 0), ("XMAX", ni == N),
            ("YMIN", nj == 0), ("YMAX", nj == N),
            ("ZMIN", nk == 0), ("ZMAX", nk == N),
        ]:
            ids = np.flatnonzero(cond) + 1
            if not len(ids):
                continue
            f.write(f"*NSET, NSET={name}\n")
            for s in range(0, len(ids), 16):
                f.write(", ".join(map(str, ids[s:s + 16])) + "\n")

        f.write(
            "*MATERIAL, NAME=LUNG_TISSUE\n"
            "*ELASTIC\n"
            "1.0, 0.45\n"  # PLACEHOLDER: replace with real tissue E, nu (mm/N/MPa)
            "*SOLID SECTION, ELSET=TISSUE, MATERIAL=LUNG_TISSUE\n"
        )

    print(f"saved {OUT.name}: {OUT.stat().st_size / 1e9:.2f} GB, "
          f"{nv:,} C3D8, {n_nodes:,} nodes", flush=True)


if __name__ == "__main__":
    main()

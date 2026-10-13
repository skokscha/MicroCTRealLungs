"""Air-interface mesh: the 2.5 mm air phase WITHOUT the 6 cube cut planes.

lung_air_2p5um_mc2.stl = alveolar wall interface (sponge) + 6 flat box faces
(the cut planes where air exits the cube - required for CFD, ugly to look at).
This script removes the box faces and saves the remaining pure interface,
so the air phase looks like the RVE500 sponge. The interface alone is OPEN
(its closure is exactly the removed box) - for CFD use the full mc2 mesh.

Output: data/preview/lung_air_2p5um_interface.stl
"""
from pathlib import Path

import numpy as np
import trimesh

PROJECT = Path(__file__).resolve().parent.parent
SRC = PROJECT / "data" / "preview" / "lung_air_2p5um_mc2.stl"
DST = PROJECT / "data" / "preview" / "lung_air_2p5um_interface.stl"
VOXEL_UM = 4.9


def main() -> None:
    m = trimesh.load(SRC)
    v, f = m.vertices, m.faces
    print(f"loaded: {len(f):,} faces", flush=True)

    # the 6 cut planes: bbox extremes per axis (mc2 is unsmoothed -> exactly planar)
    shell = np.zeros(len(f), bool)
    for ax in range(3):
        for plane in (v[:, ax].min(), v[:, ax].max()):
            shell |= (np.abs(v[f][:, :, ax] - plane) < 0.1).all(axis=1)
    print(f"box (cut-plane) faces: {shell.sum():,} ({100 * shell.mean():.1f}%)", flush=True)

    m.update_faces(~shell)
    m.remove_unreferenced_vertices()
    print(f"interface: {len(m.faces):,} faces, volume {m.volume / 1e9:.3f} mm3", flush=True)

    from trimesh.smoothing import filter_taubin

    filter_taubin(m, lamb=0.5, nu=0.53, iterations=25)
    m.export(str(DST))
    print(
        f"saved {DST.name}: {len(m.faces):,} faces, {DST.stat().st_size / 1e6:.0f} MB",
        flush=True,
    )


if __name__ == "__main__":
    main()

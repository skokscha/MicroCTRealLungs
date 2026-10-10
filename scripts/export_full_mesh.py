"""Export the full-resolution marching-cubes surface from the saved mask to a
temporary binary STL for Blender to decimate properly.
"""
from pathlib import Path

import numpy as np
from skimage.measure import marching_cubes
from trimesh.exchange.export import export_mesh
import trimesh

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "data" / "_full_surface_4.9um.stl"

mask = np.load(PROJECT / "data" / "cube_4.9um_512_mask.npz")["mask"]
verts, faces, _, _ = marching_cubes(
    mask.astype(np.float32), level=0.5, spacing=(4.9,) * 3, step_size=1
)
print(f"{len(faces) / 1e6:.1f}M faces", flush=True)
export_mesh(trimesh.Trimesh(verts, faces), str(OUT))
print("saved", OUT, f"{OUT.stat().st_size / 1e9:.2f} GB", flush=True)

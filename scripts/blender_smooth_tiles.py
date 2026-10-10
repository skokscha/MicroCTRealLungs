"""Blender headless: Taubin-style smoothing of all tile STLs.

Usage: blender --background --python blender_smooth_tiles.py -- <tiles_dir>

Reads every *_raw.stl in <tiles_dir> and writes the matching *_smooth.stl.
Smoothing = two SMOOTH modifiers applied in sequence: factor +0.5 (12 iters),
then -0.53 (12 iters) - the alternating lambda/mu pair shrinks then re-inflates,
i.e. a low-pass filter without volume shrinkage (Taubin). Face count unchanged,
no decimation.
"""
import sys
from pathlib import Path

import bpy

argv = sys.argv[sys.argv.index("--") + 1 :]
tiles_dir = Path(argv[0])
raws = sorted(tiles_dir.glob("*_raw.stl"))
print(f"smoothing {len(raws)} meshes from {tiles_dir}", flush=True)

for raw in raws:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.stl_import(filepath=str(raw))
    obj = bpy.context.selected_objects[0]
    n_faces = len(obj.data.polygons)

    mod_l = obj.modifiers.new(name="taubin_lambda", type="SMOOTH")
    mod_l.iterations = 12
    mod_l.factor = 0.5
    mod_m = obj.modifiers.new(name="taubin_mu", type="SMOOTH")
    mod_m.iterations = 12
    mod_m.factor = -0.53

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.modifier_apply(modifier="taubin_lambda")
    bpy.ops.object.modifier_apply(modifier="taubin_mu")

    out = raw.with_name(raw.name.replace("_raw.stl", "_smooth.stl"))
    bpy.ops.wm.stl_export(filepath=str(out), export_selected_objects=True)
    print(f"{raw.name}: {n_faces} faces -> {out.name}", flush=True)

print("done", flush=True)

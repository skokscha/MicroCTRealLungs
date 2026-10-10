"""Run inside Blender headless:
quadric decimate -> laplacian smooth -> shade smooth -> export STL + Cycles render.

Usage: blender --background --python blender_beauty.py -- <in.stl> <out.stl> <out.png>
"""
import math
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1 :]
in_stl, out_stl, out_png = argv[0], argv[1], argv[2]

bpy.ops.wm.read_factory_settings(use_empty=True)

bpy.ops.wm.stl_import(filepath=in_stl)
obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj
print("imported faces:", len(obj.data.polygons), flush=True)

# quadric decimation: keeps thin walls intact (unlike vertex clustering)
dec = obj.modifiers.new("dec", "DECIMATE")
dec.decimate_type = "COLLAPSE"
dec.ratio = 0.05
bpy.ops.object.modifier_apply(modifier="dec")
print("after decimate:", len(obj.data.polygons), flush=True)

# gentle laplacian smoothing to round voxel jaggies without shrinking walls
sm = obj.modifiers.new("sm", "LAPLACIANSMOOTH")
sm.iterations = 6
sm.lambda_factor = 0.35
sm.lambda_border = 0.35
bpy.ops.object.modifier_apply(modifier="sm")
print("after smooth:", len(obj.data.polygons), flush=True)

bpy.ops.object.shade_smooth()
bpy.ops.wm.stl_export(filepath=out_stl, export_selected_objects=True, apply_modifiers=False)
print("exported", out_stl, flush=True)

# ---- pretty Cycles render ----
mat = bpy.data.materials.new("tissue")
mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (0.72, 0.74, 0.70, 1.0)
bsdf.inputs["Roughness"].default_value = 0.55
obj.data.materials.append(mat)

bbox = [Vector(v) for v in obj.bound_box]
center = obj.matrix_world @ sum(bbox, Vector()) / 8
size = max((max(b[i] for b in bbox) - min(b[i] for b in bbox)) for i in range(3))

cam_data = bpy.data.cameras.new("cam")
cam_data.lens = 70
cam = bpy.data.objects.new("cam", cam_data)
bpy.context.collection.objects.link(cam)
bpy.context.scene.camera = cam

direction = Vector((1.0, -1.0, 0.55)).normalized()
cam.location = center + direction * size * 2.1
cam.rotation_euler = (-direction).to_track_quat("-Z", "Y").to_euler()

sun_data = bpy.data.lights.new("sun", "SUN")
sun_data.energy = 3.5
sun = bpy.data.objects.new("sun", sun_data)
sun.rotation_euler = (math.radians(35), 0, math.radians(25))
bpy.context.collection.objects.link(sun)

fill_data = bpy.data.lights.new("fill", "AREA")
fill_data.energy = size * size * 12
fill_data.shape = "SQUARE"
fill_data.size = size * 2.5
fill = bpy.data.objects.new("fill", fill_data)
fill.location = center + Vector((-1, 0.6, 1.2)) * size * 2
fill.rotation_euler = (math.radians(50), 0, math.radians(-140))
bpy.context.collection.objects.link(fill)

world = bpy.data.worlds.new("world")
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs[0].default_value = (0.06, 0.065, 0.07, 1.0)
bg.inputs[1].default_value = 1.0
bpy.context.scene.world = world

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 48
scene.cycles.use_denoising = True
scene.render.resolution_x = 1400
scene.render.resolution_y = 1400
scene.render.film_transparent = False
scene.render.filepath = out_png
bpy.ops.render.render(write_still=True)
print("rendered", out_png, flush=True)

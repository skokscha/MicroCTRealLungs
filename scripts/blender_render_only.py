"""Blender headless: import STL, shade smooth, pretty Cycles render. No decimation.

Usage: blender --background --python blender_render_only.py -- <in.stl> <out.png>
"""
import math
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1 :]
in_stl, out_png = argv[0], argv[1]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.stl_import(filepath=in_stl)
obj = bpy.context.selected_objects[0]
print("imported faces:", len(obj.data.polygons), flush=True)

bpy.ops.object.shade_smooth()

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
cam_data.lens = 65
cam_data.clip_start = 1.0
cam_data.clip_end = 20000.0
cam = bpy.data.objects.new("cam", cam_data)
bpy.context.collection.objects.link(cam)
bpy.context.scene.camera = cam
direction = Vector((1.0, -1.0, 0.55)).normalized()
cam.location = center + direction * size * 4.0
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
world.node_tree.nodes["Background"].inputs[0].default_value = (0.06, 0.065, 0.07, 1.0)
bpy.context.scene.world = world

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 32
scene.cycles.use_denoising = True
scene.render.resolution_x = 1200
scene.render.resolution_y = 1200
scene.render.filepath = out_png
bpy.ops.render.render(write_still=True)
print("rendered", out_png, flush=True)

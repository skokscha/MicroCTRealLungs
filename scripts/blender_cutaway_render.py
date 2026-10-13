"""Blender headless: cutaway render of the air mesh (removes camera-facing half).

Usage: blender --background --python blender_cutaway_render.py -- <in.stl> <out.png>

Proves the air STL contains the alveolar cavity network inside: boolean-difference
a slab covering the half of the cube nearest the camera, then render the exposed
interior with the same look as blender_render_only.py.
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

# camera sits at center + (1,-1,0.55)*d -> nearest corner is +x,-y,+z:
# cut away that half (y < mid) to expose the interior cavity network
bb_min, bb_max = obj.bound_box[0], obj.bound_box[6]
mid_y = (bb_min[1] + bb_max[1]) / 2
pad = (bb_max[0] - bb_min[0]) * 0.1

cut = bpy.ops.mesh.primitive_cube_add(
    size=1,
    location=((bb_min[0] + bb_max[0]) / 2, (bb_min[1] + mid_y) / 2, (bb_min[2] + bb_max[2]) / 2),
)
cutter = bpy.context.active_object
cutter.scale = (
    (bb_max[0] - bb_min[0]) * 1.2,
    (mid_y - bb_min[1]) / 2 + pad,
    (bb_max[2] - bb_min[2]) * 1.2,
)
mod = obj.modifiers.new(name="cut", type="BOOLEAN")
mod.operation = "DIFFERENCE"
mod.object = cutter
mod.solver = "FLOAT"
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
bpy.ops.object.modifier_apply(modifier="cut")
print("after boolean:", len(obj.data.polygons), "faces", flush=True)

mat = bpy.data.materials.new("air")
mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (0.72, 0.74, 0.70, 1.0)
bsdf.inputs["Roughness"].default_value = 0.55
obj.data.materials.append(mat)

center = Vector(obj.bound_box[0]) + (Vector(obj.bound_box[6]) - Vector(obj.bound_box[0])) / 2
size = max(bb_max[i] - bb_min[i] for i in range(3))

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

# A simple script that uses blender to render views of a single object by rotation the camera around it.
# Also produces depth map at the same time.

import argparse, sys, os
import json
import bpy
import numpy as np
from math import inf
         
DEBUG = False
            
VIEWS = 25
BOUNCES = 0
RESOLUTION = 256
RESULTS_PATH = 'train'
DEPTH_SCALE = 1.4
COLOR_DEPTH = 8
FORMAT = 'PNG'
RANDOM_VIEWS = False
UPPER_VIEWS = True
PHI_BOUNDS = (0 , np.pi)
THETA_BOUNDS = (0, 2* np.pi)



obj_file_path = os.environ['obj_file_path']

# 导入.obj文件
bpy.ops.import_scene.obj(filepath=obj_file_path)

# 获取导入的对象
imported_object = bpy.context.selected_objects[0]

# 计算模型在 x、y、z 方向上的尺寸
x_size = imported_object.dimensions.x
y_size = imported_object.dimensions.y
z_size = imported_object.dimensions.z

# 计算缩放比例
scale_factor = 1 / max(x_size, y_size, z_size)

# 缩放模型
imported_object.scale = (scale_factor, scale_factor, scale_factor)


def calculate_min_z(object):
    min_z = inf
    for vertex in object.data.vertices:
        z_world = object.matrix_world @ vertex.co
        if z_world.z < min_z:
            min_z = z_world.z
    return min_z

# 获取对象的底部边界的最小 Z 坐标
min_z = calculate_min_z(imported_object)

# 移动对象
imported_object.location[2] -= (min_z * scale_factor + 0.2)


fp = os.environ['img_save_path']
# from pdb import set_trace; set_trace()

def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list

if not os.path.exists(fp):
    os.makedirs(fp)

# Data to store in JSON file
out_data = {
    # the x FOV for the camera
    'camera_angle_x': bpy.data.objects['Camera'].data.angle_x,
}
out_obj_data = {}

# Render Optimizations
# accelerate the rendering by reusing data
bpy.context.scene.render.use_persistent_data = True

# Set up rendering of depth map.
# enables the use of the node-based compositing system for the current scene.
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links
bpy.context.scene.view_layers["RenderLayer"].use_pass_normal = True
bpy.context.scene.render.image_settings.file_format = str(FORMAT)
bpy.context.scene.render.image_settings.color_depth = str(COLOR_DEPTH)
# Background
bpy.context.scene.render.dither_intensity = 0.0
bpy.context.scene.render.film_transparent = True

# Create collection for objects not to render with background

    
#objs = [ob for ob in bpy.context.scene.objects if ob.type in ('EMPTY') and 'Empty' in ob.name]
#bpy.ops.object.delete({"selected_objects": objs})
def parent_obj_to_camera(b_camera):
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting

    scn = bpy.context.scene
    scn.collection.objects.link(b_empty)
    bpy.context.view_layer.objects.active = b_empty
    # scn.objects.active = b_empty
    return b_empty


scene = bpy.context.scene
scene.render.resolution_x = RESOLUTION
scene.render.resolution_y = RESOLUTION
scene.render.resolution_percentage = 100

cam = scene.objects['Camera']
cam.location = (0, 4, 0.)
cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'
b_empty = scene.objects['Empty'] #parent_obj_to_camera(cam)
cam_constraint.target = b_empty

scene.render.image_settings.file_format = 'PNG'  # set output format to .png

if os.environ['cuda_render'] == 'true':
    # Add passes for additionally dumping albedo and normals.
    bpy.context.scene.view_layers["RenderLayer"].use_pass_normal = True
    bpy.context.scene.render.image_settings.file_format = str(FORMAT)
    bpy.context.scene.render.image_settings.color_depth = str(COLOR_DEPTH)

    # Set the rendering engine to Cycles which supports CUDA
    bpy.context.scene.render.engine = 'CYCLES'

    # Enable CUDA
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'

    # List and enable CUDA devices (GPUs)
    cuda_devices = bpy.context.preferences.addons['cycles'].preferences.get_devices_for_type('CUDA')
    for device in cuda_devices:
        print(f"Found CUDA Device: {device.name}")
        # To use a specific GPU, check by name, like device.name == 'GeForce RTX 2070'
        device.use = True  # Set this to False to disable a GPU

    # Print out the devices that will be used
    enabled_gpus = [d.name for d in cuda_devices if d.use]
    print("Enabled GPUs for rendering:", ", ".join(enabled_gpus))

stepsize = 360.0 / VIEWS
phi = np.linspace(PHI_BOUNDS[0], PHI_BOUNDS[1], VIEWS)
theta = np.linspace(THETA_BOUNDS[0], THETA_BOUNDS[1], VIEWS)

phi, theta = np.meshgrid(phi, theta)

out_data['frames'] = []
out_obj_data['frames'] = []

for i in range(phi.shape[0]):
    for j in range(phi.shape[1]):

        b_empty.rotation_euler[0] = phi[i,j]
        b_empty.rotation_euler[2] = theta[i,j]

        for b in range(0, BOUNCES+1):
            bpy.context.scene.cycles.max_bounces = b
        
            img_path = os.path.join(fp, f'img_{i}_{j}_{b}')
            
            scene.render.filepath = img_path + '_b' + str(b)

            if DEBUG:
                break
            else:
                bpy.ops.render.render(write_still=True)  # render still
        
        frame_data = {
            'file_path': './' + RESULTS_PATH + '/r_' + str(i),
            'phi': float(phi[i][j]),
            'theta': float(theta[i][j]),
            'transform_matrix': listify_matrix(cam.matrix_world)
        }
        out_data['frames'].append(frame_data)

if not DEBUG:
    with open(fp + '/' + 'transforms.json', 'w') as out_file:
        json.dump(out_data, out_file, indent=4)
    # with open(fp + '/' + 'transforms_objects.json', 'w') as out_obj_file:
    #     json.dump(out_obj_data, out_obj_file, indent=4)
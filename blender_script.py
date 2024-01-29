# A simple script that uses blender to render views of a single object by rotation the camera around it.
# Also produces depth map at the same time.

import argparse, sys, os
import json
import bpy
import numpy as np
         
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

# 你的.obj文件路径
argparser = argparse.ArgumentParser()
argparser.add_argument('--obj_file_path', type=str, help='Path to the obj file to be rendered.')
argparser.add_argument('--save_path', type=str, help='Path to the obj file to be rendered.')
args = argparser.parse_args()

obj_file_path = args.obj_file_path

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

x_size = scale_factor * x_size
y_size = scale_factor * y_size
z_size = scale_factor * z_size

# 缩放模型
imported_object.scale = (x_size, y_size, z_size)

# 将对象移动到坐标原点
imported_object.location = (0, 0, z_size / 2)

# fp = bpy.path.abspath(f"//{RESULTS_PATH}")
fp = args.save_path


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

# Add passes for additionally dumping albedo and normals.
bpy.context.scene.view_layers["RenderLayer"].use_pass_normal = True
bpy.context.scene.render.image_settings.file_format = str(FORMAT)
bpy.context.scene.render.image_settings.color_depth = str(COLOR_DEPTH)

if 'Normal Output' not in tree.nodes:
    # Create input render layer node.
    render_layers = tree.nodes['Render Layers']

    depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    depth_file_output.name = 'Depth Output'
    if FORMAT == 'OPEN_EXR':
        links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
    else:
        # Remap as other types can not represent the full range of depth.
        map = tree.nodes.new(type="CompositorNodeMapRange")
        # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
        map.inputs['From Min'].default_value = 0
        map.inputs['From Max'].default_value = 8
        map.inputs['To Min'].default_value = 1
        map.inputs['To Max'].default_value = 0
        links.new(render_layers.outputs['Depth'], map.inputs[0])

        links.new(map.outputs[0], depth_file_output.inputs[0])

    normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    normal_file_output.label = 'Normal Output'
    normal_file_output.name = 'Normal Output'
    links.new(render_layers.outputs['Normal'], normal_file_output.inputs[0])

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
cam.location = (0, 1.67, 0.)
cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'
b_empty = scene.objects['Empty'] #parent_obj_to_camera(cam)
cam_constraint.target = b_empty

scene.render.image_settings.file_format = 'PNG'  # set output format to .png

from math import radians

stepsize = 360.0 / VIEWS
phi = np.linspace(PHI_BOUNDS[0], PHI_BOUNDS[1], VIEWS)
theta = np.linspace(THETA_BOUNDS[0], THETA_BOUNDS[1], VIEWS)

phi, theta = np.meshgrid(phi, theta)

if not DEBUG:
    for output_node in [tree.nodes['Depth Output'], tree.nodes['Normal Output']]:
        output_node.base_path = ''

out_data['frames'] = []
out_obj_data['frames'] = []

for i in range(phi.shape[0]):
    for j in range(phi.shape[1]):

        b_empty.rotation_euler[0] = phi[i,j]
        b_empty.rotation_euler[2] = theta[i,j]

        for b in range(0, BOUNCES+1):
            bpy.context.scene.cycles.max_bounces = b
        
            img_path = os.path.join(fp, f'img_{i}_{j}_{b}')

            tree.nodes['Depth Output'].file_slots[0].path = img_path + "_depth_"
            tree.nodes['Normal Output'].file_slots[0].path = img_path + "_normal_"
            
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
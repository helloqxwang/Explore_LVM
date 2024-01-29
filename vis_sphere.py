import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib import cm
import plotly.io as py
from utils import load_blender_pictures, get_clip_processor, get_imgfeat, get_txtfeat
import plotly.graph_objects as go
import os
import trimesh
import argparse
from typing import List

VIEW_NUM = 25
IMG_PATH = "/home/qianxu/Project/Explore_LVM/train"

def plot_mesh(verts, faces):
    return go.Mesh3d(
        x=verts[:,0], 
        y=verts[:,1], 
        z=verts[:,2], 
        i=faces[:,0], 
        j=faces[:,1], 
        k=faces[:,2], 
        color='lightblue')


def axis_angle_to_matrix(phi, theta):
    """
    Convert an axis-angle rotation into the rotation matrix form.
    
    Args:
    phi: A tensor of shape (..., 1) representing the rotation angle in the x-y plane.
    theta: A tensor of shape (..., 1) representing the rotation angle from the z-axis.
    axis: A tensor of shape (..., 3) representing the rotation axis.
    
    Returns:
    A tensor of shape (..., 3, 3) representing the rotation matrices.
    """
    # Convert angles to radians
    phi = torch.deg2rad(phi)
    theta = torch.deg2rad(theta)

    # Compute rotation matrix
    R = torch.zeros((*phi.shape[:-1], 3, 3), device=phi.device, dtype=phi.dtype)
    R[..., 0, 0] = torch.cos(phi) * torch.cos(theta)
    R[..., 0, 1] = torch.sin(phi) * torch.cos(theta)
    R[..., 0, 2] = -torch.sin(theta)
    R[..., 1, 0] = -torch.sin(phi)
    R[..., 1, 1] = torch.cos(phi)
    R[..., 1, 2] = torch.zeros_like(phi)
    R[..., 2, 0] = torch.cos(phi) * torch.sin(theta)
    R[..., 2, 1] = torch.sin(phi) * torch.sin(theta)
    R[..., 2, 2] = torch.cos(theta)

    return R

def set_axes_equal(ax, radius=0.3):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    '''
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    '''
    ax.set_xlim3d([-radius, radius])
    ax.set_ylim3d([-radius, radius])
    ax.set_zlim3d([-radius, radius])

def normalize_rgb(rgb:np.ndarray):
    """
    Normalize the rgb values to [0, 1]

    """
    rgb = (rgb - rgb.min(axis=-1, keepdims=True)) / (rgb.max(axis=-1, keepdims=True) - rgb.min(axis=-1, keepdims=True))
    return rgb

def render_sphere(cos_sim:np.ndarray, mesh_path:str, save_name:str='./results.html', img_save_name:str=None):
    """
    """

    phi = np.linspace(0, np.pi, VIEW_NUM)
    theta = np.linspace(0, 2*np.pi, VIEW_NUM)
    phi, theta = np.meshgrid(phi, theta)

    x = np.cos(phi) * np.cos(theta)
    x += 2
    y = np.cos(phi) * np.sin(theta)
    z = np.sin(phi)

    p_colors = 1 - (cos_sim - cos_sim.min()) / (cos_sim.max() - cos_sim.min())
    p_colors = p_colors.reshape(*x.shape)
    # p_colors = np.ones_like(x) purple
    sphere = go.Surface(x=x, y=y, z=z, surfacecolor=p_colors, showscale=False)

    mesh = trimesh.load_mesh(mesh_path)
    if type(mesh) == trimesh.scene.scene.Scene:
        mesh = mesh.dump(concatenate=True)
    rotation_matrix = trimesh.transformations.rotation_matrix(
        np.radians(90), [1, 0, 0], mesh.centroid
    )
    mesh.apply_transform(rotation_matrix)
    

    # # Extract the vertices and faces
    # from pdb import set_trace; set_trace()
    vertices = mesh.vertices

    faces = mesh.faces

    mesh = plot_mesh(vertices, faces)

    # Create figure with both the sphere and the scatter plot
    fig = go.Figure(data=[sphere, mesh])




    # Update layout for a better view
    fig.update_layout(
        scene=dict(
            # xaxis=dict(visible=False),
            # yaxis=dict(visible=False),
            # zaxis=dict(visible=False),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    # Show the figure
    if img_save_name is not None:
        py.write_image(fig, img_save_name)
    if save_name is not None:
        py.write_html(fig, save_name)

def vis(clip_model, processor, img_path:str, mesh_path:str, sentences:List[str], save_path:str='./results.html',shot_img_save_path:str=None):
    device = torch.device('cuda')
    images = load_blender_pictures(img_path)
    img_shape = images.shape
    images = torch.from_numpy(images).float() / 255.0
    # images = images.to(torch.dtype('float32'))
    images = images.to(device)
    
    
    img_features = get_imgfeat(clip_model, processor, images.reshape(-1, *img_shape[2:]))
    txt_features = get_txtfeat(clip_model, processor, sentences).detach().cpu()
    txt_z = txt_features / txt_features.norm(dim=-1, keepdim=True)
    img_z = img_features / img_features.norm(dim=-1, keepdim=True)
    cos_sim = (txt_z[None] * img_z[:, None]).sum(-1)
    for i in range(len(sentences)):
        # from pdb import set_trace; set_trace()
        mesh_name = os.path.split(mesh_path)[0].split('/')[-2]
        sentence = sentences[i].replace(' ', '_')
        shot_img_save_name = os.path.join(shot_img_save_path, f'{mesh_name}_{sentence}.png') if shot_img_save_path is not None else None
        render_sphere(cos_sim[:, i].cpu().detach().numpy(), mesh_path=mesh_path, 
                      save_name=os.path.join(save_path, f'{sentence}.html'), 
                      img_save_name=shot_img_save_name)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--mesh_path', type=str, help='the path of the mesh', default='/home/qianxu/Project/Explore_LVM/mesh/textured.obj')
    argparser.add_argument('--img_path', type=str, help='the path of the rendered images', default=IMG_PATH)
    argparser.add_argument('--save_path', type=str, help='the path to save the html file', default='/home/qianxu/Project/Explore_LVM/')
    argparser.add_argument('--sentences', nargs='*', type=str, help='the sentence to describe the mesh')
    args = argparser.parse_args()

    vis(args.img_path, args.mesh_path, args.sentences, args.save_path)

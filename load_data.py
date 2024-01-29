import os
import numpy as np
from PIL import Image
import json


trans_t = lambda t : np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1],
], dtype=np.float32)

rot_phi = lambda phi : np.array([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1],
], dtype=np.float32)

rot_theta = lambda th : np.array([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1],
], dtype=np.float32)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w
    


def load_blender_data(
    basedir,
    split="train",
    skip=1,
    img_size=(224, 224),
):
    with open(os.path.join(basedir, 'transforms_{}.json'.format(split)), 'r') as fp:
        meta = json.load(fp)

    imgs = []
    poses = []
        
    for frame in meta['frames'][::skip]:
        fname = os.path.join(basedir, frame['file_path'] + '.png')
        with Image.open(fname) as img:
            img = img.resize(img_size)
            img = np.asarray(img).astype(np.float32)
        imgs.append(img)
        poses.append(np.array(frame['transform_matrix']))
    imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
    # the pose should the be a 4*4 homogeneous matrix
    poses = np.array(poses).astype(np.float32)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    render_poses = np.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]],0)
        
    return imgs, poses, render_poses, [H, W, focal]

def find_files(dir:str, ext:str='obj'):
    """
    find all the files with the given extension in the given directory
    """
    # dir = os.path.abspath(dir)
    obj_files = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(ext):
                obj_files.append(os.path.join(root, file))
    return obj_files
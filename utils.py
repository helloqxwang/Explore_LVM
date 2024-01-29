import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.decomposition import PCA
from load_data import load_blender_data
import torch
import numpy as np
from scipy.spatial.transform import Rotation
import cv2
import os
import plotly.graph_objects as go
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
  
)
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.io import load_objs_as_meshes, load_obj
import matplotlib.pyplot as plt
import trimesh


def visualize_images(images):
    """
    Visualize a pile of images in numpy. 
    (n, h, w,3/4) or (n, h, w)
    """
    n = len(images) 
    rows = n // 5 
    if n % 5: rows += 1 
    fig = plt.figure(figsize=(20, rows*4)) 
    
    for i in range(n):
        ax = fig.add_subplot(rows, 5, i+1) 
        if images.shape[-1] == 4:
            ax.imshow(images[i, ..., :3]) 
        else:
            ax.imshow(images[i], cmap='gray')
        ax.axis('off') 

    plt.show()

def get_pytorch3d_RT(elev_ls:list, dist_ls:list, num:int = 10):
    """
    Get the camera pose for pytorch3d rendering 
    - R (n, 3, 3) pytorch3d2world 
    - T (n, 3) world2pytorch3d
    """
    R_ls = []
    T_ls = []
    for elev in elev_ls:
        for dist in dist_ls:
            azim = torch.linspace(-30, 30, num +1 )[:-1]
            R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
            R_ls.append(R)
            T_ls.append(T)
    R = torch.cat(R_ls, dim=0)
    T = torch.cat(T_ls, dim=0)
    return R, T

def convert_RTpytorch3d_c2w(R:torch.Tensor, T:torch.Tensor):
    """
    convert_A_B: convert A to B
    """
    c2c_pytorch3d = torch.eye(4).to(R.device)
    c2c_pytorch3d[0, 0] = -1
    c2c_pytorch3d[1, 1] = -1
    # world coordinates to pytorch3d camera coordinates
    w2c_pytorch3d = torch.eye(4).unsqueeze(0).repeat(R.shape[0], 1, 1).to(R.device)
    w2c_pytorch3d[:, :3, :3] = R.mT
    w2c_pytorch3d[:, :3, 3] = T
    # camera coordinates to world coordinates
    c2w = torch.inverse(w2c_pytorch3d) @ c2c_pytorch3d
    return c2w

def convert_c2w_RTpytorch3d(c2w:torch.Tensor):
    """
    convert_A_B: convert A to B
    """
    c2c_pytorch3d = torch.eye(4).to(R.device)
    c2c_pytorch3d[0, 0] = -1
    c2c_pytorch3d[1, 1] = -1
    w2c_pytorch3d = c2c_pytorch3d @ torch.inverse(c2w)
    R = w2c_pytorch3d[:, :3, :3].mT
    T = w2c_pytorch3d[:, :3, 3]
    return R, T

def mesh2rgbd(mesh:trimesh, R:torch.Tensor, T:torch.Tensor, 
              img_size:int=224, device=torch.device('cuda'),):
    """smampling the mesh to rgbd images using pytorch3d


    Args:
        - mesh
        - R (torch.Tensor): (N, 3, 3) pytorch3d2world
        - T (torch.Tensor): (N, 3) word2pytorch3d
        - device (torch.device, optional): Defaults to torch.device('cuda').
        - img_size (int) : the size of the image

    Returns:
        - images (np.ndarray) : (N, H, W, 3) 
        - depths (np.ndarray) : (N, H, W)
        - c2w (np.ndarray) : (N, 4, 4)
        - K (np.ndarray) : (3, 3)
        - camera_params (dict) : {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy, 'xres': img_size, 'yres': img_size}
    """
    num = R.shape[0]
    mesh_path = 'mesh/textured.obj'
    mesh = load_objs_as_meshes([mesh_path], device=device)
    # plt.figure(figsize=(7,7))
    # texturesuv_image_matplotlib(mesh.textures, subsample=None)
    # plt.savefig('./test.png')
    # plt.close()
    # exit()
    verts = mesh.verts_list()[0]
    # print(verts.shape)
    rot = Rotation.from_euler('zyx', [np.pi, 0, - np.pi/2], degrees=False)
    matrix = torch.tensor(rot.as_matrix(), dtype=verts.dtype).to(device)
    verts = (matrix @ verts.T).T

    mesh.verts_list()[0] = verts * 1.4
    tex = torch.ones_like(verts)[None, ...].to(device)  # (1, V, 3)
    mesh.textures = TexturesVertex(verts_features=tex)

    meshes = mesh.extend(num)

    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    raster_settings = RasterizationSettings(
        image_size=img_size, 
        bin_size=0,
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    materials = Materials(
    device=device,
    specular_color=[[0.0, 1.0, 0.0]],
    shininess=10.0
)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )
    ) 
    # We can pass arbitrary keyword arguments to the rasterizer/shader via the renderer
    # so the renderer does not need to be reinitialized if any of the settings change.
    images = renderer(meshes, cameras=cameras, lights=lights, materials=materials)
    fragments = renderer.rasterizer(meshes, cameras=cameras)
    depths = fragments.zbuf.mean(dim=-1)

    fx = fy = img_size / (2 * np.tan(np.pi / 3 / 2))
    cx = cy = img_size / 2
    ### construct the Inrinsics matrix
    K = np.asarray([
     [fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]])
    camera_params = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy, 'xres': img_size, 'yres': img_size}
    
    c2w = convert_RTpytorch3d_c2w(R, T).cpu().numpy()

    return images.cpu().numpy(), depths.cpu().numpy(), c2w, K, camera_params

def get_clip_processor(device=torch.device('cuda')):
    clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32", 
    # use_auth_token=token,
    cache_dir='/home/qianxu/Documents/pretrained'
    ).to(device)
    processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32",
        # use_auth_token=token,
        cache_dir='/home/qianxu/Documents/pretrained'
    )
    return clip_model, processor

def get_imgfeat(clip_model, processor, images:torch.Tensor, device=torch.device('cuda'), batch_size=200) -> torch.Tensor:
    # Initialize an empty list to hold the features
    features = []
    for i in range(0, len(images), batch_size):
        # print(1)
        # Get batch of images
        batch = images[i:min(i+batch_size, len(images)), ..., :3]

        # Prepare inputs
        inputs = processor(images=batch, return_tensors="pt", do_rescale=False)
        inputs["pixel_values"] = inputs["pixel_values"].to(device)

        # Get features and add to list
        batch_features = clip_model.get_image_features(**inputs)
        features.append(batch_features.detach().cpu())
        # torch.cuda.empty_cache()

    # Concatenate all features along the first dimension
    img_features = torch.cat(features, dim=0)
    torch.cuda.empty_cache()
    return img_features

def get_txtfeat(clip_model, processor, sentences:list, device=torch.device('cuda')) -> torch.Tensor:
    inputs = processor(text=sentences, return_tensors="pt", padding=True)
    inputs["input_ids"] = inputs["input_ids"].to(device)
    inputs["attention_mask"] = inputs["attention_mask"].to(device)
    txt_features = clip_model.get_text_features(**inputs)
    return txt_features

def load_blender_pictures(path:str):
    """
    load the blender pictures
    """
    files = os.listdir(path)

    files = sorted([f for f in files if f.endswith('.png')])

    images = []
    for filename in files:
        img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_UNCHANGED)

        images.append(img)

    images = np.array(images)

    n, m = int(np.sqrt(images.shape[0])), int(np.sqrt(images.shape[0]))
    images = images.reshape(n, m, 256, 256, -1)

    return images

if __name__ == "__main__":
    device = torch.device('cuda')
    mesh = trimesh.load('./mesh/textured.obj')
    verts = mesh.vertices
    rot = Rotation.from_euler('zyx', [np.pi, 0, - np.pi/2], degrees=False)
    matrix = np.array(rot.as_matrix(), dtype=verts.dtype)
    verts = (matrix @ verts.T).T

    mesh.vertices = verts * 1.4
    R, T = get_pytorch3d_RT([15, 30], [0.9])
    images, depths, c2w, K, camera_params = mesh2rgbd(mesh, R, T)

    visualize_images(images)

    clip_model, processor = get_clip_processor()

    img_features = get_imgfeat(clip_model, processor, images)
    feat = img_features.cpu().detach().numpy()

    pca = PCA(n_components=3)
    feat_pca = pca.fit_transform(feat)
    feat_pca = (feat_pca - feat_pca.min()) / (feat_pca.max() - feat_pca.min())
    vis_volumn(c2w[:, :3, 3], feat, feat[0], interpolation='cosine')
    vis_colorpts_mesh(c2w[:, :3, 3], feat_pca, mesh, 'results_.html')

    sentences = ["A hammer", "A rabbit with long ears"]
    txt_features = get_txtfeat(clip_model, processor, sentences)
    
    txt_z = txt_features / txt_features.norm(dim=-1, keepdim=True)
    img_z = img_features / img_features.norm(dim=-1, keepdim=True)
    cos_sim = (txt_z[None] * img_z[:, None]).sum(-1)
    cos_sim = cos_sim.detach().cpu()
    print(11111)
    for i in range(len(sentences)):
        cmap = plt.get_cmap("plasma")
        colors = cmap( 1 -(cos_sim[:, i] - cos_sim[:, i].min()) / (cos_sim[:, i].max() - cos_sim[:, i].min()))[:, :3]
        vis_colorpts_mesh(c2w[:, :3, 3], (colors * 255).astype(int), mesh, f'{sentences[i]}.html')
        vis_volumn(c2w[:, :3, 3], feat, txt_z[i].cpu().detach(), interpolation='cosine', name=f'{sentences[i]}_volumn.html')
import subprocess
from load_data import find_files
import os
import argparse
from vis_sphere import render_sphere, process
from utils import load_blender_pictures, get_clip_processor
import time
from typing import List
import numpy as np

BLENDER_PATH = "/home/qianxu/Documents/blender-3.6.8-linux-x64/blender"
SENTENCES = ["A Car", "Be careful!", "You can drive it", "Can be put into a drawer", 
             ]

def render_mesh(mesh_path:str, img_save_dir:str, use_cuda:bool=True):
    """render a mesh with blender

    Args:
        mesh_path (str): the path of the mesh
        img_save_dir (str): the DIR for saving the rendered images
        use_cuda (bool, optional): Use cuda as well. Defaults to True.
    """
    os.environ['img_save_path'] = img_save_dir
    os.environ['obj_file_path'] = mesh_path
    if use_cuda:
        os.environ['cuda_render'] = 'true'
    else:
        os.environ['cuda_render'] = 'false'
    blender_command = f"blender --background render.blend --python render.py "
    # from pdb import set_trace; set_trace()
    start = time.time()
    subprocess.run(blender_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    end = time.time()
    print(f"Blender Render Done. Time: {end-start}")

def render_models(model_paths:List[str], use_cuda:bool=True):
    for model_path in model_paths:
        img_save_dir = os.path.join(os.path.split(model_path)[0], 'img')
        os.makedirs(img_save_dir, exist_ok=True)
        render_mesh(model_path, img_save_dir, use_cuda)

def vis_models(model_paths:List[str], vis_save_base:str='./vis', 
               shot_vis_save_base:str='./vis_shot', skip_render:bool=False, skip_vis:bool=False):
    """Visualize the rendered models with the sentences

    Args:
        model_paths (List[str]): List of DIRs of shapeNet models
        vis_save_base (str, optional): DIR for all HTML Vis. Defaults to './vis'.
        shot_vis_save_path (str, optional): DIR for all shot Vis. Defaults to './vis_shot'.

    """
    """
    HTML save parttern:
        - vis_save_base
            - Detailed
                - sentence1
                    - model_name1.html
                    - model_name2.html
                    - ...
                - sentence2
                    - model_name1.html
                    - model_name2.html
                    - ...
                - ...
            - Others
    
    Shot img save parttern:
        - shot_vis_save_base
            - Detailed
                - sentence1
                    - model_name1.png
                    - model_name2.png
                    - ...
                - sentence2
                    - model_name1.png
                    - model_name2.png
                    - ...
                - ...
            - Others
    """
    clip_model, processor = get_clip_processor()
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    # Create the DIRS for saving the visualization
    for sentence in SENTENCES:
        os.makedirs(os.path.join(vis_save_base, 'Detailed', sentence), exist_ok=True)
        os.makedirs(os.path.join(shot_vis_save_base, 'Detailed', sentence), exist_ok=True)
        os.makedirs(os.path.join(vis_save_base, 'Uni_norm', sentence), exist_ok=True)
        os.makedirs(os.path.join(shot_vis_save_base, 'Uni_norm', sentence), exist_ok=True)
    feat_dict = {key: [] for key in SENTENCES}

    for model_path in model_paths:
        model_name = os.path.split(model_path)[0].split('/')[-2] # name of the mesh
        img_save_dir = os.path.join(os.path.split(model_path)[0], 'img') # DIR for rendered images

        # Render the model if not rendered and skip_render is False
        if not os.path.isdir(img_save_dir):
            if skip_render:
                print(f"Model {model_name} not rendered. Skip it.")
                continue
            else:
                print(f"Model {model_name} not rendered. Render it first.")
                os.makedirs(img_save_dir, exist_ok=True)
                render_mesh(model_path, img_save_dir)

        # Skip the Visualation if a visualization already exists and skip_vis is True
        shot_img_save_path_example = os.path.join(shot_vis_save_base, 'Detailed', SENTENCES[0], f'{model_name}.png')
        if os.path.exists(shot_img_save_path_example) and skip_vis:
            print(f"Model {model_name} already visualized. Skip it.")
            continue
        
        # VIS
        start = time.time()
        cos_sim_name = ''.join(SENTENCES).replace(' ', '')
        if os.path.isfile(os.path.join(img_save_dir, f'{cos_sim_name}.npy')): # load the cos_sim if exists
            cos_sim = np.load(os.path.join(img_save_dir, f'{cos_sim_name}.npy'))
        else:
            cos_sim = process(clip_model, processor, img_save_dir, sentences=SENTENCES).cpu().detach().numpy()
            np.save(os.path.join(img_save_dir, f'{cos_sim_name}.npy'), cos_sim) # save the cos_sim.
        cos_sim_norm = (cos_sim - cos_sim.min()) / (cos_sim.max() - cos_sim.min())

        for i, sentence in enumerate(SENTENCES):
            feat_dict[sentence].append(cos_sim[:, i])
            cos_sim_1 = cos_sim[:, i]
            cos_sim_1 = (cos_sim_1 - cos_sim_1.min()) / (cos_sim_1.max() - cos_sim_1.min())
            # Four path
            shot_img_save_path = os.path.join(shot_vis_save_base, 'Detailed', sentence, f'{model_name}.png')
            html_save_path = os.path.join(vis_save_base, 'Detailed', sentence, f'{model_name}.html')
            shot_img_norm_path = os.path.join(shot_vis_save_base, 'Uni_norm', sentence, f'{model_name}.png')
            html_norm_save_path = os.path.join(vis_save_base, 'Uni_norm', sentence, f'{model_name}.html')
            render_sphere(cos_sim_1, mesh_path=model_path, 
                          save_name=html_save_path, 
                          img_save_name=shot_img_save_path)
            render_sphere(cos_sim_norm[:, i], mesh_path=model_path, 
                          save_name=html_norm_save_path, 
                          img_save_name=shot_img_norm_path)
        end = time.time()
        print(f"Process And Vis Done: {model_name}. Time: {end-start}")

    sum_dict = {key: np.stack(feat_dict[key], axis=0).mean(axis=0) for key in feat_dict}
    for key, value in sum_dict.items():
        shot_img_save_path = os.path.join(shot_vis_save_base, f'{key}.png')
        html_save_path = os.path.join(vis_save_base, f'{key}.html')
        render_sphere(value, mesh_path=None, 
                        save_name=html_save_path, 
                        img_save_name=shot_img_save_path)
        

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--source_path', type=str, help='If vis_only, it\'s the Dir of imgs. Else, it\'s the Dir of ShapeNet models.',
                      default="./data/shapenet/02958343")
    args.add_argument('--vis_path', type=str, help='Path to save the html file', default='./vis')
    args.add_argument('--shot_img_path', type=str, help='Path to save a quick shot for the visualization', default='./vis_img')
    args.add_argument('--use_cuda', action='store_true', help='use cuda to render the models')
    args.add_argument('--render_only', action='store_true', help='only render the models')
    args.add_argument('--skip_render', action='store_true', help='skip the model if not rendered')
    args.add_argument('--skip_vis', action='store_true', help='skip the model if a vis file already exists')
    # example of sentences: --text "a chair" "can sit on it"
    args.add_argument('--text', nargs='*', type=str, default=None, help='the sentence to describe the mesh')
    args = args.parse_args()

    if args.text is not None:
        SENTENCES = args.text

    if args.render_only:
        model_paths = find_files(args.source_path, ext='obj')
        render_models(model_paths, args.use_cuda) 
    else:
        model_paths = find_files(args.source_path, ext='obj')
        model_name = os.path.split(args.source_path)[-1]
        html_dir = os.path.join(args.vis_path, model_name)
        shot_dir = os.path.join(args.shot_img_path, model_name)
        vis_models(model_paths, html_dir, shot_dir, skip_render=args.skip_render, skip_vis=args.skip_vis)
          
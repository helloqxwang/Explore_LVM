import subprocess
from load_data import find_files
import os
import argparse
from vis_sphere import vis
from utils import load_blender_pictures, get_clip_processor
import time
BLENDER_PATH = "/home/qianxu/Documents/blender-3.6.8-linux-x64/blender"
SENTENCES = ["Rubbish bin", "Standing on the ground", "Can throw trash into it"]

def render_and_process_models(dir_path:str, vis_save_path:str='./vis', shot_vis_save_path:str='./vis_img', use_cuda:bool=True):
    # html save parttern:
    #  - vis_save_path
    #    - model_name
    #      - sentence1.html
    #      - sentence2.html
    #      - ...

    #shot_vis(img) save parttern:
    #  - shot_vis_save_path
    #    - model_name_sentence1.png
    #    - model_name_sentence2.png
    #    - ...
    model_paths = find_files(dir_path, ext='obj')
    clip_model, processor = get_clip_processor()
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    for model_path in model_paths:

        model_name = os.path.split(model_path)[0].split('/')[-2]

        img_save_dir = os.path.join(os.path.split(model_path)[0], 'img')
        os.makedirs(img_save_dir, exist_ok=True)
        os.environ['img_save_path'] = img_save_dir
        os.environ['obj_file_path'] = model_path
        if use_cuda:
            os.environ['cuda_render'] = 'true'
            # exit()
        else:
            os.environ['cuda_render'] = 'false'
        blender_command = f"blender --background render.blend --python render.py "
        # from pdb import set_trace; set_trace()
        start = time.time()
        subprocess.run(blender_command, shell=True)
        end = time.time()
        print(f"Blender Render Done: {model_name}. Time: {end-start}")
        exit()

        vis_save_path = os.path.join(vis_save_path, model_name)
        os.makedirs(vis_save_path, exist_ok=True)
        os.makedirs(shot_vis_save_path, exist_ok=True)
        start = time.time()
        vis(clip_model, processor, img_save_dir, mesh_path=model_path, sentences=SENTENCES, save_path=vis_save_path, shot_img_save_path=shot_vis_save_path)
        end = time.time()
        print(f"Process Done: {model_name}. Time: {end-start}")

def render_models(dir_path:str, vis_save_path:str='./vis', shot_vis_save_path:str='./vis_img'):
    model_paths = find_files(dir_path, ext='obj')
    clip_model, processor = get_clip_processor()
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    for model_path in model_paths:

        model_name = os.path.split(model_path)[0].split('/')[-2]
        img_save_dir = os.path.join(os.path.split(model_path)[0], 'img')
        if not os.path.isdir(img_save_dir):
            raise ValueError(f"Please render the model first. {img_save_dir} not found.")
        

        vis_save_path = os.path.join(vis_save_path, model_name)
        os.makedirs(vis_save_path, exist_ok=True)
        os.makedirs(shot_vis_save_path, exist_ok=True)
        start = time.time()
        vis(clip_model, processor, img_save_dir, mesh_path=model_path, sentences=SENTENCES, save_path=vis_save_path, shot_img_save_path=shot_vis_save_path)
        end = time.time()
        print(f"Process Done: {model_name}. Time: {end-start}")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--source_path', type=str, help='If vis_only, it\'s the Dir of imgs. Else, it\'s the Dir of ShapeNet models.',
                      default="./data/shapenet/02942699")
    args.add_argument('--vis_path', type=str, help='Path to save the html file', default='./vis')
    args.add_argument('--shot_img_path', type=str, help='Path to save a quick shot for the visualization', default='./vis_img')
    args.add_argument('--use_cuda', action='store_true', help='use cuda to render the models')
    args.add_argument('--vis_only', action='store_true', help='only visualize the models')
    # example of sentences: --text "a chair" "can sit on it"
    # args.add_argument('--text', nargs='*', type=str, help='the sentence to describe the mesh')
    # args.add_argument('--render_only', action='store_true', help='only render the models')
    args = args.parse_args()

    # SENTENCES = args.text

    if args.vis_only:
        render_models(args.source_path, args.vis_path, args.shot_img_path)
    else:
        render_and_process_models(args.source_path, args.vis_path, args.shot_img_path, args.use_cuda)   
# 179
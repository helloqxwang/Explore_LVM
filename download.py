import objaverse.xl as oxl
import numpy as np 
import argparse
import pandas as pd
import os
import shutil

os.makedirs("downloads/", exist_ok=True)

parser = argparse.ArgumentParser()

parser.add_argument("--input_csv", type=str, default="requirements.csv")

args = parser.parse_args()

raw_requirements = pd.read_csv(args.input_csv)

requirements = raw_requirements["Objaverse UID 1"].tolist() + raw_requirements["Objaverse UID 2"].tolist()
requirements = [ f"https://sketchfab.com/3d-models/{x}" for x in list(set(requirements)) if isinstance(x, str) ]

# Build a dict from requirements['Objaverse UID 1'] and requirements['Objaverse UID 2'] to requirements['Object (Preference order)']
requirements_obj_dict = {}
requirements_obj_dict.update(raw_requirements.set_index("Objaverse UID 1").to_dict()["Object (Preference order)"])
requirements_obj_dict.update(raw_requirements.set_index("Objaverse UID 2").to_dict()["Object (Preference order)"])

annotations = oxl.get_annotations(download_dir="downloads/")
# Gather rows in annotations that its `fileIdentifier` is in requirements
annotations = annotations[annotations["fileIdentifier"].isin(requirements)]
annotations.to_csv("annotations.csv", index=False)

oxl.download_objects(
    objects=annotations,
    download_dir="downloads/"
)

objv_glbs_dir = "downloads/hf-objaverse-v1/glbs"
result_dir = "data"
for cat in os.listdir(objv_glbs_dir):
    for file in os.listdir(os.path.join(objv_glbs_dir, cat)):
        if file.endswith(".glb"):
            os.makedirs(os.path.join(result_dir, requirements_obj_dict[file[:-4]]), exist_ok=True)
            shutil.copy(
                os.path.join(objv_glbs_dir, cat, file),
                os.path.join(result_dir, requirements_obj_dict[file[:-4]], file)
            )

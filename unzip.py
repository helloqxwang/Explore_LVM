import os
import zipfile

def unzip_all_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".zip"):
            file_path = os.path.join(directory, filename)
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(directory)
            print(f"Unzip {file_path} done.")

# Usage example:
directory = "./data/shapenet/"
unzip_all_files(directory)

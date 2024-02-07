import os
items_dict = {
    "04379243": "table",
    "03593526": "jar",
    "04225987": "skateboard",
    "02958343": "car",
    "02876657": "bottle",
    "04460130": "tower",
    "03001627": "chair",
    "02871439": "bookshelf",
    "02942699": "camera",
    "02691156": "airplane",
    "03642806": "laptop",
    "02801938": "basket",
    "04256520": "sofa",
    "03624134": "knife",
    "02946921": "can",
    "04090263": "rifle",
    "04468005": "train",
    "03938244": "pillow",
    "03636649": "lamp",
    "02747177": "trash bin",
    "03710193": "mailbox",
    "04530566": "watercraft",
    "03790512": "motorbike",
    "03207941": "dishwasher",
    "02828884": "bench",
    "03948459": "pistol",
    "04099429": "rocket",
    "03691459": "loudspeaker",
    "03337140": "file cabinet",
    "02773838": "bag",
    "02933112": "cabinet",
    "02818832": "bed",
    "02843684": "birdhouse",
    "03211117": "display",
    "03928116": "piano",
    "03261776": "earphone",
    "04401088": "telephone",
    "04330267": "stove",
    "03759954": "microphone",
    "02924116": "bus",
    "03797390": "mug",
    "04074963": "remote",
    "02808440": "bathtub",
    "02880940": "bowl",
    "03085013": "keyboard",
    "03467517": "guitar",
    "04554684": "washer",
    "02834778": "bicycle",
    "03325088": "faucet",
    "04004475": "printer",
    "02954340": "cap"
}

def count_files_in_direct_subdirectories(directory):
    """
    Counts the number of files in each direct subdirectory of the given directory.
    
    :param directory: The path to the directory to be scanned.
    :return: A dictionary with keys as direct subdirectory names and values as the count of files within them.
    """
    file_counts = {}
    # Get all the entries in the directory
    for entry in os.listdir(directory):
        # Construct full path
        full_path = os.path.join(directory, entry)
        # Check if this is a directory
        if os.path.isdir(full_path):
            # List all files in this directory
            files = [f for f in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, f))]
            # Count the files and add to dictionary
            file_counts[entry] = len(files)
    return file_counts


# Replace 'your_directory_path' with the path of the directory you want to scan
directory_path = './data/shapenet'
counts = count_files_in_direct_subdirectories(directory_path)
# for dir_path, count in counts.items():
#     print(f"{dir_path}: {count}")

for dir_path, count in counts.items():
    if count > 500:
        print(f"{items_dict.get(dir_path, dir_path)}: {count}")

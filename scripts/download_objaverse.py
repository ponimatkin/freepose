import shutil
from pathlib import Path

import objaverse

mesh_path = Path(__file__).parent.parent / "data" / "datasets" / 'objaverse_models'
filelist_path = Path(__file__).parent.parent / "data" / "mesh_cache.txt"

objaverse.BASE_PATH = mesh_path
objaverse._VERSIONED_PATH = mesh_path / "hf-objaverse-v1"

with open(filelist_path, 'r') as f:
    files = f.readlines()
    files = [f.strip('\n') for f in files]

uids_all = objaverse.load_uids()

# filter uids from files that are not in the objaverse uids
# Not very efficient, but it works and for one time use it's fine
files = [f for f in files if f in uids_all]

objaverse.load_objects(files)

source_dir = objaverse._VERSIONED_PATH / 'glbs'

for subdir in source_dir.iterdir():
    if subdir.is_dir():
        for file in subdir.iterdir():
            shutil.copy(file, mesh_path)

shutil.rmtree(source_dir.parent)
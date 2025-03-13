# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Configuration of the BOP Toolkit."""

import os


######## Basic ########

# Folder with the BOP datasets.
datasets_path = os.environ.get('BOP_PATH', r"/scratch/project/open-30-19/bop_datasets")

# Folder with object models that were used at inference time
models_inference_path = os.environ.get('BOP_MODELS_INFERENCE_PATH', r"/scratch/project/open-30-19/models_normalized")

# Folder with pose results to be evaluated.
results_path = os.getcwd()

# Folder for the calculated pose errors and performance scores.
eval_path = os.getcwd()

######## Extended ########

# Folder for outputs (e.g. visualizations).
output_path = os.getcwd()

# For offscreen C++ rendering: Path to the build folder of bop_renderer (github.com/thodan/bop_renderer).
bop_renderer_path = r"/path/to/bop_renderer/build"

# Executable of the MeshLab server.
meshlab_server_path = r"/path/to/meshlabserver.exe"

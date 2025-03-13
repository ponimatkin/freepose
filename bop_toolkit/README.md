# Custom BOP Toolkit

A Python toolkit derived from original BOP benchmark for 6D object pose estimation
(http://bop.felk.cvut.cz).

## Installation

### Python Dependencies

To install the required python libraries, run:
```
pip install -r requirements.txt -e .
```

## Usage

### 1. Get the BOP datasets

Download the FreePose datasets (see main repo) and unpack them into a folder. Set path for those datasets via `BOP_PATH` environment variable.

### 2. Get the meshes

Download FreePose meshes (see main repo as well), and unpack it into a folder. Set path for those meshes via `BOP_MODELS_INFERENCE_PATH` environment variable.

### 3. Evaluate the pose estimates
The evaluation is build with SLURM array-based evaluation in mind. To evaluate the pose estimates, run the following command from SLURM submission script with number of tasks equal to the number of scenes in the dataset:
```
python scripts/eval_bop19.py --renderer_type=vispy --result_filenames=NAME_OF_CSV_WITH_RESULTS
```

In case you are running the evaluation on a single machine, you can run the evaluation script directly by manually setting `SLURM_ARRAY_TASK_ID` environment variable. Please note, that by evaluating on one machine the evaluation process might take a very long time.

After the evaluation is finished, rerun the evaluation script **outside** of the SLURM array job to aggregate the results.


### Differences wrt. standard bop_toolkit
- To allow different test/groundtruth meshes (denoted with `HACK` comments)
    - Remove selection of detections based on test targets file
    - Remove selection of n best detections, based on instance count in image
    - Remove sphere_projections_overlap and sphere_overlap checks (only used for quicker computations):
    -> does not make sense for meshes of different scales 
    - Render based errors (cus, vsd) take both `inf_id` and `gt_id` as input -> object ids used by the renderer for inference and groundtruth meshes respectively
    - Vertices based errors (mssd, mspd, chamger) take both `pts_e` and `pts_gt` as input -> point-clouds of inferene and groundtruth meshes respectively
    - Renderer loads both objects of groundtruth dataset and all objects used at inference

- new error function `chamfer` in `pose_error.py`

- Types of errors to consider:
    - Scale invariant: cus
    - Non-scale invariant, work with different vertices numbers: vsd, chamfer  
    - Non-scale invariant, work with same vertices number: mssd, mspd
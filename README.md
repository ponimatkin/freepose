<h1 align="center">
FreePose: 6D Object Pose Tracking in Internet Videos for Robotic Manipulation
</h1>

<div align="center">
<h3>
<a href="http://ponimatkin.github.io">Georgy Ponimatkin*</a>,
<a href="http://cifkam.github.io">Martin Cífka*</a>,
<a href="http://soczech.github.io">Tomáš Souček</a>,
<a href="https://medericfourmy.github.io">Médéric Fourmy</a>,
<a href="http://ylabbe.github.io">Yann Labbé</a>,
<a href="https://petrikvladimir.github.io">Vladimír Petrík</a>,
<a href="https://people.ciirc.cvut.cz/~sivic/">Josef Sivic</a>
<br>
<br>
The Thirteenth International Conference on Learning Representations (ICLR 2025)
<br>
<br>
<a href="https://openreview.net/pdf?id=1CIUkpoata">[Paper]</a>
<a href="https://arxiv.org/abs/2503.10307">[ArXiv]</a>
<a href="https://ponimatkin.github.io/freepose/index.html">[Project page]</a>
<br>
</h3>
</div>
 
## Preparing the environment and data
To prepare the environment run the following commands: 
```
conda env create --file environment_cuda.yaml
conda activate freepose

cd segment-anything-2 && pip install -e .
```

### Downloading 3D assets
The 3D assets can be downloaded by running the following script:
```
bash download_meshes.sh
```
This will download the 3D models that from Objaverse-LVIS and Google Scanned Object datasets, that are not pointclouds and do not cause errors when rendering the templates further on. Please note, that this operation takes a while and will require up to 2TB of disk space. After the data is downloaded and preprocessed, you can remove `google_scanned_objects` and `objaverse_models` folders to save the disk space.


### Rendering 3D model templates
The synthetic data needed for training can be rendered via the SLURM array job via the following command:
```
python -m scripts.render_templates
```
In case your SLURM environment does not allow more than `N` tasks in a single array job, you can chunk the rendering into multiple array jobs by specifying the `--offset` parameter as follows (in this example assuming that SLURM allows only 1000 tasks per array job):
```
python -m scripts.render_templates --offset 0
python -m scripts.render_templates --offset 1000
```
and so on.


### Extracting retrieval features
After rendering has finished, the per-view retrieval features can be extracted via the following command (again assuming usage of SLURM array job):
```
python -m scripts.extract_retrieval_features --feature ffa --batch_size 256 --layer 22
```
and finally the per-view retrieval features can be aggregated into per-object features via:
```
python -m scripts.merge_features --features_folder objaverse_features_ffa_22
```
Alternatively, we also provide precomputed per-object FFA features in the `data/` folder.

**We are aware that this is a very computationally expensive step, and we are working towards releasing all required data publicly as soon as possible.**

### Downloading the static image and video datasets
To download the static image and video datasets, run the following script:
```
bash download_datasets.sh
```

## Inferencing on Static Images
The first step in the pipeline is generation of object proposals. This can be done via the following command:
```
python -m scripts.extract_proposals_ground --dataset <DATASET_NAME>
```
This will generate the object proposals for the chosen dataset. The next step is to run scale estimation on the proposals:
```
python -m scripts.compute_scale --dataset <DATASET_NAME> --proposals data/results/<DATASET_NAME>/props-ground-box-0.3-text-0.5-ffa-22-top-0_<DATASET_NAME>-test.json
```
This will generate new file `props-ground-box-0.3-text-0.5-ffa-22-top-0_<DATASET_NAME>-test_gpt4_scaled.json` which will have the scale estimation for each proposal. The final step is to run the inference on the proposals (this time again under SLURM array job):
```
python -m scripts.dino_inference --dataset <DATASET_NAME> --proposals props-ground-box-0.3-text-0.5-ffa-22-top-0_<DATASET_NAME>-test_gpt4_scaled.json
```
This will generate a set of `.csv` files in `data/results/<DATASET_NAME>/` folder, which can be used for evaluation after merging them into one. The evaluation steps can be found in [bop_toolkit](bop_toolkit/README.md) section.

## Inferencing on Video
Running inference on video is similar to running on a dataset with static images. First, start by generating the proposals:
```
python -m scripts.extract_proposals_ground_video --video "$video" 
```
and continue with the scale estimation:
```
python -m scripts.compute_scale_video --video "$video" \
    --proposals props-ground-box-0.2-text-0.2-ffa-22-top-25_"$video".json
```
This creates the object proposals, which include the object scale estimates, in a file `props-ground-box-0.2-text-0.2-ffa-22-top-25_"$video"_gpt4_scaled.json`.
Next, we select only the proposals corresponding to the annotated manipulated object by running:
```
python -m scripts.filter_predictions --video "$video" \
    --proposals props-ground-box-0.2-text-0.2-ffa-22-top-25_"$video"_gpt4_scaled.json
```
Finally, we run the pose inference with:
```
python -m scripts.dino_inference_video --video "$video" \
    --proposals props-ground-box-0.2-text-0.2-ffa-22-top-25_"$video"_gpt4_scaled_best_object.json
```
and then refine the predicted poses via pose tracking:
```
python -m scripts.smooth_poses_video --video "$video" \
    --proposals props-ground-box-0.2-text-0.2-ffa-22-top-25_"$video"_gpt4_scaled_best_object.json \
    --poses props-ground-box-0.2-text-0.2-ffa-22-top-25_"$video"_gpt4_scaled_best_object_dinopose_layer_22_bbext_0.05_depth_zoedepth.csv
```
which generates csv files with coarse and refined poses in the `data/results/videos/"$video"/` directory.

### Evaluation on Videos
To run the evaluation on (multiple) videos, please run:
```
python -m scripts.eval_videos --labels <METHOD_NAME> \
    --patterns props-ground-box-0.2-text-0.2-ffa-22-top-25_{video}_gpt4_scaled_best_object_megapose_coarse_ref.csv
```
where `{video}` is a placeholder for the python string formatting, allowing us to easily evaluate on multiple videos by running a single script. By default, the script runs the evaluation on all videos, but you can specify a subset of the used videos by providing (multiple) video names to the `--videos` option. Note that the evaluation script also supports evaluation of multiple pose estimate files (obtained using different methods or their hyperparameters) at once by providing multiple method names to the `--labels` option and the same number of values to the `--patterns` options.  

## Citation
If you use this code in your research, please cite the following paper:

```
@inproceedings{ponimatkin2025d,
      title={{{6D}} {{Object}} {{Pose}} {{Tracking}} in {{Internet}} {{Videos}} for {{Robotic}} {{Manipulation}}},
      author={Georgy Ponimatkin and Martin C{\'\i}fka and Tom\'{a}\v{s} Sou\v{c}ek and M{\'e}d{\'e}ric Fourmy and Yann Labb{\'e} and Vladimir Petrik and Josef Sivic},
      booktitle={The Thirteenth International Conference on Learning Representations},
      year={2025},
}
```

This project also uses pieces of code from the following repositories:
- [MegaPose](https://github.com/megapose6d/megapose6d)
- [SAM2](https://github.com/facebookresearch/sam2)
- [CNOS](https://github.com/nv-nguyen/cnos)

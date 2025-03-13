import os
import pandas as pd
import subprocess
from pathlib import Path
from typing import List

import bop_toolkit_lib

# Need to set BOP_PATH environment variable to the right bop_datasets absolute folder

def run_bop_evaluation(results_path: Path, result_filenames: List[str], eval_path: Path, targets_filename='test_targets_bop19.json', visib_gt_min=-1):
    if isinstance(result_filenames, str):
        result_filenames = result_filenames.split(',')
    myenv = os.environ.copy()

    BOP_TOOLKIT_DIR = Path(bop_toolkit_lib.__file__).parent.parent
    POSE_EVAL_SCRIPT_PATH = BOP_TOOLKIT_DIR / "scripts/eval_bop19_pose.py"

    renderer_type = 'vispy'  # other options: 'cpp', 'python'
    cmd = [
        "python",
        POSE_EVAL_SCRIPT_PATH.as_posix(),
        "--result_filenames",
        ','.join(result_filenames),
        "--results_path",
        results_path,
        "--renderer_type",
        renderer_type,
        "--eval_path",
        eval_path,
        '--targets_filename',
        targets_filename,
        # '--visib_gt_min',
        # str(visib_gt_min)
    ]
    # subprocess.call(cmd, env=myenv, cwd=BOP_TOOLKIT_DIR.as_posix())
    subprocess.call(cmd, env=myenv, cwd=os.getcwd())

if __name__ == '__main__':
    results_path = 'data/results'
    eval_path = 'data/evals'
    result_filenames = 'restricted_ycbv-test.csv'
    targets_filename = 'test_targets_bop19_restricted.json'

    run_bop_evaluation(results_path, result_filenames, eval_path, targets_filename=targets_filename)
    print('\n\n\n\nDONE')
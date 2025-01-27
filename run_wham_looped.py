#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   run_humor_looped.py
@Time    :   2024/02/10 20:10:09
@Author  :   Siddharth Ravi
@Version :   1.0
@Contact :   siddharth.ravi@ua.es
@License :   Copyright 2022-2023, Siddharth Ravi, Distributed under terms of the MIT license
@Desc    :   Script to run WHAM on the data in a loop, before or after preprocessing (uses humor/fitting/run_fitting.py)
'''
import os
import sys
import logging
import subprocess
from tabnanny import check

ROOT_DIR = "/home/NAS-mountpoint/kinect-omni-ego/"
python_file = "/home/sid/Projects/WHAM/demo_images.py"

# Configure logging
logging.basicConfig(level=logging.INFO,
                    filename='wham_cuda_log.txt',
                    filemode='a',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)    
python_executable = "/home/sid/miniforge3/envs/wham/bin/python"

def check_cuda_error(out_dir):
    if os.path.exists(f"{out_dir}/rgb/wham_output.pkl"):
        if os.path.getsize(f"{out_dir}/rgb/wham_output.pkl") > 0:
            return True
    return False

def main():
    for root, dirs, files in os.walk(ROOT_DIR):
        if "calib" not in root and "rgb" in dirs and "out_capture" not in root:
            n = os.path.basename(root)[-1]
            image_dir = f"{root}/rgb"
            output_dir = f"{root}/out_capture{n}/wham_output"
            calib = f"/home/sid/Projects/WHAM/examples/k{n}_calib.txt"
            if check_cuda_error(output_dir):
                continue
            command = f"{python_executable} {python_file} " \
                f"--image_dir {image_dir} " \
                f"--output_pth {root}/out_capture{n}/wham_output " \
                f"--calib /home/sid/Projects/WHAM/examples/k{n}_calib.txt " \
                "--save_pkl --run_smplify --estimate_local_only"
            logger.info(command)
            print(command)
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            logger.info(result.stdout)
            if result.stderr:
                logger.error(result.stderr)


if __name__ == "__main__":
    main()

import os
import os.path as osp
from glob import glob
from tqdm import tqdm  # For progress bars

import cv2
import torch
import joblib
import numpy as np
from collections import defaultdict
from smplx import SMPL
from loguru import logger

from configs.config import get_cfg_defaults
from lib.data.datasets import CustomDataset
from lib.models import build_network, build_body_model
from lib.models.preproc.detector import DetectionModel
from lib.models.preproc.extractor import FeatureExtractor

try: 
    from lib.models.preproc.slam import SLAMModel
    _run_global = True
except ImportError as e: 
    logger.error(f'DPVO is not properly installed: {e}. Only estimate in local coordinates!')
    _run_global = False

def prepare_cfg(config_path='configs/yamls/demo.yaml'):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_path)
    return cfg

def load_video(video):
    cap = cv2.VideoCapture(video)
    assert cap.isOpened(), f'Failed to load video file {video}'
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return cap, fps, length, width, height

class WHAM_API(object):
    def __init__(self, device=None):
        self.cfg = prepare_cfg()
        self.device = device if device else self.cfg.DEVICE.lower()
        self.network = build_network(
            self.cfg, 
            build_body_model(self.device, self.cfg.TRAIN.BATCH_SIZE * self.cfg.DATASET.SEQLEN)
        )
        self.network.eval()
        self.detector = DetectionModel(self.device)
        self.extractor = FeatureExtractor(self.device)
        self.slam = None
        self.image_paths = []  # Initialize to store image paths

    def _video_to_images(self, cap):
        try:
            while cap.isOpened():
                flag, img = cap.read()
                if not flag:
                    break
                yield img
        finally:
            cap.release()

    def _load_images_from_directory(self, image_dir):
        supported_formats = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff')
        image_paths = []
        for fmt in supported_formats:
            image_paths.extend(glob(osp.join(image_dir, fmt)))
        image_paths = sorted(image_paths)
        
        if not image_paths:
            raise ValueError(f"No images found in directory {image_dir}")
        
        # Validate image paths
        valid_image_paths = []
        for path in image_paths:
            if cv2.imread(path) is not None:
                valid_image_paths.append(path)
            else:
                logger.warning(f"Invalid image path: {path}. Skipping.")
        
        if not valid_image_paths:
            raise ValueError(f"No valid images found in directory {image_dir}")
        
        # Assume a default FPS or allow passing FPS as a parameter
        fps = self.cfg.DEFAULT_FPS if hasattr(self.cfg, 'DEFAULT_FPS') else 30.0
        self.image_paths = valid_image_paths  # Store for extractor
        return valid_image_paths, fps

    def preprocessing(self, input_data, fps, length, output_dir, input_type='video'):
        tracking_pth = osp.join(output_dir, 'tracking_results.pth')
        slam_pth = osp.join(output_dir, 'slam_results.pth')

        if not (osp.exists(tracking_pth) and osp.exists(slam_pth)):
            if input_type == 'video':
                cap = cv2.VideoCapture(input_data)
                image_generator = self._video_to_images(cap)
            else:
                image_generator = (cv2.imread(p) for p in input_data)

            for img in image_generator:
                if img is None:
                    continue
                # 2D detection and tracking
                self.detector.track(img, fps, length)
                
                # SLAM
                if self.slam is not None: 
                    self.slam.track()

            tracking_results = self.detector.process(fps)
            
            if self.slam is not None: 
                slam_results = self.slam.process()
            else:
                slam_results = np.zeros((length, 7))
                slam_results[:, 3] = 1.0    # Unit quaternion
        
            # Extract image features
            tracking_results = self.extractor.run(input_data, tracking_results)
            # Save the processed data
            joblib.dump(tracking_results, tracking_pth)
            joblib.dump(slam_results, slam_pth)
        
        else:
            tracking_results = joblib.load(tracking_pth)
            slam_results = joblib.load(slam_pth)

        return tracking_results, slam_results

    @torch.no_grad()
    def wham_inference(self, tracking_results, slam_results, width, height, fps, output_dir):
        # Build dataset
        dataset = CustomDataset(self.cfg, tracking_results, slam_results, width, height, fps)
        
        # Run WHAM
        results = defaultdict(dict)
        for batch in tqdm(dataset, desc="Running WHAM Inference"):
            if batch is None: 
                break

            _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = batch
            
            # Inference
            pred = self.network(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=True, **kwargs)
            
            # Store results
            results[_id]['poses_body'] = pred['poses_body'].cpu().squeeze(0).numpy()
            results[_id]['poses_root_cam'] = pred['poses_root_cam'].cpu().squeeze(0).numpy()
            results[_id]['betas'] = pred['betas'].cpu().squeeze(0).numpy()
            results[_id]['verts_cam'] = (pred['verts_cam'] + pred['trans_cam'].unsqueeze(1)).cpu().numpy()
            results[_id]['poses_root_world'] = pred['poses_root_world'].cpu().squeeze(0).numpy()
            results[_id]['trans_world'] = pred['trans_world'].cpu().squeeze(0).numpy()
            results[_id]['frame_id'] = frame_id
        
        # Optionally save WHAM-specific results
        joblib.dump(slam_results, osp.join(output_dir, 'wham_results.pth'))
        return results

    @torch.no_grad()
    def __call__(self, input_path, output_dir='output/demo', calib=None, run_global=True, visualize=False, input_type='images'):
        os.makedirs(output_dir, exist_ok=True)

        if input_type == 'images':
            image_paths, fps = self._load_images_from_directory(input_path)
            length = len(image_paths)
            first_img = cv2.imread(image_paths[0])
            if first_img is None:
                raise ValueError(f"Failed to load the first image: {image_paths[0]}")
            height, width = first_img.shape[:2]
        else:
            raise ValueError("input_type must be 'images'")

        run_global = run_global and _run_global
        if run_global: 
            self.slam = SLAMModel(input_path, output_dir, width, height, calib)
        
        tracking_results, slam_results = self.preprocessing(image_paths, fps, length, output_dir, input_type=input_type)

        results = self.wham_inference(tracking_results, slam_results, width, height, fps, output_dir)
        
        if visualize:
            from lib.vis.run_vis import run_vis_on_demo_images
            run_vis_on_demo_images(self.cfg, input_path, results, output_dir, self.network.smpl, vis_global=run_global)
        
        return results, tracking_results, slam_results



# Example usage
if __name__ == '__main__':
    wham_model = WHAM_API()
    
    # For video input
    # input_video_path = 'examples/IMG_9732.mov'
    # results, tracking_results, slam_results = wham_model(input_video_path, input_type='video')
    
    # For image directory input
    input_image_folder = "/home/NAS-mountpoint/kinect-omni-ego/2022-10-07/at-a02/bedroom/a02/capture0/rgb/"
    # input_image_folder = "/home/sid/Projects/dataset/test/"
    results, tracking_results, slam_results = wham_model(
        input_image_folder, 
        output_dir='output/test',  
        calib=None, 
        run_global=False, 
        visualize=True, 
        input_type="images"
    )

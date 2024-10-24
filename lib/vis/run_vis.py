import os
import os.path as osp
import cv2
import torch
import imageio
import numpy as np
from progress.bar import Bar
from lib.vis.renderer import Renderer, get_global_cameras


def run_vis_on_demo_images(cfg, imagedir, results, output_pth, smpl, vis_global=True):
    # to torch tensor
    tt = lambda x: torch.from_numpy(x).float().to(cfg.DEVICE)
    
    # List all image files in the directory
    image_files = sorted([f for f in os.listdir(imagedir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    length = len(image_files)
    first_image = cv2.imread(os.path.join(imagedir, image_files[0]))
    height, width = first_image.shape[:2]
    
    # create renderer with cliff focal length estimation
    focal_length = (width ** 2 + height ** 2) ** 0.5
    renderer = Renderer(width, height, focal_length, cfg.DEVICE, smpl.faces)
    
    if vis_global:
        # setup global coordinate subject
        n_frames = {k: len(results[k]['frame_id']) for k in results.keys()}
        sid = max(n_frames, key=n_frames.get)
        global_output = smpl.get_output(
            body_pose=tt(results[sid]['pose_world'][:, 3:]), 
            global_orient=tt(results[sid]['pose_world'][:, :3]),
            betas=tt(results[sid]['betas']),
            transl=tt(results[sid]['trans_world']))
        verts_glob = global_output.vertices.cpu()
        verts_glob[..., 1] = verts_glob[..., 1] - verts_glob[..., 1].min()
        cx, cz = (verts_glob.mean(1).max(0)[0] + verts_glob.mean(1).min(0)[0])[[0, 2]] / 2.0
        sx, sz = (verts_glob.mean(1).max(0)[0] - verts_glob.mean(1).min(0)[0])[[0, 2]]
        scale = max(sx.item(), sz.item()) * 1.5
        
        # set default ground
        renderer.set_ground(scale, cx.item(), cz.item())
        
        # build global camera
        global_R, global_T, global_lights = get_global_cameras(verts_glob, cfg.DEVICE)
    
    # build default camera
    default_R, default_T = torch.eye(3), torch.zeros(3)
    
    writer = imageio.get_writer(
        os.path.join(output_pth, 'output.mp4'), 
        fps=30, mode='I', format='FFMPEG', macro_block_size=1
    )
    bar = Bar('Rendering results ...', fill='#', max=length)
    
    frame_i = 0
    _global_R, _global_T = None, None
    # run rendering
    for image_file in image_files:
        image_path = os.path.join(imagedir, image_file)
        org_img = cv2.imread(image_path)
        if org_img is None: break
        img = org_img[..., ::-1].copy()
        
        # render onto the input image
        renderer.create_camera(default_R, default_T)
        for _id, val in results.items():
            # render onto the image
            frame_i2 = np.where(val['frame_id'] == frame_i)[0]
            if len(frame_i2) == 0: continue
            frame_i2 = frame_i2[0]
            img = renderer.render_mesh(torch.from_numpy(val['verts'][frame_i2]).to(cfg.DEVICE), img)
        
        if vis_global:
            # render the global coordinate
            if frame_i in results[sid]['frame_id']:
                frame_i3 = np.where(results[sid]['frame_id'] == frame_i)[0]
                verts = verts_glob[[frame_i3]].to(cfg.DEVICE)
                faces = renderer.faces.clone().squeeze(0)
                colors = torch.ones((1, 4)).float().to(cfg.DEVICE); colors[..., :3] *= 0.9
                
                if _global_R is None:
                    _global_R = global_R[frame_i3].clone(); _global_T = global_T[frame_i3].clone()
                cameras = renderer.create_camera(global_R[frame_i3], global_T[frame_i3])
                img_glob = renderer.render_with_ground(verts, faces, colors, cameras, global_lights)
            
            try: img = np.concatenate((img, img_glob), axis=1)
            except: img = np.concatenate((img, np.ones_like(img) * 255), axis=1)
        
        writer.append_data(img)
        bar.next()
        frame_i += 1
    writer.close()


def run_vis_on_demo_images_old(cfg, image_dir, results, output_pth, smpl, vis_global=True):
    # to torch tensor
    tt = lambda x: torch.from_numpy(x).float().to(cfg.DEVICE)

    # Get list of image paths
    image_paths = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
    if not image_paths:
        raise ValueError(f"No images found in directory {image_dir}")

    # Read the first image to get the dimensions
    img_sample = cv2.imread(os.path.join(image_dir, image_paths[0]))
    height, width = img_sample.shape[:2]
    length = len(image_paths)

    # create renderer with cliff focal length estimation
    focal_length = (width ** 2 + height ** 2) ** 0.5
    renderer = Renderer(width, height, focal_length, cfg.DEVICE, smpl.faces)

    if vis_global:
        
        # setup global coordinate subject
        # current implementation only visualize the subject appeared longest
        n_frames = {k: len(results[k]['frame_id']) for k in results.keys()}
        sid = max(n_frames, key=n_frames.get)
        print(results[sid].keys())

        # Ensure correct tensor shapes
        poses_body = tt(results[sid]['poses_body']).view(-1, 23, 3)
        poses_root_world = tt(results[sid]['poses_root_world']).view(-1, 3)
        betas = tt(results[sid]['betas']).view(-1, 10)
        trans_world = tt(results[sid]['trans_world']).view(-1, 3)
        
        # Convert rotation matrices to 6D rotation representation
        poses_body_6d = torch.cat([poses_body[..., :2].flatten(start_dim=-2)], dim=-1)
        poses_root_world_6d = torch.cat([poses_root_world[..., :2]], dim=-1)
        
        global_output = smpl(
            body_pose=poses_body_6d,
            global_orient=poses_root_world_6d,
            betas=betas,
            transl=trans_world,
            pred_rot6d=True
        )
        
        verts_glob = global_output.vertices.cpu()
        verts_glob[..., 1] = verts_glob[..., 1] - verts_glob[..., 1].min()
        cx, cz = (verts_glob.mean(1).max(0)[0] + verts_glob.mean(1).min(0)[0])[[0, 2]] / 2.0
        sx, sz = (verts_glob.mean(1).max(0)[0] - verts_glob.mean(1).min(0)[0])[[0, 2]]
        scale = max(sx.item(), sz.item()) * 1.5

        # set default ground
        renderer.set_ground(scale, cx.item(), cz.item())

        # build global camera
        global_R, global_T, global_lights = get_global_cameras(verts_glob, cfg.DEVICE)

    # build default camera
    default_R, default_T = torch.eye(3), torch.zeros(3)

    os.makedirs(osp.join(output_pth, 'vis_output'), exist_ok=True)
    bar = Bar('Rendering results ...', fill='#', max=length)

    _global_R, _global_T = None, None
    # run rendering
    for frame_i, image_name in enumerate(image_paths):
        org_img = cv2.imread(os.path.join(image_dir, image_name))
        img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)

        # render onto the input image
        renderer.create_camera(default_R, default_T)
        for _id, val in results.items():
            # render onto the image
            frame_i2 = np.where(val['frame_id'] == frame_i)[0]
            if len(frame_i2) == 0: continue
            frame_i2 = frame_i2[0]
            img = renderer.render_mesh(torch.from_numpy(val['verts'][frame_i2]).to(cfg.DEVICE), img)

        if vis_global:
            # render the global coordinate
            if frame_i in results[sid]['frame_id']:
                frame_i3 = np.where(results[sid]['frame_id'] == frame_i)[0]
                verts = verts_glob[[frame_i3]].to(cfg.DEVICE)
                faces = renderer.faces.clone().squeeze(0)
                colors = torch.ones((1, 4)).float().to(cfg.DEVICE); colors[..., :3] *= 0.9

                if _global_R is None:
                    _global_R = global_R[frame_i3].clone(); _global_T = global_T[frame_i3].clone()
                cameras = renderer.create_camera(global_R[frame_i3], global_T[frame_i3])
                img_glob = renderer.render_with_ground(verts, faces, colors, cameras, global_lights)

            try: 
                img = np.concatenate((img, img_glob), axis=1)
            except: 
                img = np.concatenate((img, np.ones_like(img) * 255), axis=1)

        cv2.imwrite(osp.join(output_pth, 'vis_output', f'frame_{frame_i:04d}.jpg'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        bar.next()

    bar.finish()

    # Create a video from the rendered frames
    frames = []
    for i in range(length):
        frame_path = osp.join(output_pth, 'vis_output', f'frame_{i:04d}.jpg')
        frames.append(imageio.imread(frame_path))
    
    output_video_path = osp.join(output_pth, 'vis_output.mp4')
    imageio.mimsave(output_video_path, frames, fps=30)



def run_vis_on_demo(cfg, video, results, output_pth, smpl, vis_global=True):
    # to torch tensor
    tt = lambda x: torch.from_numpy(x).float().to(cfg.DEVICE)
    
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    # create renderer with cliff focal length estimation
    focal_length = (width ** 2 + height ** 2) ** 0.5
    renderer = Renderer(width, height, focal_length, cfg.DEVICE, smpl.faces)
    
    if vis_global:
        # setup global coordinate subject
        # current implementation only visualize the subject appeared longest
        n_frames = {k: len(results[k]['frame_id']) for k in results.keys()}
        sid = max(n_frames, key=n_frames.get)
        global_output = smpl.get_output(
            body_pose=tt(results[sid]['pose_world'][:, 3:]), 
            global_orient=tt(results[sid]['pose_world'][:, :3]),
            betas=tt(results[sid]['betas']),
            transl=tt(results[sid]['trans_world']))
        verts_glob = global_output.vertices.cpu()
        verts_glob[..., 1] = verts_glob[..., 1] - verts_glob[..., 1].min()
        cx, cz = (verts_glob.mean(1).max(0)[0] + verts_glob.mean(1).min(0)[0])[[0, 2]] / 2.0
        sx, sz = (verts_glob.mean(1).max(0)[0] - verts_glob.mean(1).min(0)[0])[[0, 2]]
        scale = max(sx.item(), sz.item()) * 1.5
        
        # set default ground
        renderer.set_ground(scale, cx.item(), cz.item())
        
        # build global camera
        global_R, global_T, global_lights = get_global_cameras(verts_glob, cfg.DEVICE)
    
    # build default camera
    default_R, default_T = torch.eye(3), torch.zeros(3)
    
    writer = imageio.get_writer(
        osp.join(output_pth, 'output.mp4'), 
        fps=fps, mode='I', format='FFMPEG', macro_block_size=1
    )
    bar = Bar('Rendering results ...', fill='#', max=length)
    
    frame_i = 0
    _global_R, _global_T = None, None
    # run rendering
    while (cap.isOpened()):
        flag, org_img = cap.read()
        if not flag: break
        img = org_img[..., ::-1].copy()
        
        # render onto the input video
        renderer.create_camera(default_R, default_T)
        for _id, val in results.items():
            # render onto the image
            frame_i2 = np.where(val['frame_id'] == frame_i)[0]
            if len(frame_i2) == 0: continue
            frame_i2 = frame_i2[0]
            img = renderer.render_mesh(torch.from_numpy(val['verts'][frame_i2]).to(cfg.DEVICE), img)
        
        if vis_global:
            # render the global coordinate
            if frame_i in results[sid]['frame_id']:
                frame_i3 = np.where(results[sid]['frame_id'] == frame_i)[0]
                verts = verts_glob[[frame_i3]].to(cfg.DEVICE)
                faces = renderer.faces.clone().squeeze(0)
                colors = torch.ones((1, 4)).float().to(cfg.DEVICE); colors[..., :3] *= 0.9
                
                if _global_R is None:
                    _global_R = global_R[frame_i3].clone(); _global_T = global_T[frame_i3].clone()
                cameras = renderer.create_camera(global_R[frame_i3], global_T[frame_i3])
                img_glob = renderer.render_with_ground(verts, faces, colors, cameras, global_lights)
            
            try: img = np.concatenate((img, img_glob), axis=1)
            except: img = np.concatenate((img, np.ones_like(img) * 255), axis=1)
        
        writer.append_data(img)
        bar.next()
        frame_i += 1
    writer.close()
#!/usr/bin/env python3
"""
Depth Anything V2 - Depth to Point Cloud Converter
Generates metric point clouds from single images using Depth Anything V2.

Usage:
    python depth_to_pointcloud.py \
        --encoder vitl \
        --load-from checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
        --max-depth 20 \
        --img-path ./input \
        --outdir ./output
"""

import argparse
import os
import glob
import cv2
import torch
import numpy as np
from PIL import Image
import open3d as o3d
from tqdm import tqdm

# Model configurations for different encoder sizes
MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}


def load_calibration(calibration_path):
    """Load camera calibration parameters if available."""
    if os.path.exists(calibration_path):
        calibration_data = np.load(calibration_path)
        camera_matrix = calibration_data['Camera_matrix']
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        print(f"Loaded calibration: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
        return fx, fy, cx, cy
    return None, None, None, None


def estimate_focal_length(width, height):
    """Estimate focal length based on image dimensions (assuming 60° FOV)."""
    fov_degrees = 60
    focal_length = width / (2 * np.tan(np.radians(fov_degrees / 2)))
    return focal_length, focal_length


def depth_to_pointcloud(depth, color_image, fx, fy, cx, cy):
    """Convert depth map to colored point cloud."""
    height, width = depth.shape
    
    # Create mesh grid for pixel coordinates
    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)
    
    # Convert to 3D coordinates
    z = depth
    x3d = (x - cx) * z / fx
    y3d = (y - cy) * z / fy
    
    # Stack into point cloud (N, 3)
    points = np.stack([x3d, y3d, z], axis=-1).reshape(-1, 3)
    
    # Get colors (N, 3) normalized to [0, 1]
    colors = np.array(color_image).reshape(-1, 3) / 255.0
    
    # Filter out invalid points (zero or negative depth)
    valid_mask = points[:, 2] > 0
    points = points[valid_mask]
    colors = colors[valid_mask]
    
    return points, colors


def process_images(model, args, device):
    """Process all images and generate point clouds."""
    os.makedirs(args.outdir, exist_ok=True)
    
    # Get image paths
    if os.path.isfile(args.img_path):
        image_paths = [args.img_path]
    else:
        image_paths = (
            glob.glob(os.path.join(args.img_path, '*.png')) +
            glob.glob(os.path.join(args.img_path, '*.jpg')) +
            glob.glob(os.path.join(args.img_path, '*.jpeg'))
        )
    
    if not image_paths:
        print(f"No images found in {args.img_path}")
        return
    
    # Load calibration if available
    calibration_path = args.calibration if args.calibration else "MyCalibration.npz"
    fx, fy, cx, cy = load_calibration(calibration_path)
    
    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            # Load image
            raw_img = cv2.imread(image_path)
            if raw_img is None:
                print(f"Could not load {image_path}")
                continue
            
            height, width = raw_img.shape[:2]
            
            # Use calibration or estimate focal length
            if fx is None:
                fx_img, fy_img = estimate_focal_length(width, height)
                cx_img, cy_img = width / 2, height / 2
            else:
                fx_img, fy_img, cx_img, cy_img = fx, fy, cx, cy
            
            # Infer depth (returns HxW numpy array in meters)
            with torch.no_grad():
                depth = model.infer_image(raw_img)
            
            # Convert BGR to RGB for colors
            color_image = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
            color_pil = Image.fromarray(color_image)
            
            # Generate point cloud
            points, colors = depth_to_pointcloud(
                depth, color_pil, fx_img, fy_img, cx_img, cy_img
            )
            
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Downsample for efficiency
            pcd = pcd.voxel_down_sample(voxel_size=0.01)
            
            # Save point cloud
            output_path = os.path.join(
                args.outdir,
                os.path.splitext(os.path.basename(image_path))[0] + ".ply"
            )
            o3d.io.write_point_cloud(output_path, pcd)
            print(f"Saved: {output_path} ({len(pcd.points)} points)")
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Depth Anything V2 - Depth to Point Cloud')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'],
                        help='Model encoder size: vits (small), vitb (base), vitl (large)')
    parser.add_argument('--load-from', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--max-depth', type=float, default=20.0,
                        help='Maximum depth in meters (20 for indoor, 80 for outdoor)')
    parser.add_argument('--img-path', type=str, required=True,
                        help='Input image path or directory')
    parser.add_argument('--outdir', type=str, required=True,
                        help='Output directory for point clouds')
    parser.add_argument('--calibration', type=str, default=None,
                        help='Path to camera calibration file (.npz)')
    parser.add_argument('--input-size', type=int, default=518,
                        help='Input size for model inference')
    
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    from depth_anything_v2.dpt import DepthAnythingV2
    
    model_config = MODEL_CONFIGS[args.encoder]
    model = DepthAnythingV2(**{**model_config, 'max_depth': args.max_depth})

    state_dict = torch.load(args.load_from, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    
    print(f"Loaded model: {args.encoder} encoder, max_depth={args.max_depth}m")
    process_images(model, args, device)


if __name__ == '__main__':
    main()

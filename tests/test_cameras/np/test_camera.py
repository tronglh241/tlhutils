import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

sys.path.append(os.getcwd())
from cameras.np.fisheye import Fisheye  # noqa: E402
from tests.test_cameras.utils.calib import load_calib  # noqa: E402

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--im-file', default=str(Path(__file__).parents[1].joinpath('data', 'fisheye.jpg')))
    parser.add_argument('--mask-file', default=str(Path(__file__).parents[1].joinpath('data', 'mask.png')))
    parser.add_argument('--camera-data', default=str(Path(__file__).parents[1].joinpath('data', 'cameraData.json')))
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=800)
    parser.add_argument('--out-dir', default=str(Path(__file__).parents[1].joinpath('output')))
    args = parser.parse_args()

    im = cv2.imread(args.im_file)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.astype(np.float32) / 255.
    mask = cv2.imread(args.mask_file, cv2.IMREAD_GRAYSCALE)
    mask = mask.astype(np.float32) / 255.

    calib_info = load_calib(args.camera_data)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    camera = Fisheye(
        width=args.width,
        height=args.height,
        intrinsic=calib_info['rear']['intrinsic'],
        extrinsic=calib_info['rear']['extrinsic'],
        distortion=calib_info['rear']['distortion'],
    )
    # Single depth
    depth = np.ones((args.height, args.width, 1), dtype=np.float32)

    # PCD in camera coordinate
    points_3d, valid_points = camera.im_to_cam_map(
        depth_map=depth,
        mask=mask,
    )
    assert points_3d.shape[2] == 1
    points_3d = points_3d.squeeze(2)

    points = points_3d[valid_points]
    positions = o3d.utility.Vector3dVector(points)
    pcd = o3d.geometry.PointCloud(positions)

    point_colors = im[valid_points]
    colors = o3d.utility.Vector3dVector(point_colors)
    pcd.colors = colors

    o3d.io.write_point_cloud(out_dir.joinpath('camera_single_depth_single_calib.ply'), pcd)

    # PCD in world coordinate
    points_3d = camera.cam_to_world(points_3d=points_3d)

    points = points_3d[valid_points]
    positions = o3d.utility.Vector3dVector(points)
    pcd = o3d.geometry.PointCloud(positions)

    point_colors = im[valid_points]
    colors = o3d.utility.Vector3dVector(point_colors)
    pcd.colors = colors

    o3d.io.write_point_cloud(out_dir.joinpath('world_single_depth_single_calib.ply'), pcd)

    # Multiple depth
    depth = np.ones((args.height, args.width, 3), dtype=np.float32)
    depth[..., 1] = 2.0
    depth[..., 2] = 3.0

    # PCD in camera coordinate
    points_3d, valid_points = camera.im_to_cam_map(
        depth_map=depth,
        mask=mask,
    )
    assert points_3d.shape[2] == 3

    for depth_idx in range(points_3d.shape[2]):
        points = points_3d[..., depth_idx, :][valid_points]
        positions = o3d.utility.Vector3dVector(points)
        pcd = o3d.geometry.PointCloud(positions)

        point_colors = im[valid_points]
        colors = o3d.utility.Vector3dVector(point_colors)
        pcd.colors = colors

        o3d.io.write_point_cloud(out_dir.joinpath(f'camera_multiple_depth_single_calib_{depth_idx}.ply'), pcd)

    # PCD in world coordinate
    points_3d = camera.cam_to_world(points_3d=points_3d)

    for depth_idx in range(points_3d.shape[2]):
        points = points_3d[..., depth_idx, :][valid_points]
        positions = o3d.utility.Vector3dVector(points)
        pcd = o3d.geometry.PointCloud(positions)

        point_colors = im[valid_points]
        colors = o3d.utility.Vector3dVector(point_colors)
        pcd.colors = colors

        o3d.io.write_point_cloud(out_dir.joinpath(f'world_multiple_depth_single_calib_{depth_idx}.ply'), pcd)

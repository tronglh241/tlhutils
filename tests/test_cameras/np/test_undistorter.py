import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(os.getcwd())
from cameras.np.fisheye_old.undistorter import Undistorter  # noqa: E402

from tests.test_cameras.utils.calib import load_calib  # noqa: E402
from tests.test_cameras.utils.matrix import rotation_matrix  # noqa: E402

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--im-file', default=str(Path(__file__).parents[1].joinpath('data', 'fisheye.jpg')))
    parser.add_argument('--camera-data', default=str(Path(__file__).parents[1].joinpath('data', 'cameraData.json')))
    args = parser.parse_args()

    calib_info = load_calib(args.camera_data)

    im = cv2.imread(args.im_file)

    undistorter = Undistorter(
        intrinsic=calib_info['rear']['intrinsic'],
        distortion=calib_info['rear']['distortion'],
        undistorted_im_size=(im.shape[1], im.shape[0]),
        rotation=None,
        extrinsic=calib_info['rear']['extrinsic'],
    )
    undistorted_im = undistorter(im)

    cv2.imshow('Undistorted', undistorted_im)

    new_intrinsic = calib_info['rear']['intrinsic'].copy()
    new_intrinsic[:2] /= 2
    undistorted_im_size = (im.shape[1] // 2, im.shape[0] // 2)
    undistorter = Undistorter(
        intrinsic=calib_info['rear']['intrinsic'],
        distortion=calib_info['rear']['distortion'],
        undistorted_im_size=undistorted_im_size,
        rotation=None,
        extrinsic=calib_info['rear']['extrinsic'],
        new_intrinsic=new_intrinsic,
    )
    undistorted_im = undistorter(im)

    cv2.imshow('Undistorted Customized', undistorted_im)

    undistorter = Undistorter(
        intrinsic=calib_info['rear']['intrinsic'],
        distortion=calib_info['rear']['distortion'],
        undistorted_im_size=undistorted_im_size,
        rotation=rotation_matrix(axis=(1, 0, 0), theta=-np.pi / 4),
        extrinsic=calib_info['rear']['extrinsic'],
        new_intrinsic=new_intrinsic,
    )
    undistorted_im = undistorter(im)

    cv2.imshow('Undistorted Rotated', undistorted_im)
    cv2.waitKey()
    cv2.destroyAllWindows()

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import numpy.typing as npt

sys.path.append(os.getcwd())
from cameras.fisheye.undistorter import Undistorter  # noqa: E402


def load_calib(cam_data_file: str) -> Dict[str, Dict[str, npt.NDArray[Any]]]:
    cam_mapping = {
        'left': 0,
        'front': 1,
        'rear': 2,
        'right': 3,
    }

    with open(cam_data_file) as f:
        cam_data = json.load(f)
        cam_data = {data['camPos']: data for data in cam_data['Items']}

    calib_info = {}

    for cam_name in cam_mapping:
        pos = cam_mapping[cam_name]
        data = cam_data[pos]
        R = np.array(data['matrixR'], dtype=np.float32).reshape(3, 3)
        T = np.array([data['vectT']], dtype=np.float32)
        extrinsic = np.concatenate((R, T.T), axis=1)
        extrinsic = np.concatenate((extrinsic, np.zeros((1, 4))), axis=0)
        extrinsic[-1, -1] = 1.0

        K = np.array(data['matrixK'])
        intrinsic = np.zeros((3, 3), dtype=np.float32)
        intrinsic[0, 0] = K[0]
        intrinsic[0, 2] = K[1]
        intrinsic[1, 1] = K[2]
        intrinsic[1, 2] = K[3]
        intrinsic[2, 2] = 1.0

        distortion = np.array(data['matrixD'], dtype=np.float32)

        calib_info[cam_name] = {
            'extrinsic': extrinsic,
            'intrinsic': intrinsic,
            'distortion': distortion,
        }

    return calib_info


def rotation_matrix(axis: Tuple[float, float, float], theta: float) -> npt.NDArray[Any]:
    '''
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    '''
    axis_arr: npt.NDArray[Any] = np.asarray(axis)
    axis_arr = axis_arr / np.sqrt(np.dot(axis_arr, axis_arr))
    a = np.cos(theta / 2.0)
    b, c, d = -axis_arr * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--im-file', default=str(Path(__file__).parent.joinpath('data', 'fisheye.jpg')))
    parser.add_argument('--camera-data', default=str(Path(__file__).parent.joinpath('data', 'cameraData.json')))
    args = parser.parse_args()

    im_file = args.im_file
    calib_info = load_calib(args.camera_data)

    im = cv2.imread(im_file)

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

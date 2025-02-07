import argparse
import os
import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.append(os.getcwd())
from cameras.np.camera_transformer import CameraTransformer  # noqa: E402
from cameras.np.fisheye import Fisheye  # noqa: E402
from cameras.np.pinhole import PinHole  # noqa: E402
from tests.test_cameras.utils.calib import load_calib  # noqa: E402
from tests.test_cameras.utils.matrix import rotation_matrix  # noqa: E402

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--im-file', default=str(Path(__file__).parents[1].joinpath('data', 'fisheye.jpg')))
    parser.add_argument('--camera-data', default=str(Path(__file__).parents[1].joinpath('data', 'cameraData.json')))
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=800)
    parser.add_argument('--out-dir', default=str(Path(__file__).parents[1].joinpath('output')))
    args = parser.parse_args()

    im = cv2.imread(args.im_file)

    calib_info = load_calib(args.camera_data)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_cam = Fisheye(
        width=args.width,
        height=args.height,
        intrinsic=calib_info['rear']['intrinsic'],
        extrinsic=calib_info['rear']['extrinsic'],
        distortion=calib_info['rear']['distortion'],
    )

    dst_cam = PinHole(
        width=args.width,
        height=args.height,
        intrinsic=calib_info['rear']['intrinsic'],
        extrinsic=calib_info['rear']['extrinsic'],
    )

    cam_transformer = CameraTransformer(
        src_cam=src_cam,
        dst_cam=dst_cam,
    )

    transformed_im = cam_transformer.transform(im)

    cv2.imshow('Image', transformed_im)
    cv2.waitKey()
    cv2.destroyAllWindows()

    cam_transformer = CameraTransformer(
        src_cam=dst_cam,
        dst_cam=src_cam,
    )

    transformed_im = cam_transformer.transform(transformed_im)

    cv2.imshow('Image', transformed_im)
    cv2.waitKey()
    cv2.destroyAllWindows()

    rotation = rotation_matrix(axis=(1, 0, 0), theta=-np.pi / 4)
    cam_transformer = CameraTransformer(
        src_cam=src_cam,
        dst_cam=dst_cam,
        rotation=rotation,
    )

    transformed_im = cam_transformer.transform(im)

    cv2.imshow('Image', transformed_im)
    cv2.waitKey()
    cv2.destroyAllWindows()

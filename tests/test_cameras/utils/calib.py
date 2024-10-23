import json
from typing import Any, Dict

import numpy as np
import numpy.typing as npt


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

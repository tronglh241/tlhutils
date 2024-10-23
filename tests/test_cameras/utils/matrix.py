from typing import Any, Tuple

import numpy as np
import numpy.typing as npt


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
